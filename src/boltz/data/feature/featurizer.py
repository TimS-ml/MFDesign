"""Feature extraction module for the Boltz structure prediction model.

This module converts tokenized molecular structures into tensor features suitable
for consumption by the neural network. It handles the full featurization pipeline:

1. **Token-level features**: Residue types (one-hot encoded), chain/entity/symmetry
   identifiers, inter-token bond matrices, pocket conditioning labels, and
   distogram center coordinates. Each "token" corresponds to one residue (for
   polymers) or one atom group (for non-polymers/ligands).

2. **Atom-level features**: Reference conformer positions, element types, charges,
   atom name encodings, ground-truth coordinates, local reference frames (used for
   structure module invariant point attention), pairwise distograms, and the
   mapping tensors that relate atoms to their parent tokens and tokens to their
   representative atoms.

3. **MSA features**: Multiple sequence alignment data paired across chains using
   taxonomy-based matching. Includes one-hot residue identities, deletion values,
   sequence profiles, and paired/unpaired indicators.

4. **Symmetry features**: Equivalent-atom permutation information for amino acids,
   ligands, and symmetric chain copies, used for symmetry-aware loss computation.

The main entry point is the :class:`BoltzFeaturizer` class, whose ``process``
method orchestrates all of the above steps and returns a single feature dictionary.
"""

import math
import random
from typing import Optional

import numpy as np
import torch
from torch import Tensor, from_numpy
from torch.nn.functional import one_hot

from boltz.data import const
from boltz.data.feature.pad import pad_dim
from boltz.data.feature.symmetry import (
    get_amino_acids_symmetries,
    get_chain_symmetries,
    get_ligand_symmetries,
)
from boltz.data.types import (
    MSA,
    MSADeletion,
    MSAResidue,
    MSASequence,
    Tokenized,
)
from boltz.model.modules.utils import center_random_augmentation

####################################################################################################
# HELPERS
####################################################################################################


def compute_frames_nonpolymer(
    data: Tokenized,
    coords,
    resolved_mask,
    atom_to_token,
    frame_data: list,
    resolved_frame_data: list,
) -> tuple[list, list]:
    """Compute local reference frames for non-polymer (ligand) tokens.

    Unlike proteins and nucleotides which have canonical backbone atoms to define
    a local coordinate frame (N/CA/C for proteins, C1'/C3'/C4' for nucleotides),
    non-polymer molecules (small-molecule ligands, ions, etc.) lack predefined
    frame atoms. This function constructs frames by sorting each atom's neighbors
    by Euclidean distance and picking the two nearest resolved neighbors.

    The algorithm for each non-polymer chain:
      1. Compute the all-pairs distance matrix among atoms in the chain.
      2. Penalize unresolved atom pairs by adding infinity to their distances,
         so that resolved atoms are always preferred as frame references.
      3. Sort neighbors by (penalized) distance for each atom.
      4. Construct a 3-atom frame: [nearest neighbor, self, second-nearest neighbor].
         This gives three non-degenerate points to define a local coordinate system.
      5. Mark the frame as resolved only if all three frame atoms are experimentally
         resolved.

    After computing frames for all non-polymer chains, a collinearity check
    filters out degenerate frames where the three atoms are nearly collinear.

    Parameters
    ----------
    data : Tokenized
        The tokenized data containing chain and atom information.
    coords : np.ndarray
        Atom coordinates, shape (1, N_atoms, 3) or similar.
    resolved_mask : np.ndarray
        Boolean mask indicating which atoms have resolved (experimental) positions.
    atom_to_token : list or np.ndarray
        Mapping from each atom index to its parent token index.
    frame_data : list
        Pre-populated frame atom indices for polymer tokens (will be updated
        in-place for non-polymer tokens). Each entry is [atom_a, atom_b, atom_c].
    resolved_frame_data : list
        Pre-populated boolean mask indicating whether each token's frame atoms
        are all resolved (will be updated in-place for non-polymer tokens).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        - frame_data: Array of shape (N_tokens, 3) with global atom indices for
          the three frame-defining atoms per token.
        - resolved_frame_data: Boolean array of shape (N_tokens,) that is True
          only when all three frame atoms are resolved AND non-collinear.

    """
    frame_data = np.array(frame_data)
    resolved_frame_data = np.array(resolved_frame_data)
    asym_id_token = data.tokens["asym_id"]
    asym_id_atom = data.tokens["asym_id"][atom_to_token]
    token_idx = 0
    atom_idx = 0
    for id in np.unique(data.tokens["asym_id"]):
        mask_chain_token = asym_id_token == id
        mask_chain_atom = asym_id_atom == id
        num_tokens = mask_chain_token.sum()
        num_atoms = mask_chain_atom.sum()
        # Skip polymer chains and chains with fewer than 3 atoms (cannot form a frame)
        if (
            data.tokens[token_idx]["mol_type"] != const.chain_type_ids["NONPOLYMER"]
            or num_atoms < 3
        ):
            token_idx += num_tokens
            atom_idx += num_atoms
            continue
        # Step 1: Compute pairwise Euclidean distances among atoms in this chain
        dist_mat = (
            (
                coords.reshape(-1, 3)[mask_chain_atom][:, None, :]
                - coords.reshape(-1, 3)[mask_chain_atom][None, :, :]
            )
            ** 2
        ).sum(-1) ** 0.5
        # Step 2: Penalize unresolved atom pairs with infinite distance so that
        # resolved atoms are always chosen as frame references when available.
        # resolved_pair[i,j] == 0 when both atoms i and j are resolved, 1 otherwise.
        resolved_pair = 1 - (
            resolved_mask[mask_chain_atom][None, :]
            * resolved_mask[mask_chain_atom][:, None]
        ).astype(np.float32)
        resolved_pair[resolved_pair == 1] = math.inf
        # Step 3: Sort by penalized distance; closest resolved neighbors come first
        indices = np.argsort(dist_mat + resolved_pair, axis=1)
        # Step 4: Build frame as [nearest_neighbor, self, second_nearest_neighbor].
        # indices[:,0] is the atom itself (distance 0), indices[:,1] is the nearest
        # neighbor, and indices[:,2] is the second nearest. The frame ordering
        # [1st-nearest, self, 2nd-nearest] is chosen so that the central atom
        # is atom_b (the origin of the local frame).
        frames = (
            np.concatenate(
                [
                    indices[:, 1:2],
                    indices[:, 0:1],
                    indices[:, 2:3],
                ],
                axis=1,
            )
            + atom_idx
        )
        # Step 5: Write frames back into the global arrays and mark as resolved
        # only if all three reference atoms have experimental coordinates
        frame_data[token_idx : token_idx + num_atoms, :] = frames
        resolved_frame_data[token_idx : token_idx + num_atoms] = resolved_mask[
            frames
        ].all(axis=1)
        token_idx += num_tokens
        atom_idx += num_atoms

    # Expand frame indices to actual 3D coordinates for collinearity checking
    frames_expanded = coords.reshape(-1, 3)[frame_data]

    # Filter out degenerate frames where the three atoms are nearly collinear,
    # as collinear points cannot define a proper 2D plane / coordinate frame.
    mask_collinear = compute_collinear_mask(
        frames_expanded[:, 1] - frames_expanded[:, 0],
        frames_expanded[:, 1] - frames_expanded[:, 2],
    )
    return frame_data, resolved_frame_data & mask_collinear


def compute_collinear_mask(v1, v2):
    """Check whether pairs of vectors are approximately collinear.

    A local reference frame requires three non-collinear points. This function
    tests whether the angle between vectors v1 and v2 (typically frame_b->frame_a
    and frame_b->frame_c) is large enough to define a valid plane.

    The check has two parts:
      1. **Angle test**: |cos(angle)| < 0.9063, which corresponds to an angle
         greater than approximately 25 degrees from being perfectly parallel or
         anti-parallel. If the vectors are too close to collinear, the cross
         product used to build the frame would be numerically unstable.
      2. **Overlap test**: Both vectors must have length > 0.01 Angstroms to
         ensure the frame atoms are not overlapping (which would make the
         direction undefined).

    Parameters
    ----------
    v1 : np.ndarray
        First set of vectors, shape (N, 3). Typically (frame_b - frame_a).
    v2 : np.ndarray
        Second set of vectors, shape (N, 3). Typically (frame_b - frame_c).

    Returns
    -------
    np.ndarray
        Boolean mask of shape (N,). True where the vectors are sufficiently
        non-collinear and non-overlapping to form a valid reference frame.
    """
    norm1 = np.linalg.norm(v1, axis=1, keepdims=True)
    norm2 = np.linalg.norm(v2, axis=1, keepdims=True)
    # Normalize to unit vectors for cosine similarity computation
    v1 = v1 / (norm1 + 1e-6)
    v2 = v2 / (norm2 + 1e-6)
    # cos(25 deg) ~ 0.9063; reject frames where the angle is within ~25 deg
    # of being perfectly parallel or anti-parallel
    mask_angle = np.abs(np.sum(v1 * v2, axis=1)) < 0.9063
    # Ensure neither vector is degenerate (atoms nearly overlapping)
    mask_overlap1 = norm1.reshape(-1) > 1e-2
    mask_overlap2 = norm2.reshape(-1) > 1e-2
    return mask_angle & mask_overlap1 & mask_overlap2


def dummy_msa(residues: np.ndarray) -> MSA:
    """Create a single-sequence "dummy" MSA for a chain that lacks alignments.

    Some chains (e.g., ligands, ions, or chains for which no homologous sequences
    were found) do not have a real MSA. To keep the downstream pairing logic
    uniform, we create a trivial MSA containing only the query sequence itself,
    with no deletions. The single sequence entry uses taxonomy=-1 to indicate
    that it has no organism annotation and therefore cannot participate in
    taxonomy-based cross-chain pairing.

    Parameters
    ----------
    residues : np.ndarray
        Structured array of residues for the chain, each containing a ``res_type``
        field with the integer residue identity.

    Returns
    -------
    MSA
        A minimal MSA object with one sequence (the query), zero deletions,
        and no taxonomy information.

    """
    # Extract just the residue type integers for the single query sequence
    residues = [res["res_type"] for res in residues]
    deletions = []
    # MSASequence fields: (seq_idx, taxonomy, res_start, num_residues, del_start, del_end)
    # taxonomy=-1 means "unknown / not applicable"
    sequences = [(0, -1, 0, len(residues), 0, 0)]
    return MSA(
        residues=np.array(residues, dtype=MSAResidue),
        deletions=np.array(deletions, dtype=MSADeletion),
        sequences=np.array(sequences, dtype=MSASequence),
    )


def construct_paired_msa(  # noqa: C901, PLR0915, PLR0912
    data: Tokenized,
    max_seqs: int,
    max_pairs: int = 8192,
    max_total: int = 16384,
    random_subset: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    """Construct a cross-chain paired MSA from per-chain alignments.

    In multi-chain complexes, each chain may have its own MSA from sequence search.
    To model inter-chain co-evolution, we need to "pair" MSA rows across chains so
    that sequences from the same organism end up in the same row. This function
    implements taxonomy-based MSA pairing, following a strategy similar to
    AlphaFold-Multimer:

    **Algorithm overview:**

    1. **Collect per-chain MSAs**: For each chain in the crop, retrieve its MSA.
       Chains without alignments (e.g., ligands) get a single-sequence dummy MSA.

    2. **Build taxonomy map**: Group all MSA sequences by their taxonomy ID
       (organism annotation). Each entry maps a taxonomy ID to a list of
       (chain_id, sequence_index) pairs -- i.e., which sequences from which
       chains belong to that organism.

    3. **Create paired rows** (up to ``max_pairs``):
       - Sort taxonomies by the number of *distinct chains* they cover (most
         chains first), so we prioritize organisms that provide the broadest
         cross-chain coverage.
       - For each taxonomy, create one or more MSA rows. In each row, chains
         that have a sequence from this organism are marked as "paired" and
         assigned that sequence. Chains absent from this taxonomy get their
         next available unpaired sequence (or a gap if exhausted).
       - If a taxonomy has multiple sequences from the same chain (paralogs),
         we create multiple rows, cycling through them with modular indexing
         to maximize sequence diversity.

    4. **Create unpaired rows** (up to ``max_total``):
       - Remaining sequences that were not consumed during pairing are added
         as unpaired rows. Each chain independently contributes its next
         available sequence; chains with no remaining sequences emit a gap.

    5. **Downsample**: If the total exceeds ``max_seqs``, either randomly
       subsample (preserving the query row at index 0) or deterministically
       truncate.

    6. **Materialize**: Convert the pairing plan into actual residue-type and
       deletion tensors by looking up each token's residue in the appropriate
       MSA sequence.

    Parameters
    ----------
    data : Tokenized
        The tokenized structure data, including per-chain MSA objects.
    max_seqs : int
        Maximum number of MSA rows to return after downsampling.
    max_pairs : int
        Maximum number of taxonomy-paired rows to generate (default 8192).
    max_total : int
        Maximum total rows (paired + unpaired) before downsampling (default 16384).
    random_subset : bool
        If True, randomly subsample rows; otherwise truncate deterministically.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        - msa_data: Integer tensor of shape (N_tokens, N_seqs) with residue type
          indices for each token at each MSA row.
        - del_data: Float tensor of shape (N_tokens, N_seqs) with deletion counts.
        - paired_data: Float tensor of shape (N_tokens, N_seqs) indicating which
          entries are taxonomy-paired (1.0) vs unpaired/gap-filled (0.0).

    """
    # -------------------------------------------------------------------------
    # Step 1: Gather per-chain MSAs. Ensure tokens are ordered by chain (asym_id
    # is non-decreasing) so we can iterate chains in a consistent order.
    # -------------------------------------------------------------------------
    assert np.all(np.diff(data.tokens["asym_id"], n=1) >= 0)
    chain_ids = np.unique(data.tokens["asym_id"])

    # Retrieve existing MSAs and create dummy single-sequence MSAs for chains
    # that lack alignments (e.g., ligands, ions, or chains with no homologs).
    msa = {k: data.msa[k] for k in chain_ids if k in data.msa}
    for chain_id in chain_ids:
        if chain_id not in msa:
            chain = data.structure.chains[chain_id]
            res_start = chain["res_idx"]
            res_end = res_start + chain["res_num"]
            residues = data.structure.residues[res_start:res_end]
            msa[chain_id] = dummy_msa(residues)

    # -------------------------------------------------------------------------
    # Step 2: Build the taxonomy map. For each taxonomy ID (organism), collect
    # all (chain_id, seq_idx) pairs that share that organism annotation.
    # Sequences with taxonomy == -1 (unknown) are excluded from pairing.
    # -------------------------------------------------------------------------
    taxonomy_map: dict[str, list] = {}
    for chain_id, chain_msa in msa.items():
        sequences = chain_msa.sequences
        # Filter to sequences with known taxonomy (taxonomy != -1)
        sequences = sequences[sequences["taxonomy"] != -1]
        for sequence in sequences:
            seq_idx = sequence["seq_idx"]
            taxon = sequence["taxonomy"]
            taxonomy_map.setdefault(taxon, []).append((chain_id, seq_idx))

    # Remove singleton taxonomies (present in only one chain's MSA -- cannot pair)
    # and sort by descending chain coverage so that organisms spanning the most
    # chains are paired first, maximizing co-evolutionary signal.
    taxonomy_map = {k: v for k, v in taxonomy_map.items() if len(v) > 1}
    taxonomy_map = sorted(
        taxonomy_map.items(),
        key=lambda x: len({c for c, _ in x[1]}),
        reverse=True,
    )

    # -------------------------------------------------------------------------
    # Step 3: Track which sequences have been assigned to paired rows ("visited")
    # and which remain available for filling gaps or creating unpaired rows.
    # We skip index 0 (the query sequence) since it is always placed in row 0.
    # -------------------------------------------------------------------------
    visited = {(c, s) for c, items in taxonomy_map for s in items}
    available = {}
    for c in chain_ids:
        available[c] = [
            i for i in range(1, len(msa[c].sequences)) if (c, i) not in visited
        ]

    # -------------------------------------------------------------------------
    # Step 4: Build the pairing plan. Each "row" is a dict mapping chain_id to
    # a sequence index (or -1 for gap). is_paired tracks which entries in each
    # row are taxonomy-matched (1) vs gap-filled (0).
    # -------------------------------------------------------------------------
    is_paired = []
    pairing = []

    # Row 0: the query sequence for every chain (always paired, always present)
    is_paired.append({c: 1 for c in chain_ids})
    pairing.append({c: 0 for c in chain_ids})

    # Generate paired rows from taxonomy groups (up to max_pairs rows total).
    # Each taxonomy group may produce multiple rows if a single chain has
    # multiple sequences from that organism (e.g., paralogs).
    for _, pairs in taxonomy_map:
        # Group sequences by chain_id within this taxonomy, since one chain
        # may have multiple homologs from the same organism
        chain_occurences = {}
        for chain_id, seq_idx in pairs:
            chain_occurences.setdefault(chain_id, []).append(seq_idx)

        # Create one row per paralog (cycling with modular indexing for chains
        # that have fewer sequences than the maximum)
        max_occurences = max(len(v) for v in chain_occurences.values())
        for i in range(max_occurences):
            row_pairing = {}
            row_is_paired = {}

            # Assign paired sequences for chains present in this taxonomy
            for chain_id, seq_idxs in chain_occurences.items():
                # Modular indexing: cycle through available sequences to maximize
                # diversity when one chain has fewer paralogs than another
                idx = i % len(seq_idxs)
                seq_idx = seq_idxs[idx]

                row_pairing[chain_id] = seq_idx
                row_is_paired[chain_id] = 1

            # Fill in chains that are NOT present in this taxonomy group.
            # These get the next available unpaired sequence or a gap (-1).
            for chain_id in chain_ids:
                if chain_id not in row_pairing:
                    row_is_paired[chain_id] = 0
                    if available[chain_id]:
                        seq_idx = available[chain_id].pop(0)
                        row_pairing[chain_id] = seq_idx
                    else:
                        # No more sequences available, we place a gap
                        row_pairing[chain_id] = -1

            pairing.append(row_pairing)
            is_paired.append(row_is_paired)

            if len(pairing) >= max_pairs:
                break

        if len(pairing) >= max_pairs:
            break

    # -------------------------------------------------------------------------
    # Step 5: Add unpaired rows from remaining (unconsumed) sequences.
    # These rows have is_paired=0 for all chains and simply drain the leftover
    # sequences in order, filling gaps where a chain has been exhausted.
    # -------------------------------------------------------------------------
    max_left = max(len(v) for v in available.values())
    for _ in range(min(max_total - len(pairing), max_left)):
        row_pairing = {}
        row_is_paired = {}
        for chain_id in chain_ids:
            row_is_paired[chain_id] = 0
            if available[chain_id]:
                seq_idx = available[chain_id].pop(0)
                row_pairing[chain_id] = seq_idx
            else:
                # No more sequences available, we place a gap
                row_pairing[chain_id] = -1

        pairing.append(row_pairing)
        is_paired.append(row_is_paired)

        if len(pairing) >= max_total:
            break

    # -------------------------------------------------------------------------
    # Step 6: Downsample to max_seqs if needed.
    # Row 0 (query) is always kept. Random mode shuffles; deterministic truncates.
    # -------------------------------------------------------------------------
    if random_subset:
        num_seqs = len(pairing)
        if num_seqs > max_seqs:
            indices = np.random.choice(list(range(1, num_seqs)), replace=False)  # noqa: NPY002
            pairing = [pairing[0]] + [pairing[i] for i in indices]
            is_paired = [is_paired[0]] + [is_paired[i] for i in indices]
    else:
        # Deterministic downsample to max_seqs
        pairing = pairing[:max_seqs]
        is_paired = is_paired[:max_seqs]

    # -------------------------------------------------------------------------
    # Step 7: Materialize the pairing plan into tensors. For each token (residue
    # position) and each MSA row, look up the residue type and deletion count
    # from the corresponding chain's MSA.
    # -------------------------------------------------------------------------
    msa_data = []
    del_data = []
    paired_data = []
    gap_token_id = const.token_ids["-"]

    # Pre-build a lookup table: (chain_id, seq_idx, res_idx) -> deletion count
    deletions = {}
    for chain_id, chain_msa in msa.items():
        chain_deletions = chain_msa.deletions
        for sequence in chain_msa.sequences:
            del_start = sequence["del_start"]
            del_end = sequence["del_end"]
            chain_deletions = chain_msa.deletions[del_start:del_end]
            for deletion_data in chain_deletions:
                seq_idx = sequence["seq_idx"]
                res_idx = deletion_data["res_idx"]
                deletion = deletion_data["deletion"]
                deletions[(chain_id, seq_idx, res_idx)] = deletion

    # For each token, build one column of the MSA matrix (across all rows)
    for token in data.tokens:
        token_res_types = []
        token_deletions = []
        token_is_paired = []
        for row_pairing, row_is_paired in zip(pairing, is_paired):
            res_idx = int(token["res_idx"])
            chain_id = int(token["asym_id"])
            seq_idx = row_pairing[chain_id]
            token_is_paired.append(row_is_paired[chain_id])

            # Look up the residue type from the chain's MSA, or emit a gap
            if seq_idx == -1:
                # This chain has no sequence in this row -> gap character
                token_res_types.append(gap_token_id)
                token_deletions.append(0)
            else:
                sequence = msa[chain_id].sequences[seq_idx]
                res_start = sequence["res_start"]
                res_type = msa[chain_id].residues[res_start + res_idx][0]
                # if seq_idx == 0 and res_type == 22: # Maybe unnecessary
                #     res_type = token["res_type"]
                deletion = deletions.get((chain_id, seq_idx, res_idx), 0)
                token_res_types.append(res_type)
                token_deletions.append(deletion)

        msa_data.append(token_res_types)
        del_data.append(token_deletions)
        paired_data.append(token_is_paired)

    # Convert to tensors: shape (N_tokens, N_seqs) for all three
    msa_data = torch.tensor(msa_data, dtype=torch.long)
    del_data = torch.tensor(del_data, dtype=torch.float)
    paired_data = torch.tensor(paired_data, dtype=torch.float)

    return msa_data, del_data, paired_data


####################################################################################################
# FEATURES
####################################################################################################


def select_subset_from_mask(mask, p):
    """Randomly select a geometrically-distributed subset of True entries from a mask.

    Used during pocket conditioning training augmentation to vary the number of
    pocket residues visible to the model. The subset size is drawn from a
    geometric distribution with parameter ``p``, shifted by +1 so the minimum
    is always 2 (at least one pocket residue is always shown). This stochastic
    masking encourages the model to generalize across different pocket sizes.

    Parameters
    ----------
    mask : np.ndarray
        Boolean mask where True indicates a candidate pocket residue.
    p : float
        Parameter of the geometric distribution controlling subset size.
        Smaller p -> larger expected subsets; larger p -> smaller subsets.

    Returns
    -------
    np.ndarray
        A new boolean mask of the same shape with at most ``num_true`` entries
        set to True, where the number selected follows Geometric(p) + 1.
    """
    num_true = np.sum(mask)
    # Draw subset size from geometric distribution (minimum 2, capped at num_true)
    v = np.random.geometric(p) + 1
    k = min(v, num_true)

    true_indices = np.where(mask)[0]

    # Randomly select k indices from the true_indices
    selected_indices = np.random.choice(true_indices, size=k, replace=False)

    new_mask = np.zeros_like(mask)
    new_mask[selected_indices] = 1

    return new_mask


def process_token_features(
    data: Tokenized,
    max_tokens: Optional[int] = None,
    binder_pocket_conditioned_prop: Optional[float] = 0.0,
    binder_pocket_cutoff: Optional[float] = 6.0,
    binder_pocket_sampling_geometric_p: Optional[float] = 0.0,
    only_ligand_binder_pocket: Optional[bool] = False,
    inference_binder: Optional[list[int]] = None,
    inference_pocket: Optional[list[tuple[int, int]]] = None,
) -> dict[str, Tensor]:
    """Extract token-level features from the tokenized structure.

    Each "token" corresponds to one residue in a polymer chain or one atom group
    in a non-polymer/ligand. This function produces:

    - **Identity features**: residue type (one-hot), molecule type, chain/entity/
      symmetry IDs, and sequential token/residue indices used for relative
      positional encoding.
    - **Bond matrix**: A symmetric (N_tokens x N_tokens x 1) tensor indicating
      covalent bonds between tokens (e.g., peptide bonds, disulfide bridges,
      inter-residue ligand bonds).
    - **Pocket conditioning**: A per-token label encoding whether each token is
      part of the "binder" molecule, part of the binding "pocket" on the receptor,
      or neither. This feature enables pocket-conditioned generation, where the
      model is told which molecule to dock and which receptor residues form the
      binding site.
    - **Masks**: Padding mask (1 for real tokens, 0 for padding), resolved mask
      (1 if experimentally resolved), and distogram mask.
    - **Distogram center**: Coordinates used as the center for distogram
      computation at the token level.

    **Pocket conditioning explained:**

    Pocket conditioning assigns one of four labels to each token:
      - UNSPECIFIED (default): No pocket information is provided. Used when
        pocket conditioning is disabled.
      - BINDER: The token belongs to the molecule being docked (the ligand or
        the chain designated as the "binder").
      - POCKET: The token is a receptor residue within the binding pocket
        (within ``binder_pocket_cutoff`` Angstroms of the binder).
      - UNSELECTED: The token is part of the receptor but NOT in the pocket.

    During training, pocket conditioning is stochastically activated with
    probability ``binder_pocket_conditioned_prop``. When activated:
      1. A random ligand (non-polymer) chain is chosen as the binder. If no
         ligands exist and ``only_ligand_binder_pocket`` is False, any chain
         may be chosen.
      2. All polymer residues within ``binder_pocket_cutoff`` Angstroms of any
         binder atom are labeled POCKET. Optionally, a random subset of pocket
         residues is selected (geometric distribution) to augment training.
      3. All remaining residues are labeled UNSELECTED.

    During inference, the binder and pocket are specified explicitly via
    ``inference_binder`` and ``inference_pocket``.

    Parameters
    ----------
    data : Tokenized
        The tokenized data.
    max_tokens : int, optional
        If provided, pad all token-dimension tensors to this length.
    binder_pocket_conditioned_prop : float
        Probability of activating pocket conditioning during training.
    binder_pocket_cutoff : float
        Distance threshold (Angstroms) for defining pocket residues.
    binder_pocket_sampling_geometric_p : float
        Geometric distribution parameter for stochastic pocket subset selection.
        0.0 disables subset sampling (all pocket residues are kept).
    only_ligand_binder_pocket : bool
        If True, only non-polymer (ligand) chains can be chosen as the binder.
    inference_binder : list[int], optional
        Chain asym_ids designated as the binder during inference.
    inference_pocket : list[tuple[int, int]], optional
        List of (asym_id, res_idx) tuples defining pocket residues during inference.

    Returns
    -------
    dict[str, Tensor]
        Dictionary of token-level feature tensors.

    """
    # Token data
    token_data = data.tokens
    token_bonds = data.bonds

    # -------------------------------------------------------------------------
    # Core identity features: these identify each token and provide the model
    # with residue-level information for embedding and relative position encoding.
    # -------------------------------------------------------------------------
    token_index = torch.arange(len(token_data), dtype=torch.long)
    residue_index = from_numpy(token_data["res_idx"]).long()   # Position within chain
    asym_id = from_numpy(token_data["asym_id"]).long()         # Chain identifier
    entity_id = from_numpy(token_data["entity_id"]).long()     # Entity (unique sequence) ID
    sym_id = from_numpy(token_data["sym_id"]).long()           # Symmetry copy index
    mol_type = from_numpy(token_data["mol_type"]).long()       # Molecule type (protein/DNA/RNA/ligand)
    res_type = from_numpy(token_data["res_type"]).long()       # Integer residue type
    res_type = one_hot(res_type, num_classes=const.num_tokens) # One-hot encode residue type
    disto_center = from_numpy(token_data["disto_coords"])      # Center coords for distogram

    # -------------------------------------------------------------------------
    # Mask features
    # -------------------------------------------------------------------------
    pad_mask = torch.ones(len(token_data), dtype=torch.float)           # 1 = real token, 0 = padding
    resolved_mask = from_numpy(token_data["resolved_mask"]).float()     # 1 = experimentally resolved
    disto_mask = from_numpy(token_data["disto_mask"]).float()           # 1 = distogram data available

    # -------------------------------------------------------------------------
    # Inter-token bond matrix: a symmetric (N_tokens x N_tokens) adjacency matrix
    # encoding covalent connectivity between tokens (e.g., backbone peptide bonds,
    # disulfide bridges, ligand inter-group bonds). Expanded to shape
    # (N_tokens, N_tokens, 1) for use as a pair feature.
    # -------------------------------------------------------------------------
    if max_tokens is not None:
        pad_len = max_tokens - len(token_data)
        num_tokens = max_tokens if pad_len > 0 else len(token_data)
    else:
        num_tokens = len(token_data)

    tok_to_idx = {tok["token_idx"]: idx for idx, tok in enumerate(token_data)}
    bonds = torch.zeros(num_tokens, num_tokens, dtype=torch.float)
    for token_bond in token_bonds:
        token_1 = tok_to_idx[token_bond["token_1"]]
        token_2 = tok_to_idx[token_bond["token_2"]]
        bonds[token_1, token_2] = 1
        bonds[token_2, token_1] = 1

    bonds = bonds.unsqueeze(-1)  # Shape: (N_tokens, N_tokens, 1)

    # -------------------------------------------------------------------------
    # Pocket conditioning feature.
    #
    # This feature tells the model which tokens are the "binder" (molecule to
    # dock), which are "pocket" residues (receptor residues near the binder),
    # which are "unselected" (receptor residues far from the binder), and which
    # are "unspecified" (pocket conditioning not active).
    #
    # Three modes:
    #   1. Inference mode (inference_binder is not None): use explicit labels.
    #   2. Training with stochastic activation: randomly pick a binder chain
    #      and compute pocket from distance threshold.
    #   3. Disabled (default): all tokens get UNSPECIFIED.
    # -------------------------------------------------------------------------
    pocket_feature = (
        np.zeros(len(token_data)) + const.pocket_contact_info["UNSPECIFIED"]
    )
    if inference_binder is not None:
        # Inference mode: binder and pocket residues are explicitly specified
        assert inference_pocket is not None
        pocket_residues = set(inference_pocket)
        for idx, token in enumerate(token_data):
            if token["asym_id"] in inference_binder:
                pocket_feature[idx] = const.pocket_contact_info["BINDER"]
            elif (token["asym_id"], token["res_idx"]) in pocket_residues:
                pocket_feature[idx] = const.pocket_contact_info["POCKET"]
            else:
                pocket_feature[idx] = const.pocket_contact_info["UNSELECTED"]
    elif (
        binder_pocket_conditioned_prop > 0.0
        and random.random() < binder_pocket_conditioned_prop
    ):
        # Training mode: stochastically activate pocket conditioning.
        # Step A: Choose a binder chain. Prefer ligands (non-polymer chains);
        # fall back to any chain if no ligands exist and only_ligand_binder_pocket
        # is False.
        binder_asym_ids = np.unique(
            token_data["asym_id"][
                token_data["mol_type"] == const.chain_type_ids["NONPOLYMER"]
            ]
        )

        if len(binder_asym_ids) == 0:
            if not only_ligand_binder_pocket:
                binder_asym_ids = np.unique(token_data["asym_id"])

        if len(binder_asym_ids) > 0:
            # Randomly select one chain as the binder
            pocket_asym_id = random.choice(binder_asym_ids)
            binder_mask = token_data["asym_id"] == pocket_asym_id

            # Step B: Collect all atom coordinates of the binder chain
            binder_coords = []
            for token in token_data:
                if token["asym_id"] == pocket_asym_id:
                    binder_coords.append(
                        data.structure.atoms["coords"][
                            token["atom_idx"] : token["atom_idx"] + token["atom_num"]
                        ]
                    )
            binder_coords = np.concatenate(binder_coords, axis=0)

            # Step C: For each non-binder polymer token, compute the minimum
            # distance between its atoms and any binder atom. Tokens closer
            # than binder_pocket_cutoff are labeled as pocket residues.
            token_dist = np.zeros(len(token_data)) + 1000
            for i, token in enumerate(token_data):
                if (
                     token["mol_type"] != const.chain_type_ids["NONPOLYMER"]
                    and token["asym_id"] != pocket_asym_id
                    and token["resolved_mask"] == 1
                ):
                    token_coords = data.structure.atoms["coords"][
                        token["atom_idx"] : token["atom_idx"] + token["atom_num"]
                    ]

                    # find chain and apply chain transformation
                    for chain in data.structure.chains:
                        if chain["asym_id"] == token["asym_id"]:
                            break

                    # Minimum atom-to-atom distance between this token and binder
                    token_dist[i] = np.min(
                        np.linalg.norm(
                            token_coords[:, None, :] - binder_coords[None, :, :],
                            axis=-1,
                        )
                    )

            pocket_mask = token_dist < binder_pocket_cutoff

            # Step D: If any pocket residues were found, assign labels
            if np.sum(pocket_mask) > 0:
                pocket_feature = (
                    np.zeros(len(token_data)) + const.pocket_contact_info["UNSELECTED"]
                )
                pocket_feature[binder_mask] = const.pocket_contact_info["BINDER"]

                # Optionally subsample the pocket residues for augmentation
                if binder_pocket_sampling_geometric_p > 0.0:
                    pocket_mask = select_subset_from_mask(
                        pocket_mask, binder_pocket_sampling_geometric_p
                    )

                pocket_feature[pocket_mask] = const.pocket_contact_info["POCKET"]

    # One-hot encode the pocket labels (UNSPECIFIED / BINDER / POCKET / UNSELECTED)
    pocket_feature = from_numpy(pocket_feature).long()
    pocket_feature = one_hot(pocket_feature, num_classes=len(const.pocket_contact_info))

    # -------------------------------------------------------------------------
    # Pad all features to max_tokens if specified (for batched training)
    # -------------------------------------------------------------------------
    if max_tokens is not None:
        pad_len = max_tokens - len(token_data)
        if pad_len > 0:
            token_index = pad_dim(token_index, 0, pad_len)
            residue_index = pad_dim(residue_index, 0, pad_len)
            asym_id = pad_dim(asym_id, 0, pad_len)
            entity_id = pad_dim(entity_id, 0, pad_len)
            sym_id = pad_dim(sym_id, 0, pad_len)
            mol_type = pad_dim(mol_type, 0, pad_len)
            res_type = pad_dim(res_type, 0, pad_len)
            disto_center = pad_dim(disto_center, 0, pad_len)
            pad_mask = pad_dim(pad_mask, 0, pad_len)
            resolved_mask = pad_dim(resolved_mask, 0, pad_len)
            disto_mask = pad_dim(disto_mask, 0, pad_len)
            pocket_feature = pad_dim(pocket_feature, 0, pad_len)

    # Assemble the final token feature dictionary
    token_features = {
        "token_index": token_index,           # Sequential index: 0, 1, 2, ...
        "residue_index": residue_index,       # Position within chain (for relative PE)
        "asym_id": asym_id,                   # Chain identifier (for inter-chain masking)
        "entity_id": entity_id,               # Entity ID (chains with same sequence share this)
        "sym_id": sym_id,                     # Symmetry copy index
        "mol_type": mol_type,                 # Molecule type (protein/DNA/RNA/ligand)
        "res_type": res_type,                 # One-hot residue type (N_tokens, num_tokens)
        "disto_center": disto_center,         # Center coordinates for distogram
        "token_bonds": bonds,                 # Inter-token bond adjacency (N_tokens, N_tokens, 1)
        "token_pad_mask": pad_mask,           # Padding mask (1=real, 0=pad)
        "token_resolved_mask": resolved_mask, # Experimentally resolved mask
        "token_disto_mask": disto_mask,       # Distogram data availability mask
        "pocket_feature": pocket_feature,     # Pocket conditioning one-hot labels
    }
    return token_features


def process_atom_features(
    data: Tokenized,
    atoms_per_window_queries: int = 32,
    min_dist: float = 2.0,
    max_dist: float = 22.0,
    num_bins: int = 64,
    max_atoms: Optional[int] = None,
    max_tokens: Optional[int] = None,
) -> dict[str, Tensor]:
    """Extract atom-level features from the tokenized structure.

    This function converts the atom-level data (coordinates, element types, charges,
    atom names, conformer reference positions) into tensors for the neural network.
    It also constructs the critical mapping tensors that bridge the two-level
    (token / atom) hierarchy in the model:

    **Atom-to-token mapping (atom_to_token):**
      Each atom belongs to exactly one token. ``atom_to_token`` is a one-hot matrix
      of shape (N_atoms, N_tokens) that maps every atom to its parent token. This
      is used by the model to aggregate atom-level representations into token-level
      ones (e.g., via weighted sum or attention pooling).

    **Token-to-representative-atom mapping (token_to_rep_atom):**
      Each token has one designated "representative" atom used for the distogram
      (inter-token distance prediction). For proteins this is typically the CA
      atom; for nucleotides it may be C1'. ``token_to_rep_atom`` is a one-hot
      matrix of shape (N_tokens, N_atoms) that selects this atom. It is used to
      project token-level representations back to specific atom positions.

    **Resolved-set representative atoms (r_set_to_rep_atom):**
      A subset of token_to_rep_atom restricted to resolved polymer tokens. This
      is used to compute the center of mass for coordinate centering, excluding
      ligands and unresolved residues.

    **Reference frames:**
      Each token has a local coordinate frame defined by three atoms. These frames
      are used by the structure module for invariant point attention (IPA):
        - **Proteins**: N, CA, C backbone atoms define the frame. CA is the origin
          (frame_b), N defines one axis (frame_a), C defines the plane (frame_c).
        - **Nucleotides (DNA/RNA)**: C1', C3', C4' sugar ring atoms define the
          frame, analogous to the protein backbone frame.
        - **Non-polymers (ligands)**: No canonical frame atoms exist; frames are
          computed by ``compute_frames_nonpolymer`` using nearest-neighbor sorting.
        - **Unknown/padding/small tokens**: Frame is degenerate (all indices = 0)
          and marked as unresolved.

    **Distogram:**
      A pairwise distance histogram between representative atoms of all tokens,
      discretized into ``num_bins`` bins between ``min_dist`` and ``max_dist``
      Angstroms. Used as a training target for the distogram head.

    Parameters
    ----------
    data : Tokenized
        The tokenized data containing atoms, tokens, and structure information.
    atoms_per_window_queries : int
        Window size for atom attention. Atom count is padded to a multiple of this.
    min_dist : float
        Minimum distance (Angstroms) for distogram binning.
    max_dist : float
        Maximum distance (Angstroms) for distogram binning.
    num_bins : int
        Number of distance bins in the distogram.
    max_atoms : int, optional
        If provided, pad atom-dimension tensors to this length.
    max_tokens : int, optional
        If provided, pad token-dimension tensors to this length.

    Returns
    -------
    dict[str, Tensor]
        Dictionary of atom-level feature tensors including reference positions,
        element types, coordinates, mapping matrices, frames, and distograms.

    """
    # -------------------------------------------------------------------------
    # Iterate over tokens and collect per-atom data, building the atom-to-token
    # and token-to-representative-atom mappings along the way.
    # -------------------------------------------------------------------------
    atom_data = []
    ref_space_uid = []       # Unique ID for each residue's reference space (for grouping atoms)
    coord_data = []
    frame_data = []          # Per-token frame: [atom_a_idx, atom_b_idx, atom_c_idx]
    resolved_frame_data = [] # Per-token bool: True if all frame atoms are resolved
    atom_to_token = []       # Maps each atom index -> its parent token index
    token_to_rep_atom = []   # Maps each token -> its representative (distogram) atom index
    r_set_to_rep_atom = []   # Like token_to_rep_atom but only for resolved polymer tokens
    disto_coords = []        # Coordinates of each token's representative atom (for distogram)
    atom_idx = 0             # Running global atom index

    # Assign a unique reference-space ID to each (chain, residue) pair.
    # Atoms sharing the same ref_space_uid are in the same local coordinate frame
    # (e.g., all atoms of a single amino acid residue share one reference space).
    chain_res_ids = {}
    for token_id, token in enumerate(data.tokens):
        chain_idx, res_id = token["asym_id"], token["res_idx"]
        chain = data.structure.chains[chain_idx]

        if (chain_idx, res_id) not in chain_res_ids:
            new_idx = len(chain_res_ids)
            chain_res_ids[(chain_idx, res_id)] = new_idx
        else:
            new_idx = chain_res_ids[(chain_idx, res_id)]

        # Every atom in this token gets the same reference space UID and token index
        ref_space_uid.extend([new_idx] * token["atom_num"])
        atom_to_token.extend([token_id] * token["atom_num"])

        # Retrieve the atoms belonging to this token from the structure
        start = token["atom_idx"]
        end = token["atom_idx"] + token["atom_num"]
        token_atoms = data.structure.atoms[start:end]

        # ---- Representative atom mapping ----
        # token["disto_idx"] is the global atom index of this token's representative
        # atom (used for distogram). Convert to local (cropped) atom index.
        token_to_rep_atom.append(atom_idx + token["disto_idx"] - start)
        # For the resolved-set mapping, only include resolved polymer tokens.
        # token["center_idx"] points to the atom used for center-of-mass computation.
        if (chain["mol_type"] != const.chain_type_ids["NONPOLYMER"]) and token[
            "resolved_mask"
        ]:
            r_set_to_rep_atom.append(atom_idx + token["center_idx"] - start)

        # Get token coordinates (shape: 1 x num_atoms_in_token x 3)
        token_coords = np.array([token_atoms["coords"]])
        coord_data.append(token_coords)

        # ---- Local reference frame construction ----
        # The frame is defined by three atoms (a, b, c) where b is the origin,
        # (b-a) defines one axis, and (b-c) helps define the plane.
        res_type = const.tokens[token["res_type"]]

        if token["atom_num"] < 3 or res_type in ["PAD", "UNK", "-"]:
            # Cannot form a valid frame: too few atoms or unknown residue type.
            # Use degenerate frame (all indices point to first atom) and mark invalid.
            idx_frame_a, idx_frame_b, idx_frame_c = 0, 0, 0
            mask_frame = False
        elif (token["mol_type"] == const.chain_type_ids["PROTEIN"]) and (
            res_type in const.ref_atoms
        ):
            # PROTEIN FRAME: Use backbone atoms N, CA, C.
            # CA (alpha-carbon) is the frame origin (b), N defines one arm (a),
            # and C (carbonyl carbon) defines the second arm (c). Together they
            # specify the peptide plane orientation for this residue.
            idx_frame_a, idx_frame_b, idx_frame_c = (
                const.ref_atoms[res_type].index("N"),
                const.ref_atoms[res_type].index("CA"),
                const.ref_atoms[res_type].index("C"),
            )
            # Frame is valid only if all three backbone atoms are experimentally resolved
            mask_frame = (
                token_atoms["is_present"][idx_frame_a]
                and token_atoms["is_present"][idx_frame_b]
                and token_atoms["is_present"][idx_frame_c]
            )
        elif (
            token["mol_type"] == const.chain_type_ids["DNA"]
            or token["mol_type"] == const.chain_type_ids["RNA"]
        ) and (res_type in const.ref_atoms):
            # NUCLEOTIDE FRAME: Use sugar ring atoms C1', C3', C4'.
            # These three atoms on the deoxyribose/ribose ring define the
            # nucleotide's local orientation. C3' is the frame origin (b),
            # C1' defines one arm (a), and C4' defines the second arm (c).
            # This is analogous to the N/CA/C frame for proteins but adapted
            # to the nucleic acid backbone geometry.
            idx_frame_a, idx_frame_b, idx_frame_c = (
                const.ref_atoms[res_type].index("C1'"),
                const.ref_atoms[res_type].index("C3'"),
                const.ref_atoms[res_type].index("C4'"),
            )
            mask_frame = (
                token_atoms["is_present"][idx_frame_a]
                and token_atoms["is_present"][idx_frame_b]
                and token_atoms["is_present"][idx_frame_c]
            )
        else:
            # Unrecognized molecule type or residue: degenerate frame.
            # Non-polymer frames will be overwritten by compute_frames_nonpolymer() below.
            idx_frame_a, idx_frame_b, idx_frame_c = 0, 0, 0
            mask_frame = False

        # Store frame indices as global atom indices (offset by atom_idx)
        frame_data.append(
            [idx_frame_a + atom_idx, idx_frame_b + atom_idx, idx_frame_c + atom_idx]
        )
        resolved_frame_data.append(mask_frame)

        # Get distogram coordinates (representative atom position for this token)
        disto_coords_tok = data.structure.atoms[token["disto_idx"]]["coords"]
        disto_coords.append(disto_coords_tok)

        # Update atom data. This is technically never used again (we rely on coord_data),
        # but we update for consistency and to make sure the Atom object has valid, transformed coordinates.
        token_atoms = token_atoms.copy()
        token_atoms["coords"] = token_coords[0]  # atom has a copy of first coords
        atom_data.append(token_atoms)
        atom_idx += len(token_atoms)

    disto_coords = np.array(disto_coords)

    # -------------------------------------------------------------------------
    # Distogram computation: pairwise distance histogram between representative
    # atoms of all tokens, discretized into num_bins uniform bins from min_dist
    # to max_dist. This serves as a training target for the distogram head,
    # which predicts inter-residue distance distributions.
    # -------------------------------------------------------------------------
    t_center = torch.Tensor(disto_coords)
    t_dists = torch.cdist(t_center, t_center)
    boundaries = torch.linspace(min_dist, max_dist, num_bins - 1)
    # Bin assignment: count how many boundaries each distance exceeds
    distogram = (t_dists.unsqueeze(-1) > boundaries).sum(dim=-1).long()
    disto_target = one_hot(distogram, num_classes=num_bins)

    # Concatenate per-token arrays into global arrays
    atom_data = np.concatenate(atom_data)
    coord_data = np.concatenate(coord_data, axis=1)
    ref_space_uid = np.array(ref_space_uid)

    # -------------------------------------------------------------------------
    # Convert raw atom properties to tensor features
    # -------------------------------------------------------------------------
    ref_atom_name_chars = from_numpy(atom_data["name"]).long()   # Atom name character codes
    ref_element = from_numpy(atom_data["element"]).long()        # Element type (integer)
    ref_charge = from_numpy(atom_data["charge"])                 # Formal charge
    ref_pos = from_numpy(
        atom_data["conformer"].copy()                            # Reference conformer positions
    )  # not sure why I need to copy here..
    ref_space_uid = from_numpy(ref_space_uid)                    # Reference space group IDs
    coords = from_numpy(coord_data.copy())                       # Ground-truth coordinates
    resolved_mask = from_numpy(atom_data["is_present"])          # 1 = atom experimentally resolved
    pad_mask = torch.ones(len(atom_data), dtype=torch.float)     # 1 = real atom, 0 = padding
    atom_to_token = torch.tensor(atom_to_token, dtype=torch.long)
    token_to_rep_atom = torch.tensor(token_to_rep_atom, dtype=torch.long)
    r_set_to_rep_atom = torch.tensor(r_set_to_rep_atom, dtype=torch.long)

    # Compute reference frames for non-polymer (ligand) tokens using
    # nearest-neighbor distance sorting, since they lack canonical frame atoms.
    # Polymer frames (set above) are preserved; only non-polymer entries are overwritten.
    frame_data, resolved_frame_data = compute_frames_nonpolymer(
        data,
        coord_data,
        atom_data["is_present"],
        atom_to_token,
        frame_data,
        resolved_frame_data,
    )
    frames = from_numpy(frame_data.copy())
    frame_resolved_mask = from_numpy(resolved_frame_data.copy())

    # -------------------------------------------------------------------------
    # One-hot encoding of atom features and mapping matrices.
    # After one-hot encoding, atom_to_token becomes (N_atoms, N_tokens) and
    # token_to_rep_atom becomes (N_tokens, N_atoms), enabling differentiable
    # aggregation / projection between the atom and token levels via matrix
    # multiplication.
    # -------------------------------------------------------------------------
    ref_atom_name_chars = one_hot(
        ref_atom_name_chars % num_bins, num_classes=num_bins
    )  # Modulo handles lower-case letters that exceed the bin count
    ref_element = one_hot(ref_element, num_classes=const.num_elements)
    atom_to_token = one_hot(atom_to_token, num_classes=token_id + 1)
    token_to_rep_atom = one_hot(token_to_rep_atom, num_classes=len(atom_data))
    r_set_to_rep_atom = one_hot(r_set_to_rep_atom, num_classes=len(atom_data))

    # -------------------------------------------------------------------------
    # Coordinate centering: subtract the center of mass (over resolved atoms)
    # from ground-truth coordinates so the structure is origin-centered.
    # -------------------------------------------------------------------------
    center = (coords * resolved_mask[None, :, None]).sum(dim=1)
    center = center / resolved_mask.sum().clamp(min=1)
    coords = coords - center[:, None]

    # Apply a random rotation and translation to the reference conformer
    # positions. This data augmentation ensures the model does not overfit to
    # a particular orientation of the input reference structure.
    ref_pos = center_random_augmentation(
        ref_pos[None], resolved_mask[None], centering=False
    )[0]

    # -------------------------------------------------------------------------
    # Padding: pad atom-dimension tensors to either max_atoms or the next
    # multiple of atoms_per_window_queries (for efficient windowed attention).
    # -------------------------------------------------------------------------
    if max_atoms is not None:
        assert max_atoms % atoms_per_window_queries == 0
        pad_len = max_atoms - len(atom_data)
    else:
        # Round up to the next multiple of atoms_per_window_queries
        pad_len = (
            (len(atom_data) - 1) // atoms_per_window_queries + 1
        ) * atoms_per_window_queries - len(atom_data)

    if pad_len > 0:
        pad_mask = pad_dim(pad_mask, 0, pad_len)
        ref_pos = pad_dim(ref_pos, 0, pad_len)
        resolved_mask = pad_dim(resolved_mask, 0, pad_len)
        ref_element = pad_dim(ref_element, 0, pad_len)
        ref_charge = pad_dim(ref_charge, 0, pad_len)
        ref_atom_name_chars = pad_dim(ref_atom_name_chars, 0, pad_len)
        ref_space_uid = pad_dim(ref_space_uid, 0, pad_len)
        coords = pad_dim(coords, 1, pad_len)
        atom_to_token = pad_dim(atom_to_token, 0, pad_len)
        token_to_rep_atom = pad_dim(token_to_rep_atom, 1, pad_len)
        r_set_to_rep_atom = pad_dim(r_set_to_rep_atom, 1, pad_len)

    # Pad token-dimension of cross-mapping tensors and token-indexed features
    if max_tokens is not None:
        pad_len = max_tokens - token_to_rep_atom.shape[0]
        if pad_len > 0:
            atom_to_token = pad_dim(atom_to_token, 1, pad_len)
            token_to_rep_atom = pad_dim(token_to_rep_atom, 0, pad_len)
            r_set_to_rep_atom = pad_dim(r_set_to_rep_atom, 0, pad_len)
            disto_target = pad_dim(pad_dim(disto_target, 0, pad_len), 1, pad_len)
            frames = pad_dim(frames, 0, pad_len)
            frame_resolved_mask = pad_dim(frame_resolved_mask, 0, pad_len)

    return {
        "ref_pos": ref_pos,                       # Reference conformer positions (N_atoms, 3)
        "atom_resolved_mask": resolved_mask,       # Which atoms are experimentally resolved (N_atoms,)
        "ref_element": ref_element,                # One-hot element type (N_atoms, num_elements)
        "ref_charge": ref_charge,                  # Formal charge per atom (N_atoms,)
        "ref_atom_name_chars": ref_atom_name_chars,# One-hot atom name characters (N_atoms, 4, num_bins)
        "ref_space_uid": ref_space_uid,            # Reference space group ID per atom (N_atoms,)
        "coords": coords,                         # Ground-truth coordinates, centered (1, N_atoms, 3)
        "atom_pad_mask": pad_mask,                 # Padding mask (N_atoms,): 1=real, 0=pad
        "atom_to_token": atom_to_token,            # One-hot atom->token mapping (N_atoms, N_tokens)
        "token_to_rep_atom": token_to_rep_atom,    # One-hot token->representative atom (N_tokens, N_atoms)
        "r_set_to_rep_atom": r_set_to_rep_atom,    # Resolved polymer token->atom (N_resolved, N_atoms)
        "disto_target": disto_target,              # Distogram target (N_tokens, N_tokens, num_bins)
        "frames_idx": frames,                      # Frame atom indices (N_tokens, 3)
        "frame_resolved_mask": frame_resolved_mask,# Frame validity mask (N_tokens,)
    }


def process_msa_features(
    data: Tokenized,
    max_seqs_batch: int,
    max_seqs: int,
    max_tokens: Optional[int] = None,
    pad_to_max_seqs: bool = False,
) -> dict[str, Tensor]:
    """Prepare MSA features for the model from the paired cross-chain alignment.

    This function:
      1. Calls ``construct_paired_msa`` to produce the taxonomy-paired MSA matrix
         (combining paired and unpaired rows across all chains).
      2. One-hot encodes residue identities in the MSA (num_tokens classes).
      3. Computes a **sequence profile**: the column-wise average of the one-hot
         MSA, giving a soft distribution over residue types at each position.
      4. Processes deletion information: raw deletion counts are transformed via
         ``(pi/2) * arctan(deletion / 3)`` to compress large values into [0, pi/2],
         and a binary ``has_deletion`` indicator is computed.
      5. Computes ``deletion_mean``: the column-wise mean of transformed deletions.
      6. Pads tensors along both the MSA (sequence) and token (residue) dimensions.

    Parameters
    ----------
    data : Tokenized
        The tokenized data containing per-chain MSAs.
    max_seqs_batch : int
        Maximum number of MSA sequences for this batch (may be randomly sampled
        during training to vary MSA depth).
    max_seqs : int
        Absolute maximum number of MSA sequences (for padding dimension).
    max_tokens : int, optional
        If provided, pad token dimension to this length.
    pad_to_max_seqs : bool
        If True, pad the MSA dimension to exactly ``max_seqs`` rows.

    Returns
    -------
    dict[str, Tensor]
        Dictionary containing:
        - msa: One-hot MSA residues, shape (N_seqs, N_tokens, num_tokens)
        - msa_paired: Pairing indicator, shape (N_seqs, N_tokens)
        - deletion_value: Transformed deletion values, shape (N_seqs, N_tokens)
        - has_deletion: Binary deletion indicator, shape (N_seqs, N_tokens)
        - deletion_mean: Mean deletion per position, shape (N_tokens,)
        - profile: Sequence profile (mean one-hot), shape (N_tokens, num_tokens)
        - msa_mask: Mask for valid MSA entries, shape (N_seqs, N_tokens)

    """
    # Build the cross-chain paired MSA (taxonomy-matched + unpaired rows)
    msa, deletion, paired = construct_paired_msa(data, max_seqs_batch)
    # Transpose from (N_tokens, N_seqs) to (N_seqs, N_tokens) for standard
    # MSA layout where dim 0 is the sequence axis and dim 1 is the position axis
    msa, deletion, paired = (
        msa.transpose(1, 0),
        deletion.transpose(1, 0),
        paired.transpose(1, 0),
    )

    # One-hot encode residue types in the MSA (including gap character)
    msa = torch.nn.functional.one_hot(msa, num_classes=const.num_tokens)
    # MSA mask: 1 for all real MSA entries (before padding)
    msa_mask = torch.ones_like(msa[:, :, 0])
    # Sequence profile: column-wise mean of one-hot MSA, giving the frequency
    # of each residue type at each position across all MSA sequences
    profile = msa.float().mean(dim=0)
    # Deletion features: binary indicator and compressed continuous value.
    # The arctan transformation maps raw deletion counts (potentially large
    # integers) to a bounded range [0, pi/2], preventing numerical issues.
    has_deletion = deletion > 0
    deletion = np.pi / 2 * np.arctan(deletion / 3)
    deletion_mean = deletion.mean(axis=0)

    # Pad along MSA sequence dimension (dim=0) to fixed size for batching
    if pad_to_max_seqs:
        pad_len = max_seqs - msa.shape[0]
        if pad_len > 0:
            msa = pad_dim(msa, 0, pad_len, const.token_ids["-"])
            paired = pad_dim(paired, 0, pad_len)
            msa_mask = pad_dim(msa_mask, 0, pad_len)
            has_deletion = pad_dim(has_deletion, 0, pad_len)
            deletion = pad_dim(deletion, 0, pad_len)

    # Pad along token/position dimension (dim=1) to fixed size for batching
    if max_tokens is not None:
        pad_len = max_tokens - msa.shape[1]
        if pad_len > 0:
            msa = pad_dim(msa, 1, pad_len, const.token_ids["-"])
            paired = pad_dim(paired, 1, pad_len)
            msa_mask = pad_dim(msa_mask, 1, pad_len)
            has_deletion = pad_dim(has_deletion, 1, pad_len)
            deletion = pad_dim(deletion, 1, pad_len)
            profile = pad_dim(profile, 0, pad_len)
            deletion_mean = pad_dim(deletion_mean, 0, pad_len)

    return {
        "msa": msa,                   # One-hot MSA (N_seqs, N_tokens, num_tokens)
        "msa_paired": paired,          # Pairing indicator per entry (N_seqs, N_tokens)
        "deletion_value": deletion,    # Compressed deletion values (N_seqs, N_tokens)
        "has_deletion": has_deletion,  # Binary deletion indicator (N_seqs, N_tokens)
        "deletion_mean": deletion_mean,# Mean deletion per position (N_tokens,)
        "profile": profile,           # Sequence profile (N_tokens, num_tokens)
        "msa_mask": msa_mask,          # Valid-entry mask (N_seqs, N_tokens)
    }


def process_symmetry_features(
    cropped: Tokenized, symmetries: dict
) -> dict[str, Tensor]:
    """Compute symmetry-related features for symmetry-aware loss computation.

    Molecular structures often contain symmetric elements that are physically
    indistinguishable. This function collects three types of symmetry information:

    1. **Chain symmetries**: Permutations of identical chains (e.g., homodimer
       subunits A and B can be swapped without changing the structure).
    2. **Amino acid symmetries**: Symmetric side-chain atom swaps within
       individual residues (e.g., the two carboxyl oxygens of Asp/Glu, or the
       ring nitrogens of Phe/Tyr).
    3. **Ligand symmetries**: Atom permutations within small-molecule ligands
       that produce chemically equivalent configurations.

    These symmetry features enable the model's loss function to consider all
    equivalent atom orderings and pick the one with lowest RMSD, avoiding
    penalizing correct predictions that happen to use a different but equivalent
    atom labeling.

    Parameters
    ----------
    cropped : Tokenized
        The (possibly cropped) tokenized data.
    symmetries : dict
        Pre-computed symmetry information for ligands.

    Returns
    -------
    dict[str, Tensor]
        Dictionary of symmetry feature tensors (permutation matrices, etc.).

    """
    features = get_chain_symmetries(cropped)
    features.update(get_amino_acids_symmetries(cropped))
    features.update(get_ligand_symmetries(cropped, symmetries))

    return features


class BoltzFeaturizer:
    """Main featurizer class that orchestrates all feature computation.

    This class provides the ``process`` method which is the single entry point
    for converting a tokenized molecular structure into the complete set of
    tensor features consumed by the Boltz neural network. It delegates to the
    individual feature-processing functions defined in this module.
    """

    def process(
        self,
        data: Tokenized,
        training: bool,
        max_seqs: int = 4096,
        atoms_per_window_queries: int = 32,
        min_dist: float = 2.0,
        max_dist: float = 22.0,
        num_bins: int = 64,
        max_tokens: Optional[int] = None,
        max_atoms: Optional[int] = None,
        pad_to_max_seqs: bool = False,
        compute_symmetries: bool = False,
        symmetries: Optional[dict] = None,
        binder_pocket_conditioned_prop: Optional[float] = 0.0,
        binder_pocket_cutoff: Optional[float] = 6.0,
        binder_pocket_sampling_geometric_p: Optional[float] = 0.0,
        only_ligand_binder_pocket: Optional[bool] = False,
        inference_binder: Optional[int] = None,
        inference_pocket: Optional[list[tuple[int, int]]] = None,
    ) -> dict[str, Tensor]:
        """Compute all features for the Boltz model from tokenized structure data.

        This is the main featurization pipeline. It executes four stages in order
        and merges their outputs into a single feature dictionary:

        **Stage 1 -- Token features** (``process_token_features``):
          Extracts residue-level identities (one-hot residue type, molecule type),
          structural metadata (chain/entity/symmetry IDs, residue indices),
          inter-token bond adjacency, pocket conditioning labels, and masks.
          These features feed into the trunk's token-level (pair + single)
          representations.

        **Stage 2 -- Atom features** (``process_atom_features``):
          Extracts atom-level data: reference conformer positions, element types,
          charges, atom names, ground-truth coordinates, local reference frames
          (for IPA), pairwise distograms (training target), and the mapping
          tensors (atom_to_token, token_to_rep_atom) that bridge the atom and
          token representation levels.

        **Stage 3 -- MSA features** (``process_msa_features``):
          Constructs the cross-chain paired MSA using taxonomy matching, then
          produces one-hot encoded MSA rows, deletion features, and sequence
          profiles. During training, the MSA depth is randomly varied (sampled
          uniformly from [1, max_seqs]) to improve robustness to different
          alignment depths.

        **Stage 4 -- Symmetry features** (``process_symmetry_features``, optional):
          Computes permutation information for symmetric chains, amino acid
          side-chain atoms, and ligand atoms. Only computed when
          ``compute_symmetries=True`` (typically during training for the
          symmetry-aware loss).

        Parameters
        ----------
        data : Tokenized
            The tokenized data (tokens, atoms, MSA, structure).
        training : bool
            Whether the model is in training mode. Affects MSA depth sampling
            and pocket conditioning stochasticity.
        max_seqs : int
            Maximum number of MSA sequences.
        atoms_per_window_queries : int
            Window size for atom-level windowed attention (atom count is padded
            to a multiple of this).
        min_dist : float
            Minimum distance for distogram binning (Angstroms).
        max_dist : float
            Maximum distance for distogram binning (Angstroms).
        num_bins : int
            Number of distance bins in the distogram.
        max_tokens : int, optional
            If set, pad token-dimension tensors to this fixed size.
        max_atoms : int, optional
            If set, pad atom-dimension tensors to this fixed size.
        pad_to_max_seqs : bool
            If True, pad the MSA sequence dimension to ``max_seqs``.
        compute_symmetries : bool
            Whether to compute symmetry features (for symmetry-aware loss).
        symmetries : dict, optional
            Pre-computed ligand symmetry information.
        binder_pocket_conditioned_prop : float
            Probability of activating pocket conditioning during training.
        binder_pocket_cutoff : float
            Distance cutoff (Angstroms) for defining pocket residues.
        binder_pocket_sampling_geometric_p : float
            Geometric distribution parameter for pocket residue subsampling.
        only_ligand_binder_pocket : bool
            If True, only ligands can be chosen as the binder for pocket conditioning.
        inference_binder : int or list[int], optional
            Chain ID(s) of the binder molecule(s) during inference.
        inference_pocket : list[tuple[int, int]], optional
            Explicit pocket residues (asym_id, res_idx) during inference.

        Returns
        -------
        dict[str, Tensor]
            Merged dictionary of all feature tensors (token + atom + MSA +
            symmetry), ready to be consumed by the model's forward pass.

        """
        # During training, randomly vary MSA depth by sampling uniformly from
        # [1, max_seqs]. This makes the model robust to different alignment
        # depths and prevents overfitting to a fixed MSA size. During inference,
        # use the full max_seqs.
        if training and max_seqs is not None:
            max_seqs_batch = np.random.randint(1, max_seqs + 1)  # noqa: NPY002
        else:
            max_seqs_batch = max_seqs

        # Stage 1: Token-level features (residue types, chain IDs, bonds,
        # pocket conditioning, masks, distogram centers)
        token_features = process_token_features(
            data,
            max_tokens,
            binder_pocket_conditioned_prop,
            binder_pocket_cutoff,
            binder_pocket_sampling_geometric_p,
            only_ligand_binder_pocket,
            inference_binder=inference_binder,
            inference_pocket=inference_pocket,
        )

        # Stage 2: Atom-level features (reference positions, element types,
        # coordinates, frames, distograms, atom<->token mappings)
        atom_features = process_atom_features(
            data,
            atoms_per_window_queries,
            min_dist,
            max_dist,
            num_bins,
            max_atoms,
            max_tokens,
        )

        # Stage 3: MSA features (one-hot sequences, deletions, profiles,
        # paired indicators)
        msa_features = process_msa_features(
            data,
            max_seqs_batch,
            max_seqs,
            max_tokens,
            pad_to_max_seqs,
        )

        # Stage 4: Symmetry features (optional -- chain/residue/ligand
        # permutation information for symmetry-aware loss computation)
        symmetry_features = {}
        if compute_symmetries:
            symmetry_features = process_symmetry_features(data, symmetries)

        # Merge all feature dictionaries into a single flat dictionary
        return {
            **token_features,
            **atom_features,
            **msa_features,
            **symmetry_features,
        }
