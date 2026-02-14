"""Boltz tokenizer for converting molecular structures into token sequences.

This module converts an input molecular structure (atoms, residues, chains) into
a flat sequence of tokens suitable for the Boltz model. The tokenization strategy
differs based on whether a residue is *standard* or *non-standard*:

Standard residues (canonical amino acids, nucleotides):
    Each standard residue becomes **one token**. The token's center and distogram
    atoms (e.g., C-alpha and C-beta for proteins) are recorded for spatial
    operations like cropping and distogram prediction. All atoms of the residue
    are mapped to this single token index.

Non-standard residues (modified residues, ligands, non-canonical molecules):
    Each individual atom becomes a **separate token**, using the unknown protein
    token type. The atom itself serves as both the center and distogram atom.
    This per-atom tokenization allows the model to handle arbitrary chemical
    entities without requiring a fixed residue vocabulary.

The tokenizer also produces:
    - A ``token_mask`` indicating which tokens correspond to CDR residues
      (for antibody design applications).
    - A ``token_bonds`` array mapping covalent bonds (both intra-residue ligand
      bonds and inter-residue connections) to their token-level endpoints.
"""

from dataclasses import astuple, dataclass

import numpy as np

from boltz.data import const
from boltz.data.tokenize.tokenizer import Tokenizer
from boltz.data.types import Input, Token, TokenBond, Tokenized


@dataclass
class TokenData:
    """Intermediate data structure holding all fields for a single token.

    This dataclass is used during tokenization to collect per-token attributes
    before they are converted into a structured NumPy array of dtype ``Token``.

    Attributes
    ----------
    token_idx : int
        Global index of this token in the output sequence.
    atom_idx : int
        Index of the first atom belonging to this token in the atom table.
    atom_num : int
        Number of atoms belonging to this token (1 for non-standard per-atom
        tokens, variable for standard residue tokens).
    res_idx : int
        Residue index within the chain.
    res_type : int
        Integer residue type ID (from the token vocabulary).
    sym_id : int
        Symmetry ID for the chain (used for symmetry-aware training).
    asym_id : int
        Asymmetric unit chain ID.
    entity_id : int
        Entity ID grouping identical chains.
    mol_type : int
        Molecule type (protein, RNA, DNA, or nonpolymer).
    center_idx : int
        Atom index of the center atom (e.g., C-alpha for proteins).
    disto_idx : int
        Atom index of the distogram atom (e.g., C-beta for proteins).
    center_coords : np.ndarray
        3D coordinates of the center atom.
    disto_coords : np.ndarray
        3D coordinates of the distogram atom.
    resolved_mask : bool
        Whether this token has experimentally resolved coordinates.
    disto_mask : bool
        Whether the distogram atom has resolved coordinates.
    """

    token_idx: int
    atom_idx: int
    atom_num: int
    res_idx: int
    res_type: int
    sym_id: int
    asym_id: int
    entity_id: int
    mol_type: int
    center_idx: int
    disto_idx: int
    center_coords: np.ndarray
    disto_coords: np.ndarray
    resolved_mask: bool
    disto_mask: bool


class BoltzTokenizer(Tokenizer):
    """Tokenize an input structure for training.

    Converts a molecular structure into a sequence of tokens, handling both
    standard residues (one token per residue) and non-standard residues
    (one token per atom). Also builds the token-level bond graph and CDR mask.
    """

    def tokenize(self, data: Input) -> Tokenized:
        """Tokenize the input data.

        Parameters
        ----------
        data : Input
            The input data containing structure and MSA information.

        Returns
        -------
        tuple[Tokenized, np.ndarray]
            A tuple of (tokenized_data, token_mask) where token_mask is a
            boolean array indicating CDR residue tokens.

        """
        # Get structure data
        struct = data.structure

        # Accumulator for token records (will be converted to structured array)
        token_data = []

        # Mapping from atom index to token index, used for bond conversion
        token_idx = 0
        atom_to_token = {}

        # Filter to valid chains only (chains that pass the structure mask)
        chains = struct.chains[struct.mask]

        # Boolean mask: True for tokens that are CDR residues (antibody design)
        token_mask = []

        for chain in chains:
            # Determine the range of residues belonging to this chain
            res_start = chain["res_idx"]
            res_end = chain["res_idx"] + chain["res_num"]

            for res in struct.residues[res_start:res_end]:
                # Determine the range of atoms belonging to this residue
                atom_start = res["atom_idx"]
                atom_end = res["atom_idx"] + res["atom_num"]

                # --------------------------------------------------------
                # Standard residues: one token per residue
                # --------------------------------------------------------
                if res["is_standard"]:
                    # Look up the designated center atom (e.g., C-alpha) and
                    # distogram atom (e.g., C-beta) for this residue
                    center = struct.atoms[res["atom_center"]]
                    disto = struct.atoms[res["atom_disto"]]

                    # A token is considered "resolved" only if BOTH the residue
                    # is present AND its center atom has coordinates
                    is_present = res["is_present"] & center["is_present"]
                    # Distogram mask requires both residue and disto atom present
                    is_disto_present = res["is_present"] & disto["is_present"]

                    # Extract 3D coordinates for the center and distogram atoms
                    c_coords = center["coords"]
                    d_coords = disto["coords"]

                    # Create the token record with all required fields
                    token = TokenData(
                        token_idx=token_idx,
                        atom_idx=res["atom_idx"],
                        atom_num=res["atom_num"],
                        res_idx=res["res_idx"],
                        res_type=res["res_type"],
                        sym_id=chain["sym_id"],
                        asym_id=chain["asym_id"],
                        entity_id=chain["entity_id"],
                        mol_type=chain["mol_type"],
                        center_idx=res["atom_center"],
                        disto_idx=res["atom_disto"],
                        center_coords=c_coords,
                        disto_coords=d_coords,
                        resolved_mask=is_present,
                        disto_mask=is_disto_present,
                    )
                    token_data.append(astuple(token))

                    # Mark CDR residues in the token mask (used for antibody tasks)
                    if res["is_cdr_residue"]:
                        token_mask.append(True)
                    else:
                        token_mask.append(False)

                    # Map every atom in this residue to the same token index
                    for atom_idx in range(atom_start, atom_end):
                        atom_to_token[atom_idx] = token_idx

                    token_idx += 1

                # --------------------------------------------------------
                # Non-standard residues: one token per atom
                # --------------------------------------------------------
                else:
                    # Use the unknown protein token type for all non-standard atoms
                    unk_token = const.unk_token["PROTEIN"]
                    unk_id = const.token_ids[unk_token]

                    # Pre-fetch all atom coordinates for this residue
                    atom_data = struct.atoms[atom_start:atom_end]
                    atom_coords = atom_data["coords"]

                    # Create a separate token for each atom in the residue
                    for i, atom in enumerate(atom_data):
                        # Token presence requires both residue and atom to be present
                        is_present = res["is_present"] & atom["is_present"]
                        index = atom_start + i

                        # For per-atom tokens, the atom itself serves as both
                        # the center and distogram atom (center_idx == disto_idx)
                        token = TokenData(
                            token_idx=token_idx,
                            atom_idx=index,
                            atom_num=1,
                            res_idx=res["res_idx"],
                            res_type=unk_id,
                            sym_id=chain["sym_id"],
                            asym_id=chain["asym_id"],
                            entity_id=chain["entity_id"],
                            mol_type=chain["mol_type"],
                            center_idx=index,
                            disto_idx=index,
                            center_coords=atom_coords[i],
                            disto_coords=atom_coords[i],
                            resolved_mask=is_present,
                            disto_mask=is_present,
                        )
                        token_data.append(astuple(token))

                        # Mark CDR atoms in the token mask
                        if atom["is_cdr_atom"]:
                            token_mask.append(True)
                        else:
                            token_mask.append(False)

                        # Each atom maps to its own unique token
                        atom_to_token[index] = token_idx
                        token_idx += 1

                # Sanity check: residue index should be within the chain bounds
                assert res["res_idx"] < chain["res_num"]

        # ----------------------------------------------------------------
        # Build token-level bond graph
        # ----------------------------------------------------------------
        token_bonds = []

        # Convert atom-atom bonds (e.g., within ligands) to token-token bonds.
        # For standard residues, intra-residue bonds map to self-loops (same token).
        # For non-standard residues, each atom is its own token so bonds are
        # between different tokens.
        for bond in struct.bonds:
            if (
                bond["atom_1"] not in atom_to_token
                or bond["atom_2"] not in atom_to_token
            ):
                # Skip bonds involving atoms outside valid chains
                continue
            token_bond = (
                atom_to_token[bond["atom_1"]],
                atom_to_token[bond["atom_2"]],
            )
            token_bonds.append(token_bond)

        # Convert inter-residue covalent connections (e.g., disulfide bonds,
        # peptide bonds between non-standard residues) to token-token bonds
        for conn in struct.connections:
            if (
                conn["atom_1"] not in atom_to_token
                or conn["atom_2"] not in atom_to_token
            ):
                continue
            token_bond = (
                atom_to_token[conn["atom_1"]],
                atom_to_token[conn["atom_2"]],
            )
            token_bonds.append(token_bond)

        # Verify consistency: token_mask length must match token_data length
        assert len(token_data) == len(token_mask)

        # Convert to NumPy structured arrays for efficient downstream processing
        token_mask = np.array(token_mask, dtype=bool)
        token_data = np.array(token_data, dtype=Token)
        token_bonds = np.array(token_bonds, dtype=TokenBond)

        # Package everything into a Tokenized object
        tokenized = Tokenized(
            token_data,
            token_bonds,
            data.structure,
            data.msa,
        )
        return tokenized, token_mask
