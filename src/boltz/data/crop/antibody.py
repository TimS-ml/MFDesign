"""Antibody-specific cropping strategy for structure prediction.

This module implements a cropping strategy tailored for antibody structures.
The key design decisions are:

1. **CDR preservation**: All residues belonging to the heavy (H) and light (L)
   antibody chains are always included in the crop, ensuring that the
   complementarity-determining regions (CDRs) -- the critical binding loops --
   are never discarded during cropping.

2. **Neighbor sampling for antigen context**: When ``add_antigen`` is enabled,
   nearby antigen residues are added around a randomly chosen query token from
   the antibody. A *neighborhood size* is sampled uniformly from a configurable
   range; smaller sizes yield tighter spatial crops while larger sizes include
   more contiguous chain segments.

3. **H / L chain handling**: Heavy and light chain residues are unconditionally
   added to the crop first. Only after both antibody chains are fully included
   does the cropper attempt to fill the remaining token budget with antigen
   residues sorted by spatial proximity to the antibody.

The cropper inherits from :class:`Cropper` and is intended to be used during
training data preparation for antibody design models.
"""

from dataclasses import replace
from typing import Optional

import numpy as np
from scipy.spatial.distance import cdist

from boltz.data import const
from boltz.data.crop.cropper import Cropper
from boltz.data.types import Tokenized


def pick_chain_token(
    tokens: np.ndarray,
    token_mask: np.ndarray,
    chain_id: int,
    random: np.random.RandomState,
) -> np.ndarray:
    """Pick a random CDR token from a specified chain.

    Filters the token array to those belonging to ``chain_id`` that are also
    flagged in ``token_mask`` (i.e., CDR residues), then returns one at random.
    This token is later used as the spatial query center for neighbor sampling.

    Parameters
    ----------
    tokens : np.ndarray
        Structured array of all resolved tokens.
    token_mask : np.ndarray
        Boolean mask indicating which tokens are CDR residues.
    chain_id : int
        The asymmetric chain ID to filter on.
    random : np.random.RandomState
        Random state for reproducibility.

    Returns
    -------
    np.ndarray
        A single token record chosen at random from the filtered set.
    """
    # Select tokens that belong to the target chain AND are CDR residues
    cdr_tokens = tokens[(tokens["asym_id"] == chain_id) & token_mask]
    return cdr_tokens[random.randint(len(cdr_tokens))]


class AntibodyCropper(Cropper):
    """Antibody-aware cropper that preserves full H/L chains and optionally
    adds spatially proximal antigen residues.

    The cropping strategy works in two phases:

    Phase 1 -- Antibody inclusion:
        All resolved tokens from the heavy chain and light chain are
        unconditionally added to the crop. This guarantees that CDR1, CDR2,
        and CDR3 loops on both chains are always preserved.

    Phase 2 -- Antigen neighbor sampling (optional, controlled by ``add_antigen``):
        A random CDR token is chosen as a spatial query point. All antigen
        (non-antibody) tokens are sorted by Euclidean distance to that query.
        For each antigen token (nearest first), a contiguous neighborhood of
        residues from the same chain is selected and added to the crop, up to
        the token/atom budget.

    The neighborhood size controls the trade-off between spatial and contiguous
    cropping: a neighborhood of 0 adds only the single closest residue, while
    larger values include contiguous stretches of the antigen chain.
    """

    def __init__(self, add_antigen: Optional[bool] = False,
                 min_neighborhood: int = 0, max_neighborhood: int = 40) -> None:
        """Initialize the antibody cropper.

        Parameters
        ----------
        add_antigen : bool, optional
            Whether to include antigen residues in the crop. Default is False.
        min_neighborhood : int, optional
            Minimum contiguous neighborhood size for antigen residues.
        max_neighborhood : int, optional
            Maximum contiguous neighborhood size for antigen residues.
        """
        self.add_antigen = add_antigen
        # Build list of even-numbered neighborhood sizes from min to max (inclusive)
        sizes = list(range(min_neighborhood, max_neighborhood + 1, 2))
        self.neighborhood_sizes = sizes

    def crop(  # noqa: PLR0915
        self,
        data: Tokenized,
        token_mask: np.ndarray,
        token_region: np.ndarray,
        max_tokens: int,
        random: np.random.RandomState,
        max_atoms: Optional[int] = None,
        chain_id: Optional[int] = None,
        h_chain_id: Optional[int] = None,
        l_chain_id: Optional[int] = None,
    ):
        """Crop the data to a maximum number of tokens.

        The method first adds all resolved H-chain and L-chain tokens, then
        optionally fills remaining capacity with antigen residues selected by
        spatial proximity to a randomly chosen CDR token.

        Parameters
        ----------
        data : Tokenized
            The tokenized data.
        token_mask : np.ndarray
            Boolean mask indicating CDR tokens.
        token_region : np.ndarray
            Region annotations per token.
        max_tokens : int
            The maximum number of tokens to crop.
        random : np.random.RandomState
            The random state for reproducibility.
        max_atoms : int, optional
            The maximum number of atoms to consider.
        chain_id : int, optional
            The chain ID to use as spatial query origin. If None and
            add_antigen is True, a chain is randomly chosen from H/L.
        h_chain_id : int, optional
            The heavy chain asymmetric ID.
        l_chain_id : int, optional
            The light chain asymmetric ID.

        Returns
        -------
        tuple[Tokenized, np.ndarray, np.ndarray]
            The cropped tokenized data, the cropped token mask, and the
            cropped token region array.

        Raises
        ------
        ValueError
            If no valid tokens exist, or if the antibody chains alone
            exceed the budget.
        """

        # Get token data
        token_data = data.tokens
        token_bonds = data.bonds

        # Filter to resolved tokens (those with experimentally determined coordinates)
        valid_tokens = token_data[token_data["resolved_mask"]]
        valid_masks = token_mask[token_data["resolved_mask"]]

        # Check if we have any valid tokens
        if not valid_tokens.size:
            msg = "No valid tokens in structure"
            raise ValueError(msg)

        # ----------------------------------------------------------------
        # Phase 1: Unconditionally include ALL resolved antibody tokens
        # ----------------------------------------------------------------
        # This ensures CDR regions (H1, H2, H3, L1, L2, L3) and framework
        # residues are never lost during cropping.
        cropped: set[int] = set()
        total_atoms = 0

        # Add all resolved heavy chain tokens
        if h_chain_id is not None:
            h_chain_tokens = token_data[(token_data["asym_id"] == h_chain_id) &
                                        token_data["resolved_mask"]]
            cropped.update(h_chain_tokens["token_idx"])
            total_atoms += np.sum(h_chain_tokens["atom_num"])

        # Add all resolved light chain tokens
        if l_chain_id is not None:
            l_chain_tokens = token_data[(token_data["asym_id"] == l_chain_id) &
                                        token_data["resolved_mask"]]
            cropped.update(l_chain_tokens["token_idx"])
            # NOTE: This uses h_chain_tokens["atom_num"] -- appears to be a
            # pre-existing bug (should likely be l_chain_tokens), but we
            # preserve original logic as-is.
            total_atoms += np.sum(h_chain_tokens["atom_num"])

        # ----------------------------------------------------------------
        # Phase 2: Optionally add spatially proximal antigen residues
        # ----------------------------------------------------------------
        if self.add_antigen:
            # Gather all non-antibody tokens that are resolved and pass the mask
            epitope_tokens = token_data[(token_data["asym_id"] != h_chain_id) &
                                        (token_data["asym_id"] != l_chain_id) &
                                        token_mask & token_data["resolved_mask"]]

            # Randomly choose a neighborhood size for this crop.
            # Smaller values = tighter spatial crop around the query;
            # larger values = more contiguous chain residues included.
            neighborhood_size = random.choice(self.neighborhood_sizes)

            # Pick a random CDR token from the antibody as the spatial query center
            if chain_id is None:
                chain_id = random.choice([x for x in [h_chain_id, l_chain_id] if x is not None])
            query = pick_chain_token(valid_tokens, valid_masks, chain_id, random)

            # Compute distances from each antigen token to the query token
            # and sort by ascending distance (nearest first)
            dists = epitope_tokens["center_coords"] - query["center_coords"]
            indices = np.argsort(np.linalg.norm(dists, axis=1))

            # Iterate over antigen tokens in order of proximity to the antibody
            for idx in indices:
                token = epitope_tokens[idx]

                # Get ALL tokens from the same chain as this antigen token
                chain_tokens = token_data[token_data["asym_id"] == token["asym_id"]]

                # If the chain is small enough, include it entirely
                if len(chain_tokens) <= neighborhood_size:
                    new_tokens = chain_tokens
                else:
                    # Select a contiguous window of residues centered on the
                    # current antigen token, expanding symmetrically by
                    # res_idx until the neighborhood_size is reached.

                    # First, limit to a maximum window (2x neighborhood) for
                    # efficiency -- avoids scanning the entire chain repeatedly
                    min_idx = token["res_idx"] - neighborhood_size
                    max_idx = token["res_idx"] + neighborhood_size

                    max_token_set = chain_tokens
                    max_token_set = max_token_set[max_token_set["res_idx"] >= min_idx]
                    max_token_set = max_token_set[max_token_set["res_idx"] <= max_idx]

                    # Start with just the anchor residue
                    new_tokens = max_token_set[max_token_set["res_idx"] == token["res_idx"]]

                    # Expand the window one residue at a time in both directions
                    # until we have enough tokens. Using res_idx (not token_idx)
                    # ensures modified residues and ligands at the same position
                    # are fully included.
                    min_idx = max_idx = token["res_idx"]
                    while new_tokens.size < neighborhood_size:
                        min_idx = min_idx - 1
                        max_idx = max_idx + 1
                        new_tokens = max_token_set
                        new_tokens = new_tokens[new_tokens["res_idx"] >= min_idx]
                        new_tokens = new_tokens[new_tokens["res_idx"] <= max_idx]

                # Only consider tokens not already in the crop
                new_indices = set(new_tokens["token_idx"]) - cropped
                new_tokens = token_data[list(new_indices)]
                new_atoms = np.sum(new_tokens["atom_num"])

                # Stop if adding these tokens would exceed the budget
                if (len(new_indices) > (max_tokens - len(cropped))) or (
                    (max_atoms is not None) and ((total_atoms + new_atoms) > max_atoms)
                ):
                    break

                # Accept the new tokens into the crop
                cropped.update(new_indices)
                total_atoms += new_atoms

        # Sanity checks: ensure we did not exceed budgets
        if max_tokens is not None and len(cropped) > max_tokens:
            raise ValueError("Cropped tokens exceed maximum")
        if max_atoms is not None and total_atoms > max_atoms:
            raise ValueError("Cropped atoms exceed maximum")

        # Sort cropped indices to maintain original token ordering
        token_data = token_data[sorted(cropped)]
        token_mask = token_mask[sorted(cropped)]
        token_region = token_region[sorted(cropped)]

        # Filter bonds: only keep bonds where both endpoints are in the crop
        indices = token_data["token_idx"]
        token_bonds = token_bonds[np.isin(token_bonds["token_1"], indices)]
        token_bonds = token_bonds[np.isin(token_bonds["token_2"], indices)]

        # Return the cropped tokenized data along with updated masks
        return replace(data, tokens=token_data, bonds=token_bonds), token_mask, token_region
