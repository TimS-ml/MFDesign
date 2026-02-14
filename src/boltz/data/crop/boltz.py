"""General-purpose Boltz cropping strategy for molecular structure prediction.

This module implements the default spatial/contiguous cropping approach used by
the Boltz model. The cropper selects a subset of tokens from a tokenized
molecular structure, respecting maximum token and atom budgets.

Cropping proceeds as follows:

1. **Query selection**: A single token is chosen as the spatial center of the
   crop. The query can be selected from a specific chain, a specific interface,
   or (by default) from a randomly chosen interface or chain.

2. **Distance-based ordering**: All resolved tokens are sorted by Euclidean
   distance to the query token's center coordinates.

3. **Greedy neighborhood expansion**: Tokens are added to the crop in order
   of proximity. For each token, a contiguous neighborhood of residues from
   the same chain is included. The neighborhood size is sampled uniformly at
   initialization and controls the trade-off between spatial tightness
   (small neighborhood) and chain contiguity (large neighborhood).

4. **Budget enforcement**: Expansion stops when the maximum number of tokens
   or atoms would be exceeded by the next addition.

Interface-aware query selection uses pairwise distance computation (via
``cdist``) to identify tokens near the boundary between two chains, biasing
the crop toward interaction sites.
"""

from dataclasses import replace
from typing import Optional

import numpy as np
from scipy.spatial.distance import cdist

from boltz.data import const
from boltz.data.crop.cropper import Cropper
from boltz.data.types import Tokenized


def pick_random_token(
    tokens: np.ndarray,
    random: np.random.RandomState,
) -> np.ndarray:
    """Pick a random token from the data.

    Parameters
    ----------
    tokens : np.ndarray
        The token data.
    random : np.ndarray
        The random state for reproducibility.

    Returns
    -------
    np.ndarray
        The selected token.

    """
    return tokens[random.randint(len(tokens))]


def pick_chain_token(
    tokens: np.ndarray,
    chain_id: int,
    random: np.random.RandomState,
) -> np.ndarray:
    """Pick a random token from a specific chain.

    If the chain has no tokens (e.g., it was filtered out), falls back to
    picking a random token from all available tokens.

    Parameters
    ----------
    tokens : np.ndarray
        The token data.
    chain_id : int
        The chain ID.
    random : np.ndarray
        The random state for reproducibility.

    Returns
    -------
    np.ndarray
        The selected token.

    """
    # Filter to chain
    chain_tokens = tokens[tokens["asym_id"] == chain_id]

    # Pick from chain, fallback to all tokens
    if chain_tokens.size:
        query = pick_random_token(chain_tokens, random)
    else:
        query = pick_random_token(tokens, random)

    return query


def pick_interface_token(
    tokens: np.ndarray,
    interface: np.ndarray,
    random: np.random.RandomState,
) -> np.ndarray:
    """Pick a random token from a chain-chain interface.

    An interface token is one whose center coordinates are within a distance
    cutoff of a token on the partner chain. This biases the crop toward
    biologically relevant interaction sites.

    The method handles edge cases where one or both chains may have no tokens
    by falling back to the available chain or all tokens, respectively.

    Parameters
    ----------
    tokens : np.ndarray
        The token data.
    interface : np.ndarray
        The interface record containing chain_1 and chain_2 identifiers.
    random : np.ndarray
        The random state for reproducibility.

    Returns
    -------
    np.ndarray
        The selected token from the interface region.

    """
    # Extract the two chain IDs forming the interface
    chain_1 = int(interface["chain_1"])
    chain_2 = int(interface["chain_2"])

    tokens_1 = tokens[tokens["asym_id"] == chain_1]
    tokens_2 = tokens[tokens["asym_id"] == chain_2]

    # Handle cases where one or both chains have no resolved tokens
    if tokens_1.size and (not tokens_2.size):
        query = pick_random_token(tokens_1, random)
    elif tokens_2.size and (not tokens_1.size):
        query = pick_random_token(tokens_2, random)
    elif (not tokens_1.size) and (not tokens_2.size):
        query = pick_random_token(tokens, random)
    else:
        # Both chains have tokens -- compute pairwise distances to find
        # tokens near the interface boundary
        tokens_1_coords = tokens_1["center_coords"]
        tokens_2_coords = tokens_2["center_coords"]

        # Compute all-vs-all pairwise distances between the two chains
        dists = cdist(tokens_1_coords, tokens_2_coords)

        # Apply the interface distance cutoff to identify contacting tokens
        cuttoff = dists < const.interface_cutoff

        # In rare cases, the interface cutoff is slightly too small
        # (e.g., for loosely packed interfaces). Expand by 5 Angstroms
        # as a fallback to ensure we get at least some interface tokens.
        if not np.any(cuttoff):
            cuttoff = dists < (const.interface_cutoff + 5.0)

        # Filter each chain to only tokens within the cutoff of the other chain.
        # axis=1 checks if token_1[i] is close to ANY token in chain_2.
        # axis=0 checks if token_2[j] is close to ANY token in chain_1.
        tokens_1 = tokens_1[np.any(cuttoff, axis=1)]
        tokens_2 = tokens_2[np.any(cuttoff, axis=0)]

        # Combine interface tokens from both chains and pick one at random
        candidates = np.concatenate([tokens_1, tokens_2])
        query = pick_random_token(candidates, random)

    return query


class BoltzCropper(Cropper):
    """General Boltz cropper that interpolates between contiguous and spatial crops.

    The neighborhood size parameter controls the cropping behavior:

    - **Neighborhood = 0**: Pure spatial cropping. Each token is added
      individually, producing a crop of the spatially closest tokens
      regardless of chain contiguity.
    - **Large neighborhood**: More contiguous cropping. When a token is added,
      a window of neighboring residues (by residue index) from the same chain
      is also included, producing longer contiguous chain segments.
    - **Mixed range**: Sampling the neighborhood uniformly from a range (e.g.,
      0 to 40) produces a mixture of spatial and contiguous crops across
      different training examples.
    """

    def __init__(self, min_neighborhood: int = 0, max_neighborhood: int = 40) -> None:
        """Initialize the cropper.

        Modulates the type of cropping to be performed.
        Smaller neighborhoods result in more spatial
        cropping. Larger neighborhoods result in more
        continuous cropping. A mix can be achieved by
        providing a range over which to sample.

        Parameters
        ----------
        min_neighborhood : int
            The minimum neighborhood size, by default 0.
        max_neighborhood : int
            The maximum neighborhood size, by default 40.

        """
        # Build a list of candidate neighborhood sizes (step of 2)
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
        interface_id: Optional[int] = None,
    ) -> Tokenized:
        """Crop the data to a maximum number of tokens.

        Parameters
        ----------
        data : Tokenized
            The tokenized data.
        max_tokens : int
            The maximum number of tokens to crop.
        random : np.random.RandomState
            The random state for reproducibility.
        max_atoms : int, optional
            The maximum number of atoms to consider.
        chain_id : int, optional
            The chain ID to crop.
        interface_id : int, optional
            The interface ID to crop.

        Returns
        -------
        Tokenized
            The cropped data.

        """
        # Mutual exclusivity: only one targeting mode at a time
        if chain_id is not None and interface_id is not None:
            msg = "Only one of chain_id or interface_id can be provided."
            raise ValueError(msg)

        # Sample a neighborhood size for this crop instance
        neighborhood_size = random.choice(self.neighborhood_sizes)

        # Get token data
        token_data = data.tokens
        token_bonds = data.bonds
        mask = data.structure.mask
        chains = data.structure.chains
        interfaces = data.structure.interfaces

        # Filter to valid (non-masked) chains
        valid_chains = chains[mask]

        # Filter to valid interfaces (both endpoint chains must be valid)
        valid_interfaces = interfaces
        if valid_interfaces.size:
            valid_interfaces = valid_interfaces[mask[valid_interfaces["chain_1"]]]
            valid_interfaces = valid_interfaces[mask[valid_interfaces["chain_2"]]]

        # Filter to resolved tokens (experimentally determined coordinates)
        valid_tokens = token_data[token_data["resolved_mask"]]

        # Check if we have any valid tokens
        if not valid_tokens.size:
            msg = "No valid tokens in structure"
            raise ValueError(msg)

        # ----------------------------------------------------------------
        # Query token selection: determines the spatial center of the crop
        # ----------------------------------------------------------------
        # Priority: explicit chain_id > explicit interface_id > random interface > random chain
        if chain_id is not None:
            # User specified a particular chain -- pick a random token from it
            query = pick_chain_token(valid_tokens, chain_id, random)
        elif interface_id is not None:
            # User specified a particular interface -- pick from interface region
            interface = interfaces[interface_id]
            query = pick_interface_token(valid_tokens, interface, random)
        elif valid_interfaces.size:
            # Default: randomly choose one of the valid interfaces, then
            # pick a token from that interface region
            idx = random.randint(len(valid_interfaces))
            interface = valid_interfaces[idx]
            query = pick_interface_token(valid_tokens, interface, random)
        else:
            # No interfaces available: randomly choose a chain, then pick
            # a token from that chain
            idx = random.randint(len(valid_chains))
            chain_id = valid_chains[idx]["asym_id"]
            query = pick_chain_token(valid_tokens, chain_id, random)

        # ----------------------------------------------------------------
        # Sort all resolved tokens by Euclidean distance to the query token
        # ----------------------------------------------------------------
        dists = valid_tokens["center_coords"] - query["center_coords"]
        indices = np.argsort(np.linalg.norm(dists, axis=1))

        # ----------------------------------------------------------------
        # Greedy crop expansion: add tokens nearest-first with neighborhoods
        # ----------------------------------------------------------------
        cropped: set[int] = set()
        total_atoms = 0
        for idx in indices:
            # Get the current token
            token = valid_tokens[idx]

            # Get all tokens from this chain (including unresolved ones)
            chain_tokens = token_data[token_data["asym_id"] == token["asym_id"]]

            # Pick the whole chain if possible, otherwise select
            # a contiguous subset centered at the query token
            if len(chain_tokens) <= neighborhood_size:
                # Chain is small enough to include entirely
                new_tokens = chain_tokens
            else:
                # First limit to the maximum set of tokens, with the
                # neighborhood on both sides to handle edges. This
                # is mostly for efficiency with the while loop below.
                min_idx = token["res_idx"] - neighborhood_size
                max_idx = token["res_idx"] + neighborhood_size

                max_token_set = chain_tokens
                max_token_set = max_token_set[max_token_set["res_idx"] >= min_idx]
                max_token_set = max_token_set[max_token_set["res_idx"] <= max_idx]

                # Start by adding just the query token
                new_tokens = max_token_set[max_token_set["res_idx"] == token["res_idx"]]

                # Expand the neighborhood until we have enough tokens, one
                # by one to handle some edge cases with non-standard chains.
                # We switch to the res_idx instead of the token_idx to always
                # include all tokens from modified residues or from ligands.
                min_idx = max_idx = token["res_idx"]
                while new_tokens.size < neighborhood_size:
                    min_idx = min_idx - 1
                    max_idx = max_idx + 1
                    new_tokens = max_token_set
                    new_tokens = new_tokens[new_tokens["res_idx"] >= min_idx]
                    new_tokens = new_tokens[new_tokens["res_idx"] <= max_idx]

            # Compute new tokens and new atoms (exclude already-cropped)
            new_indices = set(new_tokens["token_idx"]) - cropped
            new_tokens = token_data[list(new_indices)]
            new_atoms = np.sum(new_tokens["atom_num"])

            # Stop if we exceed the max number of tokens or atoms
            if (len(new_indices) > (max_tokens - len(cropped))) or (
                (max_atoms is not None) and ((total_atoms + new_atoms) > max_atoms)
            ):
                break

            # Add new indices to the crop
            cropped.update(new_indices)
            total_atoms += new_atoms

        # Sort cropped indices to maintain original token ordering
        token_data = token_data[sorted(cropped)]
        token_mask = token_mask[sorted(cropped)]
        token_region = token_region[sorted(cropped)]

        # Only keep bonds where both endpoints remain in the crop
        indices = token_data["token_idx"]
        token_bonds = token_bonds[np.isin(token_bonds["token_1"], indices)]
        token_bonds = token_bonds[np.isin(token_bonds["token_2"], indices)]

        # Return the cropped tokens with updated bond list
        return replace(data, tokens=token_data, bonds=token_bonds), token_mask, token_region
