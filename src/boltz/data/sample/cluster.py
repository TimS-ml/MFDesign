"""Cluster-based weighted sampling strategy for training data selection.

This module implements the weighted sampling approach described in AlphaFold 3.
The key idea is to prevent overrepresentation of common protein families during
training by assigning each chain or interface a sampling weight that is
inversely proportional to its cluster size.

Weight formula for a chain:
    w = beta_chain / N_cluster * alpha_type

Weight formula for an interface:
    w = beta_interface / N_cluster * (alpha_prot * n_prot + alpha_nucl * n_nucl
        + alpha_ligand * n_ligand)

Where:
    - N_cluster: number of chains/interfaces sharing the same cluster ID
    - alpha_prot, alpha_nucl, alpha_ligand: type-specific scaling factors that
      control the relative frequency of protein, nucleic acid, and ligand
      samples
    - beta_chain, beta_interface: base weights that control the relative
      frequency of single-chain vs. interface samples
    - n_prot, n_nucl, n_ligand: count of each chain type in the interface
      (0, 1, or 2 for each)

The sampling produces an infinite stream of ``Sample`` objects, each pointing
to either a chain (via ``chain_id``) or an interface (via ``interface_id``).
"""

from typing import Dict, Iterator, List

import numpy as np
from numpy.random import RandomState

from boltz.data import const
from boltz.data.types import ChainInfo, InterfaceInfo, Record
from boltz.data.sample.sampler import Sample, Sampler


def get_chain_cluster(chain: ChainInfo, record: Record) -> str:  # noqa: ARG001
    """Get the cluster id for a chain.

    Each chain is pre-assigned a cluster_id (e.g., from sequence-based
    clustering). Chains in the same cluster share high sequence similarity.

    Parameters
    ----------
    chain : ChainInfo
        The chain id to get the cluster id for.
    record : Record
        The record the interface is part of.

    Returns
    -------
    str
        The cluster id of the chain.

    """
    return chain.cluster_id


def get_interface_cluster(interface: InterfaceInfo, record: Record) -> str:
    """Get the cluster id for an interface.

    The interface cluster is defined as the sorted tuple of the two
    participating chains' cluster IDs. Sorting ensures that the pair
    (A, B) and (B, A) map to the same cluster, avoiding double-counting.

    Parameters
    ----------
    interface : InterfaceInfo
        The interface to get the cluster id for.
    record : Record
        The record the interface is part of.

    Returns
    -------
    str
        The cluster id of the interface (a sorted 2-tuple of chain cluster IDs).

    """
    chain1 = record.chains[interface.chain_1]
    chain2 = record.chains[interface.chain_2]

    cluster_1 = str(chain1.cluster_id)
    cluster_2 = str(chain2.cluster_id)

    # Sort to ensure (chain_A, chain_B) == (chain_B, chain_A)
    cluster_id = (cluster_1, cluster_2)
    cluster_id = tuple(sorted(cluster_id))

    return cluster_id


def get_chain_weight(
    chain: ChainInfo,
    record: Record,  # noqa: ARG001
    clusters: Dict[str, int],
    beta_chain: float,
    alpha_prot: float,
    alpha_nucl: float,
    alpha_ligand: float,
) -> float:
    """Compute the sampling weight for a single chain.

    The weight is inversely proportional to the cluster size (so rare chains
    are upweighted) and scaled by a molecule-type-specific alpha factor.

    Formula: w = (beta_chain / cluster_size) * alpha_type

    Parameters
    ----------
    chain : ChainInfo
        The chain to get the weight for.
    record : Record
        The record the chain is part of.
    clusters : Dict[str, int]
        Mapping from cluster ID to cluster size (number of members).
    beta_chain : float
        The beta value for chains (base weight).
    alpha_prot : float
        The alpha value for proteins.
    alpha_nucl : float
        The alpha value for nucleic acids.
    alpha_ligand : float
        The alpha value for ligands.

    Returns
    -------
    float
        The weight of the chain.

    """
    # Look up molecule type IDs from the constant definitions
    prot_id = const.chain_type_ids["PROTEIN"]
    rna_id = const.chain_type_ids["RNA"]
    dna_id = const.chain_type_ids["DNA"]
    ligand_id = const.chain_type_ids["NONPOLYMER"]

    # Base weight: inversely proportional to how many chains share this cluster
    weight = beta_chain / clusters[chain.cluster_id]

    # Scale by molecule type: proteins and nucleic acids get higher weight
    # than ligands by default (alpha_prot=3, alpha_nucl=3, alpha_ligand=1)
    if chain.mol_type == prot_id:
        weight *= alpha_prot
    elif chain.mol_type in [rna_id, dna_id]:
        weight *= alpha_nucl
    elif chain.mol_type == ligand_id:
        weight *= alpha_ligand

    return weight


def get_interface_weight(
    interface: InterfaceInfo,
    record: Record,
    clusters: Dict[str, int],
    beta_interface: float,
    alpha_prot: float,
    alpha_nucl: float,
    alpha_ligand: float,
) -> float:
    """Compute the sampling weight for a chain-chain interface.

    Similar to chain weights, but accounts for both chains in the interface.
    The weight combines molecule-type contributions from both sides.

    Formula: w = (beta_interface / cluster_size) *
                 (alpha_prot * n_prot + alpha_nucl * n_nucl + alpha_ligand * n_ligand)

    Where n_prot, n_nucl, n_ligand count how many of the two chains are of
    each type (each can be 0, 1, or 2).

    Parameters
    ----------
    interface : InterfaceInfo
        The interface to get the weight for.
    record : Record
        The record the interface is part of.
    clusters : Dict[str, int]
        Mapping from cluster ID to cluster size.
    beta_interface : float
        The beta value for interfaces.
    alpha_prot : float
        The alpha value for proteins.
    alpha_nucl : float
        The alpha value for nucleic acids.
    alpha_ligand : float
        The alpha value for ligands.

    Returns
    -------
    float
        The weight of the interface.

    """
    # Look up molecule type IDs
    prot_id = const.chain_type_ids["PROTEIN"]
    rna_id = const.chain_type_ids["RNA"]
    dna_id = const.chain_type_ids["DNA"]
    ligand_id = const.chain_type_ids["NONPOLYMER"]

    chain1 = record.chains[interface.chain_1]
    chain2 = record.chains[interface.chain_2]

    # Count how many of the two interface chains are of each molecule type.
    # Each counter will be 0, 1, or 2 (e.g., a protein-protein interface
    # has n_prot=2, n_nuc=0, n_ligand=0).
    n_prot = (chain1.mol_type) == prot_id
    n_nuc = chain1.mol_type in [rna_id, dna_id]
    n_ligand = chain1.mol_type == ligand_id

    n_prot += chain2.mol_type == prot_id
    n_nuc += chain2.mol_type in [rna_id, dna_id]
    n_ligand += chain2.mol_type == ligand_id

    # Combine inverse cluster size with molecule type contributions
    weight = beta_interface / clusters[get_interface_cluster(interface, record)]
    weight *= alpha_prot * n_prot + alpha_nucl * n_nuc + alpha_ligand * n_ligand
    return weight


class ClusterSampler(Sampler):
    """Weighted sampler that accounts for cluster sizes, as described in AF3.

    Each chain / interface is given a weight according
    to the following formula, and sampled accordingly:

    w = b / n_clust *(a_prot * n_prot + a_nuc * n_nuc
        + a_ligand * n_ligand)

    The sampler maintains both chains and interfaces in a single pool, with
    their weights normalized to form a probability distribution. Sampling is
    done with replacement using ``numpy.random.choice``.

    The ``kind`` field in each item distinguishes chains (kind=0) from
    interfaces (kind=1), determining whether the yielded ``Sample`` carries
    a ``chain_id`` or an ``interface_id``.
    """

    def __init__(
        self,
        alpha_prot: float = 3.0,
        alpha_nucl: float = 3.0,
        alpha_ligand: float = 1.0,
        beta_chain: float = 0.5,
        beta_interface: float = 1.0,
    ) -> None:
        """Initialize the sampler.

        Parameters
        ----------
        alpha_prot : float, optional
            The alpha value for proteins.
        alpha_nucl : float, optional
            The alpha value for nucleic acids.
        alpha_ligand : float, optional
            The alpha value for ligands.
        beta_chain : float, optional
            The beta value for chains.
        beta_interface : float, optional
            The beta value for interfaces.

        """
        self.alpha_prot = alpha_prot
        self.alpha_nucl = alpha_nucl
        self.alpha_ligand = alpha_ligand
        self.beta_chain = beta_chain
        self.beta_interface = beta_interface

    def sample(self, records: List[Record], random: RandomState) -> Iterator[Sample]:  # noqa: C901, PLR0912
        """Sample a structure from the dataset infinitely.

        The method first computes cluster sizes across the entire dataset,
        then assigns weights to each chain and interface. Sampling proceeds
        infinitely by drawing from the weighted distribution.

        Parameters
        ----------
        records : List[Record]
            The records to sample from.
        random : RandomState
            The random state for reproducibility.

        Yields
        ------
        Sample
            A data sample, with either chain_id or interface_id set.

        """
        # ----------------------------------------------------------------
        # Step 1: Compute chain cluster sizes across the full dataset.
        # This counts how many valid chains belong to each cluster,
        # enabling inverse-frequency weighting.
        # ----------------------------------------------------------------
        chain_clusters: Dict[str, int] = {}
        for record in records:
            for chain in record.chains:
                if not chain.valid:
                    continue
                cluster_id = get_chain_cluster(chain, record)
                if cluster_id not in chain_clusters:
                    chain_clusters[cluster_id] = 0
                chain_clusters[cluster_id] += 1

        # ----------------------------------------------------------------
        # Step 2: Compute interface cluster sizes across the full dataset.
        # Interface clusters are pairs of chain clusters (sorted).
        # ----------------------------------------------------------------
        interface_clusters: Dict[str, int] = {}
        for record in records:
            for interface in record.interfaces:
                if not interface.valid:
                    continue
                cluster_id = get_interface_cluster(interface, record)
                if cluster_id not in interface_clusters:
                    interface_clusters[cluster_id] = 0
                interface_clusters[cluster_id] += 1

        # ----------------------------------------------------------------
        # Step 3: Build the sampling pool with weights.
        # Each item is (record, kind, index) where:
        #   kind=0 -> chain sample, index is chain_id
        #   kind=1 -> interface sample, index is interface_id
        # ----------------------------------------------------------------
        items, weights = [], []
        for record in records:
            # Add valid chains to the pool
            for chain_id, chain in enumerate(record.chains):
                if not chain.valid:
                    continue
                weight = get_chain_weight(
                    chain,
                    record,
                    chain_clusters,
                    self.beta_chain,
                    self.alpha_prot,
                    self.alpha_nucl,
                    self.alpha_ligand,
                )
                items.append((record, 0, chain_id))
                weights.append(weight)

            # Add valid interfaces to the pool
            for int_id, interface in enumerate(record.interfaces):
                if not interface.valid:
                    continue
                weight = get_interface_weight(
                    interface,
                    record,
                    interface_clusters,
                    self.beta_interface,
                    self.alpha_prot,
                    self.alpha_nucl,
                    self.alpha_ligand,
                )
                items.append((record, 1, int_id))
                weights.append(weight)

        # Normalize weights to form a valid probability distribution
        weights = np.array(weights) / np.sum(weights)

        # Infinite sampling loop: draw from weighted distribution
        while True:
            item_idx = random.choice(len(items), p=weights)
            record, kind, index = items[item_idx]
            if kind == 0:
                # Chain sample: set chain_id in the yielded Sample
                yield Sample(record=record, chain_id=index)
            else:
                # Interface sample: set interface_id in the yielded Sample
                yield Sample(record=record, interface_id=index)
