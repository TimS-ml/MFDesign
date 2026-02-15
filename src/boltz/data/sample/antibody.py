"""Antibody-specific sampling strategy for training data selection.

This module implements a simple random sampler with replacement that is
tailored for antibody datasets. Unlike the general cluster-based sampler,
this sampler treats each valid antibody chain (heavy or light) as an
independent sampling unit.

Sampling logic:
    1. During initialization, the sampler iterates over all records and
       collects (record, chain_id) pairs for each valid H-chain and L-chain.
       This means a single antibody with both H and L chains produces two
       separate items in the sampling pool.
    2. At each iteration, one item is drawn uniformly at random (with
       replacement), yielding a ``Sample`` whose ``chain_id`` identifies the
       selected chain. This chain_id is later used by the cropper to decide
       the spatial query center.

This approach ensures that both heavy and light chains are equally represented
in the training batches, regardless of whether they come from the same or
different antibody records.
"""

from dataclasses import replace
from typing import Iterator, List

from numpy.random import RandomState

from boltz.data.types import Record, AntibodyInfo
from boltz.data.sample.sampler import Sample, Sampler


class AntibodySampler(Sampler):
    """A simple random sampler with replacement for antibody structures.

    Each valid heavy (H) and light (L) chain in the dataset is treated as
    an independent sampling unit. The sampler yields (record, chain_id) pairs
    infinitely, drawing uniformly at random with replacement.
    """

    def sample(self, records: List[Record], random: RandomState) -> Iterator[Sample]:
        """Sample a structure from the antibody dataset infinitely.

        Parameters
        ----------
        records : List[Record]
            The records to sample from.
        random : RandomState
            The random state for reproducibility.

        Yields
        ------
        Sample
            A data sample containing the record and the selected chain ID.

        """
        # Build the pool of (record, chain_id) items from valid H and L chains.
        # Each valid chain becomes an independent entry so that both chains
        # of a single antibody can be sampled independently.
        items = []
        for record in records:
            # This sampler requires AntibodyInfo records, which provide
            # H_chain_id and L_chain_id attributes.
            assert isinstance(record.structure, AntibodyInfo)
            h_chain_id = record.structure.H_chain_id
            l_chain_id = record.structure.L_chain_id

            # Add the heavy chain if it exists and is marked as valid
            if h_chain_id is not None and record.chains[h_chain_id].valid:
                items.append((record, h_chain_id))

            # Add the light chain if it exists and is marked as valid
            if l_chain_id is not None and record.chains[l_chain_id].valid:
                items.append((record, l_chain_id))

        # Infinite sampling loop: draw uniformly at random with replacement
        while True:
            item_idx = random.randint(len(items))
            record, index = items[item_idx]
            yield Sample(record=record, chain_id=index)
