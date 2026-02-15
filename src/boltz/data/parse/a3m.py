"""A3M format Multiple Sequence Alignment (MSA) parser.

This module parses MSA files in the A3M format, which is a compressed variant
of the FASTA alignment format commonly produced by HHblits and MMSeqs2.

A3M format conventions:
    - Lines starting with '>' are header lines containing sequence identifiers
      and optional metadata (e.g., UniRef100 accession IDs).
    - Lines starting with '#' are comment lines and are skipped.
    - Uppercase letters and '-' (gaps) represent aligned columns that
      correspond to positions in the query sequence.
    - Lowercase letters represent insertions relative to the query -- these
      are NOT aligned columns. They are counted as "deletions" in the internal
      representation (deletion = insertion in the hit relative to query).

The parser handles:
    - Plain text (.a3m) and gzip-compressed (.a3m.gz) files.
    - Taxonomy extraction from UniRef100 headers when a taxonomy database
      is provided.
    - Sequence deduplication (case-insensitive, gap-stripped).
    - Optional maximum sequence count limiting.

The output is an ``MSA`` object containing three flat NumPy arrays:
    - ``residues``: Token IDs for each aligned position across all sequences.
    - ``deletions``: (position, count) pairs recording insertion lengths.
    - ``sequences``: Metadata per sequence (index, taxonomy, residue/deletion ranges).
"""

import gzip
from pathlib import Path
from typing import Optional, TextIO

import numpy as np

from boltz.data import const
from boltz.data.types import MSA, MSADeletion, MSAResidue, MSASequence


def _parse_a3m(  # noqa: C901
    lines: TextIO,
    taxonomy: Optional[dict[str, str]],
    max_seqs: Optional[int] = None,
) -> MSA:
    """Parse an A3M-formatted MSA from a line iterator.

    This is the core parsing function that processes the A3M file line by line.
    It handles header parsing, taxonomy lookup, sequence deduplication, and
    tokenization of aligned residues and insertions.

    Parameters
    ----------
    lines : TextIO
        An iterable of text lines (from a file handle or gzip reader).
    taxonomy : dict[str, str], optional
        A dictionary mapping UniRef100 member IDs to taxonomy IDs. If None,
        taxonomy information is not extracted.
    max_seqs : int, optional
        The maximum number of unique sequences to include. Parsing stops
        once this limit is reached.

    Returns
    -------
    MSA
        The MSA object containing tokenized residues, deletion profiles,
        and per-sequence metadata.

    """
    # Track visited sequences (gap-free, uppercase) to skip duplicates
    visited = set()

    # Flat accumulators for the three MSA arrays
    sequences = []   # Per-sequence metadata tuples
    deletions = []   # (position, count) deletion entries across all sequences
    residues = []    # Token IDs for aligned positions across all sequences

    seq_idx = 0
    for line in lines:
        line: str
        line = line.strip()  # noqa: PLW2901

        # Skip empty lines and comment lines
        if not line or line.startswith("#"):
            continue

        # Parse header lines to extract taxonomy information
        if line.startswith(">"):
            header = line.split()[0]
            # Attempt to extract taxonomy from UniRef100 headers.
            # Header format: ">UniRef100_<member_id> ..."
            if taxonomy and header.startswith(">UniRef100"):
                uniref_id = header.split("_")[1]
                taxonomy_id = taxonomy.get(uniref_id)
                if taxonomy_id is None:
                    taxonomy_id = -1
            else:
                # No taxonomy database or non-UniRef header
                taxonomy_id = -1
            continue

        # Process the sequence line (non-header, non-comment, non-empty)

        # Deduplicate: compare gap-stripped uppercase sequences
        str_seq = line.replace("-", "").upper()
        if str_seq not in visited:
            visited.add(str_seq)
        else:
            continue

        # Tokenize the sequence character by character:
        # - Uppercase + '-': aligned columns -> convert to token IDs
        # - Lowercase (not '-'): insertions -> count as deletions
        residue = []
        deletion = []
        count = 0       # Running count of consecutive lowercase (insertion) chars
        res_idx = 0     # Index into aligned columns
        for c in line:
            if c != "-" and c.islower():
                # Lowercase = insertion relative to query; accumulate count
                count += 1
                continue
            # Convert aligned character to token ID
            token = const.prot_letter_to_token[c]
            token = const.token_ids[token]
            residue.append(token)
            # If there were insertions before this position, record them
            if count > 0:
                deletion.append((res_idx, count))
                count = 0
            res_idx += 1

        # Record index ranges for this sequence in the global flat arrays
        res_start = len(residues)
        res_end = res_start + len(residue)

        del_start = len(deletions)
        del_end = del_start + len(deletion)

        # Append sequence metadata: (index, taxonomy, residue range, deletion range)
        sequences.append((seq_idx, taxonomy_id, res_start, res_end, del_start, del_end))
        residues.extend(residue)
        deletions.extend(deletion)

        seq_idx += 1
        # Stop if we have reached the maximum number of sequences
        if (max_seqs is not None) and (seq_idx >= max_seqs):
            break

    # Convert flat lists to structured NumPy arrays
    msa = MSA(
        residues=np.array(residues, dtype=MSAResidue),
        deletions=np.array(deletions, dtype=MSADeletion),
        sequences=np.array(sequences, dtype=MSASequence),
    )
    return msa


def parse_a3m(
    path: Path,
    taxonomy: Optional[dict[str, str]],
    max_seqs: Optional[int] = None,
) -> MSA:
    """Parse an A3M file, supporting both plain text and gzip compression.

    This is the public entry point that handles file I/O and delegates to
    ``_parse_a3m`` for the actual parsing logic.

    Parameters
    ----------
    path : Path
        The path to the A3M file. If the suffix is '.gz', the file is
        opened with gzip decompression.
    taxonomy : dict[str, str], optional
        The taxonomy database mapping UniRef100 IDs to taxonomy IDs.
    max_seqs : int, optional
        The maximum number of sequences to include.

    Returns
    -------
    MSA
        The MSA object.

    """
    # Open the file with appropriate handler based on compression
    if path.suffix == ".gz":
        with gzip.open(str(path), "rt") as f:
            msa = _parse_a3m(f, taxonomy, max_seqs)
    else:
        with path.open("r") as f:
            msa = _parse_a3m(f, taxonomy, max_seqs)

    return msa
