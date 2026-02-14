"""CSV-based MSA parser with antibody CDR-aware filtering.

This module provides two CSV parsers for Multiple Sequence Alignment (MSA) data:

1. ``parse_csv``: A general-purpose parser that reads a CSV file with columns
   ``sequence`` and ``key``, deduplicates sequences, and converts them into
   the internal ``MSA`` representation.

2. ``parse_csv_for_ab_design``: An antibody-design-specific parser that
   additionally filters out MSA sequences sharing high sequence identity with
   the query's CDR (Complementarity-Determining Region) loops. This prevents
   the MSA from leaking CDR information that the model should learn to predict.

MSA filtering logic (``filter_msa_df``):
    - The query sequence (first row) has its CDR positions identified by 'X'
      characters in the reference masked sequence.
    - Three CDR regions (CDR1, CDR2, CDR3) are extracted by finding contiguous
      runs of 'X' positions with gaps between them.
    - For each MSA hit, its CDR regions are compared to the query's CDRs. If
      ANY CDR region has sequence identity >= the threshold (default 0.2), the
      hit is **removed** from the MSA.
    - The first entry (query itself) is always retained.

Sequence processing details:
    - Lowercase characters in sequences represent insertions relative to the
      query and are counted as deletions but not included in the alignment.
    - Gap characters ('-') are kept as normal residue tokens (gap token).
    - Duplicate sequences (ignoring gaps and case) are removed.
    - Taxonomy IDs from the ``key`` column are preserved when available.
"""

from pathlib import Path
from typing import Optional, Tuple
from typing import List
import numpy as np
import pandas as pd

from boltz.data import const
from boltz.data.types import MSA, MSADeletion, MSAResidue, MSASequence

def get_cdr_indices(ref_masked_seq: str) -> List[int]:
    """Extract the start and end indices of three CDR regions from a masked sequence.

    The masked sequence uses 'X' characters to mark CDR positions. This function
    identifies three contiguous CDR regions by finding breakpoints (positions
    where consecutive 'X' characters are non-adjacent).

    Parameters
    ----------
    ref_masked_seq : str
        The reference masked sequence where CDR positions are marked with 'X'
        and framework positions use standard amino acid letters.

    Returns
    -------
    List[int]
        A flat list of 6 integers: [CDR1_start, CDR1_end, CDR2_start, CDR2_end,
        CDR3_start, CDR3_end]. Each (start, end) pair defines a half-open
        interval [start, end) for slicing.
    """
    # Find all positions marked as 'X' (CDR residues)
    x_indices = [i for i, char in enumerate(ref_masked_seq) if char == 'X']

    # Identify breakpoints where the gap between consecutive X positions is > 1,
    # indicating the boundary between two distinct CDR regions
    breaks = []
    for i in range(1, len(x_indices)):
        if x_indices[i] - x_indices[i-1] > 1:
            breaks.append(i)

    # Extract start/end indices for each of the three CDR regions.
    # End indices are +1 to create half-open intervals for Python slicing.
    cdr_indices = [
        x_indices[0], x_indices[breaks[0]-1] + 1,  # CDR1: [start, end)
        x_indices[breaks[0]], x_indices[breaks[1]-1] + 1,  # CDR2: [start, end)
        x_indices[breaks[1]], x_indices[-1] + 1  # CDR3: [start, end)
    ]

    return cdr_indices



def filter_msa_df(input_df: pd.DataFrame, query_seq: str, ref_masked_seq: str, msa_filtering_threshold: float) -> pd.DataFrame:
    """Filter MSA sequences that share high CDR identity with the query.

    This function removes MSA hits whose CDR regions are too similar to the
    query antibody's CDRs. This is critical for antibody design: if MSA
    sequences contain near-identical CDR loops, the model could simply copy
    them rather than learning to generate novel sequences.

    The filtering logic:
        1. Replace the query sequence in the DataFrame with the masked version
           (CDR positions replaced with 'X').
        2. For each MSA hit, extract the CDR1, CDR2, CDR3 subsequences.
        3. If ANY CDR has sequence identity >= threshold with the query CDR,
           discard the hit.
        4. Return only the surviving sequences.

    Parameters
    ----------
    input_df : pd.DataFrame
        MSA DataFrame with 'sequence' and 'key' columns.
    query_seq : str
        The original (unmasked) query sequence for CDR comparison.
    ref_masked_seq : str
        The reference masked sequence with 'X' at CDR positions.
    msa_filtering_threshold : float
        Maximum allowed CDR sequence identity (0.0 to 1.0). Sequences with
        identity >= this value in any CDR are removed.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only sequences that pass the CDR
        identity filter.
    """
    # Replace the first entry (query) with the masked version so downstream
    # processing sees 'X' at CDR positions instead of amino acid letters
    input_df.at[0, "sequence"] = ref_masked_seq

    # Always keep the query sequence (index 0)
    reserved_entry_indices = [0]

    # Determine CDR boundaries from the masked reference sequence
    CDR_indices = get_cdr_indices(ref_masked_seq)

    # Extract the query's CDR subsequences for comparison
    query_cdr1 = query_seq[CDR_indices[0]:CDR_indices[1]]
    query_cdr2 = query_seq[CDR_indices[2]:CDR_indices[3]]
    query_cdr3 = query_seq[CDR_indices[4]:CDR_indices[5]]



    def calculate_similarity(seq1: str, seq2: str) -> float:
        """Calculate the fraction of identical residues between two sequences.

        Parameters
        ----------
        seq1 : str
            First sequence (query CDR).
        seq2 : str
            Second sequence (hit CDR, must be same length as seq1).

        Returns
        -------
        float
            Fraction of positions with matching amino acids.
        """
        matches = sum(a == b for a, b in zip(seq1, seq2))
        return float(matches) / len(seq1)

    # Iterate over all MSA hits (skip the query at index 0)
    for idx, entry in enumerate(input_df.iloc[1:].iterrows(), start=1):
        # Remove lowercase characters (insertions in A3M format) to get the
        # aligned sequence that matches the query length
        entry_seq = ''.join([c for c in entry[1]["sequence"] if not c.islower()])

        # Extract CDR regions from the MSA hit at the same positions as the query
        entry_cdr1 = entry_seq[CDR_indices[0]:CDR_indices[1]]
        entry_cdr2 = entry_seq[CDR_indices[2]:CDR_indices[3]]
        entry_cdr3 = entry_seq[CDR_indices[4]:CDR_indices[5]]

        # Discard the hit if ANY CDR exceeds the similarity threshold.
        # This aggressive filtering ensures no CDR information leaks
        # from the MSA into the model during antibody design.
        if calculate_similarity(query_cdr1, entry_cdr1) >= msa_filtering_threshold or \
            calculate_similarity(query_cdr2, entry_cdr2) >= msa_filtering_threshold or \
            calculate_similarity(query_cdr3, entry_cdr3) >= msa_filtering_threshold:
            continue
        else:
            reserved_entry_indices.append(idx)

    return input_df.iloc[reserved_entry_indices]

def parse_csv(
    path: Path,
    max_seqs: Optional[int] = None,
) -> MSA:
    """Parse a CSV file containing MSA data into an MSA object.

    Reads a CSV with columns 'sequence' and 'key', deduplicates sequences,
    and converts them into the internal MSA representation with residue tokens,
    deletion counts, and taxonomy information.

    Parameters
    ----------
    path : Path
        The path to the CSV file.
    max_seqs : int, optional
        The maximum number of sequences to include.

    Returns
    -------
    MSA
        The MSA object containing residues, deletions, and sequence metadata.

    """
    # Read file
    data = pd.read_csv(path)

    # Validate CSV format: must have exactly 'sequence' and 'key' columns
    if tuple(sorted(data.columns)) != ("key", "sequence"):
        msg = "Invalid CSV format, expected columns: ['sequence', 'key']"
        raise ValueError(msg)

    # Track visited sequences for deduplication (gap-free, uppercase)
    visited = set()
    sequences = []
    deletions = []
    residues = []

    seq_idx = 0
    for line, key in zip(data["sequence"], data["key"]):
        line: str
        line = line.strip()  # noqa: PLW2901
        if not line:
            continue

        # Extract taxonomy ID from the 'key' column if available.
        # Use -1 as sentinel for missing taxonomy.
        taxonomy_id = -1
        if (str(key) != "nan") and (key is not None) and (key != ""):
            taxonomy_id = key

        # Deduplicate: skip sequences that, after removing gaps and
        # converting to uppercase, have already been seen
        str_seq = line.replace("-", "").upper()
        if str_seq not in visited:
            visited.add(str_seq)
        else:
            continue

        # Process the alignment sequence character by character.
        # - Uppercase and '-': treated as aligned residues, converted to token IDs
        # - Lowercase (not '-'): insertions relative to query, counted as deletions
        residue = []
        deletion = []
        count = 0
        res_idx = 0
        for c in line:
            if c != "-" and c.islower():
                # Lowercase = insertion in A3M format, count as deletion
                count += 1
                continue
            # Convert the character to its token ID via the protein letter mapping
            token = const.prot_letter_to_token[c]
            token = const.token_ids[token]
            residue.append(token)
            # Record accumulated deletion count at this position
            if count > 0:
                deletion.append((res_idx, count))
                count = 0
            res_idx += 1

        # Record the index ranges for this sequence's residues and deletions
        # in the global flat arrays
        res_start = len(residues)
        res_end = res_start + len(residue)

        del_start = len(deletions)
        del_end = del_start + len(deletion)

        sequences.append((seq_idx, taxonomy_id, res_start, res_end, del_start, del_end))
        residues.extend(residue)
        deletions.extend(deletion)

        seq_idx += 1
        if (max_seqs is not None) and (seq_idx >= max_seqs):
            break

    # Create MSA object from flat arrays with structured dtypes
    msa = MSA(
        residues=np.array(residues, dtype=MSAResidue),
        deletions=np.array(deletions, dtype=MSADeletion),
        sequences=np.array(sequences, dtype=MSASequence),
    )
    return msa



def parse_csv_for_ab_design(
    path: Path,
    entry_info: dict,
    max_seqs: Optional[int] = None,
    msa_filtering_threshold: float = 0.2,
) -> MSA:
    """Parse a CSV MSA file with antibody CDR-aware filtering.

    This function extends ``parse_csv`` by first filtering out MSA sequences
    that share high CDR identity with the query antibody. The entity_id is
    extracted from the filename to determine whether the MSA belongs to the
    heavy chain (entity_id=0) or light chain (entity_id=1).

    Parameters
    ----------
    path : Path
        The path to the CSV file (filename format: ``*_<entity_id>.csv``).
    entry_info : dict
        Dictionary containing antibody information with keys:
        - ``H_chain_seq``: heavy chain sequence
        - ``H_chain_masked_seq``: heavy chain with CDRs masked as 'X'
        - ``L_chain_id``: light chain ID (None if no light chain)
        - ``L_chain_seq``: light chain sequence
        - ``L_chain_masked_seq``: light chain with CDRs masked as 'X'
    max_seqs : int, optional
        The maximum number of sequences to include.
    msa_filtering_threshold : float, optional
        CDR identity threshold for filtering (default 0.2 = 20% identity).

    Returns
    -------
    tuple[MSA, int, int]
        A tuple of (MSA object, num_sequences_before_filtering,
        num_sequences_after_filtering).

    """
    # Read file
    data = pd.read_csv(path)
    num_before_filtering = len(data)
    # Check columns
    if tuple(sorted(data.columns)) != ("key", "sequence"):
        msg = "Invalid CSV format, expected columns: ['sequence', 'key']"
        raise ValueError(msg)

    # Determine which chain this MSA belongs to based on the entity_id
    # encoded in the filename (e.g., "msa_0.csv" -> entity_id "0")
    file_name = path.stem
    entity_id = file_name.split("_")[-1]

    # Apply CDR-aware filtering based on chain type:
    # entity_id "0" = heavy chain, entity_id "1" = light chain (if present)
    if entity_id == "0":
        data = filter_msa_df(data, entry_info["H_chain_seq"], entry_info["H_chain_masked_seq"], msa_filtering_threshold)
    elif entity_id == "1" and not entry_info["L_chain_id"] == None:
        data = filter_msa_df(data, entry_info["L_chain_seq"], entry_info["L_chain_masked_seq"], msa_filtering_threshold)

    num_after_filtering = len(data)

    # The rest of the parsing is identical to parse_csv:
    # deduplicate, tokenize, and build the MSA object.

    # Track visited sequences for deduplication
    visited = set()
    sequences = []
    deletions = []
    residues = []

    seq_idx = 0
    for line, key in zip(data["sequence"], data["key"]):
        line: str
        line = line.strip()  # noqa: PLW2901
        if not line:
            continue

        # Get taxonomy, if annotated
        taxonomy_id = -1
        if (str(key) != "nan") and (key is not None) and (key != ""):
            taxonomy_id = key

        # Skip if duplicate sequence
        str_seq = line.replace("-", "").upper()
        if str_seq not in visited:
            visited.add(str_seq)
        else:
            continue

        # Process sequence character by character
        residue = []
        deletion = []
        count = 0
        res_idx = 0
        for c in line:
            if c != "-" and c.islower():
                # Lowercase characters represent insertions (A3M format);
                # they are not aligned columns but are tracked as deletions
                count += 1
                continue
            # Convert character to three-letter token name (e.g., 'A' -> 'ALA')
            token = const.prot_letter_to_token[c]

            # Convert the token name to its integer ID
            token = const.token_ids[token]

            residue.append(token)
            # Gap character '-' is also treated as a valid residue token
            if count > 0:
                # If count > 0, there were insertions (deletions in MSA terms)
                # immediately before the current aligned position
                deletion.append((res_idx, count))
                count = 0
            res_idx += 1

        # Record index ranges into the global flat residue/deletion arrays
        res_start = len(residues)
        res_end = res_start + len(residue)
        del_start = len(deletions)
        del_end = del_start + len(deletion)

        sequences.append((seq_idx, taxonomy_id, res_start, res_end, del_start, del_end))
        residues.extend(residue)
        deletions.extend(deletion)

        seq_idx += 1
        if (max_seqs is not None) and (seq_idx >= max_seqs):
            break

    # Create MSA object from flat arrays
    msa = MSA(
        residues=np.array(residues, dtype=MSAResidue),
        deletions=np.array(deletions, dtype=MSADeletion),
        sequences=np.array(sequences, dtype=MSASequence),
    )
    return msa, num_before_filtering, num_after_filtering
