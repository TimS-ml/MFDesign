"""MMSeqs2 server integration for remote MSA computation.

This module provides a client for the ColabFold MMSeqs2 API, which performs
fast sequence database searches to generate Multiple Sequence Alignments (MSAs).
The code is adapted from the ColabFold project:
https://github.com/sokrypton/ColabFold/blob/main/colabfold/colabfold.py

Workflow overview:
    1. **Submit**: Send query sequences to the MMSeqs2 API server. Each
       sequence is assigned a numeric ID (starting from 101) and formatted
       as FASTA for submission.
    2. **Poll**: Check the job status periodically until it completes or
       fails. Handles transient errors, rate limits, and maintenance windows.
    3. **Download**: Retrieve the results as a tar.gz archive containing
       A3M-formatted MSA files.
    4. **Parse**: Extract A3M files and organize results by sequence ID.

The API supports several search modes:
    - ``env``: Search against environmental databases (default).
    - ``all``: Search against all databases without environmental filtering.
    - ``nofilter`` / ``env-nofilter``: Disable result filtering.
    - ``pairgreedy`` / ``paircomplete``: Paired MSA modes for multi-chain inputs.

Result format:
    - For unpaired mode: Returns per-sequence A3M lines from UniRef and
      optionally BFD/Mgnify environmental databases.
    - For paired mode: Returns paired A3M lines where homologs are matched
      across query sequences (sequences without a match get "DUMMY").

Error handling:
    - Timeout errors trigger automatic retries.
    - General errors are retried up to 5 times before raising.
    - API status errors (ERROR, MAINTENANCE) raise descriptive exceptions.

Caching:
    - Results are cached as tar.gz files on disk. If the archive already
      exists, the API call is skipped and cached results are used directly.
"""

# From https://github.com/sokrypton/ColabFold/blob/main/colabfold/colabfold.py

import logging
import os
import random
import tarfile
import time
from typing import Union

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

TQDM_BAR_FORMAT = (
    "{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed} remaining: {remaining}]"
)


def run_mmseqs2(  # noqa: PLR0912, D103, C901, PLR0915
    x: Union[str, list[str]],
    prefix: str = "tmp",
    use_env: bool = True,
    use_filter: bool = True,
    use_pairing: bool = False,
    pairing_strategy: str = "greedy",
    host_url: str = "https://api.colabfold.com",
) -> tuple[list[str], list[str]]:
    """Run MMSeqs2 sequence search via the ColabFold API server.

    Submits one or more protein sequences to the remote MMSeqs2 server,
    waits for completion, downloads the results, and returns per-sequence
    A3M-formatted MSA lines.

    Parameters
    ----------
    x : str or list[str]
        A single sequence string or a list of sequence strings to search.
    prefix : str
        Directory prefix for caching results on disk.
    use_env : bool
        Whether to include environmental database results (BFD, Mgnify, etc.).
    use_filter : bool
        Whether to apply MMSeqs2 result filtering.
    use_pairing : bool
        Whether to use paired MSA mode for multi-chain complexes.
    pairing_strategy : str
        Pairing strategy: 'greedy' (default) or 'complete'.
    host_url : str
        The ColabFold API server URL.

    Returns
    -------
    list[str]
        A list of A3M-formatted strings, one per input sequence. Each string
        contains the query sequence followed by its MSA hits in A3M format.
    """
    # Choose the API endpoint based on pairing mode
    submission_endpoint = "ticket/pair" if use_pairing else "ticket/msa"

    # Identify this client to the server
    headers = {}
    headers["User-Agent"] = "boltz"

    def submit(seqs, mode, N=101):
        """Submit sequences to the MMSeqs2 API.

        Formats the sequences as a multi-FASTA string with numeric headers
        (starting from N) and POSTs them to the server.

        Parameters
        ----------
        seqs : list[str]
            The sequences to submit.
        mode : str
            The search mode (e.g., 'env', 'all', 'pairgreedy').
        N : int
            Starting sequence ID number (default 101).

        Returns
        -------
        dict
            The JSON response containing job ID and status.
        """
        # Build multi-FASTA query string with numeric identifiers
        n, query = N, ""
        for seq in seqs:
            query += f">{n}\n{seq}\n"
            n += 1

        # Retry loop: keeps submitting until a successful response is received
        while True:
            error_count = 0
            try:
                # Use a timeout slightly larger than a multiple of 3 (best practice
                # for connect timeouts per the requests library documentation)
                res = requests.post(
                    f"{host_url}/{submission_endpoint}",
                    data={"q": query, "mode": mode},
                    timeout=6.02,
                    headers=headers,
                )
            except requests.exceptions.Timeout:
                logger.warning("Timeout while submitting to MSA server. Retrying...")
                continue
            except Exception as e:
                error_count += 1
                logger.warning(
                    f"Error while fetching result from MSA server. Retrying... ({error_count}/5)"
                )
                logger.warning(f"Error: {e}")
                time.sleep(5)
                if error_count > 5:
                    raise
                continue
            break

        try:
            out = res.json()
        except ValueError:
            logger.error(f"Server didn't reply with json: {res.text}")
            out = {"status": "ERROR"}
        return out

    def status(ID):
        """Poll the server for the current status of a submitted job.

        Parameters
        ----------
        ID : str
            The job identifier returned by the submit endpoint.

        Returns
        -------
        dict
            The JSON response containing the current job status.
        """
        while True:
            error_count = 0
            try:
                res = requests.get(
                    f"{host_url}/ticket/{ID}", timeout=6.02, headers=headers
                )
            except requests.exceptions.Timeout:
                logger.warning(
                    "Timeout while fetching status from MSA server. Retrying..."
                )
                continue
            except Exception as e:
                error_count += 1
                logger.warning(
                    f"Error while fetching result from MSA server. Retrying... ({error_count}/5)"
                )
                logger.warning(f"Error: {e}")
                time.sleep(5)
                if error_count > 5:
                    raise
                continue
            break
        try:
            out = res.json()
        except ValueError:
            logger.error(f"Server didn't reply with json: {res.text}")
            out = {"status": "ERROR"}
        return out

    def download(ID, path):
        """Download the result archive for a completed job.

        Parameters
        ----------
        ID : str
            The job identifier.
        path : str
            Local file path to save the downloaded tar.gz archive.
        """
        error_count = 0
        while True:
            try:
                res = requests.get(
                    f"{host_url}/result/download/{ID}", timeout=6.02, headers=headers
                )
            except requests.exceptions.Timeout:
                logger.warning(
                    "Timeout while fetching result from MSA server. Retrying..."
                )
                continue
            except Exception as e:
                error_count += 1
                logger.warning(
                    f"Error while fetching result from MSA server. Retrying... ({error_count}/5)"
                )
                logger.warning(f"Error: {e}")
                time.sleep(5)
                if error_count > 5:
                    raise
                continue
            break
        with open(path, "wb") as out:
            out.write(res.content)

    # Normalize input: ensure we always work with a list of sequences
    seqs = [x] if isinstance(x, str) else x

    # Determine the search mode based on configuration flags.
    # Modes control which databases are searched and whether filtering is applied.
    if use_filter:
        mode = "env" if use_env else "all"
    else:
        mode = "env-nofilter" if use_env else "nofilter"

    # Override mode for paired MSA searches (multi-chain complexes)
    if use_pairing:
        mode = ""
        if pairing_strategy == "greedy":
            mode = "pairgreedy"
        elif pairing_strategy == "complete":
            mode = "paircomplete"
        if use_env:
            mode = mode + "-env"

    # Create output directory for caching results
    path = f"{prefix}_{mode}"
    if not os.path.isdir(path):
        os.mkdir(path)

    # Path for the downloaded result archive
    tar_gz_file = f"{path}/out.tar.gz"

    # N=101 is the starting sequence ID; REDO controls the submission retry loop
    N, REDO = 101, True

    # Deduplicate input sequences while preserving order.
    # This avoids redundant searches for identical chains.
    seqs_unique = []
    [seqs_unique.append(x) for x in seqs if x not in seqs_unique]

    # Map each input sequence to its deduplicated ID.
    # Ms maps input order -> sequence ID (101, 102, ...) for result retrieval.
    Ms = [N + seqs_unique.index(seq) for seq in seqs]

    # Only submit if we do not already have cached results
    if not os.path.isfile(tar_gz_file):
        # Estimate total time based on ~150 seconds per unique sequence
        TIME_ESTIMATE = 150 * len(seqs_unique)

        with tqdm(total=TIME_ESTIMATE, bar_format=TQDM_BAR_FORMAT) as pbar:
            while REDO:
                pbar.set_description("SUBMIT")

                # Submit the job; retry if server returns UNKNOWN or RATELIMIT
                out = submit(seqs_unique, mode, N)
                while out["status"] in ["UNKNOWN", "RATELIMIT"]:
                    sleep_time = 5 + random.randint(0, 5)
                    logger.error(f"Sleeping for {sleep_time}s. Reason: {out['status']}")
                    time.sleep(sleep_time)
                    out = submit(seqs_unique, mode, N)

                # Handle fatal server errors
                if out["status"] == "ERROR":
                    msg = (
                        "MMseqs2 API is giving errors. Please confirm your "
                        " input is a valid protein sequence. If error persists, "
                        "please try again an hour later."
                    )
                    raise Exception(msg)

                # Handle server maintenance windows
                if out["status"] == "MAINTENANCE":
                    msg = (
                        "MMseqs2 API is undergoing maintenance. "
                        "Please try again in a few minutes."
                    )
                    raise Exception(msg)

                # Poll for job completion
                ID, TIME = out["id"], 0
                pbar.set_description(out["status"])
                while out["status"] in ["UNKNOWN", "RUNNING", "PENDING"]:
                    t = 5 + random.randint(0, 5)
                    logger.error(f"Sleeping for {t}s. Reason: {out['status']}")
                    time.sleep(t)
                    out = status(ID)
                    pbar.set_description(out["status"])
                    if out["status"] == "RUNNING":
                        TIME += t
                        pbar.update(n=t)

                # Job finished successfully
                if out["status"] == "COMPLETE":
                    if TIME < TIME_ESTIMATE:
                        pbar.update(n=(TIME_ESTIMATE - TIME))
                    REDO = False

                # Job failed after running
                if out["status"] == "ERROR":
                    REDO = False
                    msg = (
                        "MMseqs2 API is giving errors. Please confirm your "
                        " input is a valid protein sequence. If error persists, "
                        "please try again an hour later."
                    )
                    raise Exception(msg)

            # Download the results archive to disk
            download(ID, tar_gz_file)

    # Determine which A3M files to expect based on search mode
    if use_pairing:
        # Paired mode produces a single file with interleaved paired alignments
        a3m_files = [f"{path}/pair.a3m"]
    else:
        # Unpaired mode produces UniRef results, plus optionally environmental DB results
        a3m_files = [f"{path}/uniref.a3m"]
        if use_env:
            a3m_files.append(f"{path}/bfd.mgnify30.metaeuk30.smag30.a3m")

    # Extract the archive if any expected A3M files are missing
    if any(not os.path.isfile(a3m_file) for a3m_file in a3m_files):
        with tarfile.open(tar_gz_file) as tar_gz:
            tar_gz.extractall(path)

    # Parse A3M files and organize results by sequence ID.
    # Each A3M file may contain results for multiple query sequences,
    # delimited by null characters (\x00) between sequences.
    a3m_lines = {}
    for a3m_file in a3m_files:
        update_M, M = True, None
        for line in open(a3m_file, "r"):
            if len(line) > 0:
                if "\x00" in line:
                    # Null character separates results for different query sequences
                    line = line.replace("\x00", "")
                    update_M = True
                if line.startswith(">") and update_M:
                    # Extract the numeric sequence ID from the header
                    M = int(line[1:].rstrip())
                    update_M = False
                    if M not in a3m_lines:
                        a3m_lines[M] = []
                # Accumulate all lines for the current sequence
                a3m_lines[M].append(line)

    # Reorder results to match the original input sequence order.
    # Ms maps each input sequence to its assigned numeric ID.
    a3m_lines = ["".join(a3m_lines[n]) for n in Ms]

    return a3m_lines
