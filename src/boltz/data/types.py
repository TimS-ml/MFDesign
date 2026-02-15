"""Data type definitions for the Boltz structural prediction pipeline.

This module defines all core data types used throughout the codebase, organized
into several categories:

Serialization base classes
--------------------------
- ``NumpySerializable`` : Base class for dataclasses that serialize to/from
  compressed NumPy ``.npz`` files.
- ``JSONSerializable``  : Base class (via mashumaro) for dataclasses that
  serialize to/from JSON files.

Structure types (NumPy structured arrays)
-----------------------------------------
These are lists of ``(field_name, dtype)`` tuples used as dtypes for NumPy
structured arrays.  They define the columnar schema for each structural table:

- ``Atom``       : Per-atom data -- name (4 chars encoded as int8), element
  number, formal charge, 3-D coordinates, conformer coordinates, presence
  mask, chirality type, and CDR atom flag.
- ``Bond``       : Covalent bond between two atoms -- global atom indices for
  both endpoints and bond type (single / double / triple / aromatic / other).
- ``Residue``    : Per-residue data -- name, token type, residue index within
  chain, starting atom index, atom count, center atom index, disto atom index,
  standard flag, presence flag, and CDR residue flag.
- ``Chain``      : Per-chain data -- chain name, molecule type, entity ID,
  symmetry ID, asymmetric-unit ID, starting atom/residue indices, and counts.
- ``Connection`` : Cross-chain covalent connection (e.g. disulfide) specified
  by chain, residue, and atom indices for both endpoints.
- ``Interface``  : A pair of chain indices that form a biological interface.

Structure dataclass
-------------------
- ``Structure``  : Immutable container holding all the above NumPy arrays plus
  a per-chain boolean mask.  Supports loading from ``.npz`` and removing
  invalid (masked-out) chains while re-indexing all references.

MSA types (NumPy structured arrays)
------------------------------------
- ``MSAResidue``  : Single residue in an MSA row -- just the token type.
- ``MSADeletion`` : Records insertion/deletion counts at a given residue
  position within an MSA sequence.
- ``MSASequence`` : Per-sequence metadata -- taxonomy ID, residue/deletion
  start and end indices into the flat residue/deletion arrays.
- ``MSA``         : Immutable container holding sequences, deletions, and
  residues arrays.

Metadata / record types
-----------------------
- ``StructureInfo``    : Optional experimental metadata (resolution, method,
  deposition/release/revision dates, chain and interface counts).
- ``AntibodyInfo``     : Extends ``StructureInfo`` with heavy-chain ID,
  light-chain ID, and antigen chain IDs for antibody-specific tasks.
- ``ChainInfo``        : Per-chain metadata (ID, name, molecule type, cluster
  ID, MSA ID, residue count, validity flag, entity ID).
- ``InterfaceInfo``    : Metadata for a chain-chain interface.
- ``InferenceOptions`` : Binder chain indices and pocket contact residue pairs
  used to condition structure prediction.
- ``Record``           : Top-level metadata record combining structure info,
  chain infos, interface infos, and inference options.  JSON-serializable.

Target type
-----------
- ``Target`` : Combines a ``Record`` with a ``Structure`` and optional
  per-entity sequences.  This is the primary input object consumed by the
  data pipeline.

Manifest type
-------------
- ``Manifest`` : A list of ``Record`` objects, loadable from JSON.  Used to
  index a dataset of targets.

Input type
----------
- ``Input`` : Groups a ``Structure`` with per-chain MSA data and an optional
  ``Record``.  Represents a fully prepared model input.

Tokenized types (NumPy structured arrays)
------------------------------------------
- ``Token``     : Per-token data -- token index, atom range, residue index,
  token type, symmetry/asymmetry/entity IDs, molecule type, center/disto
  atom indices and coordinates, and resolved/disto masks.
- ``TokenBond`` : Bond between two tokens (by token index).
- ``Tokenized`` : Mutable container holding token and bond arrays together
  with the underlying ``Structure`` and MSA data.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
from mashumaro.mixins.dict import DataClassDictMixin

####################################################################################################
# SERIALIZABLE
####################################################################################################


class NumpySerializable:
    """Serializable datatype."""

    @classmethod
    def load(cls: "NumpySerializable", path: Path) -> "NumpySerializable":
        """Load the object from an NPZ file.

        Parameters
        ----------
        path : Path
            The path to the file.

        Returns
        -------
        Serializable
            The loaded object.

        """
        return cls(**np.load(path))

    def dump(self, path: Path) -> None:
        """Dump the object to an NPZ file.

        Parameters
        ----------
        path : Path
            The path to the file.

        """
        np.savez_compressed(str(path), **asdict(self))


class JSONSerializable(DataClassDictMixin):
    """Serializable datatype."""

    @classmethod
    def load(cls: "JSONSerializable", path: Path) -> "JSONSerializable":
        """Load the object from a JSON file.

        Parameters
        ----------
        path : Path
            The path to the file.

        Returns
        -------
        Serializable
            The loaded object.

        """
        with path.open("r") as f:
            return cls.from_dict(json.load(f))

    def dump(self, path: Path) -> None:
        """Dump the object to a JSON file.

        Parameters
        ----------
        path : Path
            The path to the file.

        """
        with path.open("w") as f:
            json.dump(self.to_dict(), f)


####################################################################################################
# STRUCTURE
####################################################################################################

# --- Atom structured array dtype ---
# Each element represents one heavy atom in the structure.
# "name" is a 4-character atom name encoded as 4 x int8 (ASCII offset by -32).
# "element" is the atomic number (1-118).
# "charge" is the formal charge.
# "coords" are the experimental 3-D coordinates (x, y, z) as float32.
# "conformer" are the reference/ideal conformer coordinates from CCD.
# "is_present" indicates whether the atom was experimentally resolved.
# "chirality" encodes the chirality type (see const.chirality_type_ids).
# "is_cdr_atom" flags atoms belonging to CDR (complementarity-determining region) residues.
Atom = [
    ("name", np.dtype("4i1")),         # 4-char atom name as int8 array
    ("element", np.dtype("i1")),       # atomic number
    ("charge", np.dtype("i1")),        # formal charge
    ("coords", np.dtype("3f4")),       # experimental (x, y, z) coordinates
    ("conformer", np.dtype("3f4")),    # ideal/reference conformer coordinates
    ("is_present", np.dtype("?")),     # True if atom is resolved
    ("chirality", np.dtype("i1")),     # chirality type index
    ("is_cdr_atom", np.dtype("?")),    # True if in CDR / design region
]

# --- Bond structured array dtype ---
# Each element represents one covalent bond.
# "atom_1" and "atom_2" are global atom indices (into the flat Atom array).
# "type" is the bond order (0=OTHER, 1=SINGLE, 2=DOUBLE, 3=TRIPLE, 4=AROMATIC).
Bond = [
    ("atom_1", np.dtype("i4")),        # global index of first bonded atom
    ("atom_2", np.dtype("i4")),        # global index of second bonded atom
    ("type", np.dtype("i1")),          # bond type / order
]

# --- Residue structured array dtype ---
# Each element represents one residue (amino acid, nucleotide, or CCD component).
# "name" is the residue name (up to 5 Unicode characters, e.g. "ALA", "DA").
# "res_type" is the token vocabulary index from const.token_ids.
# "res_idx" is the residue index within its parent chain.
# "atom_idx" is the global index of the first atom belonging to this residue.
# "atom_num" is the number of atoms in this residue.
# "atom_center" is the global index of the center atom (CA / C1').
# "atom_disto" is the global index of the distogram atom (CB / base atom).
# "is_standard" is True for residues in the standard token vocabulary.
# "is_present" is True if the residue has resolved coordinates.
# "is_cdr_residue" flags residues in the CDR / design region.
Residue = [
    ("name", np.dtype("<U5")),         # residue name (3-letter code or CCD code)
    ("res_type", np.dtype("i1")),      # token vocabulary index
    ("res_idx", np.dtype("i4")),       # index within parent chain
    ("atom_idx", np.dtype("i4")),      # global index of first atom
    ("atom_num", np.dtype("i4")),      # number of atoms in residue
    ("atom_center", np.dtype("i4")),   # global index of center atom
    ("atom_disto", np.dtype("i4")),    # global index of disto atom
    ("is_standard", np.dtype("?")),    # True if standard residue
    ("is_present", np.dtype("?")),     # True if experimentally resolved
    ("is_cdr_residue", np.dtype("?")), # True if in CDR / design region
]

# --- Chain structured array dtype ---
# Each element represents one chain in the biological assembly.
# "name" is the author-assigned chain name (up to 5 Unicode characters).
# "mol_type" is the molecule type (0=PROTEIN, 1=DNA, 2=RNA, 3=NONPOLYMER).
# "entity_id" groups chains that share the same sequence (symmetric copies).
# "sym_id" is the copy index among chains with the same entity_id.
# "asym_id" is the unique chain index across the entire assembly.
# "atom_idx" / "atom_num" define the atom range in the flat Atom array.
# "res_idx" / "res_num" define the residue range in the flat Residue array.
Chain = [
    ("name", np.dtype("<U5")),         # chain name (e.g. "A", "B")
    ("mol_type", np.dtype("i1")),      # molecule type index
    ("entity_id", np.dtype("i4")),     # entity group identifier
    ("sym_id", np.dtype("i4")),        # symmetry copy index within entity
    ("asym_id", np.dtype("i4")),       # unique chain index
    ("atom_idx", np.dtype("i4")),      # global start index into Atom array
    ("atom_num", np.dtype("i4")),      # number of atoms in chain
    ("res_idx", np.dtype("i4")),       # global start index into Residue array
    ("res_num", np.dtype("i4")),       # number of residues in chain
]

# --- Connection structured array dtype ---
# Each element represents a cross-chain covalent connection (e.g. disulfide bond,
# or a user-specified bond constraint).  Both endpoints are fully specified by
# their chain, residue, and atom global indices.
Connection = [
    ("chain_1", np.dtype("i4")),       # chain index of first endpoint
    ("chain_2", np.dtype("i4")),       # chain index of second endpoint
    ("res_1", np.dtype("i4")),         # residue index of first endpoint
    ("res_2", np.dtype("i4")),         # residue index of second endpoint
    ("atom_1", np.dtype("i4")),        # atom index of first endpoint
    ("atom_2", np.dtype("i4")),        # atom index of second endpoint
]

# --- Interface structured array dtype ---
# Each element represents a biological interface between two chains,
# identified by their global chain indices.
Interface = [
    ("chain_1", np.dtype("i4")),       # first chain index
    ("chain_2", np.dtype("i4")),       # second chain index
]


@dataclass(frozen=True)
class Structure(NumpySerializable):
    """Immutable container holding all structural data for a biological assembly.

    A ``Structure`` owns the flat NumPy structured arrays for atoms, bonds,
    residues, and chains, as well as cross-chain connections, interfaces, and
    a per-chain validity mask.  It can be serialized to / loaded from a
    compressed ``.npz`` file.

    Attributes
    ----------
    atoms : np.ndarray
        Structured array with dtype ``Atom``.  One row per heavy atom.
    bonds : np.ndarray
        Structured array with dtype ``Bond``.  One row per covalent bond
        (intra-residue bonds from CCD components).
    residues : np.ndarray
        Structured array with dtype ``Residue``.  One row per residue.
    chains : np.ndarray
        Structured array with dtype ``Chain``.  One row per chain.
    connections : np.ndarray
        Structured array with dtype ``Connection``.  Cross-chain covalent
        connections and user-specified bond constraints.
    interfaces : np.ndarray
        Structured array with dtype ``Interface``.  Chain-chain interface
        pairs for training / evaluation.
    mask : np.ndarray
        Boolean array of shape ``(num_chains,)`` indicating which chains
        are valid (True) or masked out (False).
    """

    atoms: np.ndarray
    bonds: np.ndarray
    residues: np.ndarray
    chains: np.ndarray
    connections: np.ndarray
    interfaces: np.ndarray
    mask: np.ndarray

    @classmethod
    def load(cls: "Structure", path: Path) -> "Structure":
        """Load a structure from an NPZ file.

        Parameters
        ----------
        path : Path
            The path to the file.

        Returns
        -------
        Structure
            The loaded structure.

        """
        structure = np.load(path)
        return cls(
            atoms=structure["atoms"],
            bonds=structure["bonds"],
            residues=structure["residues"],
            chains=structure["chains"],
            connections=structure["connections"].astype(Connection),
            interfaces=structure["interfaces"],
            mask=structure["mask"],
        )

    def remove_invalid_chains(self) -> "Structure":  # noqa: PLR0915
        """Remove invalid chains.

        Parameters
        ----------
        structure : Structure
            The structure to process.

        Returns
        -------
        Structure
            The structure with masked chains removed.

        """
        entity_counter = {}
        atom_idx, res_idx, chain_idx = 0, 0, 0
        atoms, residues, chains = [], [], []
        atom_map, res_map, chain_map = {}, {}, {}
        for i, chain in enumerate(self.chains):
            # Skip masked chains
            if not self.mask[i]:
                continue

            # Update entity counter
            entity_id = chain["entity_id"]
            if entity_id not in entity_counter:
                entity_counter[entity_id] = 0
            else:
                entity_counter[entity_id] += 1

            # Update the chain
            new_chain = chain.copy()
            new_chain["atom_idx"] = atom_idx
            new_chain["res_idx"] = res_idx
            new_chain["asym_id"] = chain_idx
            new_chain["sym_id"] = entity_counter[entity_id]
            chains.append(new_chain)
            chain_map[i] = chain_idx
            chain_idx += 1

            # Add the chain residues
            res_start = chain["res_idx"]
            res_end = chain["res_idx"] + chain["res_num"]
            for j, res in enumerate(self.residues[res_start:res_end]):
                # Update the residue
                new_res = res.copy()
                new_res["atom_idx"] = atom_idx
                new_res["atom_center"] = (
                    atom_idx + new_res["atom_center"] - res["atom_idx"]
                )
                new_res["atom_disto"] = (
                    atom_idx + new_res["atom_disto"] - res["atom_idx"]
                )
                residues.append(new_res)
                res_map[res_start + j] = res_idx
                res_idx += 1

                # Update the atoms
                start = res["atom_idx"]
                end = res["atom_idx"] + res["atom_num"]
                atoms.append(self.atoms[start:end])
                atom_map.update({k: atom_idx + k - start for k in range(start, end)})
                atom_idx += res["atom_num"]

        # Concatenate the tables
        atoms = np.concatenate(atoms, dtype=Atom)
        residues = np.array(residues, dtype=Residue)
        chains = np.array(chains, dtype=Chain)

        # Update bonds
        bonds = []
        for bond in self.bonds:
            atom_1 = bond["atom_1"]
            atom_2 = bond["atom_2"]
            if (atom_1 in atom_map) and (atom_2 in atom_map):
                new_bond = bond.copy()
                new_bond["atom_1"] = atom_map[atom_1]
                new_bond["atom_2"] = atom_map[atom_2]
                bonds.append(new_bond)

        # Update connections
        connections = []
        for connection in self.connections:
            chain_1 = connection["chain_1"]
            chain_2 = connection["chain_2"]
            res_1 = connection["res_1"]
            res_2 = connection["res_2"]
            atom_1 = connection["atom_1"]
            atom_2 = connection["atom_2"]
            if (atom_1 in atom_map) and (atom_2 in atom_map):
                new_connection = connection.copy()
                new_connection["chain_1"] = chain_map[chain_1]
                new_connection["chain_2"] = chain_map[chain_2]
                new_connection["res_1"] = res_map[res_1]
                new_connection["res_2"] = res_map[res_2]
                new_connection["atom_1"] = atom_map[atom_1]
                new_connection["atom_2"] = atom_map[atom_2]
                connections.append(new_connection)

        # Create arrays
        bonds = np.array(bonds, dtype=Bond)
        connections = np.array(connections, dtype=Connection)
        interfaces = np.array([], dtype=Interface)
        mask = np.ones(len(chains), dtype=bool)

        return Structure(
            atoms=atoms,
            bonds=bonds,
            residues=residues,
            chains=chains,
            connections=connections,
            interfaces=interfaces,
            mask=mask,
        )


####################################################################################################
# MSA
####################################################################################################


# --- MSAResidue structured array dtype ---
# A single residue position within a flattened MSA.  All residues across all
# sequences in the MSA are stored in one contiguous array; individual sequences
# reference slices of this array via MSASequence.res_start / res_end.
MSAResidue = [
    ("res_type", np.dtype("i1")),      # token vocabulary index for this MSA position
]

# --- MSADeletion structured array dtype ---
# Records the number of insertions (deletions in the query frame) at a given
# residue position within an MSA sequence.  Stored sparsely -- only positions
# with non-zero deletion counts appear.
MSADeletion = [
    ("res_idx", np.dtype("i2")),       # residue position index within the sequence
    ("deletion", np.dtype("i2")),      # number of deleted / inserted residues
]

# --- MSASequence structured array dtype ---
# Per-sequence metadata for the MSA.  Each sequence references contiguous
# slices of the flat MSAResidue and MSADeletion arrays.
MSASequence = [
    ("seq_idx", np.dtype("i2")),       # sequence index within the MSA
    ("taxonomy", np.dtype("i4")),      # NCBI taxonomy ID of the source organism
    ("res_start", np.dtype("i4")),     # start index into the MSAResidue array
    ("res_end", np.dtype("i4")),       # end index (exclusive) into MSAResidue array
    ("del_start", np.dtype("i4")),     # start index into the MSADeletion array
    ("del_end", np.dtype("i4")),       # end index (exclusive) into MSADeletion array
]


@dataclass(frozen=True)
class MSA(NumpySerializable):
    """Immutable container for a Multiple Sequence Alignment (MSA).

    The MSA is stored in a column-oriented layout: all residues across all
    sequences are packed into a single flat ``residues`` array, and per-sequence
    metadata in ``sequences`` records the slice boundaries.  Deletions are
    stored sparsely in the ``deletions`` array.

    Attributes
    ----------
    sequences : np.ndarray
        Structured array with dtype ``MSASequence``.  One row per aligned
        sequence in the MSA.
    deletions : np.ndarray
        Structured array with dtype ``MSADeletion``.  Sparse deletion counts.
    residues : np.ndarray
        Structured array with dtype ``MSAResidue``.  Flat array of all
        residue tokens across all sequences.
    """

    sequences: np.ndarray
    deletions: np.ndarray
    residues: np.ndarray


####################################################################################################
# RECORD
####################################################################################################


@dataclass(frozen=True)
class StructureInfo:
    """Experimental metadata for a macromolecular structure.

    All fields are optional because this information may not be available
    when the input is a designed / predicted structure rather than an
    experimentally determined one.

    Attributes
    ----------
    resolution : float or None
        Experimental resolution in angstroms (e.g. from X-ray or cryo-EM).
    method : str or None
        Experimental method (e.g. "X-RAY DIFFRACTION", "ELECTRON MICROSCOPY").
    deposited : str or None
        PDB deposition date string.
    released : str or None
        PDB release date string.
    revised : str or None
        PDB revision date string.
    num_chains : int or None
        Total number of chains in the assembly.
    num_interfaces : int or None
        Number of chain-chain interfaces.
    """

    resolution: Optional[float] = None
    method: Optional[str] = None
    deposited: Optional[str] = None
    released: Optional[str] = None
    revised: Optional[str] = None
    num_chains: Optional[int] = None
    num_interfaces: Optional[int] = None

@dataclass(frozen=True)
class AntibodyInfo(StructureInfo):
    """Extended structure metadata for antibody-specific tasks.

    Inherits all fields from ``StructureInfo`` and adds identifiers for the
    heavy chain, light chain, and antigen chains, which are needed by the
    antibody design pipeline.

    Attributes
    ----------
    H_chain_id : int or None
        ``asym_id`` (global chain index) of the heavy chain.
    L_chain_id : int or None
        ``asym_id`` (global chain index) of the light chain.
    antigen_chain_ids : list[int] or None
        List of ``asym_id`` values for antigen chains.
    """
    H_chain_id: Optional[int] = None
    L_chain_id: Optional[int] = None
    antigen_chain_ids: Optional[list[int]] = None

@dataclass(frozen=False)
class ChainInfo:
    """Per-chain metadata stored in the ``Record``.

    This is a mutable dataclass (``frozen=False``) so that validity and
    other fields can be updated during data filtering.

    Attributes
    ----------
    chain_id : int
        Unique chain index (same as ``asym_id`` in the Chain array).
    chain_name : str
        Author-assigned chain name (e.g. "A", "H").
    mol_type : int
        Molecule type (0=PROTEIN, 1=DNA, 2=RNA, 3=NONPOLYMER).
    cluster_id : str or int
        Sequence cluster identifier used for dataset splitting.
    msa_id : str or int
        MSA identifier -- either a file path (custom MSA), 0 (auto-generate),
        or -1 (no MSA / single-sequence mode).
    num_residues : int
        Number of residues in this chain.
    valid : bool
        Whether this chain passes quality filters.  Defaults to True.
    entity_id : str, int, or None
        Entity identifier grouping symmetric chain copies.
    """

    chain_id: int
    chain_name: str
    mol_type: int
    cluster_id: Union[str, int]
    msa_id: Union[str, int]
    num_residues: int
    valid: bool = True
    entity_id: Optional[Union[str, int]] = None


@dataclass(frozen=True)
class InterfaceInfo:
    """Metadata for a chain-chain interface.

    Attributes
    ----------
    chain_1 : int
        Global chain index of the first chain in the interface.
    chain_2 : int
        Global chain index of the second chain in the interface.
    valid : bool
        Whether this interface passes quality filters.  Defaults to True.
    """

    chain_1: int
    chain_2: int
    valid: bool = True


@dataclass(frozen=True)
class InferenceOptions:
    """Options that condition the inference / generation process.

    Attributes
    ----------
    binders : list[int]
        List of chain indices designated as "binder" chains (e.g. the
        ligand or small molecule to dock).
    pocket : list[tuple[int, int]] or None
        List of (chain_index, residue_index) pairs that define the
        binding pocket on the receptor.  None if no pocket is specified.
    """
    binders: list[int]
    pocket: Optional[list[tuple[int, int]]]


@dataclass(frozen=True)
class Record(JSONSerializable):
    """Top-level metadata record for a single target.

    A ``Record`` is JSON-serializable and contains all the metadata needed to
    identify and describe a target without loading its full atomic structure.
    It is stored alongside the structure ``.npz`` file and is used for dataset
    indexing, filtering, and manifesting.

    Attributes
    ----------
    id : str
        Unique identifier for this target (typically the input file stem).
    structure : AntibodyInfo or StructureInfo
        Experimental / structural metadata.
    chains : list[ChainInfo]
        Per-chain metadata for every chain in the assembly.
    interfaces : list[InterfaceInfo]
        Metadata for chain-chain interfaces.
    inference_options : InferenceOptions or None
        Optional inference conditioning (binder chains, pocket contacts).
    """

    id: str
    structure: Union[AntibodyInfo, StructureInfo]
    chains: list[ChainInfo]
    interfaces: list[InterfaceInfo]
    inference_options: Optional[InferenceOptions] = None


####################################################################################################
# TARGET
####################################################################################################


@dataclass(frozen=True)
class Target:
    """A complete prediction target combining metadata and structural data.

    This is the primary data object produced by the input parsing pipeline
    (``parse_boltz_schema``) and consumed by the data loading / featurization
    stages.

    Attributes
    ----------
    record : Record
        Metadata record (chain info, structure info, inference options).
    structure : Structure
        Full atomic structure (atoms, bonds, residues, chains, etc.).
    sequences : dict[str, str] or None
        Mapping from entity ID to the original one-letter-code sequence
        string.  Used for MSA generation and sequence logging.
    """

    record: Record
    structure: Structure
    sequences: Optional[dict[str, str]] = None


@dataclass(frozen=True)
class Manifest(JSONSerializable):
    """A collection of ``Record`` objects, used as a dataset index / manifest.

    Can be loaded from a JSON file that is either a dict with a ``records``
    key or a bare list of record dicts.

    Attributes
    ----------
    records : list[Record]
        The list of target records in the manifest.
    """

    records: list[Record]

    @classmethod
    def load(cls: "JSONSerializable", path: Path) -> "JSONSerializable":
        """Load the object from a JSON file.

        Parameters
        ----------
        path : Path
            The path to the file.

        Returns
        -------
        Serializable
            The loaded object.

        Raises
        ------
        TypeError
            If the file is not a valid manifest file.

        """
        with path.open("r") as f:
            data = json.load(f)
            if isinstance(data, dict):
                manifest = cls.from_dict(data)
            elif isinstance(data, list):
                records = [Record.from_dict(r) for r in data]
                manifest = cls(records=records)
            else:
                msg = "Invalid manifest file."
                raise TypeError(msg)

        return manifest


####################################################################################################
# INPUT
####################################################################################################


@dataclass(frozen=True)
class Input:
    """A fully prepared model input combining structure and MSA data.

    Attributes
    ----------
    structure : Structure
        The atomic structure to predict / evaluate.
    msa : dict[str, MSA]
        Mapping from MSA identifier (chain name or MSA file path) to the
        corresponding ``MSA`` object.
    record : Record or None
        Optional metadata record.  May be None during inference when only
        the structure and MSA are needed.
    """

    structure: Structure
    msa: dict[str, MSA]
    record: Optional[Record] = None


####################################################################################################
# TOKENS
####################################################################################################

# --- Token structured array dtype ---
# The tokenized representation collapses each residue into a single token for
# the transformer trunk.  Each token carries references back to its atom range,
# residue, chain, and key coordinate information.
Token = [
    ("token_idx", np.dtype("i4")),       # global token index
    ("atom_idx", np.dtype("i4")),        # global start index into the Atom array
    ("atom_num", np.dtype("i4")),        # number of atoms owned by this token
    ("res_idx", np.dtype("i4")),         # global residue index
    ("res_type", np.dtype("i1")),        # token vocabulary index (const.token_ids)
    ("sym_id", np.dtype("i4")),          # symmetry copy index within entity
    ("asym_id", np.dtype("i4")),         # unique chain index
    ("entity_id", np.dtype("i4")),       # entity group identifier
    ("mol_type", np.dtype("i1")),        # molecule type (0=protein, 1=DNA, ...)
    ("center_idx", np.dtype("i4")),      # global atom index of center atom
    ("disto_idx", np.dtype("i4")),       # global atom index of disto atom
    ("center_coords", np.dtype("3f4")),  # 3-D coordinates of center atom
    ("disto_coords", np.dtype("3f4")),   # 3-D coordinates of disto atom
    ("resolved_mask", np.dtype("?")),    # True if center atom is resolved
    ("disto_mask", np.dtype("?")),       # True if disto atom is resolved
]

# --- TokenBond structured array dtype ---
# Represents a bond between two tokens (e.g. peptide bond between consecutive
# residues, or a cross-chain connection).
TokenBond = [
    ("token_1", np.dtype("i4")),         # global index of first token
    ("token_2", np.dtype("i4")),         # global index of second token
]


@dataclass(frozen=False)
class Tokenized:
    """Mutable container for the tokenized representation of a target.

    This is the output of the tokenization step and serves as the direct
    input to the model's data featurization pipeline.  It is mutable
    (``frozen=False``) so that downstream transforms (cropping, masking,
    noise injection) can modify the arrays in place.

    Attributes
    ----------
    tokens : np.ndarray
        Structured array with dtype ``Token``.  One row per token.
    bonds : np.ndarray
        Structured array with dtype ``TokenBond``.  Token-level bonds.
    structure : Structure
        The underlying atomic structure (kept for atom-level operations).
    msa : dict[str, MSA]
        Per-chain MSA data.
    """

    tokens: np.ndarray
    bonds: np.ndarray
    structure: Structure
    msa: dict[str, MSA]
