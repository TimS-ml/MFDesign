"""Global constants for the Boltz structural prediction pipeline.

This module centralizes all hard-coded constants, vocabularies, and lookup
tables used across the data loading, parsing, tokenization, and evaluation
stages.  The constants are organized into the following sections:

Chains
------
- ``chain_types`` / ``chain_type_ids`` : Molecule type vocabulary.
  PROTEIN = 0, DNA = 1, RNA = 2, NONPOLYMER = 3.
- ``STANDARD_RESIDUE_SUBSTITUTIONS_INCASEOF_NON_STANDARD_RESIDUE`` : Mapping
  from non-standard (modified) amino-acid three-letter codes to their closest
  standard amino-acid equivalent (e.g. MSE -> MET, SEP -> SER).
- ``out_types`` / ``out_types_weights`` / ``out_types_weights_af3`` : Interface
  categories (e.g. "ligand_protein", "protein_protein") and their associated
  weights used in validation metric aggregation.  Two weight schemes are
  provided: the default Boltz weights and an AlphaFold3-style weighting.
- ``out_single_types`` : Per-chain output categories ("protein", "ligand",
  "dna", "rna").

Residues and Tokens
-------------------
- ``tokens`` : The 33-element token vocabulary used by the model:
  [<pad>, gap(-), 20 standard amino acids, UNK, 5 RNA bases (A/G/C/U/N),
  5 DNA bases (DA/DG/DC/DT/DN)].
- ``token_ids`` / ``num_tokens`` : Token-to-index mapping and vocabulary size.
- ``unk_token`` / ``unk_token_ids`` : Per-molecule-type unknown token names
  and their indices.
- ``prot_letter_to_token`` / ``prot_token_to_letter`` : Bidirectional mapping
  between one-letter amino acid codes and three-letter token names.
- ``rna_letter_to_token`` / ``dna_letter_to_token`` : Similar mappings for
  nucleic acids.

Atoms
-----
- ``num_elements`` : Maximum atomic number supported (128).
- ``chirality_types`` / ``chirality_type_ids`` : Chirality type vocabulary
  (UNSPECIFIED, TETRAHEDRAL_CW, TETRAHEDRAL_CCW, OTHER).
- ``ref_atoms`` : Canonical heavy-atom names for every standard residue type,
  in the order used by the model.  Protein residues list backbone atoms
  (N, CA, C, O) followed by side-chain atoms.  Nucleotides list phosphate,
  sugar, and base atoms.
- ``ref_symmetries`` : Atom-index swap lists for residues with chemically
  equivalent atoms (e.g. ASP OD1/OD2, PHE CD1/CD2 + CE1/CE2).  Used to
  compute symmetry-aware losses.  Nucleotide phosphate oxygens OP1/OP2 are
  also symmetric.
- ``res_to_center_atom`` / ``res_to_center_atom_id`` : Maps each standard
  residue name to its center atom name (CA for protein, C1' for nucleotides)
  and the corresponding index into ``ref_atoms``.
- ``res_to_disto_atom`` / ``res_to_disto_atom_id`` : Maps each standard
  residue name to its distogram atom name (CB for most protein residues,
  CA for GLY, base atoms for nucleotides) and index.

Bonds
-----
- ``atom_interface_cutoff`` : Distance cutoff (5.0 A) for determining
  atom-level contacts at an interface.
- ``interface_cutoff`` : Distance cutoff (15.0 A) for determining whether
  two chains form an interface.
- ``bond_types`` / ``bond_type_ids`` : Bond-order vocabulary
  (OTHER=0, SINGLE=1, DOUBLE=2, TRIPLE=3, AROMATIC=4).

Contacts
--------
- ``pocket_contact_info`` : Residue-level pocket annotation types:
  UNSPECIFIED=0 (not annotated), UNSELECTED=1 (annotated but not in pocket),
  POCKET=2 (pocket residue on receptor), BINDER=3 (binder/ligand residue).

MSA
---
- ``max_msa_seqs`` : Maximum number of MSA sequences to retain (16384).
- ``max_paired_seqs`` : Maximum number of paired MSA sequences (8192).

Chunking
--------
- ``chunk_size_threshold`` : Token count threshold (384) above which
  inference is performed with memory-efficient chunked attention.
"""

####################################################################################################
# CHAINS
####################################################################################################

# Mapping from non-standard (modified) amino-acid three-letter codes to the
# closest standard amino-acid equivalent.  This is used during PDB parsing to
# normalize modified residues so they can be tokenized with the standard
# 20-amino-acid vocabulary.  The final entries are identity mappings for the
# 20 standard amino acids plus UNK.
STANDARD_RESIDUE_SUBSTITUTIONS_INCASEOF_NON_STANDARD_RESIDUE = {
    '2AS':'ASP', '3AH':'HIS', '5HP':'GLU', 'ACL':'ARG', 'AGM':'ARG', 'AIB':'ALA', 'ALM':'ALA', 'ALO':'THR', 'ALY':'LYS', 'ARM':'ARG',
    'ASA':'ASP', 'ASB':'ASP', 'ASK':'ASP', 'ASL':'ASP', 'ASQ':'ASP', 'AYA':'ALA', 'BCS':'CYS', 'BHD':'ASP', 'BMT':'THR', 'BNN':'ALA',
    'BUC':'CYS', 'BUG':'LEU', 'C5C':'CYS', 'C6C':'CYS', 'CAS':'CYS', 'CCS':'CYS', 'CEA':'CYS', 'CGU':'GLU', 'CHG':'ALA', 'CLE':'LEU', 
    'CME':'CYS', 'CSD':'ALA', 'CSO':'CYS', 'CSP':'CYS', 'CSS':'CYS', 'CSW':'CYS', 'CSX':'CYS', 'CXM':'MET', 'CY1':'CYS', 'CY3':'CYS', 
    'CYG':'CYS', 'CYM':'CYS', 'CYQ':'CYS', 'DAH':'PHE', 'DAL':'ALA', 'DAR':'ARG', 'DAS':'ASP', 'DCY':'CYS', 'DGL':'GLU', 'DGN':'GLN', 
    'DHA':'ALA', 'DHI':'HIS', 'DIL':'ILE', 'DIV':'VAL', 'DLE':'LEU', 'DLY':'LYS', 'DNP':'ALA', 'DPN':'PHE', 'DPR':'PRO', 'DSN':'SER', 
    'DSP':'ASP', 'DTH':'THR', 'DTR':'TRP', 'DTY':'TYR', 'DVA':'VAL', 'EFC':'CYS', 'FLA':'ALA', 'FME':'MET', 'GGL':'GLU', 'GL3':'GLY', 
    'GLZ':'GLY', 'GMA':'GLU', 'GSC':'GLY', 'HAC':'ALA', 'HAR':'ARG', 'HIC':'HIS', 'HIP':'HIS', 'HMR':'ARG', 'HPQ':'PHE', 'HTR':'TRP', 
    'HYP':'PRO', 'IAS':'ASP', 'IIL':'ILE', 'IYR':'TYR', 'KCX':'LYS', 'LLP':'LYS', 'LLY':'LYS', 'LTR':'TRP', 'LYM':'LYS', 'LYZ':'LYS', 
    'MAA':'ALA', 'MEN':'ASN', 'MHS':'HIS', 'MIS':'SER', 'MLE':'LEU', 'MPQ':'GLY', 'MSA':'GLY', 'MSE':'MET', 'MVA':'VAL', 'NEM':'HIS', 
    'NEP':'HIS', 'NLE':'LEU', 'NLN':'LEU', 'NLP':'LEU', 'NMC':'GLY', 'OAS':'SER', 'OCS':'CYS', 'OMT':'MET', 'PAQ':'TYR', 'PCA':'GLU', 
    'PEC':'CYS', 'PHI':'PHE', 'PHL':'PHE', 'PR3':'CYS', 'PRR':'ALA', 'PTR':'TYR', 'PYX':'CYS', 'SAC':'SER', 'SAR':'GLY', 'SCH':'CYS', 
    'SCS':'CYS', 'SCY':'CYS', 'SEL':'SER', 'SEP':'SER', 'SET':'SER', 'SHC':'CYS', 'SHR':'LYS', 'SMC':'CYS', 'SOC':'CYS', 'STY':'TYR', 
    'SVA':'SER', 'TIH':'ALA', 'TPL':'TRP', 'TPO':'THR', 'TPQ':'ALA', 'TRG':'LYS', 'TRO':'TRP', 'TYB':'TYR', 'TYI':'TYR', 'TYQ':'TYR', 
    'TYS':'TYR', 'TYY':'TYR', 'ALA':'ALA', 'ARG':'ARG', 'ASN':'ASN', 'ASP':'ASP', 'CYS':'CYS', 'GLU':'GLU', 'GLN':'GLN', 'GLY':'GLY', 
    'HIS':'HIS', 'ILE':'ILE', 'LEU':'LEU', 'LYS':'LYS', 'MET':'MET', 'PHE':'PHE', 'PRO':'PRO', 'SER':'SER', 'THR':'THR', 'TRP':'TRP', 
    'TYR':'TYR', 'VAL':'VAL', 'UNK':'UNK'
}

# Molecule type vocabulary.  The integer index is used throughout the codebase
# to identify the type of a chain (e.g. in Chain["mol_type"]).
# Index:  PROTEIN=0, DNA=1, RNA=2, NONPOLYMER=3
chain_types = [
    "PROTEIN",      # 0
    "DNA",          # 1
    "RNA",          # 2
    "NONPOLYMER",  # 3  (ligands, ions, etc.)
]
chain_type_ids = {chain: i for i, chain in enumerate(chain_types)}

# --- Output interface categories ---
# These define the possible pairwise interaction types between chains (or within
# a single chain for "intra_*" categories).  Used to stratify evaluation metrics
# (e.g. lDDT, DockQ) by interaction type.
out_types = [
    "dna_protein",
    "rna_protein",
    "ligand_protein",
    "dna_ligand",
    "rna_ligand",
    "intra_ligand",
    "intra_dna",
    "intra_rna",
    "intra_protein",
    "protein_protein",
]

# AlphaFold3-style weights for aggregating per-interface-type metrics into a
# single overall score.  Higher weights emphasise certain interaction types.
out_types_weights_af3 = {
    "dna_protein": 10.0,
    "rna_protein": 10.0,
    "ligand_protein": 10.0,
    "dna_ligand": 5.0,
    "rna_ligand": 5.0,
    "intra_ligand": 20.0,
    "intra_dna": 4.0,
    "intra_rna": 16.0,
    "intra_protein": 20.0,
    "protein_protein": 20.0,
}

# Default Boltz weights for per-interface-type metric aggregation.  Compared to
# AF3 weights, ligand_protein is up-weighted and nucleic-acid interactions are
# down-weighted, reflecting the priorities of the Boltz training objective.
out_types_weights = {
    "dna_protein": 5.0,
    "rna_protein": 5.0,
    "ligand_protein": 20.0,
    "dna_ligand": 2.0,
    "rna_ligand": 2.0,
    "intra_ligand": 20.0,
    "intra_dna": 2.0,
    "intra_rna": 8.0,
    "intra_protein": 20.0,
    "protein_protein": 20.0,
}

# Per-chain (single-chain) output categories, used for single-chain metrics.
out_single_types = ["protein", "ligand", "dna", "rna"]

####################################################################################################
# RESIDUES & TOKENS
####################################################################################################

# --- Token vocabulary ---
# The model uses a fixed vocabulary of 33 tokens (indices 0-32).
# Layout:
#   [0]      <pad>   -- padding token (used for batching)
#   [1]      -       -- gap token (used in MSA alignments)
#   [2-21]   ALA..VAL -- 20 standard amino acids (alphabetical by 3-letter code)
#   [22]     UNK     -- unknown / non-standard protein residue
#   [23-26]  A,G,C,U -- 4 standard RNA bases
#   [27]     N       -- unknown RNA base
#   [28-31]  DA,DG,DC,DT -- 4 standard DNA bases
#   [32]     DN      -- unknown DNA base
tokens = [
    "<pad>",   # 0  : padding
    "-",       # 1  : gap / insertion
    "ALA",     # 2  : alanine
    "ARG",     # 3  : arginine
    "ASN",     # 4  : asparagine
    "ASP",     # 5  : aspartate
    "CYS",     # 6  : cysteine
    "GLN",     # 7  : glutamine
    "GLU",     # 8  : glutamate
    "GLY",     # 9  : glycine
    "HIS",     # 10 : histidine
    "ILE",     # 11 : isoleucine
    "LEU",     # 12 : leucine
    "LYS",     # 13 : lysine
    "MET",     # 14 : methionine
    "PHE",     # 15 : phenylalanine
    "PRO",     # 16 : proline
    "SER",     # 17 : serine
    "THR",     # 18 : threonine
    "TRP",     # 19 : tryptophan
    "TYR",     # 20 : tyrosine
    "VAL",     # 21 : valine
    "UNK",     # 22 : unknown protein token
    "A",       # 23 : RNA adenine
    "G",       # 24 : RNA guanine
    "C",       # 25 : RNA cytosine
    "U",       # 26 : RNA uracil
    "N",       # 27 : unknown RNA base
    "DA",      # 28 : DNA adenine
    "DG",      # 29 : DNA guanine
    "DC",      # 30 : DNA cytosine
    "DT",      # 31 : DNA thymine
    "DN",      # 32 : unknown DNA base
]

# Reverse lookup: token name -> index
token_ids = {token: i for i, token in enumerate(tokens)}
# Total vocabulary size (33)
num_tokens = len(tokens)

# Per-molecule-type "unknown" token name and index.  Used when a residue name
# cannot be resolved to any standard token.
unk_token = {"PROTEIN": "UNK", "DNA": "DN", "RNA": "N"}
unk_token_ids = {m: token_ids[t] for m, t in unk_token.items()}

# --- Protein one-letter to three-letter code mapping ---
# Standard 20 amino acids plus ambiguous / non-standard one-letter codes that
# map to UNK: X (unknown), J, B, Z, O, U.  The gap character "-" maps to the
# gap token.
prot_letter_to_token = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "E": "GLU",
    "Q": "GLN",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
    "X": "UNK",    # explicit unknown
    "J": "UNK",    # ambiguous leucine / isoleucine
    "B": "UNK",    # ambiguous asparagine / aspartate
    "Z": "UNK",    # ambiguous glutamine / glutamate
    "O": "UNK",    # pyrrolysine -> unknown
    "U": "UNK",    # selenocysteine -> unknown
    "-": "-",       # gap
}

# Reverse mapping: three-letter token -> one-letter code.
# UNK explicitly maps to "X"; <pad> maps to empty string.
prot_token_to_letter = {v: k for k, v in prot_letter_to_token.items()}
prot_token_to_letter["UNK"] = "X"
prot_token_to_letter["<pad>"] = ""

# --- RNA one-letter to token mapping ---
# 4 standard bases (A, G, C, U) plus N for unknown.
rna_letter_to_token = {
    "A": "A",
    "G": "G",
    "C": "C",
    "U": "U",
    "N": "N",
}
rna_token_to_letter = {v: k for k, v in rna_letter_to_token.items()}

# --- DNA one-letter to token mapping ---
# 4 standard bases (A, G, C, T) plus N for unknown.
# DNA tokens are prefixed with "D" to distinguish from RNA tokens.
dna_letter_to_token = {
    "A": "DA",
    "G": "DG",
    "C": "DC",
    "T": "DT",
    "N": "DN",
}
dna_token_to_letter = {v: k for k, v in dna_letter_to_token.items()}

####################################################################################################
# ATOMS
####################################################################################################

# Maximum supported atomic number.  Atom elements are stored as int8, but this
# constant caps the range of valid element indices used in embeddings.
num_elements = 128

# --- Chirality type vocabulary ---
# Encodes the CIP (R/S) chirality of tetrahedral stereocenters.
# Index: UNSPECIFIED=0, CW=1 (clockwise / R), CCW=2 (counter-clockwise / S), OTHER=3
chirality_types = [
    "CHI_UNSPECIFIED",       # 0: chirality not determined or not applicable
    "CHI_TETRAHEDRAL_CW",    # 1: clockwise tetrahedral (R-configuration)
    "CHI_TETRAHEDRAL_CCW",   # 2: counter-clockwise tetrahedral (S-configuration)
    "CHI_OTHER",             # 3: other / non-tetrahedral chirality
]
chirality_type_ids = {chirality: i for i, chirality in enumerate(chirality_types)}
unk_chirality_type = "CHI_UNSPECIFIED"

# --- Reference atoms per residue type ---
# Canonical ordering of heavy atoms for each standard residue in the token
# vocabulary.  This ordering defines the atom dimension of the model's
# per-residue atom representation.
#
# Protein residues: backbone (N, CA, C, O) followed by side-chain atoms.
# RNA residues:     phosphate (P, OP1, OP2) + sugar (O5', C5', ..., C1') + base.
# DNA residues:     same as RNA but without the 2'-OH (O2') on the sugar ring.
# PAD / gap:        empty atom lists.
#
# The key "UNK" uses the minimal 5-atom backbone+CB template so that unknown
# protein residues can still be represented.
# fmt: off
ref_atoms = {
    "PAD": [],
    "UNK": ["N", "CA", "C", "O", "CB"],
    "-": [],
    "ALA": ["N", "CA", "C", "O", "CB"],
    "ARG": ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
    "ASN": ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"],
    "ASP": ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"],
    "CYS": ["N", "CA", "C", "O", "CB", "SG"],
    "GLN": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"],
    "GLU": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"],
    "GLY": ["N", "CA", "C", "O"],
    "HIS": ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"],
    "ILE": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"],
    "LEU": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"],
    "LYS": ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"],
    "MET": ["N", "CA", "C", "O", "CB", "CG", "SD", "CE"],
    "PHE": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "PRO": ["N", "CA", "C", "O", "CB", "CG", "CD"],
    "SER": ["N", "CA", "C", "O", "CB", "OG"],
    "THR": ["N", "CA", "C", "O", "CB", "OG1", "CG2"],
    "TRP": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],  # noqa: E501
    "TYR": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
    "VAL": ["N", "CA", "C", "O", "CB", "CG1", "CG2"],
    "A": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"],  # noqa: E501
    "G": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"],  # noqa: E501
    "C": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"],  # noqa: E501
    "U": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6"],  # noqa: E501
    "N": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'"],  # noqa: E501
    "DA": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"],  # noqa: E501
    "DG": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"],  # noqa: E501
    "DC": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"],  # noqa: E501
    "DT": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "N1", "C2", "O2", "N3", "C4", "O4", "C5", "C7", "C6"],  # noqa: E501
    "DN": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'"]
}

# --- Reference symmetries for residues with chemically equivalent atoms ---
# Some residues have atoms that are chemically indistinguishable and can be
# swapped without changing the physical structure.  Each entry is a list of
# permutations, where each permutation is a list of (old_idx, new_idx) pairs
# (indices into the ref_atoms list for that residue).
#
# Examples:
#   ASP: OD1 (idx 6) <-> OD2 (idx 7)           -- carboxylate oxygens
#   GLU: OE1 (idx 7) <-> OE2 (idx 8)           -- carboxylate oxygens
#   PHE: CD1<->CD2 (6<->7), CE1<->CE2 (8<->9)  -- ring symmetry
#   TYR: same ring symmetry as PHE
#   Nucleotides: OP1 (idx 1) <-> OP2 (idx 2)    -- phosphate oxygens
#
# An empty list means no swappable symmetric atoms.
ref_symmetries = {
    "PAD": [],
    "ALA": [],
    "ARG": [],
    "ASN": [],
    "ASP": [[(6, 7), (7, 6)]],
    "CYS": [],
    "GLN": [],
    "GLU": [[(7, 8), (8, 7)]],
    "GLY": [],
    "HIS": [],
    "ILE": [],
    "LEU": [],
    "LYS": [],
    "MET": [],
    "PHE": [[(6, 7), (7, 6), (8, 9), (9, 8)]],
    "PRO": [],
    "SER": [],
    "THR": [],
    "TRP": [],
    "TYR": [[(6, 7), (7, 6), (8, 9), (9, 8)]],
    "VAL": [],
    "A": [[(1, 2), (2, 1)]],
    "G": [[(1, 2), (2, 1)]],
    "C": [[(1, 2), (2, 1)]],
    "U": [[(1, 2), (2, 1)]],
    "N": [[(1, 2), (2, 1)]],
    "DA": [[(1, 2), (2, 1)]],
    "DG": [[(1, 2), (2, 1)]],
    "DC": [[(1, 2), (2, 1)]],
    "DT": [[(1, 2), (2, 1)]],
    "DN": [[(1, 2), (2, 1)]]
}


# --- Center atoms per residue type ---
# The "center atom" is the representative atom used for pairwise residue-level
# distance calculations and spatial hashing.
#   Protein residues: CA (alpha carbon)
#   Nucleotide residues: C1' (anomeric carbon on the sugar ring)
res_to_center_atom = {
    "UNK": "CA",
    "ALA": "CA",
    "ARG": "CA",
    "ASN": "CA",
    "ASP": "CA",
    "CYS": "CA",
    "GLN": "CA",
    "GLU": "CA",
    "GLY": "CA",
    "HIS": "CA",
    "ILE": "CA",
    "LEU": "CA",
    "LYS": "CA",
    "MET": "CA",
    "PHE": "CA",
    "PRO": "CA",
    "SER": "CA",
    "THR": "CA",
    "TRP": "CA",
    "TYR": "CA",
    "VAL": "CA",
    "A": "C1'",
    "G": "C1'",
    "C": "C1'",
    "U": "C1'",
    "N": "C1'",
    "DA": "C1'",
    "DG": "C1'",
    "DC": "C1'",
    "DT": "C1'",
    "DN": "C1'"
}

# --- Distogram atoms per residue type ---
# The "disto atom" is used for distogram (inter-residue distance distribution)
# predictions.  It provides directional information about the side chain or base.
#   Protein: CB (beta carbon) for all residues except GLY which uses CA
#   Purines (A, G, DA, DG): C4 (base atom)
#   Pyrimidines (C, U, DC, DT): C2 (base atom)
#   Unknown nucleotides (N, DN): C1' (sugar, same as center -- no base info)
res_to_disto_atom = {
    "UNK": "CB",
    "ALA": "CB",
    "ARG": "CB",
    "ASN": "CB",
    "ASP": "CB",
    "CYS": "CB",
    "GLN": "CB",
    "GLU": "CB",
    "GLY": "CA",
    "HIS": "CB",
    "ILE": "CB",
    "LEU": "CB",
    "LYS": "CB",
    "MET": "CB",
    "PHE": "CB",
    "PRO": "CB",
    "SER": "CB",
    "THR": "CB",
    "TRP": "CB",
    "TYR": "CB",
    "VAL": "CB",
    "A": "C4",
    "G": "C4",
    "C": "C2",
    "U": "C2",
    "N": "C1'",
    "DA": "C4",
    "DG": "C4",
    "DC": "C2",
    "DT": "C2",
    "DN": "C1'"
}

# Precomputed index of the center atom within the ref_atoms list for each
# residue type.  E.g. for "ALA", CA is at index 1 in ref_atoms["ALA"].
res_to_center_atom_id = {
    res: ref_atoms[res].index(atom)
    for res, atom in res_to_center_atom.items()
}

# Precomputed index of the disto atom within the ref_atoms list for each
# residue type.  E.g. for "ALA", CB is at index 4 in ref_atoms["ALA"].
res_to_disto_atom_id = {
    res: ref_atoms[res].index(atom)
    for res, atom in res_to_disto_atom.items()
}

# fmt: on

####################################################################################################
# BONDS
####################################################################################################

# Distance cutoff (in angstroms) for determining atom-level contacts at a
# chain-chain interface.  Two atoms from different chains are considered in
# contact if their distance is less than this value.
atom_interface_cutoff = 5.0

# Distance cutoff (in angstroms) for determining whether two chains form a
# biological interface.  If any pair of center atoms between two chains is
# within this distance, the chain pair is recorded as an interface.
interface_cutoff = 15.0

# --- Bond type vocabulary ---
# Encodes covalent bond order.  Index: OTHER=0, SINGLE=1, DOUBLE=2,
# TRIPLE=3, AROMATIC=4.  "OTHER" is used as the fallback for unrecognized
# or non-standard bond types.
bond_types = [
    "OTHER",      # 0
    "SINGLE",     # 1
    "DOUBLE",     # 2
    "TRIPLE",     # 3
    "AROMATIC",   # 4
]
bond_type_ids = {bond: i for i, bond in enumerate(bond_types)}
unk_bond_type = "OTHER"


####################################################################################################
# Contacts
####################################################################################################

# --- Pocket contact annotation types ---
# Used to label residues in the context of pocket / binding-site conditioning
# during inference.
#   UNSPECIFIED (0): residue has no pocket annotation (default).
#   UNSELECTED  (1): residue was considered but not included in the pocket.
#   POCKET      (2): residue is part of the receptor binding pocket.
#   BINDER      (3): residue belongs to the binder / ligand chain.
pocket_contact_info = {
    "UNSPECIFIED": 0,
    "UNSELECTED": 1,
    "POCKET": 2,
    "BINDER": 3,
}


####################################################################################################
# MSA
####################################################################################################

# Maximum number of sequences to retain in a single-chain MSA after filtering
# and subsampling.  Sequences beyond this limit are discarded.
max_msa_seqs = 16384

# Maximum number of sequences to retain in a paired (cross-chain) MSA,
# constructed by matching sequences from different chains by taxonomy.
max_paired_seqs = 8192


####################################################################################################
# CHUNKING
####################################################################################################

# Token count threshold for memory-efficient chunked attention during inference.
# If the total number of tokens in a target exceeds this value, the attention
# computation is broken into chunks to reduce peak GPU memory usage.
chunk_size_threshold = 384
