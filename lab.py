"""Shared loading code and conventions for the PHOIBLE vowel lab (PHOIBLE v2.0)."""
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
DATA = ROOT / "phoible.csv"  # v2.0, sha256 cc19b7a1...b618 (identical to the v2.0 tag)
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)


def load_raw() -> pd.DataFrame:
    """One row per (inventory, phoneme). Note: pandas turns the string "NA" into NaN."""
    return pd.read_csv(DATA, low_memory=False)


def select_inventories(df: pd.DataFrame) -> pd.DataFrame:
    """Convention (Q4): one inventory per Glottocode, the one with the highest InventoryID.

    The 2 inventories with no Glottocode (2281 Modern Aramaic, 2729 Djindewal) are dropped
    *explicitly* here rather than silently by groupby.
    """
    df = df[df.Glottocode.notna()]
    keep = df.groupby("Glottocode").InventoryID.max()
    return df[df.InventoryID.isin(keep)]


def vowels(df: pd.DataFrame) -> pd.DataFrame:
    """Convention (Q7): a vowel is any row with SegmentClass == 'vowel', whatever Marginal is."""
    return df[df.SegmentClass == "vowel"]


def load() -> pd.DataFrame:
    """The reference dataset: one inventory per language, all segments."""
    return select_inventories(load_raw())


def presence_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Binary language x phoneme matrix (rows = Glottocode)."""
    return pd.crosstab(df.Glottocode, df.Phoneme).clip(upper=1).astype(bool)


def nfd(s: str) -> str:
    """PHOIBLE is NFD-normalised: a precomposed 'ã' typed in source code matches nothing."""
    return unicodedata.normalize("NFD", s)


# Criterion A (string-based): strip diacritics that modify a vowel without changing its
# basic quality, then compare the base letter. Diphthongs never count. PHOIBLE is in NFD.
MODIFIERS = {
    "\u02d0", "\u02d1", "\u0306",            # long, half-long, extra-short
    "\u0303",                                # nasal
    "\u0324", "\u0330", "\u0325", "\u030a",  # breathy, creaky, voiceless
    "\u031e", "\u031d", "\u031f", "\u0320",  # lowered, raised, advanced, retracted
    "\u0318", "\u0319",                      # ATR, RTR
    "\u031c", "\u0339",                      # less / more rounded
    "\u033d", "\u0353",                      # mid-centralised, x below
    "\u02e4", "\u02de", "\u02b2", "\u02e0",  # pharyngealised, rhotic, palatalised, velarised
    "\u02c0", "\u02b1", "\u02bc", "\u1d31",  # glottalised, breathy-h, ejective, capital E
}
CENTRALISED = "\u0308"  # ä, ï ...: changes front/back quality, so kept as part of the letter
NON_SYLLABIC = "\u032f"


def base(p):
    """Return the base vowel letter of a monophthong, or None for a diphthong."""
    if NON_SYLLABIC in p:
        return None
    core = "".join(c for c in p if c not in MODIFIERS)
    return core if len(core) == 1 or (len(core) == 2 and core[1] == CENTRALISED) else None
