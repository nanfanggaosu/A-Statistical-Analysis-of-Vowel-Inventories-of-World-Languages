"""Section 4: are languages independent observations? (Q29-Q32)

Genealogy: Glottolog 4.1 CLDF languages.csv (glottolog/glottolog-cldf, tag v4.1).
Isolates (no Family_ID) are their own family; dialects inherit family and macroarea
from their parent language (Language_ID).
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lab import OUT, ROOT, base, load, vowels

rng = np.random.default_rng(0)
G = pd.read_csv(ROOT / "glottolog_languages_v4.1.csv", keep_default_na=False, na_values=[""]).set_index("ID")

df = load()
v = vowels(df).assign(base=lambda d: d.Phoneme.map(base))
langs = pd.Index(df.Glottocode.unique(), name="Glottocode")

# --- genealogy -------------------------------------------------------------------
L = G.loc[langs, ["Name", "Family_ID", "Macroarea", "Language_ID"]].copy()
parent = L.Language_ID.fillna(pd.Series(L.index, index=L.index))  # dialect -> language
L["Macroarea"] = L.Macroarea.fillna(G.Macroarea.reindex(parent).set_axis(L.index))
L["Family_ID"] = L.Family_ID.fillna(G.Family_ID.reindex(parent).set_axis(L.index))
L["family"] = L.Family_ID.fillna(parent)  # isolates = own family
L["family_name"] = G.Name.reindex(L.family).to_numpy()
print("Q29 families in PHOIBLE sample:", L.family.nunique(), "| isolates:", L.Family_ID.isna().sum(),
      "| missing macroarea:", L.Macroarea.isna().sum())

# World baseline: Glottolog languages (level 'language'), excluding non-genealogical pseudo-families
PSEUDO = {"Sign Language", "Unclassifiable", "Artificial Language", "Pidgin", "Mixed Language",
          "Speech Register", "Bookkeeping", "Unattested", "Spurious"}
pseudo_ids = set(G[G.Name.isin(PSEUDO)].index)
# a "language" = no parent language, not itself a family, not in a pseudo-family, has a macroarea
W = G[G.Language_ID.isna() & ~G.index.isin(G.Family_ID.dropna().unique()) & ~G.Family_ID.isin(pseudo_ids)
      & ~G.index.isin(pseudo_ids) & G.Macroarea.notna()]
W = W.assign(family=W.Family_ID.fillna(pd.Series(W.index, index=W.index)))
print("Q29 Glottolog baseline languages:", len(W), "families (incl. isolates):", W.family.nunique())

area = pd.DataFrame({"PHOIBLE": L.Macroarea.value_counts(), "Glottolog": W.Macroarea.value_counts()})
area["share_PHOIBLE"] = area.PHOIBLE / area.PHOIBLE.sum()
area["share_world"] = area.Glottolog / area.Glottolog.sum()
area["ratio"] = area.share_PHOIBLE / area.share_world
area["coverage"] = area.PHOIBLE / area.Glottolog
print("\nQ29 macroareas:\n", area.round(3).sort_values("ratio", ascending=False).to_string())

fam = pd.DataFrame({"PHOIBLE": L.family.value_counts(), "Glottolog": W.family.value_counts()}).fillna(0)
fam = fam[fam.PHOIBLE > 0]
fam["name"] = G.Name.reindex(fam.index).to_numpy()
fam["share_PHOIBLE"] = fam.PHOIBLE / len(L)
fam["share_world"] = fam.Glottolog / len(W)
fam["ratio"] = fam.share_PHOIBLE / fam.share_world
fam["coverage"] = fam.PHOIBLE / fam.Glottolog
fam = fam.sort_values("PHOIBLE", ascending=False)
print("\nQ29 top 15 families:\n", fam.head(15).round(3).to_string())
print(f"Q29 top 5 families hold {fam.PHOIBLE.head(5).sum() / len(L):.1%} of PHOIBLE languages "
      f"vs {fam.Glottolog.head(5).sum() / len(W):.1%} of the world's")
print(f"Q29 families with 1 language in PHOIBLE: {(fam.PHOIBLE == 1).sum()} of {len(fam)}")
big = fam[fam.Glottolog >= 20]
print("Q29 most over-represented (families with >=20 languages):\n",
      big.sort_values("ratio", ascending=False).head(8)[["name", "PHOIBLE", "Glottolog", "coverage", "ratio"]].round(2).to_string())
print("Q29 most under-represented (families with >=20 languages):\n",
      big.sort_values("ratio").head(8)[["name", "PHOIBLE", "Glottolog", "coverage", "ratio"]].round(2).to_string())
fam.to_csv(OUT / "q29_families.csv")
area.to_csv(OUT / "q29_macroareas.csv")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
a = area.sort_values("ratio")
axes[0].barh(a.index, a.share_world, height=.4, align="edge", label="Glottolog (world)")
axes[0].barh(a.index, a.share_PHOIBLE, height=-.4, align="edge", label="PHOIBLE sample")
axes[0].set_xlabel("share of languages")
axes[0].legend()
axes[0].set_title("Macroareas")
top = fam.head(15)[::-1]
y = np.arange(len(top))
axes[1].barh(y + .2, top.share_world, height=.4, label="Glottolog (world)")
axes[1].barh(y - .2, top.share_PHOIBLE, height=.4, label="PHOIBLE sample")
axes[1].set_yticks(y, top.name)
axes[1].set_xlabel("share of languages")
axes[1].set_title("15 largest families in PHOIBLE")
axes[1].legend()
fig.tight_layout()
fig.savefig(OUT / "q29_representation.png", dpi=150)

# --- properties per language (from Section 3) ---------------------------------------
has = lambda bases: pd.Series(langs.isin(v[v.base.isin(bases)].Glottocode), index=langs)
NASAL = "\u0303"
inv = v.groupby("Glottocode").Phoneme.apply(set)
oral_bases = v[~v.Phoneme.str.contains(NASAL)].groupby("Glottocode").base.apply(set)


def nasal_ok(g):
    """Every nasal vowel has an oral counterpart (same string minus tilde, or same base quality)."""
    nas = [p for p in inv[g] if NASAL in p]
    return all(p.replace(NASAL, "") in inv[g] or base(p) in oral_bases.get(g, set()) for p in nas)


has_nasal = pd.Series([any(NASAL in p for p in inv[g]) for g in langs], index=langs)
props = pd.DataFrame({
    "i": has({"i"}), "a": has({"a"}), "u": has({"u"}),
}, index=langs)
props["i&a&u"] = props[["i", "a", "u"]].all(axis=1)
props["i&a&u (broad B)"] = has({"i", "ɪ"}) & has({"a", "ɑ", "ɐ", "ä", "æ"}) & has({"u", "ʊ"})
props["y"] = has({"y"})
props["n_vowels"] = v.groupby("Glottocode").size().reindex(langs)
conditional = {  # implication: P(B | A), computed only among languages with A
    "y -> i&u": (props.y, props.i & props.u),
    "nasal -> oral": (has_nasal, pd.Series([nasal_ok(g) for g in langs], index=langs)),
}


def stats(idx):
    s = props.loc[idx]
    out = {c: s[c].mean() for c in ["i", "a", "u", "i&a&u", "i&a&u (broad B)", "y"]}
    out["mean vowels"] = s.n_vowels.mean()
    for k, (A, B) in conditional.items():
        a = A.loc[idx]
        out[k] = B.loc[idx][a].mean() if a.any() else np.nan
        out[k + " (n)"] = a.sum()
    return out


# --- Q30/Q31: one language per family, many times --------------------------------
fam_groups = L.groupby("family").groups
fam_keys = list(fam_groups)
print(f"\nQ30 one language per family -> samples of {len(fam_keys)} languages")


def one_per_family():
    return pd.Index([rng.choice(fam_groups[f]) for f in fam_keys])


full = stats(langs)
one = stats(one_per_family())
reps = pd.DataFrame([stats(one_per_family()) for _ in range(1000)])
# family-weighted mean = expected value of the one-per-family estimator
famw = props.groupby(L.family).mean(numeric_only=True).mean()
area_bal = props.groupby(L.Macroarea).mean(numeric_only=True).mean()
summary = pd.DataFrame({
    "all languages": full,
    "one sample": one,
    "1000 samples: mean": reps.mean(),
    "2.5%": reps.quantile(.025),
    "97.5%": reps.quantile(.975),
    "sd": reps.std(),
})
summary.loc[famw.index.intersection(summary.index), "family-weighted"] = famw
summary.loc[area_bal.index.intersection(summary.index), "macroarea-weighted"] = area_bal
print(summary.round(3).to_string())
summary.to_csv(OUT / "q30_31_balanced_samples.csv")
print("\nQ31 per-macroarea proportions:\n", props.groupby(L.Macroarea)[["i", "a", "u", "i&a&u", "i&a&u (broad B)", "y", "n_vowels"]].mean().round(3).to_string())

fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))
for ax, col in zip(axes, ["i", "a", "u", "i&a&u"]):
    ax.hist(reps[col], bins=30, color="C0", alpha=.8)
    ax.axvline(full[col], color="r", ls="--", label="all languages")
    ax.set_title(f"[{col}]")
    ax.set_xlabel("proportion")
axes[0].legend(fontsize=8)
fig.suptitle("1000 genealogically balanced samples (one language per family)")
fig.tight_layout()
fig.savefig(OUT / "q31_balanced_distributions.png", dpi=150)
