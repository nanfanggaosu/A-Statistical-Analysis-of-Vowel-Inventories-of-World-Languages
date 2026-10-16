"""Section 3: testing the two universals (Q11-Q19)."""
import pandas as pd

from lab import OUT, base, load, vowels

df = load()
v = vowels(df).copy()
langs = pd.Index(df.Glottocode.unique())
N = len(langs)
meta = df.drop_duplicates("Glottocode").set_index("Glottocode")[["LanguageName", "InventoryID", "Source"]]


def has(phonemes):
    """Boolean Series over languages: does the language have at least one of `phonemes`?"""
    return pd.Series(langs.isin(v[v.Phoneme.isin(phonemes)].Glottocode), index=langs)


# --- Q11 ---------------------------------------------------------------
freq = v.groupby("Phoneme").Glottocode.nunique().sort_values(ascending=False)
print("Q11 most frequent vowels (languages, share):")
print(pd.DataFrame({"n": freq, "share": (freq / N).round(3)}).head(12).to_string())
freq.rename("n_languages").to_csv(OUT / "vowel_frequencies.csv")

# --- Q12: strict identity ------------------------------------------------------
strict = pd.DataFrame({p: has([p]) for p in "iau"})
print("\nQ12 strict proportions:", strict.mean().round(3).to_dict(), "| all three:", round(strict.all(axis=1).mean(), 3))

# --- Q13: "close equivalents" ------------------------------------------------------
# Criterion A (string-based): see lab.base()
v["base"] = v.Phoneme.map(base)
LEVELS = {
    "strict": {"i": {"i"}, "a": {"a"}, "u": {"u"}},
    "A: +diacritics": {"i": {"i"}, "a": {"a"}, "u": {"u"}},
    "B: +neighbouring qualities": {"i": {"i", "ɪ"}, "a": {"a", "ɑ", "ɐ", "ä", "æ"}, "u": {"u", "ʊ"}},
}


def has_base(bases):
    return pd.Series(langs.isin(v[v.base.isin(bases)].Glottocode), index=langs)


# Criterion C (feature-based natural classes on monophthongs, using PHOIBLE's own features):
#   i-like: [+high, -back, +front, -round]; u-like: [+high, +back, +round]; a-like: [+low, -round]
mono = v[~v.high.astype(str).str.contains(",")]
feat = {
    "i": mono[(mono.high == "+") & (mono.front == "+") & (mono.back == "-") & (mono["round"] != "+")],
    "u": mono[(mono.high == "+") & (mono.back == "+") & (mono["round"] == "+")],
    "a": mono[(mono.low == "+") & (mono["round"] != "+")],
}
rows = {"strict": strict}
for lvl in ["A: +diacritics", "B: +neighbouring qualities"]:
    rows[lvl] = pd.DataFrame({k: has_base(s) for k, s in LEVELS[lvl].items()})
rows["C: feature classes"] = pd.DataFrame({k: pd.Series(langs.isin(d.Glottocode), index=langs) for k, d in feat.items()})
summary = pd.DataFrame({k: {**t.mean().round(3).to_dict(), "all three": round(t.all(axis=1).mean(), 3),
                            "violators": int((~t.all(axis=1)).sum())} for k, t in rows.items()}).T
print("\nQ13 proportions under the four criteria:\n", summary.to_string())
summary.to_csv(OUT / "q13_triangle_criteria.csv")
print("Q13 phonemes that criterion C adds for 'a':", sorted(set(feat["a"].Phoneme) - {"a"})[:40])

# --- Q14/Q15: violators ---------------------------------------------------------------
tri = rows["A: +diacritics"]
viol = tri[~tri.all(axis=1)]
print(f"\nQ14 violators under criterion A: {len(viol)} / {N} ({len(viol) / N:.1%})")
print("Q14 which vowel is missing:", {k: int((~viol[k]).sum()) for k in "iau"})
vinv = v.groupby("Glottocode").Phoneme.apply(lambda s: " ".join(sorted(s)))
vt = meta.loc[viol.index].assign(missing=viol.apply(lambda r: "".join(k for k in "iau" if not r[k]), axis=1),
                                 vowels=vinv.reindex(viol.index))
print("Q15 violators by source (share of that source's languages):")
src_all = meta.Source.value_counts()
print(pd.DataFrame({"violators": vt.Source.value_counts(), "of": src_all,
                    "rate": (vt.Source.value_counts() / src_all).round(3)}).dropna().sort_values("rate", ascending=False))
vt.to_csv(OUT / "q14_violators.csv")
viol_C = rows["C: feature classes"]
still = viol_C[~viol_C.all(axis=1)].index
print("Q15 violators under criterion C by source:\n",
      (meta.loc[viol_C[~viol_C.all(axis=1)].index].Source.value_counts() / src_all).round(3).dropna().sort_values(ascending=False).to_string())
print(f"Q15 violators that remain even under feature criterion C: {len(still)}")
print(meta.loc[still].assign(vowels=vinv.reindex(still)).to_string())

# Is a violation caused only by a Marginal phoneme? (would the language pass if Marginal were ignored?)
marg = v[v.Marginal == True]
print("\nQ15 marginal vowels among the three corners:", marg[marg.base.isin(["i", "a", "u"])][["Glottocode", "Phoneme"]].values.tolist())

# Other descriptions of the same language: do they also violate?
raw_v = vowels(pd.read_csv("phoible.csv", low_memory=False))
raw_v = raw_v[raw_v.Glottocode.isin(viol.index)].assign(base=lambda d: d.Phoneme.map(base))
other = raw_v.groupby(["Glottocode", "InventoryID"]).base.apply(lambda s: {"i", "a", "u"} <= set(s)).unstack(level=1)
multi = other[other.notna().sum(axis=1) > 1]
print(f"Q15 violators with several descriptions: {len(multi)}; "
      f"of which another description has all three: {(multi == True).any(axis=1).sum()}")

# --- Q16/Q17: implications ------------------------------------------------------------
# v.nasal == '+' for nasal vowels; v.long == '+' for long ones (checked against the
# diacritics ̃ and ː, see below).
print("\nQ16 sanity: nasal feature vs tilde:",
      pd.crosstab(v.nasal, v.Phoneme.str.contains("̃")).to_dict())
print("Q16 sanity: long feature vs ː:", pd.crosstab(v.long, v.Phoneme.str.contains("ː")).to_dict())

inv_sets = v.groupby("Glottocode").Phoneme.apply(set)


def implication(name, marked_mask, counterpart):
    """For each language having >=1 marked vowel, check that every marked vowel has its counterpart."""
    rows = []
    for g, sub in v[marked_mask].groupby("Glottocode"):
        inv = inv_sets[g]
        missing = [p for p in sub.Phoneme if not (counterpart(p) & inv)]
        rows.append((g, len(sub), len(missing), " ".join(missing)))
    t = pd.DataFrame(rows, columns=["Glottocode", "n_marked", "n_missing", "missing"]).set_index("Glottocode")
    t["all_ok"] = t.n_missing == 0
    per_vowel = t.n_missing.sum() / t.n_marked.sum()
    print(f"Q17 {name}: {len(t)} languages; satisfy for every vowel: {t.all_ok.mean():.1%} "
          f"({(~t.all_ok).sum()} exceptions); marked vowels lacking counterpart: {per_vowel:.1%}")
    return t


NASAL, LONG = "̃", "ː"
nas = implication("nasal -> oral", v.Phoneme.str.contains(NASAL),
                  lambda p: {p.replace(NASAL, "")})
# Looser: any oral vowel of the same base quality (ĩː satisfied by i, iː, i̞ ...)
nas_loose = implication("nasal -> oral (same base quality)", v.Phoneme.str.contains(NASAL),
                        lambda p: {p.replace(NASAL, "")} | set(v[(v.base == base(p)) & ~v.Phoneme.str.contains(NASAL)].Phoneme))
lng = implication("long -> short", v.Phoneme.str.contains(LONG) & v.base.notna(),
                  lambda p: {p.replace(LONG, "")})
lng_loose = implication("long -> short (same base quality)", v.Phoneme.str.contains(LONG) & v.base.notna(),
                        lambda p: {p.replace(LONG, "")} | set(v[(v.base == base(p)) & ~v.Phoneme.str.contains(LONG)].Phoneme))
FRONT_ROUNDED = {"y", "ʏ", "ø", "œ", "ɶ"}
fr = v.base.isin(FRONT_ROUNDED)
# Q16: one row per language having >=1 marked vowel, listing its marked vowels of each type
marked = {"nasal": v.Phoneme.str.contains(NASAL),
          "long": v.Phoneme.str.contains(LONG) & v.base.notna(),  # long monophthongs
          "front_rounded": fr}
q16 = pd.DataFrame({k: v[m].groupby("Glottocode").Phoneme.apply(lambda s: " ".join(sorted(s)))
                    for k, m in marked.items()})
q16 = meta.join(q16, how="right")
q16.to_csv(OUT / "q16_marked_vowel_languages.csv")
print("Q16 languages with nasal / long / front rounded vowels:", q16[list(marked)].notna().sum().to_dict())
has_y = has_base({"y"})
has_i, has_u = has_base({"i"}), has_base({"u"})
for label, cond in [("y -> i", has_i), ("y -> u", has_u), ("y -> i and u", has_i & has_u)]:
    ok = cond[has_y]
    print(f"Q17 {label}: {has_y.sum()} languages with [y]; satisfied {ok.mean():.1%}, exceptions {(~ok).sum()}")
# Generalisation: every front rounded vowel -> its unrounded front counterpart (y->i, ø->e, œ->ɛ ...)
UNROUND = {"y": "i", "ʏ": "ɪ", "ø": "e", "œ": "ɛ", "ɶ": "a"}
fr_imp = implication("front rounded -> front unrounded (same height)", fr,
                     lambda p: set(v[v.base == UNROUND[base(p)]].Phoneme))
bad_y = has_y & ~(has_i & has_u)
print("Q17 [y] exceptions:\n", meta.loc[bad_y[bad_y].index].assign(vowels=vinv.reindex(bad_y[bad_y].index)).to_string())
for name, t in [("nasal", nas), ("long", lng)]:
    ex = t[~t.all_ok].join(meta)
    print(f"\nQ17 {name} exceptions by source:\n", ex.Source.value_counts().to_string())
    print(ex.head(10)[["LanguageName", "Source", "missing"]].to_string())
    ex.to_csv(OUT / f"q17_{name}_exceptions.csv")

# --- Q18: the 2x2 table of an implication --------------------------------------------
A = pd.Series(langs.isin(nas.index), index=langs)  # has a nasal vowel
B = pd.Series(langs.isin(v[~v.Phoneme.str.contains(NASAL)].Glottocode), index=langs)  # has an oral vowel
print("\nQ18 nasal (A) vs oral (B) -- trivial: every language has an oral vowel\n", pd.crosstab(A, B))
print("Q18 [y] (A) vs [i]&[u] (B)\n", pd.crosstab(has_y.rename("has y"), (has_i & has_u).rename("has i&u")))
ab = pd.crosstab(has_y, has_i & has_u)
p_b_given_a = ab.loc[True, True] / ab.loc[True].sum()
p_a_given_b = ab.loc[True, True] / ab[True].sum()
print(f"Q18 P(i&u | y) = {p_b_given_a:.3f}   but   P(y | i&u) = {p_a_given_b:.3f}")
