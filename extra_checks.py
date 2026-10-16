"""Robustness checks used in the report (Q4, Q10, Q14, Q17-Q19)."""
import numpy as np
import pandas as pd

from lab import base, load, load_raw, vowels

rng = np.random.default_rng(0)
raw = load_raw()
raw = raw[raw.Glottocode.notna()]
rv = vowels(raw).assign(base=lambda d: d.Phoneme.map(base))
by_inv = rv.groupby("InventoryID")
inv_lang = raw.drop_duplicates("InventoryID").set_index("InventoryID").Glottocode


def triangle(ids):
    b = rv[rv.InventoryID.isin(ids)].groupby("InventoryID").base.apply(set)
    return np.mean([{"i", "a", "u"} <= s for s in b]), rv[rv.InventoryID.isin(ids)].groupby("InventoryID").size().mean()


# --- Q4: sensitivity of two headline numbers to the inventory-selection convention ----------
groups = inv_lang.groupby(inv_lang).groups
rules = {"max InventoryID (ours)": [max(g) for g in groups.values()],
         "min InventoryID": [min(g) for g in groups.values()]}
for label, ids in rules.items():
    t, m = triangle(ids)
    print(f"Q4 {label:<24} i&a&u (criterion A) = {t:.3f}  mean vowels = {m:.2f}")
rand = [triangle([rng.choice(g) for g in groups.values()]) for _ in range(100)]
rand = np.array(rand)
print(f"Q4 random choice (100 draws) i&a&u = {rand[:, 0].mean():.3f} [{np.quantile(rand[:, 0], .025):.3f}, {np.quantile(rand[:, 0], .975):.3f}]"
      f"  mean vowels = {rand[:, 1].mean():.2f} [{np.quantile(rand[:, 1], .025):.2f}, {np.quantile(rand[:, 1], .975):.2f}]")

# --- selected data ------------------------------------------------------------------------
df = load()
v = vowels(df).assign(base=lambda d: d.Phoneme.map(base))
langs = pd.Index(df.Glottocode.unique())
inv = v.groupby("Glottocode").Phoneme.apply(set)
bases = v.groupby("Glottocode").base.apply(lambda s: set(s.dropna()))
meta = df.drop_duplicates("Glottocode").set_index("Glottocode")

# --- Q10: smallest inventories -----------------------------------------------------------------
n = v.groupby("Glottocode").size()
for g in n[n <= 2].index:
    print(f"Q10 2-vowel language: {meta.LanguageName[g]} ({meta.Source[g]}): {' '.join(sorted(inv[g]))}")

# --- Q14: a few violators, full vowel lists ---------------------------------------------------------
for g in ["squa1248", "nort2957", "plat1254", "dhuw1249", "tzel1254", "alab1237", "wari1268"]:
    print(f"Q14 {meta.LanguageName[g]} [{meta.Source[g]}]: {' '.join(sorted(inv[g]))}")

# --- Q17-Q19: implication vs base rate ------------------------------------------------------------
# For every marked vowel token, compare P(counterpart present | language has the marked vowel)
# with P(counterpart present) in all languages: an implication is informative only if the first
# is clearly above the second.
NASAL, LONG = "̃", "ː"


def base_rate(marker):
    cond, baseline = [], []
    for g, s in inv.items():
        for p in s:
            if marker in p and base(p):
                c = p.replace(marker, "")
                cond.append(c in s or base(p) in {base(q) for q in s if marker not in q})
                baseline.append(np.mean([c in t or base(p) in {base(q) for q in t if marker not in q}
                                         for t in inv.sample(300, random_state=0)]))
    return np.mean(cond), np.mean(baseline)


for name, m in [("nasal -> oral", NASAL), ("long -> short", LONG)]:
    c, b = base_rate(m)
    print(f"Q19 {name}: per-vowel P(counterpart | marked) = {c:.3f} vs base rate P(counterpart) = {b:.3f}")
has_i_u = bases.map(lambda s: {"i", "u"} <= s)
has_y = bases.map(lambda s: "y" in s)
print(f"Q19 y -> i&u: P(i&u | y) = {has_i_u[has_y].mean():.3f} vs P(i&u) = {has_i_u.mean():.3f} vs P(i&u | no y) = {has_i_u[~has_y].mean():.3f}")

# --- Q5: disagreement on the whole inventory, not only vowels --------------------------------
multi = raw.groupby("Glottocode").InventoryID.nunique()
multi = multi[multi > 1].index
sets = {c: raw[raw.SegmentClass.isin(cls)].groupby("InventoryID").Phoneme.apply(set)
        for c, cls in [("all", ["consonant", "vowel", "tone"]), ("consonant", ["consonant"]),
                       ("vowel", ["vowel"]), ("tone", ["tone"])]}
rows = []
for g in multi:
    ids = sorted(inv_lang[inv_lang == g].index)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            r = {"Glottocode": g}
            for c, s in sets.items():
                x, y = s.get(a, set()), s.get(b, set())
                r[c] = len(x & y) / len(x | y) if x | y else np.nan
                r[c + "_same_size"] = len(x) == len(y)
            rows.append(r)
J = pd.DataFrame(rows)
print(f"\nQ5 {len(multi)} languages, {len(J)} pairs of descriptions")
for c in ["all", "consonant", "vowel", "tone"]:
    j = J[c].dropna()
    print(f"Q5 {c:<9} Jaccard mean={j.mean():.2f} median={j.median():.2f} identical sets={(j == 1).mean():.1%} "
          f"same size={J.loc[j.index, c + '_same_size'].mean():.1%} (n={len(j)})")
t = sets["tone"]
one_side = sum((a in t.index) != (b in t.index)
               for g in multi for ids in [sorted(inv_lang[inv_lang == g].index)]
               for i, a in enumerate(ids) for b in ids[i + 1:] if a in t.index or b in t.index)
print(f"Q5 tone pairs where only one description lists tones at all: {one_side} of {J.tone.notna().sum()}")
