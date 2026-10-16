"""Section 2: getting the data (Q2, Q4-Q8, Q10)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from lab import OUT, load_raw, select_inventories, vowels

raw = load_raw()

# --- Q2 -------------------------------------------------------------------
print("Q2 rows:", len(raw))
print("Q2 inventories:", raw.InventoryID.nunique())
print("Q2 Glottocodes (nunique, NaN excluded):", raw.Glottocode.nunique())
print("Q2 Glottocodes (nunique, NaN included):", raw.Glottocode.nunique(dropna=False))
inv = raw.drop_duplicates("InventoryID")
print("Q2 inventories without Glottocode:\n", inv[inv.Glottocode.isna()][["InventoryID", "ISO6393", "LanguageName", "Source"]])
print("Q2 groupby().ngroups on Glottocode:", raw.groupby("Glottocode").ngroups)

# --- Q4 -------------------------------------------------------------------
per_lang = inv.dropna(subset=["Glottocode"]).groupby("Glottocode").InventoryID.nunique()
print("\nQ4 languages with >1 inventory:", (per_lang > 1).sum(), "| max:", per_lang.max(), per_lang.idxmax())
df = select_inventories(raw)
print("Q4 kept inventories:", df.InventoryID.nunique(), "languages:", df.Glottocode.nunique())
kept_src = df.drop_duplicates("InventoryID").Source.value_counts()
all_src = inv.Source.value_counts()
print("Q4 source share before -> after selection:\n", pd.DataFrame({"all": all_src, "kept": kept_src}).fillna(0).astype(int))

# --- Q5: disagreement between descriptions of the same language --------------
sizes = raw.groupby(["Glottocode", "InventoryID", "Source"]).agg(
    n_seg=("Phoneme", "size"),
    n_vow=("SegmentClass", lambda s: (s == "vowel").sum()),
).reset_index()
multi = sizes[sizes.Glottocode.map(per_lang) > 1]
spread = multi.groupby("Glottocode").agg(k=("InventoryID", "size"),
                                          vow_min=("n_vow", "min"), vow_max=("n_vow", "max"),
                                          seg_min=("n_seg", "min"), seg_max=("n_seg", "max"))
spread["vow_range"] = spread.vow_max - spread.vow_min
print("\nQ5 languages with several inventories:", len(spread))
print("Q5 share with identical vowel counts across descriptions:", round((spread.vow_range == 0).mean(), 3))
print("Q5 share with identical total sizes:", round((spread.seg_max == spread.seg_min).mean(), 3))
print("Q5 vowel-count range, distribution:\n", spread.vow_range.describe())


def vset(i):
    return set(raw[(raw.InventoryID == i) & (raw.SegmentClass == "vowel")].Phoneme)


def jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else 1.0


# pairwise Jaccard on vowel sets for every language with several inventories
jac = []
for g, ids in multi.groupby("Glottocode").InventoryID:
    ids = sorted(ids)
    sets = {i: vset(i) for i in ids}
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            jac.append(jaccard(sets[ids[a]], sets[ids[b]]))
jac = pd.Series(jac)
print(f"Q5 pairwise Jaccard (vowel sets): n={len(jac)} mean={jac.mean():.2f} median={jac.median():.2f} identical={(jac == 1).mean():.2%}")

for g in ["stan1293", "nort2641", "hind1269", "osse1243"]:  # English, Kurmanji, Hindi, Ossetian
    rows = sizes[sizes.Glottocode == g]
    if rows.empty:
        continue
    name = raw[raw.Glottocode == g].LanguageName.iloc[0]
    print(f"\nQ5 {name} ({g}):")
    for _, r in rows.iterrows():
        print(f"  inv {r.InventoryID:>4} [{r.Source:>6}] {r.n_vow:>2} vowels / {r.n_seg:>3} segs: {' '.join(sorted(vset(r.InventoryID)))}")

# --- Q6: inventory size by source ---------------------------------------------
print("\nQ6 total/vowel inventory size by Source (all 3020 inventories):")
print(sizes.groupby("Source")[["n_seg", "n_vow"]].agg(["count", "median", "mean"]).round(1))
# same-language comparison removes the "different languages" confound
pair = multi.pivot_table(index="Glottocode", columns="Source", values="n_vow", aggfunc="mean")
print("\nQ6 same-language comparison, mean(n_vow[A] - n_vow[B]) over languages described by both:")
srcs = pair.columns
res = []
for a in srcs:
    for b in srcs:
        if a < b:
            both = pair[[a, b]].dropna()
            if len(both) >= 20:
                res.append((a, b, len(both), round((both[a] - both[b]).mean(), 2)))
print(pd.DataFrame(res, columns=["A", "B", "n_lang", "mean_diff_vowels"]).sort_values("n_lang", ascending=False).to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 4))
order = sizes.groupby("Source").n_vow.median().sort_values().index
ax.boxplot([sizes[sizes.Source == s].n_vow for s in order], tick_labels=order, showfliers=True)
ax.set_ylabel("vowels per inventory")
ax.set_title("Vowel inventory size by PHOIBLE source (all inventories)")
fig.tight_layout()
fig.savefig(OUT / "q6_vowels_by_source.png", dpi=150)

# --- Q7: effect of the Marginal convention -------------------------------------
v = vowels(df)
print("\nQ7 vowel rows by Marginal (selected inventories):\n", v.Marginal.value_counts(dropna=False))
print("Q7 vowel rows with Marginal NA, by source:\n", v[v.Marginal.isna()].Source.value_counts())
for label, sub in [("keep all", v), ("drop Marginal==True", v[v.Marginal != True]),
                   ("drop Marginal NA", v[v.Marginal.notna()])]:
    n = sub.groupby("Glottocode").size().reindex(df.Glottocode.unique(), fill_value=0)
    print(f"Q7 {label:<22} languages with >=1 vowel: {(n > 0).sum():>4}  mean vowels: {n.mean():.2f}")

# --- Q8 -------------------------------------------------------------------
nv = v.groupby("Glottocode").size().reindex(df.Glottocode.unique(), fill_value=0)
print("\nQ8 languages:", len(nv), "| languages with 0 vowels:", (nv == 0).sum())
print("Q8 vowel inventory size:\n", nv.describe().round(2))
print("Q8 mode:", nv.mode().tolist())
print("Q8 frequency table:\n", nv.value_counts().sort_index().to_string())
nv.rename("n_vowels").to_csv(OUT / "vowel_counts.csv")

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(nv, bins=range(0, nv.max() + 2), align="left", edgecolor="white")
ax.axvline(nv.mean(), color="k", ls="--", label=f"mean = {nv.mean():.2f}")
ax.axvline(nv.median(), color="r", ls=":", label=f"median = {nv.median():.0f}")
ax.set_xlabel("number of vowels")
ax.set_ylabel("number of languages")
ax.legend()
ax.set_title(f"Vowel inventory sizes ({len(nv)} languages, PHOIBLE v2.0)")
fig.tight_layout()
fig.savefig(OUT / "q8_vowel_hist.png", dpi=150)

# --- Q10: IQR outliers -----------------------------------------------------------
q1, q3 = nv.quantile([0.25, 0.75])
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
out = nv[(nv < lo) | (nv > hi)].sort_values()
print(f"\nQ10 Q1={q1} Q3={q3} IQR={iqr} fences=[{lo}, {hi}] -> {len(out)} outliers ({(nv < lo).sum()} low, {(nv > hi).sum()} high)")
meta = df.drop_duplicates("Glottocode").set_index("Glottocode")[["LanguageName", "InventoryID", "Source"]]
tab = meta.loc[out.index].assign(n_vowels=out)
print("Q10 outliers by source:\n", tab.Source.value_counts())
print("Q10 top 15:\n", tab.sort_values("n_vowels", ascending=False).head(15).to_string())
print("Q10 low side:\n", tab[tab.n_vowels < lo].to_string())
tab.to_csv(OUT / "q10_outliers.csv")
