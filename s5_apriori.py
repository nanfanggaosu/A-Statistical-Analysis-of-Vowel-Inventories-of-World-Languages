"""Section 5: frequent itemsets and association rules (Q33-Q46)."""
import itertools

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

from lab import OUT, load, nfd, presence_matrix, vowels

df = load()
V = presence_matrix(vowels(df))  # Q39: binary language x vowel matrix
N = len(V)
MIN_SUP = 0.10

# --- Q33-Q38 by hand -------------------------------------------------------------
sup1 = V.mean().sort_values(ascending=False)  # Q33
print(f"Q33 {len(sup1)} single vowels; support >= 10%: {(sup1 >= MIN_SUP).sum()}")
print(sup1.head(20).round(3).to_string())

items = sup1[sup1 >= MIN_SUP].index  # a pair can only be frequent if both items are (Apriori property)
sup2 = pd.Series({(a, b): (V[a] & V[b]).mean() for a, b in itertools.combinations(items, 2)})  # Q34
freq2 = sup2[sup2 >= MIN_SUP].sort_values(ascending=False)  # Q35
all_pairs = len(sup1) * (len(sup1) - 1) // 2
print(f"\nQ34 pairs of all {len(sup1)} vowels: {all_pairs:,} (only {len(sup2)} need computing, Apriori property)")
print(f"Q36 frequent pairs (support >= {MIN_SUP:.0%}): {len(freq2)}")
print(freq2.head(25).round(3).to_string())


def rule(x, y):  # Q37
    x, y = nfd(x), nfd(y)
    s_xy, s_x, s_y = (V[x] & V[y]).mean(), V[x].mean(), V[y].mean()
    return dict(rule=f"{x} -> {y}", support=s_xy, confidence=s_xy / s_x, lift=s_xy / (s_x * s_y))


print("\nQ37 hand-computed rules:")
print(pd.DataFrame([rule(*r) for r in [("i", "u"), ("u", "i"), ("e", "o"), ("o", "e"), ("ɛ", "ɔ"),
                                       ("iː", "uː"), ("ĩ", "ã"), ("e", "i"), ("ə", "a")]]).round(3).to_string(index=False))

# --- Q40-Q42 with mlxtend ---------------------------------------------------------
fi = apriori(V, min_support=MIN_SUP, use_colnames=True)
fi["size"] = fi.itemsets.map(len)
print("\nQ40 frequent itemsets by size (10%):", fi["size"].value_counts().sort_index().to_dict())
# cross-check mlxtend against the hand computation
mx2 = {tuple(sorted(s)): x for s, x in zip(fi.itemsets, fi.support) if len(s) == 2}
assert len(mx2) == len(freq2) and all(abs(mx2[tuple(sorted(k))] - x) < 1e-12 for k, x in freq2.items())
print("Q40 largest itemsets:\n", fi[fi["size"] == fi["size"].max()].assign(itemsets=lambda d: d.itemsets.map(lambda s: " ".join(sorted(s))))
      .sort_values(["support", "itemsets"], ascending=[False, True]).head(10).to_string(index=False))
fi_out = fi.assign(itemsets=fi.itemsets.map(lambda s: " ".join(sorted(s)))).sort_values(["size", "support"], ascending=[True, False])
fi_out.to_csv(OUT / "q41_frequent_itemsets_vowels_10pct.csv", index=False)  # Q41

rules = association_rules(fi, metric="confidence", min_threshold=0.0)
rules["rule"] = rules.antecedents.map(lambda s: " ".join(sorted(s))) + " -> " + rules.consequents.map(lambda s: " ".join(sorted(s)))
cols = ["rule", "support", "confidence", "lift"]
print(f"\nQ42 rules from 10% itemsets: {len(rules)}")
print("Q42 top lift (confidence >= .8):\n", rules[rules.confidence >= .8].sort_values(["lift", "rule"], ascending=[False, True]).head(12)[cols].round(3).to_string(index=False))
print("Q42 lowest lift:\n", rules.sort_values(["lift", "rule"]).head(8)[cols].round(3).to_string(index=False))

# --- Q45: high confidence, lift ~ 1 ---------------------------------------------------
hc = rules[(rules.confidence >= .9) & (rules.lift.between(.97, 1.03))]
print(f"\nQ45 rules with confidence >= .9 and lift in [.97, 1.03]: {len(hc)}")
print(hc.sort_values(["confidence", "rule"], ascending=[False, True]).head(10)[cols + ["consequent support"]].round(3).to_string(index=False))

# --- Q44: threshold sensitivity ---------------------------------------------------------
print("\nQ44 threshold sensitivity (vowels):")
for s in [0.05, 0.10, 0.25, 0.50]:
    f = apriori(V, min_support=s, use_colnames=True)
    sz = f.itemsets.map(len).value_counts().sort_index().to_dict()
    singles = sorted(" ".join(x) for x in f.itemsets if len(x) == 1)
    print(f"  {s:.0%}: {len(f)} itemsets, by size {sz}; items: {' '.join(singles) if len(singles) < 35 else len(singles)}")

# --- Q46: do the universals show up? ------------------------------------------------------
low = apriori(V, min_support=0.02, use_colnames=True, max_len=3)
lr = association_rules(low, metric="confidence", min_threshold=0.0)
lr["rule"] = lr.antecedents.map(lambda s: " ".join(sorted(s))) + " -> " + lr.consequents.map(lambda s: " ".join(sorted(s)))
wanted = ["y -> i", "y -> u", "y -> i u", "ã -> a", "ĩ -> i", "ũ -> u", "iː -> i", "aː -> a", "uː -> u", "ø -> e", "œ -> ɛ"]
print("\nQ46 universal-type rules (min support 2%):")
wanted = [nfd(w) for w in wanted]
print(lr[lr.rule.isin(wanted)].set_index("rule").loc[lambda d: d.index.intersection(wanted)][["support", "confidence", "lift"]].round(3).to_string())
print(f"Q46 [y] support = {V['y'].mean():.3f}: invisible at 10% threshold, visible at 5% only as single item/pair")
print("Q46 highest-lift rules of size 3 (support >= 2%, conf >= .8):\n",
      lr[(lr.antecedents.map(len) == 2) & (lr.confidence >= .8)].sort_values(["lift", "rule"], ascending=[False, True]).head(10)[cols].round(3).to_string(index=False))

# --- Q43: whole inventory, consonants included -----------------------------------------------
A = presence_matrix(df)
fa = apriori(A, min_support=MIN_SUP, use_colnames=True, max_len=3)
fa["size"] = fa.itemsets.map(len)
seg_class = df.drop_duplicates("Phoneme").set_index("Phoneme").SegmentClass


def n_vowels(s):
    return sum(seg_class[p] == "vowel" for p in s)


fa["vowels_in_set"] = fa.itemsets.map(n_vowels)
print(f"\nQ43 all segments: {A.shape[1]} types; frequent (10%) itemsets up to size 3: {len(fa)}")
for k in [1, 2, 3]:
    sub = fa[fa["size"] == k]
    print(f"  size {k}: {len(sub)} itemsets, purely vowel {(sub.vowels_in_set == k).sum()}, "
          f"purely consonant {(sub.vowels_in_set == 0).sum()}, mixed {((sub.vowels_in_set > 0) & (sub.vowels_in_set < k)).sum()}")
print("Q43 top 15 pairs:\n", fa[fa["size"] == 2].sort_values("support", ascending=False).head(15)
      .assign(itemsets=lambda d: d.itemsets.map(lambda s: " ".join(sorted(s))))[["itemsets", "support"]].round(3).to_string(index=False))
top_single = A.mean().sort_values(ascending=False).head(15)
print("Q43 most frequent segments:", {k: round(x, 2) for k, x in top_single.items()})
ra = association_rules(fa, metric="confidence", min_threshold=0.0)
print("Q43 rules with a consonant antecedent and vowel consequent, top by confidence:")
cv = ra[ra.antecedents.map(lambda s: n_vowels(s) == 0) & ra.consequents.map(lambda s: n_vowels(s) == len(s))]
cv = cv.assign(rule=cv.antecedents.map(lambda s: " ".join(sorted(s))) + " -> " + cv.consequents.map(lambda s: " ".join(sorted(s))))
print(cv.sort_values(["confidence", "rule"], ascending=[False, True]).head(8)[cols].round(3).to_string(index=False))
print(f"Q43 lift range of those rules: {cv.lift.min():.2f}-{cv.lift.max():.2f}, median {cv.lift.median():.2f}")
