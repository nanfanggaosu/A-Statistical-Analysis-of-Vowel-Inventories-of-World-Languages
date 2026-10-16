"""Section 3.2: association between pairs of vowels (Q20-Q28)."""
import itertools

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, false_discovery_control, fisher_exact

from lab import OUT, load, presence_matrix, vowels

rng = np.random.default_rng(0)
df = load()
M = presence_matrix(vowels(df))  # 2184 languages x all vowel types
N = len(M)
meta = df.drop_duplicates("Glottocode").set_index("Glottocode")


def table(a, b):
    """2x2 contingency table [[A&B, A&~B], [~A&B, ~A&~B]]."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    return np.array([[(a & b).sum(), (a & ~b).sum()], [(~a & b).sum(), (~a & ~b).sum()]])


def mutual_info(t):
    """I(A;B) in bits, 0 log 0 = 0."""
    p = t / t.sum()
    pa, pb = p.sum(axis=1, keepdims=True), p.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, p * np.log2(p / (pa * pb)), 0.0)
    return terms.sum()


def phi(t):
    (a, b), (c, d) = t
    return (a * d - b * c) / np.sqrt((a + b) * (c + d) * (a + c) * (b + d))


def describe(x, y):
    t = table(M[x], M[y])
    chi2, p, _, exp = chi2_contingency(t, correction=False)
    p11, pa, pb = t[0, 0] / N, t[0].sum() / N, t[:, 0].sum() / N
    return t, chi2, p, exp, mutual_info(t), p11, pa * pb, phi(t)


# --- Q20-Q23 on a few illustrative pairs -------------------------------------------
for x, y in [("i", "u"), ("e", "o"), ("ɛ", "ɔ"), ("i", "ɪ"), ("a", "ɑ"), ("iː", "i")]:
    t, chi2, p, exp, mi, p11, pp, ph = describe(x, y)
    print(f"[{x}]-[{y}] table={t.tolist()} chi2={chi2:.1f} p={p:.2e} MI={mi:.4f} bits "
          f"p(1,1)={p11:.3f} vs p(A)p(B)={pp:.3f} -> {'+' if p11 > pp else '-'} phi={ph:+.3f} min expected={exp.min():.1f}")

# --- Q24: rare vowel, chi2 validity vs Fisher -------------------------------------------
print()
for x, y in [("y", "ø"), ("y", "i"), ("ø", "ɨ"), ("œ", "ɔ̃")]:
    t, chi2, p, exp, *_ = describe(x, y)
    _, pf = fisher_exact(t)
    print(f"Q24 [{x}]-[{y}] table={t.tolist()} expected min={exp.min():.2f} chi2 p={p:.3g} Fisher p={pf:.3g}")

# --- Q25: how many tests? ------------------------------------------------------------
V = M.shape[1]
singletons = (M.sum() == 1).sum()
print(f"\nQ25 distinct vowels: {V}; attested in a single language: {singletons} ({singletons / V:.0%})")
print(f"Q25 all pairs: {V * (V - 1) // 2:,} tests -> ~{0.05 * V * (V - 1) / 2:,.0f} false positives at 5% under H0")
F = M.loc[:, M.mean() >= 0.05]
fv = list(F.columns)
print(f"Q25 vowels in >=5% of languages: {len(fv)} -> {len(fv) * (len(fv) - 1) // 2} pairs: {' '.join(fv)}")

rows = []
for x, y in itertools.combinations(fv, 2):
    t = table(F[x], F[y])
    chi2, p, _, exp = chi2_contingency(t, correction=False)
    rows.append(dict(A=x, B=y, a11=t[0, 0], a10=t[0, 1], a01=t[1, 0], a00=t[1, 1], chi2=chi2, p=p,
                     min_exp=exp.min(), MI=mutual_info(t), phi=phi(t)))
R = pd.DataFrame(rows)
R["p_fisher"] = [fisher_exact([[r.a11, r.a10], [r.a01, r.a00]])[1] for r in R.itertuples()]
R["p_bonf"] = np.minimum(R.p * len(R), 1)
R["q_bh"] = false_discovery_control(R.p, method="bh")
print(f"Q25 min expected count < 5 in {(R.min_exp < 5).sum()} of {len(R)} pairs")
print(f"Q25 significant: raw p<.05 {(R.p < .05).sum()}, Bonferroni {(R.p_bonf < .05).sum()}, BH {(R.q_bh < .05).sum()}")

# --- Q26: effect size ranking -----------------------------------------------------------
R["rank_p"] = R.p.rank()
R["rank_absphi"] = (-R.phi.abs()).rank()
print(f"Q26 Spearman(rank by p, rank by |phi|) = {R.rank_p.corr(R.rank_absphi, method='spearman'):.3f}")
print(f"Q26 Spearman(|phi|, MI) = {R.phi.abs().corr(R.MI, method='spearman'):.3f}")
cols = ["A", "B", "a11", "a10", "a01", "a00", "phi", "MI", "p", "q_bh"]
print("Q26 strongest positive:\n", R.sort_values("phi", ascending=False).head(12)[cols].to_string(index=False))
print("Q26 strongest negative:\n", R.sort_values("phi").head(12)[cols].to_string(index=False))
print("Q26 BH-significant but |phi| < 0.1:", ((R.q_bh < .05) & (R.phi.abs() < .1)).sum())

# Negative pairs: which sources use which symbol? (transcription convention vs system)
src = meta.Source.reindex(F.index)
for x in ["a", "ɑ", "ɐ", "e", "e̞", "o", "o̞", "ɛ", "ɪ"]:
    if x in F:
        print(f"Q26 [{x}] share of languages by source:", (F[x].groupby(src).mean().round(2)).to_dict())

# --- Q27: null model -------------------------------------------------------------------
# Curveball algorithm (Strona et al. 2014): random swaps that preserve every row sum
# (vowel-inventory size) and every column sum (frequency of each vowel) of the full matrix.
rows_sets = [set(np.flatnonzero(r)) for r in M.to_numpy()]


def curveball(sets, n_trades):
    sets = [s.copy() for s in sets]
    n = len(sets)
    for _ in range(n_trades):
        i, j = rng.choice(n, 2, replace=False)
        a, b = sets[i], sets[j]
        shared = a & b
        only_a, only_b = list(a - shared), list(b - shared)
        pool = only_a + only_b
        if not only_a or not only_b:
            continue
        rng.shuffle(pool)
        k = len(only_a)
        sets[i] = shared | set(pool[:k])
        sets[j] = shared | set(pool[k:])
    return sets


fidx = [M.columns.get_loc(c) for c in fv]
iu = np.triu_indices(len(fv), 1)


def phis(X):
    """phi for every pair of columns of a boolean matrix (vectorised)."""
    X = X.astype(float)
    n = len(X)
    a = X.T @ X
    ca = X.sum(0)
    num = n * a - np.outer(ca, ca)
    den = np.sqrt(np.outer(ca * (n - ca), ca * (n - ca)))
    return (num / den)[iu]


obs = phis(F.to_numpy())
assert np.allclose(obs, R.phi.to_numpy())
n_null = 200
null = np.empty((n_null, len(obs)))
state = rows_sets
state = curveball(state, 20 * N)  # burn-in
for k in range(n_null):
    state = curveball(state, 2 * N)
    X = np.zeros((N, len(fidx)), bool)
    for r, s in enumerate(state):
        for c, j in enumerate(fidx):
            X[r, c] = j in s
    null[k] = phis(X)
# sanity: margins preserved
Xfull = np.zeros(M.shape, bool)
for r, s in enumerate(state):
    Xfull[r, list(s)] = True
assert (Xfull.sum(1) == M.sum(axis=1).to_numpy()).all() and (Xfull.sum(0) == M.sum(axis=0).to_numpy()).all()

R["null_mean"], R["null_sd"] = null.mean(0), null.std(0)
R["z_null"] = (R.phi - R.null_mean) / R.null_sd
# two-sided empirical p-value with +1 correction
R["p_null"] = (1 + (np.abs(null - R.null_mean.to_numpy()) >= np.abs(obs - R.null_mean.to_numpy())).sum(0)) / (n_null + 1)
R["q_null"] = false_discovery_control(R.p_null, method="bh")
print(f"\nQ27 null phi: mean over pairs {R.null_mean.mean():+.3f}; share of pairs with positive null mean {(R.null_mean > 0).mean():.0%}")
print(f"Q27 BH-significant vs independence: {(R.q_bh < .05).sum()} | vs curveball null (BH on empirical p): {(R.q_null < .05).sum()}")
sig_ind_pos = ((R.q_bh < .05) & (R.phi > 0)).sum()
sig_null_pos = ((R.q_null < .05) & (R.phi > R.null_mean)).sum()
print(f"Q27 positive associations: vs independence {sig_ind_pos}, vs null {sig_null_pos}")
print("Q27 largest |z| vs null:\n", R.reindex(R.z_null.abs().sort_values(ascending=False).index)
      .head(15)[["A", "B", "phi", "null_mean", "z_null", "q_null"]].round(3).to_string(index=False))
R.to_csv(OUT / "q20_28_vowel_pairs.csv", index=False)

fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(R.null_mean, R.phi, s=8, alpha=.6)
lim = [min(R.phi.min(), R.null_mean.min()), max(R.phi.max(), R.null_mean.max())]
ax.plot(lim, lim, "k--", lw=1)
ax.axhline(0, color="grey", lw=.5)
ax.set_xlabel("expected φ under curveball null")
ax.set_ylabel("observed φ")
ax.set_title("Vowel pairs (≥5% of languages)")
fig.tight_layout()
fig.savefig(OUT / "q27_phi_vs_null.png", dpi=150)
