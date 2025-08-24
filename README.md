# ZeroSumFreeSets-Z4

This repository presents explicit constructions of large subsets in \((\mathbb{Z}/4\mathbb{Z})^n\) with no three distinct elements summing to zero modulo 4. This addresses an open problem posed by Nathan Kaplan in the 2014 CANT problem session on zero-sum-free subsets in abelian groups.

## Problem
Find the largest subset \(H \subseteq (\mathbb{Z}/4\mathbb{Z})^n\) such that there are no distinct \(x, y, z \in H\) with \(x + y + z \equiv 0 \pmod{4}\) (pointwise vector addition).

No explicit lower bounds were found in literature searches up to August 2025. Trivial example: \(\{1,3\}^n\) (size \(2^n = 32\) for \(n=5\)).

## Method
AI-assisted greedy algorithm: Generate all \(4^n\) vectors, sort by a custom priority function, and add greedily if no forbidden triple forms. Two priority functions:
- **Baseline**: Favors vectors heavy in '2's, with symmetry and odd component sums (`baseline_script.py`).
- **Refined**: Enhanced with additional probes (periodic, asymmetric, balance) and tuned coefficients for higher density (`refined_script.py`).

Verification: Exhaustive check of all pairs for forbidden \(c = -(a + b) \mod 4\), distinct from \(a, b\).

AI Assistance: Priority functions refined with Grok (xAI).

## Results
- **n=5** (total 1024 vectors):
  - Baseline: Size 176 (~17% density), valid (`n5_size176_set.json`).
  - Refined: Size 512 (50% density), valid (`n5_size512_set.json`).
- **n=6** (total 4096): Refined size 636 (~15.5%), valid (`n6_size636_set.json`).
- **n=7** (total 16384): Refined size 2470 (~15.1%), valid (`n7_size2470_set.json`).

These provide new explicit lower bounds, with the refined construction achieving a notably high density for \(n=5\). Densities suggest an asymptotic lower bound around 15%.

Sample vectors (n=5 refined): (2,2,2,2,2), (2,2,2,2,0), (2,2,2,0,2), etc.

## How to Run
1. Clone the repo: `git clone https://github.com/DynMEP/ZeroSumFreeSets-Z4.git`
2. Install Python 3 (no extra packages needed).
3. Run baseline: `python baseline_script.py` (edit N_DIM/MODULO for other n).
4. Run refined: `python refined_script.py` (recommended).
5. Outputs size, verification, sample vectors, and saves full set to JSON.

## Related Work
- Erdős–Ginzburg–Ziv theorem [1].
- Zero-sum problems survey [2].
- Kaplan's 2014 CANT problem [3].

## Further Reading
- arXiv preprint: [].
- MathOverflow post: [].

## License
MIT License – feel free to use/extend.

References:  
[1] Y. Caro, A weighted Erdős–Ginzburg–Ziv theorem, J. Combin. Theory Ser. A 80(2):186–195, 1997.  
[2] W.D. Gao and A. Geroldinger, Zero-sum problems in finite abelian groups: A survey, Expo. Math. 24(4):337–369, 2006.  
[3] S.J. Miller et al., Combinatorial and additive number theory problem sessions: 2009–2016 (available online).

<p><a href="https://www.buymeacoffee.com/h1pot"> <img align="left" src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" height="50" width="210" alt="h1pot" /></a></p><br><br>




