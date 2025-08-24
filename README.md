# ZeroSumFreeSets-Z4

This repository provides explicit constructions of large subsets in \((\mathbb{Z}/4\mathbb{Z})^n\) with no three distinct elements summing to zero modulo 4, addressing an open problem posed by Nathan Kaplan in the 2014 Combinatorial and Additive Number Theory (CANT) problem session [3].

## Problem
Find the largest subset \(H \subseteq (\mathbb{Z}/4\mathbb{Z})^n\) such that there are no distinct \(x, y, z \in H\) with \(x + y + z \equiv 0 \pmod{4}\) (pointwise vector addition). Kaplan posed this for general abelian groups, suggesting greedy constructions to estimate maximum sizes and density [3]. No explicit lower bounds for this exact variant in \((\mathbb{Z}/4\mathbb{Z})^n\) were found in literature searches up to August 2025. A trivial construction is \(\{1,3\}^n\) (size \(2^n = 32\) for \(n=5\)).

## Method
AI-assisted greedy algorithm: Generate all \(4^n\) vectors, sort by a custom priority function, and add greedily if no forbidden triple forms. Two versions:
- **Baseline** (`baseline_script.py`): Favors vectors heavy in '2's (self-inverse mod 4), with symmetry and odd component sums.
- **Refined** (`refined_script.py`): Enhanced with additional probes (periodic, asymmetric, balance) and tuned coefficients for higher density.

Verification: Exhaustive check of all pairs for forbidden \(c = -(a + b) \mod 4\), distinct from \(a, b\). Priority functions were refined with Grok (xAI), inspired by AI-driven combinatorial searches like FunSearch.

## Results
- **n=5** (total 1024 vectors):
  - Baseline: Size 176 (~17.2% density), valid (`n5_size176_set.json`).
  - Refined: Size 512 (50% density), valid (`n5_size512_set.json`).
- **n=6** (total 4096): Refined size 636 (~15.5%), valid (available via `refined_script.py`).
- **n=7** (total 16384): Refined size 2470 (~15.1%), valid (available via `refined_script.py`).

The size-512 construction for \(n=5\) achieves a notably high density, providing a strong lower bound for this variant. Densities suggest an asymptotic lower bound around 15%. Sample vectors (n=5 refined): (2,2,1,2,2), (2,2,3,2,2), (2,2,2,3,2).

## How to Run
1. Clone the repo: `git clone https://github.com/DynMEP/ZeroSumFreeSets-Z4.git`
2. Install Python 3 (no extra packages needed).
3. Run baseline: `python baseline_script.py` (edit N_DIM/MODULO for other n).
4. Run refined: `python refined_script.py` (recommended).
5. Outputs size, verification, sample vectors, and saves full set to JSON.

## Files
- `baseline_script.py`: Baseline greedy algorithm (size 176 for n=5).
- `refined_script.py`: Refined algorithm (size 512 for n=5, extensible to n=6,7).
- `n5_size176_set.json`: Full 176-vector set for n=5 (baseline).
- `n5_size512_set.json`: Full 512-vector set for n=5 (refined).

## Related Work
- [1] Y. Caro, A weighted Erdős–Ginzburg–Ziv theorem, J. Combin. Theory Ser. A 80(2):186–195, 1997.
- [2] W.D. Gao and A. Geroldinger, Zero-sum problems in finite abelian groups: A survey, Expo. Math. 24(4):337–369, 2006.
- [3] S.J. Miller et al., Combinatorial and additive number theory problem sessions: 2009–2016, https://web.williams.edu/Mathematics/sjmiller/public_html/math/papers/CANTProblemSessions.pdf , 2017.

## Further Reading
- arXiv preprint: [].
- MathOverflow post: https://mathoverflow.net/questions/499530/largest-3-zero-sum-free-subset-in-mathbbz-4-mathbbzn

## License
MIT License – feel free to use/extend.

## Contact
Alfonso Davila Vera, adavila@dynmep.com, https://github.com/DynMEP

References:  
[1] Y. Caro, A weighted Erdős–Ginzburg–Ziv theorem, J. Combin. Theory Ser. A 80(2):186–195, 1997.  
[2] W.D. Gao and A. Geroldinger, Zero-sum problems in finite abelian groups: A survey, Expo. Math. 24(4):337–369, 2006.  
[3] S.J. Miller et al., Combinatorial and additive number theory problem sessions: 2009–2016 (available online).

<p><a href="https://www.buymeacoffee.com/h1pot"> <img align="left" src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" height="50" width="210" alt="h1pot" /></a></p><br><br>




