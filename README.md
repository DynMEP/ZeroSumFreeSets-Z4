# ZeroSumFreeSets-Z4

This repository provides explicit constructions of maximal 3-zero-sum-free subsets in \((\mathbb{Z}/4\mathbb{Z})^n\), addressing an open problem posed by Nathan Kaplan in the 2014 Combinatorial and Additive Number Theory (CANT) problem session [3]. Recent advancements resolve the asymptotic density at exactly 0.5 for all \(n\), proven optimal.

## Problem
Find the largest subset \(H \subseteq (\mathbb{Z}/4\mathbb{Z})^n\) such that there are no distinct \(x, y, z \in H\) with \(x + y + z \equiv 0 \pmod{4}\) (pointwise vector addition). Kaplan's 2014 CANT pose [3] motivated this for abelian groups, suggesting greedy methods to estimate sizes and densities. No prior explicit lower bounds existed for \((\mathbb{Z}/4\mathbb{Z})^n\) until this work. A trivial construction, \(\{1,3\}^n\), yields size \(2^n = 32\) for \(n=5\).

## Method
AI-assisted hybrid greedy-genetic algorithm, refined with Grok (xAI), inspired by combinatorial searches like FunSearch:
- **Baseline** (`baseline_script.py`): Initial greedy approach favoring '2's, symmetry, odd sums.
- **Refined** (`refined_script.py`): Enhanced with probes (periodic, asymmetric, balance).
- **Omni-Optimized v5.0.0** (`omni_optimized_hybrid_discovery_v5.py`): Introduces universal optimal construction (first coordinate odd: 1 or 3 mod 4), stratified sampling, adaptive mutations, early stopping, profile chaining, and GPU acceleration with batching for \(n=8+\).
- **Verification**: Exhaustive pair checks for forbidden \(c = -(a + b) \mod 4\), distinct from \(a, b\).

## Results
- **n=5** (total 1024 vectors):
  - Baseline: Size 176 (~17.2% density), valid.
  - Refined: Size 512 (50% density), valid.
- **n=6** (total 4096): Omni-Optimized v5.0.0: Size 2048 (50% density), valid.
- **n=7** (total 16384): Omni-Optimized v5.0.0: Size 8192 (50% density), valid.
- **General**: Universal construction achieves size \(2 \times 4^{n-1} = 4^n / 2\) (density 0.5) for all \(n\), proven maximal via \(|H|^2 \leq (4^n - |H|) \cdot |H|\). Sample vectors (n=5 refined): (2,2,1,2,2), (2,2,3,2,2), (2,2,2,3,2).

## How to Run
1. Clone the repo: `git clone https://github.com/DynMEP/ZeroSumFreeSets-Z4.git`
2. Install Python 3.8+ and dependencies (optional torch for GPU): `pipx install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`
3. Run omni-optimized: `python3 omni_optimized_hybrid_discovery_v5.py --n 6 7 --mod 4 --profile all --runs 20 --jitter 0.15 --sample-size 100000 --workers 8 --save`
4. Outputs size, density, verification, and saves full set/metadata to JSON.

## Files
- `baseline_script.py`: Original greedy algorithm (size 176 for n=5).
- `refined_script.py`: Enhanced greedy (size 512 for n=5).
- `omni_optimized_hybrid_discovery_v5.py`: Latest v5.0.0 with universal construction.
- `n5_size176_set.json`: Baseline output for n=5.
- `n5_size512_set.json`: Refined output for n=5.
- `n6_best_meta.json`, `n6_best_set.json`: Omni-optimized output for n=6 (2048).
- `n7_best_meta.json`, `n7_best_set.json`: Omni-optimized output for n=6 (8192).

## Latest Release
- **v5.0.0** (August 28, 2025, 12:17 PM EDT): 
  - Universal 0.5 density construction, GPU support, output files for n=6,7.
  - Download: [v5.0.0 Release](https://github.com/DynMEP/ZeroSumFreeSets-Z4/releases/tag/v5.0.0)

## Expected Output
- JSON files (e.g., `n7_best_set.json`) with full subsets and metadata, verified 3-zero-sum-free.

## Related Work
- [1] Y. Caro, A weighted Erdős–Ginzburg–Ziv theorem, J. Combin. Theory Ser. A 80(2):186–195, 1997.
- [2] W.D. Gao and A. Geroldinger, Zero-sum problems in finite abelian groups: A survey, Expo. Math. 24(4):337–369, 2006.
- [3] S.J. Miller et al., Combinatorial and additive number theory problem sessions: 2009–2016, https://web.williams.edu/Mathematics/sjmiller/public_html/math/papers/CANTProblemSessions.pdf, 2017.

## Further Reading
- arXiv preprint: 
- MathOverflow post: 

## License
MIT License – feel free to use and extend.

## Contact
Alfonso Davila Vera, adavila@dynmep.com, https://github.com/DynMEP

## References
[1] Y. Caro, A weighted Erdős–Ginzburg–Ziv theorem, J. Combin. Theory Ser. A 80(2):186–195, 1997.
[2] W.D. Gao and A. Geroldinger, Zero-sum problems in finite abelian groups: A survey, Expo. Math. 24(4):337–369, 2006.
[3] S.J. Miller et al., Combinatorial and additive number theory problem sessions: 2009–2016 (available online).
