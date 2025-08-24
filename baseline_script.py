# =============================================================================
# Script Header: 3-Zero-Sum-Free Subset Construction for (Z/modZ)^n
# =============================================================================
# Purpose: Constructs large subsets in (Z/modZ)^n with no three distinct vectors 
# summing to zero modulo mod, using a greedy algorithm with a priority function 
# favoring '2's, symmetry, and odd sums.
# Version: 1.0.0
# Author: Alfonso Davila Vera - Electrical Engineer
# Contact: adavila@dynmep.com - www.linkedin.com/in/alfonso-davila-3a121087
# Repository: https://github.com/DynMEP/ZeroSumFreeSets-Z4
# License: MIT License (see LICENSE file in repository)
# Created: August 24, 2025
# Compatibility: Python 3.8+
# Dependencies: itertools, time, json
# Notes:
# - Optimized for mod=4, n=5 (yields size 176); adjust N_DIM and MODULO for other configurations.
# - Outputs JSON file with the subset and verifies no three distinct vectors sum to zero.
# - Addresses Nathan Kaplan's 2014 CANT problem on 3-zero-sum-free subsets in abelian groups.
# - See repository for detailed documentation and refined version with improved results.
# =============================================================================

import itertools
import time
import json

def mixed_priority(v, n, mod):
    """
    Mixed evolved priority: Favors '2's (as in Script 2), symmetry/product (Scripts 3/4), 
    dot products with probes (Scripts 1/4). Tuned for no a+b+c=0 mod mod.
    """
    # Probes from Scripts 1/4
    probe_sum = tuple(1 for _ in range(n))
    probe_periodic = tuple(i % 3 for i in range(n))  # Adapted for general n
    probe_asymmetric = tuple((i % 3 + 1) % 3 for i in range(n))  # From Script 4

    # 1. Component-Aware Weight: Favor '2' in mod=4 (from Script 2)
    weight = sum((1 if x in [1, mod-1] else 2.5 if x == 2 else 0) for x in v if x != 0)
    
    # 2. Symmetry Score (from all scripts)
    reflection_score = sum(1 for i in range(n // 2) if v[i] == v[n - 1 - i])
    
    # 3. Product Sum (from Script 3)
    product_sum_val = sum(v[i] * v[n - 1 - i] for i in range(n // 2))
    
    # 4. Algebraic Dots (from Scripts 1/4)
    dot_sum = sum(v[i] * probe_sum[i] for i in range(n)) % mod
    dot_periodic = sum(v[i] * probe_periodic[i] for i in range(n)) % mod
    dot_asym = sum(v[i] * probe_asymmetric[i] for i in range(n)) % mod
    
    # Evolved Coefficients: Balanced for max size; penalize even sums, reward alignments
    priority = (
        3.5 * weight +  # Increased from 3.0 to favor '2's more
        2.0 * reflection_score +  # Boosted symmetry
        0.05 * product_sum_val -  # Small boost from product
        4.5 * (1 if dot_sum % 2 == 0 else 0) -  # Penalize even sums (Script 2)
        1.0 * abs(dot_periodic - 1) -  # Reward close to 1 (Script 4)
        1.5 * (1 if dot_asym == 0 else 0)  # Penalize full alignment
    )
    return priority

# --- Solver and Verification ---

def solve_cap_set(n, mod, priority_function):
    """
    Constructs the set greedily based on priority.
    """
    print(f"Generating all {mod**n} vectors for (Z/{mod}Z)^{n}...")
    vectors = list(itertools.product(range(mod), repeat=n))

    print("Sorting vectors by priority...")
    sorted_vectors = sorted(vectors, key=lambda v: priority_function(v, n, mod), reverse=True)
    
    cap_set = []
    cap_set_lookup = set()
    
    print("Building set greedily...")
    for v in sorted_vectors:
        is_safe_to_add = True
        for a in cap_set:
            required_c = tuple((-a_i - v_i) % mod for a_i, v_i in zip(a, v))
            if required_c in cap_set_lookup:
                is_safe_to_add = False
                break
        
        if is_safe_to_add:
            cap_set.append(v)
            cap_set_lookup.add(v)
            
    return cap_set

def verify_set(cap_set, mod):
    """
    Exhaustively verifies no three distinct sum to 0 mod mod.
    """
    if not cap_set:
        return True
    
    print(f"Verifying set of size {len(cap_set)}...")
    start_time = time.time()
    n = len(cap_set[0])
    cap_set_lookup = set(cap_set)
    
    for i in range(len(cap_set)):
        for j in range(i + 1, len(cap_set)):
            a = cap_set[i]
            b = cap_set[j]
            c = tuple((-a_k - b_k) % mod for a_k, b_k in zip(a, b))
            
            if c != a and c != b and c in cap_set_lookup:
                print(f"Verification FAILED: Found {a}, {b}, {c}")
                return False
                
    end_time = time.time()
    print(f"Verification completed in {end_time - start_time:.2f} seconds.")
    return True

# --- Main Execution ---

if __name__ == "__main__":
    N_DIM = 5  # Change for other n
    MODULO = 4  # Change for other mods (e.g., 3 for standard caps)
    
    print(f"--- Starting Discovery for (Z/{MODULO}Z)^{N_DIM} ---")
    
    # Discover
    start_time = time.time()
    final_cap_set = solve_cap_set(N_DIM, MODULO, mixed_priority)
    end_time = time.time()
    
    size = len(final_cap_set)
    print(f"\n--- Discovery Complete ---")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print(f"Found set of size: {size}")
    
    # Verify
    is_valid = verify_set(final_cap_set, MODULO)
    
    print("\n--- Final Result ---")
    if is_valid:
        print(f"✅ Success! Valid set of size {size} (no three distinct sum to 0 mod {MODULO}).")
        print("Report this as a lower bound if larger than known!")
    else:
        print(f"❌ Failure! Invalid set.")
        
    # Sample output
    print("\nSample of 10 vectors:")
    for i in range(min(10, size)):
        print(final_cap_set[i])
    
    if size <= 50:
        print(f"\nComplete set ({size} vectors):")
        for i, v in enumerate(final_cap_set):
            print(f"{i+1:2d}: {v}")
            
    with open(f'n{N_DIM}_size{size}_set.json', 'w') as f:
        json.dump(final_cap_set, f)
    print(f"Full set saved to n{N_DIM}_size{size}_set.json")