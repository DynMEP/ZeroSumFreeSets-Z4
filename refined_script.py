# =============================================================================
# Script Header: Refined 3-Zero-Sum-Free Subset Construction for (Z/modZ)^n
# =============================================================================
# Purpose: Constructs optimized subsets in (Z/modZ)^n with no three distinct vectors 
# summing to zero modulo mod, using a greedy algorithm with a refined priority function 
# enhancing favor for '2's, penalizing zeros, and incorporating a balance probe.
# Version: 1.1.0
# Author: Alfonso Davila Vera - Electrical Engineer
# Contact: adavila@dynmep.com - www.linkedin.com/in/alfonso-davila-3a121087
# Repository: https://github.com/DynMEP/ZeroSumFreeSets-Z4
# License: MIT License (see LICENSE file in repository)
# Created: August 24, 2025
# Compatibility: Python 3.8+
# Dependencies: itertools, time, json
# Notes:
# - Optimized for mod=4, n=5 (yields size 512); adjust N_DIM and MODULO for n=6,7.
# - Outputs JSON file with the subset and verifies no three distinct vectors sum to zero.
# - Addresses Nathan Kaplan's 2014 CANT problem on 3-zero-sum-free subsets in abelian groups.
# - See repository for baseline version and detailed documentation.
# =============================================================================

import itertools
import time
import json

def refined_priority(v, n, mod):
    """
    Refined version: Increase favor for '2's, add weight penalty for too many zeros, 
    adjust dots to penalize 0 more strongly, add a new probe for balance.
    """
    # Probes
    probe_sum = tuple(1 for _ in range(n))
    probe_periodic = tuple(i % mod for i in range(n))  # Use mod for periodicity
    probe_asymmetric = tuple((2 * i % mod) for i in range(n))  # Adjusted
    probe_balance = tuple(2 if i % 2 == 0 else 1 for i in range(n))  # New probe for alternation

    # 1. Weight: Strongly favor '2', moderate for 1/3, penalize 0 more
    weight = sum((0.5 if x == 0 else 1.5 if x in [1, mod-1] else 3.0 if x == 2 else 0) for x in v)
    
    # 2. Symmetry
    reflection_score = sum(1 for i in range(n // 2) if v[i] == v[n - 1 - i]) * 2.5  # Boost
    
    # 3. Product Sum
    product_sum_val = sum(v[i] * v[n - 1 - i] for i in range(n // 2)) * 0.1
    
    # 4. Dots
    dot_sum = sum(v[i] * probe_sum[i] for i in range(n)) % mod
    dot_periodic = sum(v[i] * probe_periodic[i] for i in range(n)) % mod
    dot_asym = sum(v[i] * probe_asymmetric[i] for i in range(n)) % mod
    dot_balance = sum(v[i] * probe_balance[i] for i in range(n)) % mod
    
    # Coefficients: Tune to reward odd sums, close to certain values
    priority = (
        weight +
        reflection_score +
        product_sum_val -
        5.0 * (1 if dot_sum % 2 == 0 else 0) -  # Stronger penalize even
        1.5 * abs(dot_periodic - 2) -  # Reward close to 2 instead
        2.0 * (1 if dot_asym == 0 else 0) -
        1.0 * abs(dot_balance - 1)  # New term
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
    N_DIM = 5  # Change to 6 or 7 for higher; mod=4 for Z/4Z
    MODULO = 4
    
    print(f"--- Starting Discovery for (Z/{MODULO}Z)^{N_DIM} ---")
    
    # Discover
    start_time = time.time()
    final_cap_set = solve_cap_set(N_DIM, MODULO, refined_priority)
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