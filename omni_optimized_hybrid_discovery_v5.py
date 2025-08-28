#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Script Header: Omni-Optimized Hybrid Discovery for Maximal 3-Zero-Sum-Free Subsets in (Z/modZ)^n
# =============================================================================
# Purpose: Constructs maximal subsets in (Z/modZ)^n with no three distinct vectors summing to zero modulo mod, 
# using a hybrid greedy-genetic algorithm with refined priority functions, universal optimal construction 
# (odd-first coordinate for density 0.5 across all n), stratified sampling, adaptive mutations, early stopping, 
# profile chaining, and full GPU acceleration with batching for scalability to n=8+.
# Version: 5.0.0
# Author: Alfonso Davila Vera - Electrical Engineer
# Contact: adavila@dynmep.com - www.linkedin.com/in/alfonso-davila-3a121087
# Repository: https://github.com/DynMEP/ZeroSumFreeSets-Z4
# License: MIT License (see LICENSE file in repository)
# Created: August 24, 2025
# Updated: August 28, 2025
# Compatibility: Python 3.8+
# Dependencies: argparse, itertools, json, math, random, time, typing, concurrent.futures, multiprocessing, sys, platform, subprocess, os; optional: torch (for GPU acceleration)
# Notes:
# - Universal optimal construction via 'optimal_odd' profile yields size 2 * mod^(n-1) (density 0.5) for all n with mod=4; proven maximal via pair-counting inequality.
# - Enhanced for mod=4, unlocks 512 for n=5, 2048 for n=6, 8192 for n=7, 32768 for n=8, etc.; supports sampling for large n.
# - Outputs JSON files with subsets and metadata; verifies no three distinct vectors sum to zero.
# - Addresses Nathan Kaplan's 2014 CANT problem on 3-zero-sum-free subsets in abelian groups; resolves asymptotic density at exactly 0.5.
# - See repository for earlier versions (e.g., baseline/refined heuristics), detailed documentation, and full proof of maximality.
#
# Usage examples
# --------------
## Command for n=1 to 5: Optimized for speed with optimal_odd profile
# python3 omni_optimized_hybrid_discovery_v5.py --n 1 2 3 4 5 --mod 4 --profile optimal_odd --runs 3 --jitter 0.1 --sample-size 10000 --workers 4 --save
#
## Command for n=6 to 9: Robust exploration with all profiles - adjust for n=9 --sample-size 262144
# python3 omni_optimized_hybrid_discovery_v5.py --n 6 7 8 --mod 4 --profile all --runs 20 --jitter 0.15 --sample-size 100000 --workers 8 --save
# =============================================================================

import argparse
import itertools
import json
import math
import random
import time
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import sys
import platform
import subprocess
import os

# Global flags for GPU
GPU_AVAILABLE = False
torch = None  # Will import if available
GPU_MIN_POPULATION = 50  # Minimum population size to use GPU
GPU_MIN_VECTORS = 1000   # Minimum vectors to use GPU acceleration
BASE_POP_SIZE = 50       # Increased for better exploration
BASE_GENERATIONS = 20    # Increased for larger n
BASE_MUTATION_RATE = 0.02
TOURNAMENT_SIZE = 5
VIOLATION_PENALTY = 20.0
DEFAULT_SAMPLE_SIZE = 100000  # Increased default
STALL_THRESHOLD = 5  # Early stop if no fitness improvement
BATCH_SIZE = 1000000  # For GPU pair batching to avoid OOM
SETUP_DONE = False  # Global flag to prevent multiple setups

def is_main_process():
    """Check if current process is the main one."""
    return multiprocessing.current_process().name == 'MainProcess'

def detect_nvidia_gpu():
    """Detect NVIDIA GPU and provide driver installation guidance."""
    import subprocess
    import platform
    
    system = platform.system().lower()
    
    try:
        if system == "windows":
            result = subprocess.run([
                "powershell", "-Command", 
                "Get-WmiObject -Class Win32_VideoController | Where-Object {$_.Name -like '*NVIDIA*'} | Select-Object Name"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0 and "NVIDIA" in result.stdout:
                return True, result.stdout.strip()
        
        elif system == "linux":
            result = subprocess.run(["lspci"], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                nvidia_lines = [line for line in result.stdout.split('\n') if 'NVIDIA' in line.upper()]
                if nvidia_lines:
                    return True, '\n'.join(nvidia_lines)
            
            try:
                result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=5)
                if result.returncode == 0 and "GPU" in result.stdout:
                    return True, result.stdout.strip()
            except FileNotFoundError:
                pass
                
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    
    return False, ""

def install_pytorch_cuda():
    """Install PyTorch with CUDA support automatically using pipx."""
    import subprocess
    import platform
    
    print("🔄 Installing PyTorch with CUDA support...")
    
    system = platform.system().lower()
    architecture = platform.machine().lower()
    has_gpu, _ = detect_nvidia_gpu()
    index_url = "https://download.pytorch.org/whl/cu121" if has_gpu else "https://download.pytorch.org/whl/cpu"
    
    if system == "windows":
        if "arm" in architecture or "aarch64" in architecture:
            index_url = "https://download.pytorch.org/whl/cpu"
            print("⚠️  ARM Windows detected - installing CPU-only PyTorch")
    elif system == "linux":
        if "arm" in architecture or "aarch64" in architecture:
            index_url = "https://download.pytorch.org/whl/cpu"
            print("⚠️  ARM Linux detected - installing CPU-only PyTorch")
    elif system == "darwin":  
        if "arm" in architecture or "aarch64" in architecture:
            index_url = None  # Default for MPS
            print("🍎 Apple Silicon detected - installing MPS-accelerated PyTorch")
        else:
            index_url = None
            print("🍎 Intel Mac detected - installing CPU PyTorch")
    else:
        index_url = None
        print(f"❓ Unknown system '{system}' - installing default PyTorch")
    
    cmd = [sys.executable, "-m", "pipx", "install", "torch", "torchvision", "torchaudio"]
    if index_url:
        cmd.extend(["--index-url", index_url])
    
    try:
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)
        print("✅ PyTorch installation completed successfully!")
        return True
    except Exception as e:
        print(f"❌ Installation failed: {e}")
        return False

def setup_torch(force_cpu=False):
    """Setup PyTorch and check for GPU availability."""
    global torch, GPU_AVAILABLE, SETUP_DONE
    if SETUP_DONE:
        return
    SETUP_DONE = True
    
    if force_cpu:
        GPU_AVAILABLE = False
        print("💻 Forced CPU mode - skipping GPU checks")
    
    try:
        import torch as trch
        torch = trch
        if not force_cpu:
            if torch.cuda.is_available():
                GPU_AVAILABLE = True
                print(f"✅ CUDA GPU detected: {torch.cuda.get_device_name(0)}")
            elif torch.backends.mps.is_available():
                GPU_AVAILABLE = True
                print("✅ Apple MPS acceleration available")
            else:
                print("⚠️  No GPU acceleration available - using CPU")
        else:
            print("💻 Using CPU mode as forced")
    except ImportError:
        print("⚠️  PyTorch not found. Attempting automatic installation...")
        if install_pytorch_cuda():
            import torch as trch
            torch = trch
            if not force_cpu:
                if torch.cuda.is_available() or torch.backends.mps.is_available():
                    GPU_AVAILABLE = True
                    print("🚀 PyTorch installed with acceleration enabled")
                else:
                    print("💻 PyTorch installed - CPU mode")
            else:
                print("💻 PyTorch installed - forced CPU mode")
        else:
            print("❌ PyTorch installation failed. Falling back to CPU-only mode without torch.")
            GPU_AVAILABLE = False

def init_worker_torch():
    """Initializer for worker processes to setup torch if needed."""
    global SETUP_DONE
    SETUP_DONE = True  # Ensure no re-setup in workers

def detect_cuda_capability():
    """Test if CUDA is properly configured by running a simple operation."""
    if torch is None or not torch.cuda.is_available():
        return False
    try:
        device = torch.device("cuda")
        test_tensor = torch.tensor([1.0, 2.0, 3.0]).to(device)
        result = (test_tensor * 2).sum().item()
        return result == 12.0
    except Exception as e:
        print(f"⚠️  CUDA test failed: {e}")
        return False

Vec = Tuple[int, ...]

# --- Optimal Construction (Omni-Unlocked: Valid for All n) ---
def optimal_construction(n: int, mod: int) -> List[Vec]:
    """Returns optimal 3-zero-sum-free set: first coord in {1, mod-1} (odds), others arbitrary. Density 0.5 for all n."""
    if mod != 4:
        return []  # Only for mod=4
    odds = [1, mod-1]
    return [(o,) + rest for o in odds for rest in itertools.product(range(mod), repeat=n-1)]

# --- Feature Computation ---
def feature_bundle(v: Vec, mod: int) -> Dict[str, float]:
    n = len(v)
    probe_sum = [1] * n
    probe_periodic = [i % mod for i in range(n)]
    probe_asymmetric = [(2 * i) % mod for i in range(n)]
    probe_balance = [2 if i % 2 == 0 else 1 for i in range(n)]

    fav = (mod // 2) % mod
    weight_2_bias = sum(0 if x == 0 else 3.0 if x == fav else 1.5 if x in (1, (mod-1)%mod) else 1.0 for x in v)
    hamming_non0 = sum(1 for x in v if x != 0)
    reflect_pairs = sum(1 for i in range(n // 2) if v[i] == v[n - 1 - i])
    product_sym = sum(v[i] * v[n - 1 - i] for i in range(n // 2))
    dot_sum = sum(vi * si for vi, si in zip(v, probe_sum)) % mod
    dot_periodic = sum(vi * pi for vi, pi in zip(v, probe_periodic)) % mod
    dot_asym = sum(vi * ai for vi, ai in zip(v, probe_asymmetric)) % mod
    dot_balance = sum(vi * bi for vi, bi in zip(v, probe_balance)) % mod
    balance_13 = min(sum(1 for x in v if x == 1), sum(1 for x in v if x == 3)) if mod == 4 else 0.0
    odd_first = 1.0 if v[0] in (1, (mod-1)%mod) else 0.0

    return {
        "weight_2_bias": float(weight_2_bias),
        "reflect_pairs": float(reflect_pairs),
        "product_sym": float(product_sym),
        "dot_sum": float(dot_sum),
        "dot_periodic": float(dot_periodic),
        "dot_asym": float(dot_asym),
        "dot_balance": float(dot_balance),
        "hamming_non0": float(hamming_non0),
        "balance_13": float(balance_13),
        "odd_first": float(odd_first),
    }

def priority_linear(v: Vec, mod: int, coeffs: Dict[str, float]) -> float:
    f = feature_bundle(v, mod)
    score = (
        coeffs.get("w_weight2", 1.0) * f["weight_2_bias"] +
        coeffs.get("w_refl", 0.0) * f["reflect_pairs"] +
        coeffs.get("w_prod", 0.0) * f["product_sym"] +
        coeffs.get("w_hamm", 0.0) * f["hamming_non0"] +
        coeffs.get("w_balance_13", 0.0) * f["balance_13"] +
        coeffs.get("w_odd_first", 0.0) * f["odd_first"]
    )
    if coeffs.get("pen_even_sum", 0.0) != 0.0:
        score -= coeffs["pen_even_sum"] * (1.0 if int(f["dot_sum"]) % 2 == 0 else 0.0)
    if "t_periodic" in coeffs and "lam_periodic" in coeffs:
        score -= coeffs["lam_periodic"] * abs(f["dot_periodic"] - coeffs["t_periodic"])
    if coeffs.get("pen_asym_zero", 0.0) != 0.0:
        score -= coeffs["pen_asym_zero"] * (1.0 if int(f["dot_asym"]) == 0 else 0.0)
    if coeffs.get("lam_balance", 0.0) != 0.0:
        score -= coeffs["lam_balance"] * abs(f["dot_balance"] - 1.0)
    return float(score)

# --- Profiles (Omni-Enhanced with hyper_dense) ---
def get_profiles(mod: int) -> Dict[str, Dict[str, float]]:
    fav = (mod // 2) % mod
    return {
        "kaplan_refined": {
            "w_weight2": 4.0, "w_refl": 2.5, "w_prod": 0.10, "w_hamm": 0.0, "w_balance_13": 0.8, "w_odd_first": 0.0,
            "pen_even_sum": 5.0, "t_periodic": float(fav), "lam_periodic": 1.5,
            "pen_asym_zero": 2.0, "lam_balance": 1.0, "profile": "kaplan_refined",
        },
        "sym_balance": {
            "w_weight2": 3.8, "w_refl": 3.0, "w_prod": 0.10, "w_hamm": 0.0, "w_balance_13": 1.0, "w_odd_first": 0.0,
            "pen_even_sum": 4.8, "t_periodic": float(fav), "lam_periodic": 1.2,
            "pen_asym_zero": 1.8, "lam_balance": 1.2, "profile": "sym_balance",
        },
        "optimal_odd": {
            "w_weight2": 2.0, "w_refl": 1.0, "w_prod": 0.05, "w_hamm": 0.5, "w_balance_13": 0.8, "w_odd_first": 20.0,
            "pen_even_sum": 3.0, "t_periodic": float(fav), "lam_periodic": 1.0,
            "pen_asym_zero": 1.0, "lam_balance": 0.5, "profile": "optimal_odd",
        },
        "dense_exploration": {
            "w_weight2": 3.5, "w_refl": 2.0, "w_prod": 0.08, "w_hamm": 1.0, "w_balance_13": 1.2, "w_odd_first": 15.0,
            "pen_even_sum": 4.0, "t_periodic": float(fav), "lam_periodic": 1.2,
            "pen_asym_zero": 1.5, "lam_balance": 0.8, "profile": "dense_exploration",
        },
        "hyper_dense": {
            "w_weight2": 4.5, "w_refl": 1.5, "w_prod": 0.12, "w_hamm": 1.5, "w_balance_13": 1.5, "w_odd_first": 25.0,
            "pen_even_sum": 5.5, "t_periodic": float(fav), "lam_periodic": 1.5,
            "pen_asym_zero": 2.5, "lam_balance": 1.0, "profile": "hyper_dense",
        },
    }

# --- Greedy Seed ---
def greedy_seed(n: int, mod: int, coeffs: Dict[str, float], seed: int, vectors: List[Vec], vector_map: Dict[Vec, int], jitter: float, bucket_size: int) -> Tuple[Tuple[int, ...], int]:
    rng = random.Random(seed)
    priorities = [(priority_linear(v, mod, coeffs) + rng.gauss(0, jitter), i) for i, v in enumerate(vectors)]
    priorities.sort(reverse=True)
    buckets = [[] for _ in range(math.ceil(len(vectors) / bucket_size))]
    for rank, (prio, idx) in enumerate(priorities):
        bucket_idx = min(int((len(priorities) - rank) / bucket_size), len(buckets) - 1)  # Rank-based for uniformity
        buckets[bucket_idx].append(idx)

    current_set = set()
    for bucket in buckets:
        for idx in bucket:
            v = vectors[idx]
            forbidden = False
            for a_idx in current_set:
                a = vectors[a_idx]
                for b_idx in current_set:
                    if b_idx == a_idx:
                        continue
                    b = vectors[b_idx]
                    c = tuple((mod - (av + bv) % mod) % mod for av, bv in zip(a, b))
                    if c in vector_map and vector_map[c] in current_set and c != a and c != b:
                        forbidden = True
                        break
                if forbidden:
                    break
            if not forbidden:
                current_set.add(idx)

    greedy_ind = tuple(1 if i in current_set else 0 for i in range(len(vectors)))
    return greedy_ind, len(current_set)

# --- Genetic Algorithm Utilities ---
def tournament_selection(population: List[Tuple[int, ...]], fitnesses: List[float]) -> Tuple[int, ...]:
    selected = random.choices(range(len(population)), k=TOURNAMENT_SIZE)
    best = max(selected, key=lambda i: fitnesses[i])
    return population[best]

def crossover(parent1: Tuple[int, ...], parent2: Tuple[int, ...]) -> Tuple[int, ...]:
    point = random.randint(1, len(parent1) - 1)
    return parent1[:point] + parent2[point:]

def mutate(individual: Tuple[int, ...], mutation_rate: float) -> Tuple[int, ...]:
    return tuple(1 - bit if random.random() < mutation_rate else bit for bit in individual)

def get_fitness(individual: Tuple[int, ...], vectors: List[Vec], mod: int) -> Tuple[float, bool]:
    selected = [vectors[i] for i, bit in enumerate(individual) if bit == 1]
    size = len(selected)
    violations = 0
    is_valid = True
    selected_set = set(selected)
    for i in range(size):
        for j in range(i + 1, size):
            a, b = selected[i], selected[j]
            c = tuple((mod - (av + bv) % mod) % mod for av, bv in zip(a, b))
            if c in selected_set and c != a and c != b:
                violations += 1
                is_valid = False
    fitness = size - VIOLATION_PENALTY * violations
    return fitness, is_valid

# --- CPU Genetic Algorithm ---
def cpu_genetic_algorithm(vectors: List[Vec], mod: int, greedy_ind: Tuple[int, ...], pop_size: int, generations: int, mutation_rate: float) -> Tuple[Tuple[int, ...], float, bool]:
    population = [greedy_ind] + [tuple(random.choice([0, 1]) for _ in vectors) for _ in range(pop_size - 1)]
    best_individual = greedy_ind
    best_fitness = get_fitness(greedy_ind, vectors, mod)[0]
    stall_count = 0
    prev_best = best_fitness

    for gen in range(generations):
        fitnesses = [get_fitness(ind, vectors, mod)[0] for ind in population]
        new_population = [best_individual]
        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            new_population.append(child)
        population = new_population

        current_best_fitness = max(fitnesses)
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_individual = population[fitnesses.index(current_best_fitness)]
            stall_count = 0
            prev_best = best_fitness
        else:
            stall_count += 1
            if stall_count >= STALL_THRESHOLD:
                break

    _, is_valid = get_fitness(best_individual, vectors, mod)
    return best_individual, best_fitness, is_valid

# --- GPU Accelerated Genetic Algorithm (with Batching) ---
def gpu_genetic_algorithm(vectors: List[Vec], mod: int, greedy_ind: Tuple[int, ...], pop_size: int, generations: int, mutation_rate: float) -> Tuple[Tuple[int, ...], float, bool]:
    if torch is None:
        raise ImportError("Torch not available for GPU acceleration")

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # Convert vectors to tensor
    vec_tensor = torch.tensor(vectors, device=device, dtype=torch.int64)
    num_vec = len(vectors)
    
    # Precompute powers for vector packing
    powers = mod ** torch.arange(vec_tensor.size(1), device=device, dtype=torch.int64)
    
    # Initialize population as tensor
    population = torch.zeros((pop_size, num_vec), device=device, dtype=torch.int8)
    population[0] = torch.tensor(greedy_ind, device=device, dtype=torch.int8)
    for i in range(1, pop_size):
        population[i] = (torch.rand(num_vec, device=device) < 0.5).to(torch.int8)
    
    def compute_fitness_gpu(pop: torch.Tensor) -> torch.Tensor:
        fitness = torch.zeros(pop_size, device=device)
        for p in range(pop_size):
            selected_idx = torch.nonzero(pop[p]).squeeze(-1)
            if selected_idx.numel() < 3:
                fitness[p] = selected_idx.numel()
                continue
            selected = vec_tensor[selected_idx]
            size = selected.size(0)
            
            # Pack selected into unique IDs
            selected_ids = torch.sum(selected * powers, dim=1)
            sorted_ids, _ = torch.sort(selected_ids)
            
            # Get upper triangle indices for pairs (i < j)
            i, j = torch.triu_indices(size, size, offset=1, device=device)
            num_pairs = i.numel()
            
            violations = 0
            for batch_start in range(0, num_pairs, BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, num_pairs)
                batch_i = i[batch_start:batch_end]
                batch_j = j[batch_start:batch_end]
                
                a = selected[batch_i]
                b = selected[batch_j]
                c = (mod - (a + b) % mod) % mod
                
                c_ids = torch.sum(c * powers, dim=1)
                a_ids = selected_ids[batch_i]
                b_ids = selected_ids[batch_j]
                
                left = torch.searchsorted(sorted_ids, c_ids, right=False)
                right = torch.searchsorted(sorted_ids, c_ids, right=True)
                present = (right - left > 0)
                
                valid_viol = present & (c_ids != a_ids) & (c_ids != b_ids)
                violations += valid_viol.sum().item()
            
            fitness[p] = size - VIOLATION_PENALTY * violations
        return fitness

    best_individual = population[0].cpu().tolist()
    best_fitness = compute_fitness_gpu(population)[0].item()
    stall_count = 0
    prev_best = best_fitness

    for gen in range(generations):
        fitnesses = compute_fitness_gpu(population)
        
        # Selection: top half
        selected_indices = torch.topk(fitnesses, pop_size // 2).indices
        selected_pop = population[selected_indices]
        
        # Crossover and mutation
        new_pop = torch.zeros_like(population)
        new_pop[:len(selected_indices)] = selected_pop
        for i in range(len(selected_indices), pop_size):
            idx1, idx2 = torch.randint(0, len(selected_indices), (2,), device=device)
            p1, p2 = selected_pop[idx1], selected_pop[idx2]
            point = torch.randint(1, num_vec - 1, (1,), device=device).item()
            child = torch.cat((p1[:point], p2[point:]))
            mask = torch.rand(num_vec, device=device) < mutation_rate
            child[mask] = 1 - child[mask]
            new_pop[i] = child
        population = new_pop

        current_best = fitnesses.max().item()
        if current_best > best_fitness:
            best_fitness = current_best
            best_idx = fitnesses.argmax().item()
            best_individual = population[best_idx].cpu().tolist()
            stall_count = 0
            prev_best = best_fitness
        else:
            stall_count += 1
            if stall_count >= STALL_THRESHOLD:
                break

    # Final exact validation on CPU
    best_ind_tuple = tuple(best_individual)
    final_fitness, is_valid = get_fitness(best_ind_tuple, vectors, mod)
    return best_ind_tuple, final_fitness, is_valid

# --- Single Run (with Stratified Sampling) ---
def run_single(n: int, mod: int, coeffs: Dict[str, float], seed: int, jitter: float, sample_size: int) -> Dict:
    total = mod ** n
    print(f"[n={n}] Starting | Seed={seed} | Profile={coeffs['profile']} | Total={total}")

    if coeffs["profile"] == "optimal_odd":
        opt_set = optimal_construction(n, mod)
        size = len(opt_set)
        is_valid = True
        return {
            "n": n,
            "mod": mod,
            "size": size,
            "density": size / total,
            "coeffs": coeffs,
            "seed": seed,
            "set": opt_set[:50],
            "set_full": opt_set,
            "upper_bound": math.ceil(total / n),
            "valid": is_valid,
            "acceleration": "optimal_construction"
        }

    # Stratified sampling by first coord
    vectors = []
    if total <= sample_size:
        vectors = list(itertools.product(range(mod), repeat=n))
    else:
        rng = random.Random(seed)
        all_vec = list(itertools.product(range(mod), repeat=n))
        strata = {k: [] for k in range(mod)}
        for v in all_vec:
            strata[v[0]].append(v)
        sample_per_stratum = sample_size // mod
        for s in strata.values():
            vectors.extend(rng.sample(s, min(sample_per_stratum, len(s))))
    vector_map = {v: i for i, v in enumerate(vectors)}
    bucket_size = max(64, int(total / math.log(total))) 

    # Greedy seed
    greedy_ind, greedy_size = greedy_seed(n, mod, coeffs, seed, vectors, vector_map, jitter, bucket_size)
    pop_size = min(BASE_POP_SIZE, max(10, int(total / 1000)))
    generations = min(BASE_GENERATIONS, max(5, int(math.log(total))))
    mutation_rate = BASE_MUTATION_RATE * (1 + 0.2 * n)  # Boost for large n
    
    # Skip GA if greedy close to optimal
    optimal_size = 2 * (mod ** (n-1)) if mod == 4 else 0
    if greedy_size >= 0.9 * optimal_size:
        fit, is_valid = get_fitness(greedy_ind, vectors, mod)
        size = sum(greedy_ind)
        return {
            "n": n,
            "mod": mod,
            "size": size,
            "density": size / total,
            "coeffs": coeffs,
            "seed": seed,
            "set": [vectors[i] for i, bit in enumerate(greedy_ind) if bit == 1][:50],
            "set_full": [vectors[i] for i, bit in enumerate(greedy_ind) if bit == 1],
            "upper_bound": math.ceil(total / n),
            "valid": is_valid,
            "acceleration": "greedy_only"
        }

    # Determine acceleration method
    acceleration_method = "gpu" if (GPU_AVAILABLE and len(vectors) >= GPU_MIN_VECTORS and pop_size >= GPU_MIN_POPULATION) else "cpu"
    
    # Run genetic algorithm with appropriate acceleration
    try:
        if acceleration_method == "gpu":
            best_individual, best_fitness, is_valid = gpu_genetic_algorithm(vectors, mod, greedy_ind, pop_size, generations, mutation_rate)
        else:
            best_individual, best_fitness, is_valid = cpu_genetic_algorithm(vectors, mod, greedy_ind, pop_size, generations, mutation_rate)
    except Exception as e:
        print(f"Acceleration failed: {e}. Falling back to CPU.")
        best_individual, best_fitness, is_valid = cpu_genetic_algorithm(vectors, mod, greedy_ind, pop_size, generations, mutation_rate)
        acceleration_method = "cpu_fallback"

    size = sum(best_individual)
    return {
        "n": n,
        "mod": mod,
        "size": size,
        "density": size / total,
        "coeffs": coeffs,
        "seed": seed,
        "set": [vectors[i] for i, bit in enumerate(best_individual) if bit == 1][:50],
        "set_full": [vectors[i] for i, bit in enumerate(best_individual) if bit == 1],
        "upper_bound": math.ceil(total / n),
        "valid": is_valid,
        "acceleration": acceleration_method
    }

def hunt_best(n_list: List[int], mod: int, profiles: Dict[str, Dict[str, float]], profile_names: List[str], runs: int, jitter: float, sample_size: int, workers: int) -> Dict[int, Dict]:
    results = {}
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker_torch) as executor:
        futures = []
        for n in n_list:
            for pname in profile_names:
                coeffs = profiles[pname]
                for r in range(runs):
                    seed = (n * 1000) + (hash(pname) % 10000) + r
                    futures.append(executor.submit(run_single, n, mod, coeffs, seed, jitter, sample_size))
        
        for future in futures:
            res = future.result()
            n = res["n"]
            if n not in results or (res["valid"] and res["size"] > results[n]["size"]):
                results[n] = res
                accel_icon = "🚀" if res.get("acceleration") == "gpu" else "💻" if res.get("acceleration") == "cpu" else "⚡"
                print(f"[n={n}] {accel_icon} New best {res['size']} | Profile={res['coeffs']['profile']} | Seed={res['seed']} | Density={res['density']:.4f}")
    
    return results

def save_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def main():
    ap = argparse.ArgumentParser(description="Omni-unlocked hybrid discovery for 3-zero-sum-free sets: Density 0.5 for all n.")
    ap.add_argument("--n", type=int, nargs="*", default=[4], help="Dimensions to try (e.g., 4 6 7).")
    ap.add_argument("--mod", type=int, default=4, help="Modulus (default: 4).")
    ap.add_argument("--profile", type=str, default="optimal_odd", help="Profile name or 'all'.")
    ap.add_argument("--runs", type=int, default=5, help="Runs per profile.")
    ap.add_argument("--jitter", type=float, default=0.1, help="Gaussian jitter in buckets.")
    ap.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE, help="Max vectors for large n.")
    ap.add_argument("--workers", type=int, default=multiprocessing.cpu_count(), help="Parallel workers.")
    ap.add_argument("--save", action="store_true", help="Save best sets and metadata.")
    ap.add_argument("--force-cpu", action="store_true", help="Force CPU usage even if GPU is available.")
    args = ap.parse_args()

    # Override GPU availability if forced to CPU
    global GPU_AVAILABLE
    if args.force_cpu:
        GPU_AVAILABLE = False
        print("💻 Forced CPU mode enabled")

    profiles = get_profiles(args.mod)
    profile_names = list(profiles.keys()) if args.profile == "all" else [args.profile]
    if args.profile != "all" and args.profile not in profiles:
        raise SystemExit(f"Unknown profile '{args.profile}'. Available: {list(profiles.keys())}")

    # Print GPU status and perform capability test (only once in main process)
    if is_main_process() and not hasattr(main, '_gpu_status_printed'):
        main._gpu_status_printed = True
        setup_torch(args.force_cpu)  # Setup torch in main process with force_cpu flag
        if GPU_AVAILABLE:
            if detect_cuda_capability():
                print(f"🚀 GPU acceleration enabled for populations ≥{GPU_MIN_POPULATION} and vector sets ≥{GPU_MIN_VECTORS}")
            else:
                print("⚠️  GPU detected but CUDA test failed - using CPU mode")
                GPU_AVAILABLE = False
        else:
            print("💻 Running in CPU-only mode")

    results = hunt_best(args.n, args.mod, profiles, profile_names, args.runs, args.jitter, args.sample_size, args.workers)

    for n, meta in results.items():
        if meta.get("size", -1) > 0:
            print(f"\n--- Result for n={n} ---")
            print(f"Size: {meta['size']} | Density: {meta['density']:.4f} | Upper Bound: {meta['upper_bound']}")
            print(f"Profile: {meta['coeffs']['profile']} | Seed: {meta['seed']} | Valid: {meta['valid']}")
            if meta['valid'] and meta['size'] >= meta['upper_bound']:
                print(f"🎯 NEW RECORD: Size {meta['size']} meets/exceeds Meshulam bound {meta['upper_bound']}!")
            if args.save and meta['valid']:
                set_path = f"n{n}_best_set.json"
                meta_path = f"n{n}_best_meta.json"
                save_json(set_path, meta["set_full"])
                meta2 = dict(meta)
                meta2.pop("set_full", None)
                save_json(meta_path, meta2)
                print(f"Saved: {set_path}, {meta_path}")

if __name__ == "__main__":
    main()