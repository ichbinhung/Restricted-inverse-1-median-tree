# Restricted Inverse Optimal Median Objective Problem on Trees
This repository contains the official Python implementation and benchmarking source code for the algorithms proposed in the paper:
**"Restricted Inverse Optimal Median Objective Problem on Trees under Chebyshev Norm and Hamming Distance"** (Nguyen et al., 2026).

## Requirements
The code is written in Python 3.10+ and requires the following packages:
- `networkx` (for tree topology generation)
- `numpy` & `pandas` (for vectorized bounding and data processing)
- `matplotlib` (for plotting Log-Log charts)
- `gurobipy` (Gurobi Optimizer for baseline LP/MILP comparison)

*Note: You need a valid Gurobi license (academic licenses are free) to run the baseline solvers.*

## How to Run
Simply execute the main script. The script will automatically verify dependencies, generate various tree topologies (Random, Path, Binary), and run the benchmarking process.

## bash
python riomo_benchmark.py

## Outputs
The script will output:
1. Two comprehensive console tables containing the execution times, standard deviations, and objective costs.
2. `plot_chebyshev.png`: Log-log plot comparing the proposed $O(n \log n)$ algorithm vs Gurobi LP.
3. `plot_hamming.png`: Log-log plot comparing the proposed algorithm vs Gurobi MILP.
