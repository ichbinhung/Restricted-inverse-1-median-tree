import sys
import subprocess
import os
import gc
import networkx as nx
import gurobipy as gp
import time
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Force pandas to display floats cleanly instead of scientific notation
pd.options.display.float_format = '{:.4f}'.format

# ==========================================
# 1. Helper Functions & Algorithm Logic
# ==========================================

def get_subtrees(T, s):
    """Returns subtrees components created if node s is removed."""
    subtrees = {}
    for neighbor in T.neighbors(s):
        edges = [e for e in T.edges if s not in e]
        T_sub = nx.Graph(edges)
        if neighbor in T_sub:
            subtrees[neighbor] = list(nx.node_connected_component(T_sub, neighbor))
        else:
            subtrees[neighbor] = [neighbor]
    return subtrees

def fast_calculate_bounds(lb, ub, d_arr, subtree_masks, sorted_subtrees, all_sorted):
    """Highly optimized, vectorized NumPy bounds calculator (O(n) per call)."""
    # --- Gamma Max ---
    W_total_max = np.sum(ub)
    gamma_max = np.dot(ub, d_arr)
    
    for idx, mask in enumerate(subtree_masks):
        W_u = np.sum(ub[mask])
        if W_u > W_total_max * 0.5 + 1e-7:
            delta = 2 * W_u - W_total_max
            for i in sorted_subtrees[idx]:
                available = ub[i] - lb[i]
                reduction = available if available < delta else delta
                gamma_max -= reduction * d_arr[i]
                delta -= reduction
                if delta <= 1e-7: break
            if delta > 1e-5:
                return float('inf'), float('-inf')
            break
            
    # --- Gamma Min ---
    W_total_min = np.sum(lb)
    gamma_min = np.dot(lb, d_arr)
    
    for idx, mask in enumerate(subtree_masks):
        W_u = np.sum(lb[mask])
        if W_u > W_total_min * 0.5 + 1e-7:
            delta = 2 * W_u - W_total_min
            for i in all_sorted:
                if not mask[i]:
                    available = ub[i] - lb[i]
                    increase = available if available < delta else delta
                    gamma_min += increase * d_arr[i]
                    delta -= increase
                    if delta <= 1e-7: break
            if delta > 1e-5:
                return float('inf'), float('-inf')
            break
            
    return gamma_min, gamma_max

# ==========================================
# 2. Chebyshev Norm Algorithms
# ==========================================

def solve_chebyshev_proposed(V, d, w, c, x_bar, y_bar, s, subtrees, gamma, tol=1e-7):
    """Exact O(n log n) Algorithm using Discrete Binary Search & Linear Interpolation."""
    V_len = len(V)
    d_arr = np.array([d[v] for v in range(V_len)], dtype=np.float64)
    w_arr = np.array([w[v] for v in range(V_len)], dtype=np.float64)
    c_arr = np.array([c[v] for v in range(V_len)], dtype=np.float64)
    x_bar_arr = np.array([x_bar[v] for v in range(V_len)], dtype=np.float64)
    y_bar_arr = np.array([y_bar[v] for v in range(V_len)], dtype=np.float64)
    
    subtree_masks = []
    for comp in subtrees.values():
        mask = np.zeros(V_len, dtype=bool)
        mask[comp] = True
        subtree_masks.append(mask)
        
    all_sorted = np.argsort(d_arr).tolist()
    sorted_subtrees = []
    for mask in subtree_masks:
        comp_indices = np.where(mask)[0]
        sorted_subtrees.append(comp_indices[np.argsort(d_arr[comp_indices])].tolist())
        
    def evaluate_z(z_val):
        M_over_c = z_val / c_arr
        lb = np.maximum(0.0, w_arr - np.minimum(y_bar_arr, M_over_c))
        ub = w_arr + np.minimum(x_bar_arr, M_over_c)
        return fast_calculate_bounds(lb, ub, d_arr, subtree_masks, sorted_subtrees, all_sorted)

    b_array = np.concatenate(([0.0], c_arr * x_bar_arr, c_arr * y_bar_arr))
    b_array = np.unique(b_array)
    b_array = np.sort(b_array)
    
    L_idx, R_idx = 0, len(b_array) - 1
    best_z = b_array[-1]
    
    while R_idx - L_idx > 1:
        M_idx = (L_idx + R_idx) // 2
        M_val = b_array[M_idx]
        g_min, g_max = evaluate_z(M_val)
        
        if g_min - tol <= gamma <= g_max + tol:
            R_idx = M_idx
            best_z = M_val
        else:
            L_idx = M_idx
            
    z_L = b_array[L_idx]
    z_R = b_array[R_idx]
    
    g_min_L, g_max_L = evaluate_z(z_L)
    g_min_R, g_max_R = evaluate_z(z_R)
    
    if gamma > g_max_L:
        if (g_max_R - g_max_L) > 1e-12:
            best_z = z_L + (gamma - g_max_L) * (z_R - z_L) / (g_max_R - g_max_L)
        else:
            best_z = z_R
    elif gamma < g_min_L:
        if (g_min_L - g_min_R) > 1e-12:
            best_z = z_L + (g_min_L - gamma) * (z_R - z_L) / (g_min_L - g_min_R)
        else:
            best_z = z_R
    else:
        best_z = z_L
        
    g_min_eval, g_max_eval = evaluate_z(best_z)
    if not (g_min_eval - tol <= gamma <= g_max_eval + tol):
        ref_L, ref_R = z_L, z_R
        while ref_R - ref_L > tol:
            M = (ref_L + ref_R) / 2.0
            g_min, g_max = evaluate_z(M)
            if g_min - tol <= gamma <= g_max + tol:
                best_z = M
                ref_R = M
            else:
                ref_L = M
                
    return best_z

def solve_chebyshev_lp(V, d, w, c, x_bar, y_bar, s, subtrees, gamma):
    """Standard Linear Programming formulation solved via Gurobi."""
    try:
        model = gp.Model("RIOMO_Chebyshev")
        model.setParam('OutputFlag', 0)
        model.setParam('TimeLimit', 60)
        
        x = model.addVars(V, lb=0, ub=x_bar, name="x")
        y = model.addVars(V, lb=0, ub=y_bar, name="y")
        z = model.addVar(lb=0, name="z")
        
        model.setObjective(z, gp.GRB.MINIMIZE)
        
        for v in V:
            model.addConstr(c[v] * x[v] <= z)
            model.addConstr(c[v] * y[v] <= z)
            model.addConstr(w[v] + x[v] - y[v] >= 0)
            
        obj_expr = gp.quicksum((w[v] + x[v] - y[v]) * d[v] for v in V)
        model.addConstr(obj_expr == gamma)
        
        W_total = gp.quicksum(w[v] + x[v] - y[v] for v in V)
        for u in subtrees:
            W_u = gp.quicksum(w[v] + x[v] - y[v] for v in subtrees[u])
            model.addConstr(W_u <= 0.5 * W_total)
            
        model.optimize()
        
        if model.status == gp.GRB.OPTIMAL:
            return model.objVal
        return None
    except gp.GurobiError as e:
        if "size-limited license" in str(e):
            return None
        raise e

# ==========================================
# 3. Bottleneck Hamming Distance Algorithms
# ==========================================

def solve_hamming_proposed(V, d, w, c, x_bar, y_bar, s, subtrees, gamma):
    """Optimized Vectorized O(n log n) algorithm for Bottleneck Hamming Distance."""
    V_len = len(V)
    d_arr = np.array([d[v] for v in range(V_len)], dtype=np.float64)
    w_arr = np.array([w[v] for v in range(V_len)], dtype=np.float64)
    c_arr = np.array([c[v] for v in range(V_len)], dtype=np.float64)
    x_bar_arr = np.array([x_bar[v] for v in range(V_len)], dtype=np.float64)
    y_bar_arr = np.array([y_bar[v] for v in range(V_len)], dtype=np.float64)
    
    subtree_masks = []
    for comp in subtrees.values():
        mask = np.zeros(V_len, dtype=bool)
        mask[comp] = True
        subtree_masks.append(mask)
        
    all_sorted = np.argsort(d_arr).tolist()
    sorted_subtrees = []
    for mask in subtree_masks:
        comp_indices = np.where(mask)[0]
        sorted_subtrees.append(comp_indices[np.argsort(d_arr[comp_indices])].tolist())
        
    unique_costs = np.unique(np.append(c_arr, 0.0))
    L, R = 0, len(unique_costs) - 1
    best_z = unique_costs[-1]
    
    while L <= R:
        mid = (L + R) // 2
        M = unique_costs[mid]
        
        allow_change = c_arr <= M
        lb = np.where(allow_change, np.maximum(0.0, w_arr - y_bar_arr), w_arr)
        ub = np.where(allow_change, w_arr + x_bar_arr, w_arr)
        
        g_min, g_max = fast_calculate_bounds(lb, ub, d_arr, subtree_masks, sorted_subtrees, all_sorted)
        
        if g_min - 1e-5 <= gamma <= g_max + 1e-5:
            best_z = M
            R = mid - 1
        else:
            L = mid + 1
            
    return best_z

def solve_hamming_milp(V, d, w, c, x_bar, y_bar, s, subtrees, gamma):
    """Mixed Integer Linear Programming formulation solved via Gurobi."""
    try:
        model = gp.Model("RIOMO_Hamming")
        model.setParam('OutputFlag', 0)
        model.setParam('TimeLimit', 60)
        
        x = model.addVars(V, lb=0, ub=x_bar, name="x")
        y = model.addVars(V, lb=0, ub=y_bar, name="y")
        Ix = model.addVars(V, vtype=gp.GRB.BINARY, name="Ix")
        Iy = model.addVars(V, vtype=gp.GRB.BINARY, name="Iy")
        z = model.addVar(lb=0, name="z")
        
        model.setObjective(z, gp.GRB.MINIMIZE)
        
        for v in V:
            model.addConstr(x[v] <= x_bar[v] * Ix[v])
            model.addConstr(y[v] <= y_bar[v] * Iy[v])
            model.addConstr(c[v] * Ix[v] <= z)
            model.addConstr(c[v] * Iy[v] <= z)
            model.addConstr(w[v] + x[v] - y[v] >= 0)
            
        obj_expr = gp.quicksum((w[v] + x[v] - y[v]) * d[v] for v in V)
        model.addConstr(obj_expr == gamma)
        
        W_total = gp.quicksum(w[v] + x[v] - y[v] for v in V)
        for u in subtrees:
            W_u = gp.quicksum(w[v] + x[v] - y[v] for v in subtrees[u])
            model.addConstr(W_u <= 0.5 * W_total)
            
        model.optimize()
        
        if model.status == gp.GRB.OPTIMAL:
            return model.objVal
        return None
    except gp.GurobiError as e:
        if "size-limited license" in str(e):
            return None
        raise e

# ==========================================
# 4. Topology Generation & Plotting
# ==========================================

def generate_star_graph(n):
    """Generate a star graph: one central node connected to all others."""
    return nx.star_graph(n - 1)

def generate_spider_graph(n):
    """Generate a spider graph: central hub with multiple legs (paths) extending from it."""
    num_legs = max(2, int(np.sqrt(n)))  # ~sqrt(n) legs
    leg_length = (n - 1) // num_legs  # Distribute remaining nodes among legs
    remainder = (n - 1) % num_legs
    
    T = nx.Graph()
    T.add_node(0)  # Central hub at node 0
    
    node_id = 1
    for leg_idx in range(num_legs):
        current_length = leg_length + (1 if leg_idx < remainder else 0)
        prev_node = 0
        for _ in range(current_length):
            T.add_edge(prev_node, node_id)
            prev_node = node_id
            node_id += 1
    
    return T

def generate_topology(n, topology="random"):
    """Generate tree graph topology (random, star, or spider)."""
    if topology == "star":
        return generate_star_graph(n)
    elif topology == "spider":
        return generate_spider_graph(n)
    else:  # "random"
        return nx.random_labeled_tree(n)

def generate_feasible_instance(n, topology="random"):
    """Generate a feasible instance with the specified topology."""
    while True:
        T = generate_topology(n, topology)
        V = list(T.nodes)
        s = random.choice(V)
        
        d = nx.single_source_shortest_path_length(T, s)
        subtrees = get_subtrees(T, s)
        # Extract topological degrees to simulate hierarchical real-world networks (e.g., hub-and-spoke models)
        degrees = dict(T.degree())
        max_degree = max(degrees.values()) if degrees else 1
        
        w = {}
        c = {}
        x_bar = {}
        y_bar = {}
        
        for v in V:
            # 1. GENERATE INITIAL WEIGHTS (w_v)
            # Rationale: Node demand/population is positively correlated with its topological degree.
            # Hubs (high degree) are assigned substantially larger base weights than leaf nodes.
            # A 20% uniform noise is injected to ensure heterogeneity.
            base_w = 10.0 + (degrees[v] / max_degree) * 1000.0
            w[v] = base_w * random.uniform(0.8, 1.2) 
            
            # 2. GENERATE MODIFICATION COSTS (c_v)
            # Rationale: Direct proportionality between modification cost (c_v) and initial weight (w_v).
            # This reflects the economic reality that altering parameters in high-capacity hubs 
            # requires significantly greater capital expenditure compared to minor peripheral nodes.
            cost_scaling_factor = random.uniform(0.05, 0.15) # Scaling parameter representing unit cost
            c[v] = w[v] * cost_scaling_factor
            
            # Ensure strict positivity to prevent zero-division in Chebyshev bounds calculation
            c[v] = max(0.1, c[v]) 
            
            # 3. GENERATE MODIFICATION BOUNDS (x_bar_v, y_bar_v)
            # Rationale: Upper bounds are formulated as fractional proportions of the initial weights.
            # This constraint preserves the underlying topology's weight distribution and fundamentally
            # prevents mathematically degenerate cases (e.g., reaching negative perturbed weights).
            
            # Maximum allowed expansion capacity is constrained strictly between 20% and 60% of current scale
            x_bar[v] = w[v] * random.uniform(0.2, 0.6) 
            
            # Maximum allowed reduction capacity is constrained between 10% and 50%
            # (Ensures w_v - y_bar_v >= 0.5 * w_v > 0, strictly maintaining positive node demands)
            y_bar[v] = w[v] * random.uniform(0.1, 0.5)
        V_len = len(V)
        d_arr = np.array([d[v] for v in range(V_len)], dtype=np.float64)
        w_arr = np.array([w[v] for v in range(V_len)], dtype=np.float64)
        x_bar_arr = np.array([x_bar[v] for v in range(V_len)], dtype=np.float64)
        y_bar_arr = np.array([y_bar[v] for v in range(V_len)], dtype=np.float64)
        
        lb = np.maximum(0.0, w_arr - y_bar_arr)
        ub = w_arr + x_bar_arr
        
        subtree_masks = []
        for comp in subtrees.values():
            mask = np.zeros(V_len, dtype=bool)
            mask[comp] = True
            subtree_masks.append(mask)
            
        all_sorted = np.argsort(d_arr).tolist()
        sorted_subtrees = []
        for mask in subtree_masks:
            comp_indices = np.where(mask)[0]
            sorted_subtrees.append(comp_indices[np.argsort(d_arr[comp_indices])].tolist())
            
        g_min, g_max = fast_calculate_bounds(lb, ub, d_arr, subtree_masks, sorted_subtrees, all_sorted)
        
        if g_min != float('inf') and g_max != float('-inf') and (g_min < g_max - 1e-3):
            gamma = random.uniform(g_min + 0.2*(g_max-g_min), g_max - 0.2*(g_max-g_min))
            return V, d, w, c, x_bar, y_bar, s, subtrees, gamma

def generate_tree_and_bounds(n, topology="random"):
    """Generate a graph and return both gamma_min and gamma_max instead of a single random gamma value."""
    while True:
        T = generate_topology(n, topology)
        V = list(T.nodes)
        s = random.choice(V)
        
        d = nx.single_source_shortest_path_length(T, s)
        subtrees = get_subtrees(T, s)
        
        degrees = dict(T.degree())
        max_degree = max(degrees.values()) if degrees else 1
        
        w, c, x_bar, y_bar = {}, {}, {}, {}
        for v in V:
            # 1. GENERATE INITIAL WEIGHTS (w_v)
            base_w = 10.0 + (degrees[v] / max_degree) * 1000.0
            w[v] = base_w * random.uniform(0.8, 1.2) 
            
            # 2. GENERATE MODIFICATION COSTS (c_v)
            cost_scaling_factor = random.uniform(0.05, 0.15) 
            c[v] = max(0.1, w[v] * cost_scaling_factor) 
            
            # 3. GENERATE MODIFICATION BOUNDS (x_bar_v, y_bar_v)
            x_bar[v] = w[v] * random.uniform(0.2, 0.6) 
            y_bar[v] = w[v] * random.uniform(0.1, 0.5)
            
        V_len = len(V)
        d_arr = np.array([d[v] for v in range(V_len)], dtype=np.float64)
        w_arr = np.array([w[v] for v in range(V_len)], dtype=np.float64)
        x_bar_arr = np.array([x_bar[v] for v in range(V_len)], dtype=np.float64)
        y_bar_arr = np.array([y_bar[v] for v in range(V_len)], dtype=np.float64)
        
        lb = np.maximum(0.0, w_arr - y_bar_arr)
        ub = w_arr + x_bar_arr
        
        subtree_masks =[]
        for comp in subtrees.values():
            mask = np.zeros(V_len, dtype=bool)
            mask[comp] = True
            subtree_masks.append(mask)
            
        all_sorted = np.argsort(d_arr).tolist()
        sorted_subtrees = []
        for mask in subtree_masks:
            comp_indices = np.where(mask)[0]
            sorted_subtrees.append(comp_indices[np.argsort(d_arr[comp_indices])].tolist())
            
        g_min, g_max = fast_calculate_bounds(lb, ub, d_arr, subtree_masks, sorted_subtrees, all_sorted)
        
        if g_min != float('inf') and g_max != float('-inf') and (g_min < g_max - 1e-3):
            return V, d, w, c, x_bar, y_bar, s, subtrees, g_min, g_max

def plot_loglog_results(df, title, solver_name, filename):
    """Plot Log-Log graphs comparing execution speeds."""
    plt.figure(figsize=(10, 6))
    
    # Filter out Gurobi rows with errors/timeouts (if any)
    valid_df = df[df[f'{solver_name} Mean (s)'] > 0]
    sizes = valid_df['Size (n)']
    
    # Plot proposed algorithm
    plt.plot(sizes, valid_df['Prop Mean (s)'], marker='o', color='blue', 
             label='Proposed O(n log n)', linewidth=2)
    
    # Plot Gurobi solver
    plt.plot(sizes, valid_df[f'{solver_name} Mean (s)'], marker='s', color='red', 
             label=f'Gurobi {solver_name}', linewidth=3, linestyle='--')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Graph Size n (Log Scale)')
    plt.ylabel('Execution Time in seconds (Log Scale)')
    plt.title(title)
    plt.legend(fontsize=11, handlelength=4.5, handletextpad=1.0, framealpha=0.95)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Graph saved to: {filename}")

def plot_all_topologies_comparison(df, title, solver_name, filename):
    """Plot Log-Log graphs comparing execution times for all topologies."""
    plt.figure(figsize=(12, 7))
    
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'x']
    
    topologies = df['Topology'].unique()
    
    # Determine Gurobi column based on solver_name
    gurobi_col = "LP Mean (s)" if solver_name == "LP" else "MILP Mean (s)"
    
    for idx, topo in enumerate(topologies):
        topo_df = df[df['Topology'] == topo].sort_values('Size (n)')
        sizes = topo_df['Size (n)']
        prop_times = topo_df['Prop Mean (s)']
        gurobi_times = topo_df[gurobi_col]
        
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        
        # Plot proposed algorithm line
        plt.plot(sizes, prop_times, marker=marker, color=color, 
                label=f'{topo} (Proposed)', linewidth=2.5, markersize=8, linestyle='-')
        
        # Plot Gurobi line (dashed line, same color)
        plt.plot(sizes, gurobi_times, marker=marker, color=color, 
                label=f'{topo} (Gurobi)', linewidth=3, markersize=8, linestyle='--')
    
    # Apply log scales
    plt.xscale('log')
    plt.yscale('log')
    
    plt.xlabel('Graph Size n (Log Scale)', fontsize=12)
    plt.ylabel('Execution Time in seconds (Log Scale)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc='best', fontsize=11, handlelength=5, handletextpad=1.0, columnspacing=2, labelspacing=1.2, framealpha=0.95)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Graph saved to: {filename}")

def run_topology_gamma_sensitivity():
    """
    Measure total execution time versus Gamma for each topology type.
    Fix n and use a single graph instance for each topology.
    Compare Proposed algorithm vs Gurobi solver.
    """
    fixed_n = 500  # Fixed node count to focus on gamma sensitivity, avoiding graph size effects
    topologies = ["random", "star", "spider"]
    alphas = np.linspace(0.05, 0.95, 20)  # Sweep gamma from 5% to 95%
    micro_trials = 20  # Run 20 times to filter OS noise
    
    print(f"\nStarting Topology vs Gamma Sensitivity Benchmark (Fixed n = {fixed_n})...")
    
    # Dictionary to store time arrays for each topology
    topology_times_proposed = {topo:[] for topo in topologies}
    topology_times_gurobi = {topo:[] for topo in topologies}
    topology_times_proposed_hamming = {topo:[] for topo in topologies}
    topology_times_gurobi_hamming = {topo:[] for topo in topologies}
    
    for topo in topologies:
        print(f"  Generating and evaluating FIXED graph for topology: {topo.upper()}...")
        
        # Generate a single fixed tree instance for this topology
        V, d, w, c, x_bar, y_bar, s, subtrees, g_min, g_max = generate_tree_and_bounds(fixed_n, topology=topo)
        
        for alpha in alphas:
            # Determine specific gamma target
            gamma_target = g_min + alpha * (g_max - g_min)
            
            times_proposed = []
            times_gurobi = []
            times_proposed_hamming = []
            times_gurobi_hamming = []
            
            # Disable garbage collection for accurate millisecond-level timing
            gc.disable()
            
            for _ in range(micro_trials):
                # Measure execution time for Proposed Algorithm (Chebyshev)
                start = time.perf_counter()
                solve_chebyshev_proposed(V, d, w, c, x_bar, y_bar, s, subtrees, gamma_target)
                times_proposed.append(time.perf_counter() - start)
                
                # Measure execution time for Gurobi LP Solver (Chebyshev)
                start = time.perf_counter()
                solve_chebyshev_lp(V, d, w, c, x_bar, y_bar, s, subtrees, gamma_target)
                times_gurobi.append(time.perf_counter() - start)
                
                # Measure execution time for Proposed Algorithm (Hamming)
                start = time.perf_counter()
                solve_hamming_proposed(V, d, w, c, x_bar, y_bar, s, subtrees, gamma_target)
                times_proposed_hamming.append(time.perf_counter() - start)
                
                # Measure execution time for Gurobi MILP Solver (Hamming)
                start = time.perf_counter()
                solve_hamming_milp(V, d, w, c, x_bar, y_bar, s, subtrees, gamma_target)
                times_gurobi_hamming.append(time.perf_counter() - start)
                
            gc.enable()
            
            # Take minimum time (exclude OS noise) - Chebyshev
            topology_times_proposed[topo].append(min(times_proposed))
            topology_times_gurobi[topo].append(min(times_gurobi))
            
            # Take minimum time - Hamming
            topology_times_proposed_hamming[topo].append(min(times_proposed_hamming))
            topology_times_gurobi_hamming[topo].append(min(times_gurobi_hamming))

    # --- PLOT GRAPH ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    plt.figure(figsize=(12, 7))
    
    # Define academic-standard markers and colors
    markers = {
        'random': 'o',
        'star': 'v',
        'spider': 'p'
    }
    colors = {
        'random': '#1f77b4',
        'star': '#ff7f0e',
        'spider': '#8c564b'
    }
    labels = {
        'random': 'Random Tree',
        'star': 'Star Graph',
        'spider': 'Spider Graph'
    }
    
    for topo in topologies:
        # Plot proposed algorithm line (solid line)
        plt.plot(alphas * 100, topology_times_proposed[topo], 
                 marker=markers[topo], color=colors[topo], 
                 linewidth=2.5, markersize=7, label=f'{labels[topo]} (Proposed)', linestyle='-')
        
        # Plot Gurobi line (dashed line, same color)
        plt.plot(alphas * 100, topology_times_gurobi[topo], 
                 marker=markers[topo], color=colors[topo], 
                 linewidth=3.5, markersize=7, label=f'{labels[topo]} (Gurobi)', linestyle='--')
        
    plt.xlabel(r'Target Value Position ($\gamma$) between Min and Max (%)', fontsize=12)
    plt.ylabel('Execution Time (seconds)', fontsize=12)
    plt.title(rf'Execution Time vs Target $\gamma$: Proposed vs Gurobi LP ($n = {fixed_n}$)', fontsize=14, fontweight='bold')
    plt.legend(title="Network Topology", fontsize=10, title_fontsize=12, ncol=2, handlelength=5, handletextpad=1.0, columnspacing=2, labelspacing=1.2, framealpha=0.95)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Limit Y-axis (starting from 0) to clearly see performance differences
    bottom_ylim = 0
    all_times = list(topology_times_proposed.values()) + list(topology_times_gurobi.values())
    top_ylim = max([max(times) for times in all_times]) * 1.2
    plt.ylim(bottom_ylim, top_ylim)
    
    plt.tight_layout()
    filepath = os.path.join(current_dir, f"topology_gamma_sensitivity_chebyshev_n{fixed_n}.png")
    plt.savefig(filepath, dpi=300)
    print(f"Chebyshev graph saved to: {filepath}")
    
    # --- PLOT HAMMING GRAPH ---
    plt.figure(figsize=(12, 7))
    
    for topo in topologies:
        # Plot proposed algorithm line (solid line)
        plt.plot(alphas * 100, topology_times_proposed_hamming[topo], 
                 marker=markers[topo], color=colors[topo], 
                 linewidth=2.5, markersize=7, label=f'{labels[topo]} (Proposed)', linestyle='-')
        
        # Plot Gurobi MILP line (dashed line, same color)
        plt.plot(alphas * 100, topology_times_gurobi_hamming[topo], 
                 marker=markers[topo], color=colors[topo], 
                 linewidth=3.5, markersize=7, label=f'{labels[topo]} (Gurobi MILP)', linestyle='--')
        
    plt.xlabel(r'Target Value Position ($\gamma$) between Min and Max (%)', fontsize=12)
    plt.ylabel('Execution Time (seconds)', fontsize=12)
    plt.title(rf'Execution Time vs Target $\gamma$: Proposed vs Gurobi MILP - Hamming Distance ($n = {fixed_n}$)', fontsize=14, fontweight='bold')
    plt.legend(title="Network Topology", fontsize=10, title_fontsize=12, ncol=2, handlelength=5, handletextpad=1.0, columnspacing=2, labelspacing=1.2, framealpha=0.95)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Limit Y-axis (starting from 0) to clearly see performance differences
    bottom_ylim = 0
    all_times_hamming = list(topology_times_proposed_hamming.values()) + list(topology_times_gurobi_hamming.values())
    top_ylim = max([max(times) for times in all_times_hamming]) * 1.2
    plt.ylim(bottom_ylim, top_ylim)
    
    plt.tight_layout()
    filepath_hamming = os.path.join(current_dir, f"topology_gamma_sensitivity_hamming_n{fixed_n}.png")
    plt.savefig(filepath_hamming, dpi=300)
    print(f"Hamming graph saved to: {filepath_hamming}")

# ==========================================
# 5. Benchmarking Logic
# ==========================================

def run_benchmark():
    # Use only the 3 topologies with most contrasting performance
    topologies = ["random", "star", "spider"]
    sizes = [100, 200, 500, 1000, 2000, 5000] 
    num_trials = 10  # Increased for more stable results
    
    all_results_cheb = []
    all_results_ham = []
    
    print(f"Starting Benchmarks with {num_trials} trials per size/topology...")
    print("Warning: Large graph sizes (> 5000 nodes) might take a while.\n")
    
    # Initialize Gurobi to avoid initialization overhead
    print("--- WARM-UP PHASE: Initializing Gurobi Solver ---")
    for topo in topologies[:1]:  # Warm-up with first topology (random)
        V, d, w, c, x_bar, y_bar, s, subtrees, gamma = generate_feasible_instance(50, topology=topo)
        _ = solve_chebyshev_lp(V, d, w, c, x_bar, y_bar, s, subtrees, gamma)
        _ = solve_hamming_milp(V, d, w, c, x_bar, y_bar, s, subtrees, gamma)
    print("Warm-up completed. Starting actual benchmarks...\n")
    
    for topo in topologies:
        print(f"--- EVALUATING TOPOLOGY: {topo.upper()} ---")
        for n in sizes:
            t_prop_c_list, t_lp_c_list = [], []
            t_prop_h_list, t_lp_h_list = [], []
            gap_c_sum, gap_h_sum = 0, 0
            
            for trial in range(num_trials):
                V, d, w, c, x_bar, y_bar, s, subtrees, gamma = generate_feasible_instance(n, topology=topo)
                
                # Chebyshev benchmark
                start = time.perf_counter()
                c_prop = solve_chebyshev_proposed(V, d, w, c, x_bar, y_bar, s, subtrees, gamma)
                t_prop_c_list.append(time.perf_counter() - start)
                
                start = time.perf_counter()
                c_lp = solve_chebyshev_lp(V, d, w, c, x_bar, y_bar, s, subtrees, gamma)
                if c_lp is not None:
                    t_lp_c_list.append(time.perf_counter() - start)
                    gap_c_sum += abs(c_prop - c_lp) / max(1e-5, c_lp)
                
                # Hamming benchmark
                start = time.perf_counter()
                h_prop = solve_hamming_proposed(V, d, w, c, x_bar, y_bar, s, subtrees, gamma)
                t_prop_h_list.append(time.perf_counter() - start)
                
                start = time.perf_counter()
                h_lp = solve_hamming_milp(V, d, w, c, x_bar, y_bar, s, subtrees, gamma)
                if h_lp is not None:
                    t_lp_h_list.append(time.perf_counter() - start)
                    gap_h_sum += abs(h_prop - h_lp) / max(1e-5, h_lp)

            # Aggregate Chebyshev statistics
            prop_mean_c = np.mean(t_prop_c_list)
            lp_mean_c = np.mean(t_lp_c_list) if t_lp_c_list else 0
            speedup_c = round(lp_mean_c / prop_mean_c) if prop_mean_c > 0 else 0
            all_results_cheb.append({
                "Topology": topo.capitalize(),
                "Size (n)": n,
                "Prop Mean (s)": prop_mean_c,
                "LP Mean (s)": lp_mean_c,
                "Speed Up Ratio": speedup_c,
                "Gap (%)": (gap_c_sum / max(1, len(t_lp_c_list))) * 100
            })
            
            # Aggregate Hamming statistics
            prop_mean_h = np.mean(t_prop_h_list)
            milp_mean_h = np.mean(t_lp_h_list) if t_lp_h_list else 0
            speedup_h = round(milp_mean_h / prop_mean_h) if prop_mean_h > 0 else 0
            all_results_ham.append({
                "Topology": topo.capitalize(),
                "Size (n)": n,
                "Prop Mean (s)": prop_mean_h,
                "MILP Mean (s)": milp_mean_h,
                "Speed Up Ratio": speedup_h,
                "Gap (%)": (gap_h_sum / max(1, len(t_lp_h_list))) * 100
            })
            print(f"  Finished size n = {n}")

    # Convert to DataFrame
    df_cheb = pd.DataFrame(all_results_cheb)
    df_ham = pd.DataFrame(all_results_ham)
    
    print("\n" + "="*95)
    print("Table 1: Chebyshev Norm Results (Continuous) - With Standard Deviation")
    print("="*95)
    print(df_cheb.to_string(index=False))
    
    print("\n" + "="*95)
    print("Table 2: Bottleneck Hamming Distance Results (Discrete IP) - With Std Dev")
    print("="*95)
    print(df_ham.to_string(index=False))

    # Save graphs to the same directory as the script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    chebyshev_img_path = os.path.join(current_dir, "plot_chebyshev.png")
    hamming_img_path = os.path.join(current_dir, "plot_hamming.png")
    chebyshev_all_topo_path = os.path.join(current_dir, "plot_chebyshev_all_topologies.png")
    hamming_all_topo_path = os.path.join(current_dir, "plot_hamming_all_topologies.png")
    
    # Plot Log-Log graphs (using Random topology as baseline)
    df_cheb_random = df_cheb[df_cheb['Topology'] == 'Random']
    df_ham_random = df_ham[df_ham['Topology'] == 'Random']
    
    plot_loglog_results(df_cheb_random, "Chebyshev Norm: Proposed vs Gurobi LP", "LP", chebyshev_img_path)
    plot_loglog_results(df_ham_random, "Hamming Distance: Proposed vs Gurobi MILP", "MILP", hamming_img_path)
    
    # Plot comparison graphs for all topologies
    plot_all_topologies_comparison(df_cheb, "Chebyshev Norm: Proposed Algorithm - All Topologies", "LP", chebyshev_all_topo_path)
    plot_all_topologies_comparison(df_ham, "Hamming Distance: Proposed Algorithm - All Topologies", "MILP", hamming_all_topo_path)

if __name__ == "__main__":
    # Set seed for reproducible results
    random.seed(42)
    np.random.seed(42)
    run_benchmark()
    run_topology_gamma_sensitivity()
