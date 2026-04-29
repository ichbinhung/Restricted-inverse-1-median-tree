import sys
import subprocess
import os

# ==========================================
# 0. Auto-Install Missing Dependencies
# ==========================================
def install_requirements():
    required = ['gurobipy', 'networkx', 'pandas', 'numpy', 'matplotlib']
    for package in required:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing missing library: {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install_requirements()

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
    
    subtree_masks =[]
    for comp in subtrees.values():
        mask = np.zeros(V_len, dtype=bool)
        mask[comp] = True
        subtree_masks.append(mask)
        
    all_sorted = np.argsort(d_arr).tolist()
    sorted_subtrees =[]
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
    
    subtree_masks =[]
    for comp in subtrees.values():
        mask = np.zeros(V_len, dtype=bool)
        mask[comp] = True
        subtree_masks.append(mask)
        
    all_sorted = np.argsort(d_arr).tolist()
    sorted_subtrees =[]
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

def generate_topology(n, topology="random"):
    """Sinh các dạng đồ thị cây khác nhau (Robustness Test)."""
    if topology == "path":
        return nx.path_graph(n)
    elif topology == "binary":
        return nx.full_rary_tree(2, n)
    elif topology == "star":
        return nx.star_graph(n-1)
    else: # "random"
        return nx.random_labeled_tree(n)

def generate_feasible_instance(n, topology="random"):
    """Sinh instance hợp lệ với topology được chỉ định."""
    while True:
        T = generate_topology(n, topology)
        V = list(T.nodes)
        s = random.choice(V)
        
        d = nx.single_source_shortest_path_length(T, s)
        subtrees = get_subtrees(T, s)
        
        w = {v: random.uniform(1, 10) for v in V}
        c = {v: random.uniform(1, 5) for v in V}
        x_bar = {v: random.uniform(2, 5) for v in V}
        y_bar = {v: random.uniform(0.5, w[v] - 0.1) if w[v] > 0.6 else 0 for v in V}
        
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
        sorted_subtrees =[]
        for mask in subtree_masks:
            comp_indices = np.where(mask)[0]
            sorted_subtrees.append(comp_indices[np.argsort(d_arr[comp_indices])].tolist())
            
        g_min, g_max = fast_calculate_bounds(lb, ub, d_arr, subtree_masks, sorted_subtrees, all_sorted)
        
        if g_min != float('inf') and g_max != float('-inf') and (g_min < g_max - 1e-3):
            gamma = random.uniform(g_min + 0.2*(g_max-g_min), g_max - 0.2*(g_max-g_min))
            return V, d, w, c, x_bar, y_bar, s, subtrees, gamma

def plot_loglog_results(df, title, solver_name, filename):
    """Vẽ đồ thị Log-Log so sánh tốc độ."""
    plt.figure(figsize=(10, 6))
    
    # Lọc bỏ các dòng Gurobi bị lỗi/timeout (nếu có)
    valid_df = df[df[f'{solver_name} Mean (s)'] > 0]
    sizes = valid_df['Size (n)']
    
    # Plot Thuật toán đề xuất
    plt.plot(sizes, valid_df['Prop Mean (s)'], marker='o', color='blue', 
             label='Proposed O(n log n)', linewidth=2)
    plt.fill_between(sizes, 
                     valid_df['Prop Mean (s)'] - valid_df['Prop Std'], 
                     valid_df['Prop Mean (s)'] + valid_df['Prop Std'], 
                     color='blue', alpha=0.2)
    
    # Plot Gurobi Solver
    plt.plot(sizes, valid_df[f'{solver_name} Mean (s)'], marker='s', color='red', 
             label=f'Gurobi {solver_name}', linewidth=2, linestyle='--')
    plt.fill_between(sizes, 
                     valid_df[f'{solver_name} Mean (s)'] - valid_df[f'{solver_name} Std'], 
                     valid_df[f'{solver_name} Mean (s)'] + valid_df[f'{solver_name} Std'], 
                     color='red', alpha=0.2)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Graph Size n (Log Scale)')
    plt.ylabel('Execution Time in seconds (Log Scale)')
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Đã lưu biểu đồ: {filename}")

# ==========================================
# 5. Benchmarking Logic
# ==========================================

def run_benchmark():
    topologies =["random", "path", "binary"] # Robustness test
    sizes =[100, 200, 500, 1000, 2000, 5000] # Thêm 10000, 20000 sau nếu muốn
    num_trials = 15 # Tăng số lượng mẫu để lấy độ lệch chuẩn
    
    all_results_cheb =[]
    all_results_ham =[]
    
    print(f"Starting Benchmarks with {num_trials} trials per size/topology...")
    print("Warning: MILP solver on path topologies > 2000 nodes might take a while.\n")
    
    for topo in topologies:
        print(f"--- EVALUATING TOPOLOGY: {topo.upper()} ---")
        for n in sizes:
            t_prop_c_list, t_lp_c_list = [],[]
            t_prop_h_list, t_lp_h_list = [],[]
            gap_c_sum, gap_h_sum = 0, 0
            
            for trial in range(num_trials):
                V, d, w, c, x_bar, y_bar, s, subtrees, gamma = generate_feasible_instance(n, topology=topo)
                
                # --- Chebyshev Benchmark ---
                start = time.perf_counter()
                c_prop = solve_chebyshev_proposed(V, d, w, c, x_bar, y_bar, s, subtrees, gamma)
                t_prop_c_list.append(time.perf_counter() - start)
                
                start = time.perf_counter()
                c_lp = solve_chebyshev_lp(V, d, w, c, x_bar, y_bar, s, subtrees, gamma)
                if c_lp is not None:
                    t_lp_c_list.append(time.perf_counter() - start)
                    gap_c_sum += abs(c_prop - c_lp) / max(1e-5, c_lp)
                
                # --- Hamming Benchmark ---
                start = time.perf_counter()
                h_prop = solve_hamming_proposed(V, d, w, c, x_bar, y_bar, s, subtrees, gamma)
                t_prop_h_list.append(time.perf_counter() - start)
                
                start = time.perf_counter()
                h_lp = solve_hamming_milp(V, d, w, c, x_bar, y_bar, s, subtrees, gamma)
                if h_lp is not None:
                    t_lp_h_list.append(time.perf_counter() - start)
                    gap_h_sum += abs(h_prop - h_lp) / max(1e-5, h_lp)

            # Tổng hợp thống kê Chebyshev
            all_results_cheb.append({
                "Topology": topo.capitalize(),
                "Size (n)": n,
                "Prop Mean (s)": np.mean(t_prop_c_list),
                "Prop Std": np.std(t_prop_c_list),
                "LP Mean (s)": np.mean(t_lp_c_list) if t_lp_c_list else 0,
                "LP Std": np.std(t_lp_c_list) if t_lp_c_list else 0,
                "Gap (%)": (gap_c_sum / max(1, len(t_lp_c_list))) * 100
            })
            
            # Tổng hợp thống kê Hamming
            all_results_ham.append({
                "Topology": topo.capitalize(),
                "Size (n)": n,
                "Prop Mean (s)": np.mean(t_prop_h_list),
                "Prop Std": np.std(t_prop_h_list),
                "MILP Mean (s)": np.mean(t_lp_h_list) if t_lp_h_list else 0,
                "MILP Std": np.std(t_lp_h_list) if t_lp_h_list else 0,
                "Gap (%)": (gap_h_sum / max(1, len(t_lp_h_list))) * 100
            })
            print(f"  Finished size n = {n}")

    # Chuyển thành DataFrame
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

    # --- Lưu 2 biểu đồ vào CÙNG THƯ MỤC với file script này ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    chebyshev_img_path = os.path.join(current_dir, "plot_chebyshev.png")
    hamming_img_path = os.path.join(current_dir, "plot_hamming.png")
    
    # Vẽ biểu đồ Log-Log (lấy cấu trúc Random làm đại diện)
    df_cheb_random = df_cheb[df_cheb['Topology'] == 'Random']
    df_ham_random = df_ham[df_ham['Topology'] == 'Random']
    
    plot_loglog_results(df_cheb_random, "Chebyshev Norm: Proposed vs Gurobi LP", "LP", chebyshev_img_path)
    plot_loglog_results(df_ham_random, "Hamming Distance: Proposed vs Gurobi MILP", "MILP", hamming_img_path)

if __name__ == "__main__":
    # Đặt seed để có thể tái tạo (reproduce) lại kết quả
    random.seed(42)
    np.random.seed(42)
    run_benchmark()