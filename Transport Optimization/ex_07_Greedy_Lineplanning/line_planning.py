"""
Optimization in Public Transport – Exercise Sheet 07
Exercise 3: Greedy Heuristics and Optimal Solver for (LP1)
"""

import os
import math
import heapq
import argparse
from copy import deepcopy
from collections import defaultdict

# ─────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────

def parse_giv(path, skip_comment_char='#'):
    """Read a .giv file and return list of non-comment, non-empty rows as lists."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(skip_comment_char):
                continue
            rows.append([x.strip() for x in line.split(';')])
    return rows


def load_instance(folder):
    """
    Load a PTN instance from a folder containing:
      Edge.giv, Stop.giv, Pool.giv, Pool-Cost.giv, Load.giv
    Returns a dict with all relevant data.
    """
    # Stops
    stops = {}
    for row in parse_giv(os.path.join(folder, 'Stop.giv')):
        sid = int(row[0])
        stops[sid] = {'short': row[1], 'long': row[2], 'x': float(row[3]), 'y': float(row[4])}

    # Edges
    edges = {}
    for row in parse_giv(os.path.join(folder, 'Edge.giv')):
        edge_id = int(row[0])
        edges[edge_id] = {
            'left': int(row[1]),
            'right': int(row[2]),
            'length': float(row[3]),
            'min_tt': float(row[4]),
            'max_tt': float(row[5]),
        }

    # Loads (min/max frequencies per edge)
    loads = {}
    for row in parse_giv(os.path.join(folder, 'Load.giv')):
        edge_id = int(row[0])
        loads[edge_id] = {
            'load': float(row[1]),
            'f_min': int(row[2]),
            'f_max': int(row[3]),
        }

    # Line pool: line_id -> ordered list of edge IDs
    pool_edges = defaultdict(list)
    for row in parse_giv(os.path.join(folder, 'Pool.giv')):
        line_id = int(row[0])
        # row[1] is the order index, row[2] is the edge id
        pool_edges[line_id].append((int(row[1]), int(row[2])))
    line_pool = {}
    for line_id, ordered in pool_edges.items():
        ordered.sort()
        line_pool[line_id] = [e for _, e in ordered]

    # Line costs and lengths
    line_costs = {}
    for row in parse_giv(os.path.join(folder, 'Pool-Cost.giv')):
        line_id = int(row[0])
        line_costs[line_id] = {'length': float(row[1]), 'cost': float(row[2])}

    return {
        'stops': stops,
        'edges': edges,
        'loads': loads,
        'line_pool': line_pool,   # {lid: [edge_id, ...]}
        'line_costs': line_costs, # {lid: {'length': ..., 'cost': ...}}
    }



# ─────────────────────────────────────────────
# 3. GENERIC GREEDY FRAMEWORK
# ─────────────────────────────────────────────

def run_greedy(instance, step3_fn, step4_fn, name="Greedy"):
    """
    Generic greedy heuristic for (LP1).

    step3_fn(line_pool, line_costs, f_min_current, edges) -> (lid, f_l)
        Returns the chosen line id and its frequency increment, or None if no suitable line.
    step4_fn is embedded in step3_fn for flexibility, but the interface is:
        step3 chooses l (and determines f(l) via step4 logic internally).

    For cleaner separation we pass both as one combined selector:
        select_fn(candidates, line_costs, f_min_current, edges)
            -> (lid, f_l)   or   None
    """
    loads = instance['loads']
    line_pool = instance['line_pool']
    line_costs = instance['line_costs']
    edges = instance['edges']

    # Working copy of f_min
    f_min = {edge_id: loads[edge_id]['f_min'] for edge_id in loads}
    chosen_lines = {}   # {lid: frequency}

    iteration = 0
    while True:
        # Step 2: check if all f_min are 0
        if all(v == 0 for v in f_min.values()):
            # Check f_max feasibility (frequencies must not exceed f_max)
            # (already maintained below)
            print(f"  [{name}] Feasible solution found after {iteration} iterations.")
            total_cost = sum(line_costs[l]['cost'] * f for l, f in chosen_lines.items())
            return chosen_lines, total_cost

        # Step 3: choose a line
        result = step3_fn(line_pool, line_costs, f_min, edges)
        if result is None:
            print(f"  [{name}] No feasible solution found.")
            return None, None

        line_id, f_l = result

        # Step 4: update
        if line_id not in chosen_lines:
            chosen_lines[line_id] = 0
        chosen_lines[line_id] += f_l

        for edge_id in line_pool[line_id]:
            f_min[edge_id] = max(f_min[edge_id] - f_l, 0)

        iteration += 1


# ─────────────────────────────────────────────
# 4. HEURISTIC 7
# ─────────────────────────────────────────────

def heuristic7_select(line_pool, line_costs, f_min, edges):
    """
    Heuristic 7:
      Candidates L' = L0 \\ L  (lines where at least one edge still needs coverage)
      g(l) = cost_l / length_l
      f(l) = max_{e in l} f_min_e

    Lines with f(l)=0 have all their edges already covered, so they are
    excluded from L' (they have been 'used up', in a sense).
    """
    best = None
    best_g = math.inf

    for line_id, edge_list in line_pool.items():
        # f(l) = max_{e in l} f_min_e; skip lines that bring no progress
        f_l_candidate = max(f_min[eid] for eid in edge_list)
        if f_l_candidate == 0:
            continue  # line already 'satisfied' – not in L'
        cost_l = line_costs[line_id]['cost']
        length_l = line_costs[line_id]['length']
        if length_l == 0:
            continue
        g = cost_l / length_l
        if g < best_g:
            best_g = g
            best = line_id

    if best is None:
        return None  # no line can make progress

    line_id = best
    f_l = max(f_min[edge_id] for edge_id in line_pool[line_id])
    return line_id, f_l


# ─────────────────────────────────────────────
# 5. HEURISTIC 8
# ─────────────────────────────────────────────

def heuristic8_select(line_pool, line_costs, f_min, edges):
    """
    Heuristic 8:
      Find edge e'' with maximal f_min_{e''}  (e'' must have f_min > 0)
      Candidates L' = {l in L0 : e'' in l}
      g(l) = cost_l / |{e' in l : f_min_{e'} > 0}|
      f(l) = min_{e in l, f_min_e > 0} f_min_e
    """
    # Find edge e'' with max f_min (must be > 0)
    e_star = max((eid for eid, v in f_min.items() if v > 0),
                 key=lambda eid: f_min[eid],
                 default=None)
    if e_star is None:
        return None

    # Candidates: lines containing e_star
    candidates = {line_id: edge_list for line_id, edge_list in line_pool.items()
                  if e_star in edge_list}
    if not candidates:
        return None

    best = None
    best_g = math.inf

    for line_id, edge_list in candidates.items():
        cost_l = line_costs[line_id]['cost']
        uncovered = sum(1 for edge_id in edge_list if f_min[edge_id] > 0)
        if uncovered == 0:
            continue
        g = cost_l / uncovered
        if g < best_g:
            best_g = g
            best = line_id

    if best is None:
        return None

    line_id = best
    # f(l) = min_{e in l, f_min_e > 0} f_min_e
    positive = [f_min[edge_id] for edge_id in line_pool[line_id] if f_min[edge_id] > 0]
    f_l = min(positive) if positive else 0
    if f_l == 0:
        return None
    return line_id, f_l


# ─────────────────────────────────────────────
# 6. OWN HEURISTIC (custom)
# ─────────────────────────────────────────────

def heuristic_own_select(line_pool, line_costs, f_min, edges):
    """
    Own Heuristic ("Coverage-Weighted Cost"):
      Like H8, first find the most-loaded uncovered edge e''.
      Candidates: all lines containing e''.
      g(l) = cost_l / sum_{e in l, f_min_e > 0} f_min_e
             (cost per unit of total remaining demand covered)
      f(l) = min_{e in l, f_min_e > 0} f_min_e

    Correctness argument:
      - At every iteration there exists at least one line containing e'' (from
        feasibility assumption), so the candidate set is non-empty as long as
        e'' cannot be covered. Since we always make progress (f_min_{e''} drops
        by at least f(l) > 0), we terminate.
      - Whenever all f_min are zero we output the feasible solution; whenever
        e'' is uncoverable (no candidate) we correctly report infeasibility.
    """
    # Find most-loaded uncovered edge
    e_star = max((eid for eid, v in f_min.items() if v > 0),
                 key=lambda eid: f_min[eid],
                 default=None)
    if e_star is None:
        return None

    candidates = {line_id: edge_list for line_id, edge_list in line_pool.items()
                  if e_star in edge_list}
    if not candidates:
        return None

    best = None
    best_g = math.inf

    for line_id, edge_list in candidates.items():
        cost_l = line_costs[line_id]['cost']
        total_demand = sum(f_min[edge_id] for edge_id in edge_list if f_min[edge_id] > 0)
        if total_demand == 0:
            continue
        g = cost_l / total_demand   # cost per unit of demand covered
        if g < best_g:
            best_g = g
            best = line_id

    if best is None:
        return None

    line_id = best
    positive = [f_min[edge_id] for edge_id in line_pool[line_id] if f_min[edge_id] > 0]
    f_l = min(positive) if positive else 0
    if f_l == 0:
        return None
    return line_id, f_l


# ─────────────────────────────────────────────
# 8. PRETTY PRINTING
# ─────────────────────────────────────────────

def print_solution(name, lines, cost):
    if lines is None:
        print(f"  {name}: No feasible solution found.")
        return
    print(f"  {name}: cost = {cost:.2f}")
    for lid in sorted(lines):
        print(f"    Line {lid}: frequency = {lines[lid]}")


# ─────────────────────────────────────────────
# 10. MAIN
# ─────────────────────────────────────────────

def run_instance(folder, label):
    print(f"\n{'='*60}")
    print(f"Instance: {label}  ({folder})")
    print('='*60)

    try:
        instance = load_instance(folder)
    except FileNotFoundError as e:
        print(f"  Skipping – file not found: {e}")
        return {}

    results = {}

    # Heuristic 7
    lines7, cost7 = run_greedy(instance, heuristic7_select, None, name="H7")
    print_solution("Heuristic 7", lines7, cost7)
    results['H7'] = cost7

    # Heuristic 8
    lines8, cost8 = run_greedy(instance, heuristic8_select, None, name="H8")
    print_solution("Heuristic 8", lines8, cost8)
    results['H8'] = cost8

    # Own heuristic
    lines_own, cost_own = run_greedy(instance, heuristic_own_select, None, name="Own")
    print_solution("Own Heuristic", lines_own, cost_own)
    results['Own'] = cost_own

    # Summary
    print("\n  Summary:")
    print(f"  {'Method':<18} {'Cost':>10}")
    for method, cost in results.items():
        val = f"{cost:.2f}" if cost is not None else "—"
        opt_flag = " *" if method == 'Optimal' else ""
        print(f"  {method:<18} {val:>10}{opt_flag}")

    return results


def main():
    # In main(), ersetze den data_root-Block:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    run_instance(os.path.join(script_dir, 'data', 'ring'), "ring")

    # If you have additional instances (grid, ring, bahn-01), add them:
    for inst in ['grid', 'ring', 'bahn-01']:
        inst_path = os.path.join(script_dir, 'data', inst)
        if os.path.isdir(inst_path):
            run_instance(inst_path, inst)


if __name__ == '__main__':
    main()
