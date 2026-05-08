"""
DSL Solver
==========
Existing stops are always open (x_j = 1 fixed).
Only the number of NEW candidate locations is minimised.

DSL formulation:
    min  sum_{j in Candidates}  x_j
    s.t. sum_{j: a_ij=1} x_j >= 1    for all demand points i
         x_j = 1                      for all j in Existing-Stops
         x_j in {0, 1}               for all j in Candidates
"""

import os
import gurobipy as gp
from gurobipy import GRB

RADIUS          = 75
SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
COVERING_MATRIX = os.path.join(SCRIPT_DIR, "Covering-Matrix.giv")
STOPS_FILE      = os.path.join(SCRIPT_DIR, "Existing-Stops.giv")
CANDIDATES_FILE = os.path.join(SCRIPT_DIR, "Candidates.giv")
DEMAND_FILE     = os.path.join(SCRIPT_DIR, "Demand.giv")
OUTPUT          = os.path.join(SCRIPT_DIR, "DSL-solution.giv")
# ──────────────────────────────────────────────────────────


def parse_giv(filepath):
    """id; short_name; long_name; x; y , return a list of dicts."""
    points = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(";")]
            points.append({
                "id":         parts[0],
                "short_name": parts[1],
                "long_name":  parts[2],
                "x":          float(parts[3]),
                "y":          float(parts[4]),
            })
    return points


def parse_covering_matrix(filepath):
    """Returns list of demand_ids, location_ids, a[(d_id, l_id)] -> 0/1."""
    a = {}
    demand_ids  = []
    stations_ids = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts  = [p.strip() for p in line.split(";")]
            d_id, l_id, entry = parts[0], parts[1], int(parts[2])
            a[(d_id, l_id)] = entry
            if d_id not in demand_ids:
                demand_ids.append(d_id)
            if l_id not in stations_ids:
                stations_ids.append(l_id)
    return demand_ids, stations_ids, a


# Loading data as dicts
existing_stops = parse_giv(STOPS_FILE)
stations_to_open     = parse_giv(CANDIDATES_FILE)
demand_points  = parse_giv(DEMAND_FILE)

# sets für einfachen zugriff
existing_ids   = {s["id"] for s in existing_stops}
stations_to_open_ids  = {c["id"] for c in stations_to_open}
station_info  = {loc["id"]: loc for loc in existing_stops + stations_to_open}

#load covering matrix as lists and a as a dict
demand_ids, stations_ids, a = parse_covering_matrix(COVERING_MATRIX)

print(f"Demand points : {len(demand_ids)}")
print(f"Existing stops: {len(existing_ids)}  (always open)")
print(f"Stops to open    : {len(stations_to_open_ids)} (decision variables)")
print(f"Radius        : {RADIUS}")

# create Gurobi model
model = gp.Model("DSL")
model.Params.OutputFlag = 1

x = model.addVars(stations_ids, vtype=GRB.BINARY, name="x")

# Fix existing stops to 1
for j in stations_ids:
    if j in existing_ids:
        x[j].LB = 1.0

# Objective: minimise number of opened candidate locations
model.setObjective(
    gp.quicksum(x[j] for j in stations_ids if j in stations_to_open_ids),
    GRB.MINIMIZE
)

# Coverage: every demand point must be covered by at least one open location
for i in demand_ids:
    covering = [j for j in stations_ids if a.get((i, j), 0) == 1]
    if not covering:
        print(f"  WARNING: demand point {i} cannot be covered — infeasible!")
    model.addConstr(gp.quicksum(x[j] for j in covering) >= 1, name=f"cover_{i}")

model.optimize()

# 
if model.Status == GRB.OPTIMAL:
    selected_new      = [j for j in stations_ids if j in stations_to_open_ids  and x[j].X == 1]
    selected_existing = [j for j in stations_ids if j in existing_ids]

    print(f"\nOptimal solution:")
    print(f"  Existing stops (always open) : {len(selected_existing)}")
    print(f"  New candidates opened        : {len(selected_new)} — IDs: {selected_new}")

    with open(OUTPUT, "w") as f:
        f.write("# stop-id; short-name; long-name; x-coordinate; y-coordinate\n")
        for j in selected_existing + selected_new:
            loc = station_info[j]
            f.write(f"{loc['id']}; {loc['short_name']}; {loc['long_name']}; "
                    f"{int(loc['x'])}; {int(loc['y'])}\n")


elif model.Status == GRB.INFEASIBLE:
    print("INFEASIBLE - radius too small to cover all demand points.")
