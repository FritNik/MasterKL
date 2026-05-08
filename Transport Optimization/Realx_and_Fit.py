"""
DSL Relax-and-Fix Heuristic
============================
Step 1:
Preprocessing:
    - lösche demand points, die bereits von existing stops abgedeckt werden
- für die restlichen demand points: finde die Blöcke von aufeinanderfolgenden Kandidaten, die sie abdecken
- löse die LP-Relaxierung von DSL prime

Step 2:
- für jeden demand point: finde den Block mit maximalem y_r Wert

Step 3:
- löse DSL, sodass jeder demand Point von einer Station aus dem gewählten Block aus Step 2 abgedect wird
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
OUTPUT          = os.path.join(SCRIPT_DIR, "DSL-heuristic-solution.giv")
# ──────────────────────────────────────────────────────────


# ── Parsing (recycled from (a) and (b)) ───────────────────
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
    location_ids = []
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
            if l_id not in location_ids:
                location_ids.append(l_id)
    return demand_ids, location_ids, a


# Load data
existing_stops = parse_giv(STOPS_FILE)
stations_to_open     = parse_giv(CANDIDATES_FILE)
demand_points  = parse_giv(DEMAND_FILE)

demand_ids, station_ids, a = parse_covering_matrix(COVERING_MATRIX)
existing_ids  = [s["id"] for s in existing_stops]
stations_to_open_ids = [c["id"] for c in stations_to_open]


# Demand points die von mindestens einem existing stop gedeckt werden
already_covered = {
    i for i in demand_ids
    if any(a.get((i, j), 0) == 1 for j in existing_ids)
}

# Nur die noch ungedeckten müssen noch abgedeckt werden
uncovered_demand_points = [i for i in demand_ids if i not in already_covered]

#um die Formulierung zu vereinfachen, betrachten wir nur die Teilmatrix der uncovered demand points und zu eröffnenden Stationen
a_reduced = {
    (i, j): a[(i, j)]
    for i in uncovered_demand_points
    for j in stations_to_open_ids
}
print("a_reduced:" +str(a_reduced))

#covering_stations: gibt für jeden ungedeckten Demand Point die StationsIDs zurück, die ihn abdecken
covering_stations = {
    i: [j for j in stations_to_open_ids if a_reduced.get((i, j), 0) == 1]
    for i in uncovered_demand_points
}

for i in uncovered_demand_points:
    if covering_stations[i] == []:
        print(f"Radius {RADIUS} is too small — demand point {i} cannot be covered by any candidate.")
        exit(1)


#_______________ STEP 1 _______________________

for i, station in covering_stations.items():
    print(f"  Demand point with index {i}: covered by {station}")

def get_consecutive_blocks(stations_covering, stations_all):
    """
    stations_covering: Liste von Stations-IDs, die einen Demand Point abdecken
    stations_all: Liste aller Stations-IDs (in fixer Reihenfolge)
    return: Liste von Blöcken für die Reihe des Demand points

    Grupppiert eine Liste von Stations-IDs für einen Demand Point in Blöcke von
    aufeinanderfolgenden Einsen bezogen auf alle Stationen.
    """
    if not stations_covering:
        return []
    
    indices = [stations_all.index(j) for j in stations_covering]
    indices.sort()
    
    blocks = []
    current_block = [indices[0]]
    
    for idx in indices[1:]:
        if idx == current_block[-1] + 1:  # consecutive
            current_block.append(idx)
        else:                              # gap -> new block
            blocks.append([stations_all[i] for i in current_block])
            current_block = [idx]
    blocks.append([stations_all[i] for i in current_block])
    
    return blocks


#erstelle Blöcke von Kandidaten-IDs für jeden uncovered Demand Point
print(stations_to_open_ids)
consecutive_blocks = {}
for i in uncovered_demand_points:
    consecutive_blocks[i] = get_consecutive_blocks(covering_stations[i], stations_to_open_ids)
    print(f"  Demand point with index {i}: consecutive blocks: {consecutive_blocks[i]}")
    


# LP Relaxierung DSL prime lösen
DSL_prime_relax = gp.Model("DSL_prime_relax")
DSL_prime_relax.Params.OutputFlag = 0

x_r = DSL_prime_relax.addVars(stations_to_open_ids, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="x")

y_r = {}
for i in uncovered_demand_points:
    for k, block in enumerate(consecutive_blocks[i]):
        y_r[i, k] = DSL_prime_relax.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"y_{i}_{k}")

DSL_prime_relax.setObjective(gp.quicksum(x_r[j] for j in stations_to_open_ids), GRB.MINIMIZE)

for i in uncovered_demand_points:
    for k, block in enumerate(consecutive_blocks[i]):
        DSL_prime_relax.addConstr(gp.quicksum(x_r[j] for j in block) >= y_r[i, k])

for i in uncovered_demand_points:
    DSL_prime_relax.addConstr(gp.quicksum(y_r[i, k] for k in range(len(consecutive_blocks[i]))) >= 1)

DSL_prime_relax.optimize()




#______________________ STEP 2 _______________________

# Für jeden Demand Point den Block mit maximalem y_r Wert finden
best_blocks = {}   # i -> (k*, block)
for i in uncovered_demand_points:
    k_star = max(range(len(consecutive_blocks[i])), key=lambda k: y_r[i, k].X)
    best_blocks[i] = (k_star, consecutive_blocks[i][k_star])
    print(f"  Demand {i}: k*={k_star+1}, block={consecutive_blocks[i][k_star]}, y={y_r[i, k_star].X:.4f}")
    
    


#_______________________ STEP 3 _______________________
DSL_C1P = gp.Model("DSL_with_C1P")
DSL_C1P.Params.OutputFlag = 0

# Wegen C1P reicht LP-Relaxierung -- Lösung ist automatisch ganzzahlig
x_tilde = DSL_C1P.addVars(stations_to_open_ids, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="x")

DSL_C1P.setObjective(gp.quicksum(x_tilde[j] for j in stations_to_open_ids), GRB.MINIMIZE)

# Für jeden Demand Point p: der beste Block k*(p) muss abgedeckt werden
for i in uncovered_demand_points:
    k_star, block = best_blocks[i]
    DSL_C1P.addConstr(
        gp.quicksum(x_tilde[j] for j in block) >= 1,
        name=f"cover_{i}"
    )

DSL_C1P.optimize()

x_sol = {j: x_tilde[j].X for j in stations_to_open_ids}
print(f"Relax-and-Fit solution: {sum(x_sol[j] for j in stations_to_open_ids):.0f} new stations opened.")
for j in stations_to_open_ids:
    if x_sol[j] == 1:
        print(f"  x[{j}] = 1")