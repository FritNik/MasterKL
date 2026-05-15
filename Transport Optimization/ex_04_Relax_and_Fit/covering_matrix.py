"""
Covering Matrix Generator
"""

import os

RADIUS      = 75
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DEMAND      = os.path.join(SCRIPT_DIR, "Demand.giv")
STOPS       = os.path.join(SCRIPT_DIR, "Existing-Stops.giv")
CANDIDATES  = os.path.join(SCRIPT_DIR, "Candidates.giv")
OUTPUT      = os.path.join(SCRIPT_DIR, "Covering-Matrix.giv")


def parse_giv(filepath):
    points = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(";")]
            point_id = parts[0]
            x = float(parts[3])
            y = float(parts[4])
            points.append((point_id, x, y))
    return points


def l1_distance(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)


demand_points  = parse_giv(DEMAND)
existing_stops = parse_giv(STOPS)
stops_to_open     = parse_giv(CANDIDATES)
stops      = existing_stops + stops_to_open

with open(OUTPUT, "w") as f:
    f.write("# demand-point; location; matrix-entry\n")
    for d_id, dx, dy in demand_points:
        for l_id, lx, ly in stops:
            entry = 1 if l1_distance(dx, dy, lx, ly) <= RADIUS else 0
            f.write(f"{d_id}; {l_id}; {entry}\n")

print(f"Done. Covering matrix written to '{OUTPUT}'.")