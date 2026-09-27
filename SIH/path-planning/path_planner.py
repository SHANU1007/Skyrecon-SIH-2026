"""
Path Planning — A* and Dijkstra for best rescue route to survivors
---------------------------------------------------------------------
Rescue team ko drone ke start point se survivor tak sabse safe/sasta
(least-cost) raasta batata hai, disaster map (hazard-detection module se)
ko cost-grid mein convert karke.

Grid representation:
    - 2D array, har cell ek "cost" hai us terrain se guzarne ka.
    - float('inf') = completely blocked (e.g. active fire, deep water)
    - higher number = riskier/slower (debris, damaged ground)
    - 1 = normal, clear terrain

Two algorithms provided for comparison:
    - Dijkstra: guaranteed shortest/cheapest path, explores more nodes
    - A*: usually faster (uses heuristic to guide search), same optimal
      cost path here since heuristic used is admissible (Euclidean distance)

Usage (as a script, runs a demo):
    python path_planner.py

Usage (as a module):
    from path_planner import astar, dijkstra, hazard_labels_to_cost_grid
"""

import heapq
import math
import time


# ---------------------------------------------------------------------------
# Cost grid helpers
# ---------------------------------------------------------------------------

# Default cost mapping from hazard-detection module's output labels.
# Tune these based on real-world risk assessment.
HAZARD_COST_MAP = {
    "normal": 1.0,
    "earthquake_damage": 6.0,
    "landslide_debris": 8.0,
    "flood": 12.0,
    "fire": float("inf"),   # never route rescue teams through active fire
}


def hazard_labels_to_cost_grid(label_grid):
    """
    Convert a 2D grid of hazard-detection labels (strings, one per map cell)
    into a numeric cost grid usable by astar()/dijkstra().

    label_grid: list of lists of strings, e.g.
        [["normal", "normal", "flood"],
         ["normal", "fire",   "flood"],
         ["normal", "normal", "normal"]]
    """
    return [[HAZARD_COST_MAP.get(cell, 1.0) for cell in row] for row in label_grid]


def neighbors(pos, grid):
    """8-directional neighbors (includes diagonals) that are within bounds and not blocked."""
    rows, cols = len(grid), len(grid[0])
    r, c = pos
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != float("inf"):
                step_cost = grid[nr][nc] * (math.sqrt(2) if dr != 0 and dc != 0 else 1.0)
                yield (nr, nc), step_cost


def euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


# ---------------------------------------------------------------------------
# Dijkstra
# ---------------------------------------------------------------------------
def dijkstra(grid, start, goal):
    """Returns (path, total_cost, nodes_explored) or (None, inf, n) if unreachable."""
    frontier = [(0.0, start)]
    came_from = {}
    cost_so_far = {start: 0.0}
    explored = 0

    while frontier:
        current_cost, current = heapq.heappop(frontier)
        explored += 1

        if current == goal:
            return reconstruct_path(came_from, current), current_cost, explored

        for nxt, step_cost in neighbors(current, grid):
            new_cost = cost_so_far[current] + step_cost
            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                came_from[nxt] = current
                heapq.heappush(frontier, (new_cost, nxt))

    return None, float("inf"), explored


# ---------------------------------------------------------------------------
# A*
# ---------------------------------------------------------------------------
def astar(grid, start, goal):
    """Returns (path, total_cost, nodes_explored) or (None, inf, n) if unreachable."""
    frontier = [(0.0, start)]
    came_from = {}
    cost_so_far = {start: 0.0}
    explored = 0

    while frontier:
        _, current = heapq.heappop(frontier)
        explored += 1

        if current == goal:
            return reconstruct_path(came_from, current), cost_so_far[current], explored

        for nxt, step_cost in neighbors(current, grid):
            new_cost = cost_so_far[current] + step_cost
            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                priority = new_cost + euclidean(nxt, goal)
                heapq.heappush(frontier, (priority, nxt))
                came_from[nxt] = current

    return None, float("inf"), explored


# ---------------------------------------------------------------------------
# Multi-survivor rescue ordering (nearest-next greedy)
# ---------------------------------------------------------------------------
def plan_rescue_order(grid, start, survivors, algorithm="astar"):
    """
    Given multiple survivor locations, greedily picks the cheapest-to-reach
    survivor first, then plans from there to the next-cheapest remaining one,
    and so on. Returns a list of dicts describing each leg of the rescue route.

    This is a greedy nearest-neighbor heuristic, not a full TSP solve —
    good enough for a handful of survivors found in one drone pass.
    """
    solve = astar if algorithm == "astar" else dijkstra
    remaining = list(survivors)
    current = start
    plan = []

    while remaining:
        best = None
        for s in remaining:
            path, cost, _ = solve(grid, current, s)
            if path is not None and (best is None or cost < best[1]):
                best = (s, cost, path)

        if best is None:
            # No reachable survivors left (all blocked off) — stop here.
            break

        survivor, cost, path = best
        plan.append({"survivor": survivor, "cost": cost, "path": path})
        remaining.remove(survivor)
        current = survivor

    return plan


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Example disaster map: hazard-detection labels per grid cell
    # (in practice these come from classifying drone footage frame-by-frame
    # or region-by-region and mapping GPS coordinates to grid cells)
    label_grid = [
        ["normal",             "normal",             "normal", "normal", "normal"],
        ["normal",             "fire",               "fire",   "normal", "flood"],
        ["normal",             "landslide_debris",   "normal", "normal", "flood"],
        ["earthquake_damage",  "normal",             "normal", "normal", "normal"],
        ["normal",             "normal",             "normal", "normal", "normal"],
    ]
    cost_grid = hazard_labels_to_cost_grid(label_grid)

    drone_start = (0, 0)
    survivor_location = (4, 4)

    print("=== Single survivor: A* vs Dijkstra ===")
    for name, fn in [("A*", astar), ("Dijkstra", dijkstra)]:
        t0 = time.perf_counter()
        path, cost, explored = fn(cost_grid, drone_start, survivor_location)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"\n{name}:")
        print(f"  Path: {path}")
        print(f"  Total cost: {cost:.2f}")
        print(f"  Nodes explored: {explored}")
        print(f"  Time: {elapsed_ms:.3f} ms")

    print("\n=== Multiple survivors: rescue order plan (A*) ===")
    survivors = [(4, 4), (2, 0), (0, 4)]
    plan = plan_rescue_order(cost_grid, drone_start, survivors, algorithm="astar")
    for i, leg in enumerate(plan, 1):
        print(f"Leg {i}: go to survivor at {leg['survivor']} — cost {leg['cost']:.2f} — path {leg['path']}")
