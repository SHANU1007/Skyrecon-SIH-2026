# Path Planning — Best Rescue Route to Survivors

Given where survivors were spotted (human-detection) and what hazards are on
the ground (hazard-detection), this module computes the safest/cheapest path
for a rescue team to reach them — using **A\*** and **Dijkstra**.

## How it fits with the rest of the project

```
drone-ai-project/  -->  finds WHERE the survivors are (pixel/GPS coords)
hazard-detection/  -->  finds WHAT hazards are on the ground (per region)
path-planning/     -->  finds the BEST ROUTE from rescue team to survivors,
                         avoiding fire/flood and minimizing risk through debris
```

## Files

- **`path_planner.py`** — core algorithms:
  - `dijkstra(grid, start, goal)` — classic guaranteed-shortest-cost search
  - `astar(grid, start, goal)` — same optimal cost, but explores fewer nodes
    (faster) using a distance heuristic to guide the search toward the goal
  - `plan_rescue_order(grid, start, survivors)` — for multiple survivors,
    greedily plans the order to rescue them in (nearest-cost-first)
  - `hazard_labels_to_cost_grid(label_grid)` — converts hazard-detection
    output labels into a numeric cost grid the algorithms can use

- **`visualize_path.py`** — draws the hazard map + planned path(s) and saves
  `rescue_path.png` for your presentation/report.

## Run it

```bash
pip install -r requirements.txt
python path_planner.py       # prints paths, costs, nodes explored, timing
python visualize_path.py     # saves rescue_path.png
```

## How the grid/cost map works

Each grid cell holds a **cost** — how expensive/risky it is for a rescue team
to pass through:

| Hazard label         | Cost       | Meaning                          |
|-----------------------|-----------|-----------------------------------|
| `normal`              | 1         | Clear terrain                     |
| `earthquake_damage`   | 6         | Passable but slow/risky           |
| `landslide_debris`    | 8         | Harder to pass, higher risk       |
| `flood`                | 12        | Very risky, avoided unless needed |
| `fire`                 | infinity | Never routed through              |

Tune `HAZARD_COST_MAP` in `path_planner.py` to match real-world risk
assessment (e.g. flood might be less costly than debris if the rescue team
has a boat).

## A* vs Dijkstra — why both?

Both are guaranteed to find the cheapest path here (the A* heuristic used —
straight-line distance — never overestimates the true cost, so it stays
optimal). The difference is **speed**: A* explores far fewer nodes because it
uses the heuristic to search toward the goal instead of expanding outward in
every direction like Dijkstra. Running `path_planner.py` prints a
side-by-side comparison of nodes explored and time taken for both, on the
same map — good numbers to quote in your PPT/demo.

## Connecting to real map data

Right now the grid is a small hand-made example. For a real deployment:
1. Divide the drone's aerial view / GPS-mapped area into a grid of cells.
2. For each cell, run the hazard-detection classifier on the corresponding
   image region to get a label.
3. Feed that `label_grid` into `hazard_labels_to_cost_grid()`.
4. Feed survivor GPS coordinates (converted to grid cells) into
   `plan_rescue_order()` to get the route.
5. Send the resulting path to the `dashboard/` for the rescue team to follow
   (not wired up yet — see the main project README's integration notes).
