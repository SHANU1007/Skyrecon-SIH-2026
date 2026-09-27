"""
Visualize the hazard grid + planned rescue path(s) and save as a PNG.
Useful for the SIH demo/presentation — shows the drone/rescue-team route
avoiding fire and flood, routing around debris where possible.

Usage:
    python visualize_path.py
Output:
    rescue_path.png (in the same folder)
"""

import matplotlib
matplotlib.use("Agg")  # no display needed, just save to file
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from path_planner import (
    hazard_labels_to_cost_grid,
    astar,
    plan_rescue_order,
)

# Same demo map as path_planner.py
label_grid = [
    ["normal",             "normal",             "normal", "normal", "normal"],
    ["normal",             "fire",               "fire",   "normal", "flood"],
    ["normal",             "landslide_debris",   "normal", "normal", "flood"],
    ["earthquake_damage",  "normal",             "normal", "normal", "normal"],
    ["normal",             "normal",             "normal", "normal", "normal"],
]

COLOR_MAP = {
    "normal": "#2ecc71",             # green
    "earthquake_damage": "#f1c40f",  # yellow
    "landslide_debris": "#e67e22",   # orange
    "flood": "#3498db",              # blue
    "fire": "#e74c3c",               # red
}

drone_start = (0, 0)
survivors = [(4, 4), (2, 0), (0, 4)]


def draw():
    cost_grid = hazard_labels_to_cost_grid(label_grid)
    plan = plan_rescue_order(cost_grid, drone_start, survivors, algorithm="astar")

    rows, cols = len(label_grid), len(label_grid[0])
    fig, ax = plt.subplots(figsize=(7, 7))

    # draw grid cells colored by hazard type
    for r in range(rows):
        for c in range(cols):
            label = label_grid[r][c]
            ax.add_patch(mpatches.Rectangle((c, rows - 1 - r), 1, 1,
                                             facecolor=COLOR_MAP[label],
                                             edgecolor="white", linewidth=1))

    # draw each rescue leg's path
    path_colors = ["black", "dimgray", "gray"]
    for i, leg in enumerate(plan):
        path = leg["path"]
        xs = [c + 0.5 for (r, c) in path]
        ys = [rows - 1 - r + 0.5 for (r, c) in path]
        ax.plot(xs, ys, color=path_colors[i % len(path_colors)], linewidth=3,
                 marker="o", markersize=4, label=f"Leg {i+1} -> survivor {leg['survivor']} (cost {leg['cost']:.1f})")

    # mark drone start
    sr, sc = drone_start
    ax.plot(sc + 0.5, rows - 1 - sr + 0.5, marker="^", markersize=16,
             color="black", markeredgecolor="white", label="Drone / rescue start")

    # mark survivors
    for (sr, sc) in survivors:
        ax.plot(sc + 0.5, rows - 1 - sr + 0.5, marker="*", markersize=20,
                 color="gold", markeredgecolor="black", label="_nolegend_")

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Rescue Path Planning — A* routing around hazards")

    # legend: hazard colors + path info
    hazard_patches = [mpatches.Patch(color=col, label=name.replace("_", " ").title())
                       for name, col in COLOR_MAP.items()]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=hazard_patches + handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=8)

    plt.tight_layout()
    out_path = "rescue_path.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved visualization to {out_path}")


if __name__ == "__main__":
    draw()
