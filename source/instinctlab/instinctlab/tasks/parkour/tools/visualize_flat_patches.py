# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Utility to visualize the flat patches sampled by the terrain generator's flat patch sampler.

This script builds the terrain generator used by a parkour task config (without spawning a robot
or running the simulation loop), and plots a top-down view of the generated terrain mesh together
with the sampled flat patches (e.g. the "target" patches used by the pose/velocity command).

Usage:
    ./isaaclab.sh -p source/instinctlab/instinctlab/tasks/parkour/tools/visualize_flat_patches.py \
        --task Instinct-Parkour-Target-Amp-G1-v0 --patch_key target --output flat_patches.png
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Visualize flat patches sampled by the terrain generator.")
parser.add_argument("--task", type=str, required=True, help="Name of the task whose terrain config should be used.")
parser.add_argument(
    "--patch_key", type=str, default="target", help="Key of the flat patch sampling group to visualize."
)
parser.add_argument("--output", type=str, default="flat_patches.png", help="Path to save the output image.")
parser.add_argument("--show", action="store_true", default=False, help="Show the plot in an interactive window.")
parser.add_argument("--patch_size", type=float, default=4.0, help="Marker size for the flat patch points.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.tri import Triangulation

from isaaclab.terrains import TerrainGenerator
from isaaclab_tasks.utils import parse_env_cfg

import instinctlab.tasks  # noqa: F401


def main():
    # parse the environment configuration to get the terrain generator config used by the task
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    terrain_gen_cfg = env_cfg.scene.terrain.terrain_generator
    if terrain_gen_cfg is None:
        raise RuntimeError(f"Task '{args_cli.task}' does not use a procedurally generated terrain.")

    # generate the terrain (this also samples the flat patches)
    terrain_generator = TerrainGenerator(terrain_gen_cfg, device="cpu")

    if args_cli.patch_key not in terrain_generator.flat_patches:
        raise RuntimeError(
            f"Patch key '{args_cli.patch_key}' not found in the generated flat patches."
            f" Available keys: {list(terrain_generator.flat_patches.keys())}"
        )
    # shape: (num_rows, num_cols, num_patches, 3)
    flat_patches = terrain_generator.flat_patches[args_cli.patch_key].numpy()

    # build a triangulation of the terrain mesh for a top-down height plot
    mesh = terrain_generator.terrain_mesh
    vertices = mesh.vertices
    triangulation = Triangulation(vertices[:, 0], vertices[:, 1], mesh.faces)

    fig, ax = plt.subplots(figsize=(12, 12))
    height_plot = ax.tripcolor(triangulation, vertices[:, 2], cmap="terrain", shading="gouraud")
    fig.colorbar(height_plot, ax=ax, label="Height [m]", fraction=0.03)

    # overlay the sampled flat patches, colored by terrain row (i.e. curriculum/difficulty level)
    num_rows, num_cols = flat_patches.shape[:2]
    points = flat_patches.reshape(-1, 3)
    row_ids = np.repeat(np.arange(num_rows), num_cols * flat_patches.shape[2])
    scatter = ax.scatter(
        points[:, 0],
        points[:, 1],
        c=row_ids,
        cmap="autumn",
        s=args_cli.patch_size,
        edgecolors="none",
    )
    fig.colorbar(scatter, ax=ax, label="Terrain row (level)", fraction=0.03)

    # draw the sub-terrain grid boundaries
    # rows are laid out along x, columns are laid out along y
    size_x, size_y = terrain_gen_cfg.size
    x_min = terrain_generator.terrain_origins[0, 0, 0] - size_x / 2
    y_min = terrain_generator.terrain_origins[0, 0, 1] - size_y / 2
    for row in range(1, num_rows):
        ax.axvline(x_min + row * size_x, color="black", linewidth=0.3, alpha=0.5)
    for col in range(1, num_cols):
        ax.axhline(y_min + col * size_y, color="black", linewidth=0.3, alpha=0.5)

    # label each column with its sub-terrain name (assumes curriculum ordering)
    if terrain_gen_cfg.curriculum:
        proportions = np.array([sub_cfg.proportion for sub_cfg in terrain_gen_cfg.sub_terrains.values()])
        proportions /= np.sum(proportions)
        sub_terrain_names = list(terrain_gen_cfg.sub_terrains.keys())
        x_label = x_min - 0.5
        for col in range(num_cols):
            sub_index = np.min(np.where(col / num_cols + 0.001 < np.cumsum(proportions))[0])
            y_center = terrain_generator.terrain_origins[0, col, 1]
            ax.text(
                x_label,
                y_center,
                sub_terrain_names[sub_index],
                ha="right",
                va="center",
                fontsize=6,
            )

    ax.set_aspect("equal")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title(f"Flat patches ('{args_cli.patch_key}') sampled by the terrain generator")

    fig.tight_layout()
    fig.savefig(args_cli.output, dpi=200)
    print(f"[INFO] Saved flat patch visualization to: {args_cli.output}")

    if args_cli.show:
        plt.show()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
