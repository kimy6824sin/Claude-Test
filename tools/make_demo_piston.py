"""Write the procedural demo piston to a mesh file.

Example: ``python tools/make_demo_piston.py demo_piston.stl --voxel 0.5 --noise 0.01``
"""

from __future__ import annotations

import argparse

import numpy as np

from meshrev import io as mio
from meshrev.core.bodies import MeshBody
from meshrev.core.samples import make_demo_piston


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", help="output file (.stl / .obj / .ply)")
    parser.add_argument("--voxel", type=float, default=0.5, help="marching cubes voxel size [mm]")
    parser.add_argument(
        "--noise", type=float, default=0.0, help="gaussian vertex noise sigma [mm] (scanner-like)"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    mesh = make_demo_piston(voxel_size=args.voxel)
    if args.noise > 0:
        rng = np.random.default_rng(args.seed)
        mesh.points = mesh.points + rng.normal(scale=args.noise, size=mesh.points.shape)
    mio.save([MeshBody(mesh, "demo_piston")], args.output)
    print(f"wrote {args.output}: {mesh.n_points:,} vertices, {mesh.n_cells:,} triangles")


if __name__ == "__main__":
    main()
