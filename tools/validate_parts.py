"""Reverse-engineer every sample part and grade it with the accuracy analyzer.

    python tools/validate_parts.py [3.stl 4.stl ...] [--tolerance 0.1] [--heatmaps DIR]

For each mesh all strategies of :func:`meshrev.core.workflows.reconstruct` are
run (revolve, layered extrusion, zone-wise hybrid); a Markdown table of the
area-weighted deviation statistics is printed, best strategy first.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from meshrev import io as mio
from meshrev.core import workflows as wf
from meshrev.core.samples import make_demo_piston

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="*", help="meshes (default: demo + ./[0-9]*.stl)")
    parser.add_argument("--tolerance", type=float, default=0.1)
    parser.add_argument("--max-range", type=float, default=1.0)
    parser.add_argument("--heatmaps", type=Path, help="write a PNG heat map per part here")
    args = parser.parse_args()
    files = args.files or ["demo", *sorted(str(p) for p in ROOT.glob("[0-9]*.stl"))]

    tol = f"±{args.tolerance:.2f}"
    print(
        f"| 零件 | 策略 | 得分 (全部点 {tol} 内) | 范围内点 {tol} 内 | 超范围 | RMS "
        "| 最大 + | 最大 − | 有效实体 | 用时 |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name in files:
        start = time.time()
        mesh = make_demo_piston(voxel_size=0.5) if name == "demo" else mio.load(name)[0].polydata
        results = wf.reconstruct(mesh, args.tolerance, args.max_range)
        elapsed = time.time() - start
        for i, r in enumerate(results):
            s = r.deviation.stats
            share_out = s.out_of_range / max(s.count + s.out_of_range, 1)
            solids = r.body.kernel.topology_counts(r.body.shape)["solids"]
            first = i == 0  # best strategy: name, bold method and total time
            method = f"**{r.method}**" if first else r.method
            print(
                f"| {Path(name).name if first else ''} | {method} | "
                f"{100 * r.score():.1f}% | {100 * s.within_tolerance:.1f}% | "
                f"{100 * share_out:.1f}% | {s.rms:.3f} | "
                f"{s.max_positive:+.3f} | {s.max_negative:+.3f} | "
                f"{'是' if r.valid else '否'} ({solids}) | {f'{elapsed:.0f}s' if first else ''} |"
            )
        if args.heatmaps and results:
            import pyvista as pv

            from meshrev.core.deviation import render_heatmap

            args.heatmaps.mkdir(parents=True, exist_ok=True)
            plotter = pv.Plotter(off_screen=True, shape=(1, 2), window_size=(1600, 700))
            for k, view in enumerate([(1, -1, 1), (-1, 1, -0.7)]):
                plotter.subplot(0, k)
                render_heatmap(results[0].deviation, mesh, plotter=plotter)
                plotter.view_vector(view)
            plotter.screenshot(str(args.heatmaps / f"{Path(name).stem}_deviation.png"))
            plotter.close()


if __name__ == "__main__":
    main()
