"""Rendering styles for surface bodies."""

from __future__ import annotations

from enum import Enum

import vtk

MESH_COLOR = (0.72, 0.76, 0.82)
EDGE_COLOR = (0.18, 0.20, 0.24)
HIGHLIGHT_COLOR = (1.0, 0.78, 0.10)
BACKGROUND_TOP = (0.93, 0.95, 0.98)
BACKGROUND_BOTTOM = (0.62, 0.68, 0.76)


class DisplayMode(str, Enum):
    WIREFRAME = "wireframe"
    SHADED = "shaded"  # flat shading: every triangle visible
    SMOOTH = "smooth"  # Phong shading with split vertex normals

    @property
    def label(self) -> str:
        return {
            DisplayMode.WIREFRAME: "线框",
            DisplayMode.SHADED: "着色",
            DisplayMode.SMOOTH: "平滑",
        }[self]


def apply_display_mode(prop: vtk.vtkProperty, mode: DisplayMode, show_edges: bool) -> None:
    """Configure a surface actor's property for ``mode``."""
    if mode is DisplayMode.WIREFRAME:
        prop.SetRepresentationToWireframe()
        prop.SetInterpolationToFlat()
        prop.EdgeVisibilityOff()
        prop.LightingOff()
        prop.SetLineWidth(1.0)
        return
    prop.SetRepresentationToSurface()
    prop.LightingOn()
    if mode is DisplayMode.SHADED:
        prop.SetInterpolationToFlat()
    else:
        prop.SetInterpolationToPhong()
    prop.SetEdgeVisibility(bool(show_edges))
    prop.SetEdgeColor(*EDGE_COLOR)
    prop.SetAmbient(0.12)
    prop.SetDiffuse(0.85)
    prop.SetSpecular(0.25)
    prop.SetSpecularPower(30.0)
