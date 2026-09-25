"""meshrev - mesh-to-CAD reverse engineering workbench for piston engine parts.

Package layout (dependencies only point downwards)::

    gui  ->  io  ->  core

``core`` is pure Python (numpy/scipy/pyvista, optional OCP) and never imports Qt,
so every algorithm can be used headless from scripts and tests.
"""

__version__ = "0.1.0"
