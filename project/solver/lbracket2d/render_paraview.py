"""Render a 2D L-bracket run with ParaView, in element form. Run with pvbatch, not the app's Python.

    pvbatch --opengl-window-backend OSMesa render_paraview.py <solver_dir> <out_dir> [iter,iter,...]

Reads the solver's density.xdmf (DOLFINx DG0, one value per element) as cell data, places one
cube per element with material (box glyph at the cell centre, one element deep, no
interpolation or smoothing), colours it by density and writes one PNG per standard view per
requested iteration plus manifest.json. Uses only paraview.simple, so it never imports the app's
packages.
"""
import json
import math
import os
import sys

from paraview.simple import (CellCenters, ColorBy, GetActiveViewOrCreate, GetColorTransferFunction,
                             GetScalarBar, Glyph, Outline, Render, SaveScreenshot, Show,
                             Threshold, UpdatePipeline, Xdmf3ReaderS, _DisableFirstRenderCameraReset)

solver_dir, out_dir = sys.argv[1], sys.argv[2]
summary = json.load(open(os.path.join(solver_dir, "summary.json")))
config = summary["input"]
nelx = summary["mesh_shape"][0]
h = config["lx_m"] / nelx
last = summary["completed_iterations"] - 1
wanted = [int(i) for i in sys.argv[3].split(",")] if len(sys.argv) > 3 else sorted({0, *range(50, last, 50), last})
os.makedirs(out_dir, exist_ok=True)

reader = Xdmf3ReaderS(FileName=[os.path.join(solver_dir, "states", "density.xdmf")])
reader.UpdatePipelineInformation()
times = list(reader.TimestepValues) if reader.TimestepValues else [0.0]
material = Threshold(Input=reader, Scalars=["CELLS", "f"], LowerThreshold=1e-3, UpperThreshold=1.0)
centers = CellCenters(Input=material)
centers.VertexCells = 1
cubes = Glyph(Input=centers, GlyphType="Box")
cubes.OrientationArray = ["POINTS", "No orientation array"]
cubes.ScaleArray = ["POINTS", "No scale array"]
cubes.ScaleFactor = 1.0
cubes.GlyphType.XLength, cubes.GlyphType.YLength, cubes.GlyphType.ZLength = h, h, h
cubes.GlyphMode = "All Points"
frame = Outline(Input=reader)

view = GetActiveViewOrCreate("RenderView")
view.ViewSize = [1100, 900]
view.Background = [1, 1, 1]
try:
    view.UseColorPaletteForBackground = 0
except AttributeError:
    pass
view.OrientationAxesVisibility = 1
shown = Show(cubes, view)
ColorBy(shown, ("POINTS", "f"))
lut = GetColorTransferFunction("f")
lut.ApplyPreset("Blues", True)
lut.InvertTransferFunction()  # solid elements (density 1) dark, partial boundary elements light
lut.RescaleTransferFunction(0.0, 1.0)
shown.SetScalarBarVisibility(view, True)
bar = GetScalarBar(lut, view)
bar.Title, bar.ComponentTitle = "element density", ""
bar.TitleColor = bar.LabelColor = [0.1, 0.1, 0.1]
view.OrientationAxesLabelColor = [0.1, 0.1, 0.1]
shown.Representation = "Surface With Edges"
shown.EdgeColor = [0.15, 0.2, 0.3]
outline = Show(frame, view)
outline.DiffuseColor = outline.AmbientColor = [0.55, 0.6, 0.67]

lx, ly = config["lx_m"], config["ly_m"]
# Standard views, CAD convention: y up, the front view looks along -z at the design plane.
# name: (direction from the focal point to the camera, view up)
VIEWS = {
    "isometric": ((1, 1, 1), (0, 1, 0)),
    "front": ((0, 0, 1), (0, 1, 0)),
    "back": ((0, 0, -1), (0, 1, 0)),
    "top": ((0, 1, 0), (0, 0, -1)),
    "bottom": ((0, -1, 0), (0, 0, 1)),
    "right": ((1, 0, 0), (0, 1, 0)),
    "left": ((-1, 0, 0), (0, 1, 0)),
}
center = [lx / 2, ly / 2, 0.0]
radius = 0.5 * math.sqrt(lx ** 2 + ly ** 2 + h ** 2)   # bounding sphere of the design space, one element deep
_DisableFirstRenderCameraReset()   # Render() would otherwise reset the camera, and the scale below, once
view.CameraParallelProjection = 1
view.CameraParallelScale = 1.05 * radius   # one scale for every view and frame, so they compare directly

written = []
for iteration in wanted:
    if iteration < 0 or iteration > last:
        continue
    t = min(times, key=lambda v: abs(v - iteration))
    view.ViewTime = t
    UpdatePipeline(time=t, proxy=cubes)
    files = {}
    for name, (direction, up) in VIEWS.items():
        norm = math.sqrt(sum(c * c for c in direction))
        view.CameraFocalPoint = center
        view.CameraPosition = [center[i] + 4 * radius * direction[i] / norm for i in range(3)]
        view.CameraViewUp = list(up)
        Render(view)
        files[name] = f"elements_{iteration:04d}_{name}.png"
        SaveScreenshot(os.path.join(out_dir, files[name]), view, ImageResolution=[1100, 900])
    written.append({"iteration": iteration, "views": files})
json.dump({"renderer": "ParaView pvbatch", "source": "states/density.xdmf (cell data f)", "frames": written,
           "views": list(VIEWS), "camera": "orthographic; y up; front looks along -z; one scale for all views",
           "representation": "one box glyph per element with density > 0.001, edge = element size, no smoothing"},
          open(os.path.join(out_dir, "manifest.json"), "w"), indent=2)
print("rendered", len(written), "frames")
