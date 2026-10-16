"""Render a 2D L-bracket run with ParaView, in element form. Run with pvbatch, not the app's Python.

    pvbatch --opengl-window-backend OSMesa render_paraview.py <solver_dir> <out_dir> [iter,iter,...]

Reads the solver's density.xdmf (DOLFINx DG0, one value per element) as cell data, places one
cube per element with material (box glyph at the cell centre, one element deep, no
interpolation or smoothing), colours it by density and writes one PNG per requested iteration
plus manifest.json. Uses only paraview.simple, so it never imports the app's packages.
"""
import json
import os
import sys

from paraview.simple import (CellCenters, ColorBy, GetActiveViewOrCreate, GetColorTransferFunction,
                             GetScalarBar, Glyph, Outline, Render, SaveScreenshot, Show,
                             Threshold, UpdatePipeline, Xdmf3ReaderS)

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
view.CameraFocalPoint = [lx / 2, ly / 2, 0]
view.CameraPosition = [lx * 1.25, -ly * 0.95, lx * 1.35]
view.CameraViewUp = [0, 0, 1]
view.CameraParallelProjection = 0

written = []
for iteration in wanted:
    if iteration < 0 or iteration > last:
        continue
    t = min(times, key=lambda v: abs(v - iteration))
    view.ViewTime = t
    UpdatePipeline(time=t, proxy=cubes)
    Render(view)
    name = f"elements_{iteration:04d}.png"
    SaveScreenshot(os.path.join(out_dir, name), view, ImageResolution=[1100, 900])
    written.append({"iteration": iteration, "file": name})
json.dump({"renderer": "ParaView pvbatch", "source": "states/density.xdmf (cell data f)", "frames": written,
           "representation": "one box glyph per element with density > 0.001, edge = element size, no smoothing"},
          open(os.path.join(out_dir, "manifest.json"), "w"), indent=2)
print("rendered", len(written), "frames")
