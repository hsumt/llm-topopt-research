"""ParaView renders of one design in the standard views, with an overlay to switch between them.

pvbatch renders every view (render_paraview.py); this only picks which image to show, in the
browser, so switching is instant and the chosen view stays put while the iteration changes.
The overlay (view_hud) lays the six orthographic views out as an unfolded cube with the isometric
view in the free corner; keys 1-7 select front, back, left, right, top, bottom and isometric while
the view has focus. The render is a white "paper" figure in either theme; only the frame and the
caption follow the Streamlit theme.
"""
import base64
import json
from pathlib import Path

import streamlit as st

from project.ui.components import view_hud

_CSS = r"""
.pv { position: relative; aspect-ratio: 1100 / 900; background: #fff; border: 1px solid var(--st-border-color, #d6d6d6);
      border-radius: 10px; overflow: hidden; outline: none; color: #1f2328;
      font-family: var(--st-font, 'Source Sans 3', system-ui, sans-serif); }
.pv:focus-visible { box-shadow: 0 0 0 2px color-mix(in srgb, var(--st-primary-color, #1f4e8c) 50%, transparent); }
.pv img { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: contain; }
.pv img[hidden] { display: none; }
.badge { position: absolute; top: 10px; left: 10px; background: #ffffffe0; padding: 3px 8px;
         border-radius: 6px; font-size: 13px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
         max-width: calc(100% - 100px); }
.pv.open .badge { max-width: calc(100% - 246px); }   /* stops short of the open view panel */
.badge b { font-weight: 600; }
.caption { margin-top: 6px; text-align: center; font-size: 14px; opacity: .7;
           font-family: var(--st-font, 'Source Sans 3', system-ui, sans-serif); color: var(--st-text-color, inherit); }
"""

_MODULE = r"""
const states = {};

export default function (component) {
  const { data, parentElement } = component;
  const key = data.key || "paraview", images = data.images || {}, names = Object.keys(images);
  const state = states[key] || (states[key] = {view: "isometric"});
  const current = () => images[state.view] ? state.view : images.isometric ? "isometric" : names[0];

  parentElement.innerHTML = "";
  const style = document.createElement("style");
  style.textContent = CSS;   // in the module, not registered: clearing parentElement would drop a registered sheet
  const box = document.createElement("div");
  box.className = "pv"; box.tabIndex = 0;
  box.setAttribute("aria-label", "ParaView render. Keys 1 to 7 select front, back, left, right, top, bottom and isometric views.");
  const caption = document.createElement("div");
  caption.className = "caption"; caption.textContent = data.caption || "";
  parentElement.append(style, box, caption);

  const imgs = {};
  for (const name of names) {
    const img = document.createElement("img");
    img.src = images[name]; img.alt = `ParaView render, ${name} view`; img.draggable = false;
    box.appendChild(img); imgs[name] = img;
  }
  const badge = document.createElement("div");
  badge.className = "badge";
  box.appendChild(badge);
  const hud = viewHud(box, {state, paper: true, available: name => !!images[name], current,
                            select: name => { state.view = name; update(); },
                            missing: name => `${hudTitle(name)} view was not rendered for this run`});

  function update() {
    const view = current();
    for (const name of names) imgs[name].hidden = name !== view;
    hud();
    badge.innerHTML = HUD_LOOK[view] ? `<b>${hudTitle(view)}</b> · ${HUD_LOOK[view]} · orthographic`
      : "Single view from an earlier render; re-render the run for the standard views";
    badge.title = badge.textContent;   // the full text when a narrow view cuts it short
  }
  update();
}
"""

_JS = view_hud.JS + f"const CSS = {json.dumps(view_hud.CSS + _CSS)};\n" + _MODULE


def _component(**kwargs):
    # Registered on each render: registration belongs to the running app, which tests and reruns may replace.
    return st.components.v2.component("ato_paraview_view", js=_JS)(**kwargs)


def render(views: dict[str, str], *, caption: str, key: str) -> None:
    """views: view name -> PNG path, from the run's paraview/manifest.json. key: one per run, so the
    chosen view is kept while the iteration changes."""
    images = {}
    for name, path in views.items():
        try:
            images[name] = "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()
        except OSError:
            continue
    _component(data={"images": images, "caption": caption, "key": key}, key=f"paraview_{key}")
