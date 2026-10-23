"""Standard-view overlay shared by the design viewer (result_view) and the ParaView renders (paraview_view).

The six orthographic views are laid out as an unfolded cube (third-angle arrangement: Top above
Front, Left and Right beside it, Back at the end, Bottom below) with the isometric view in the free
corner. Keys 1-7 select front, back, left, right, top, bottom and isometric while the frame has focus;
a toggle folds the buttons away. Directions follow render_paraview.py (VIEWS): y up, front looks along
-z, top along -y with -z up, bottom along +y with +z up, right along -x, left along +x, isometric from
(1, 1, 1).

Each component concatenates JS into its module and CSS into its own <style> element. The overlay is
styled from the Streamlit theme (--st-* variables) on themed chrome, or with fixed light colours
("paper") where it sits on a white ParaView render.
"""

CSS = r"""
.hud { position: absolute; top: 10px; right: 10px; z-index: 2; background: var(--hud-bg); border: 1px solid var(--hud-line);
       border-radius: 8px; padding: 6px 7px 7px; box-shadow: 0 1px 4px #0000001f; color: var(--hud-ink);
       font-family: var(--st-font, 'Source Sans 3', system-ui, sans-serif); }
.hud-paper { --hud-bg: #ffffffeb; --hud-line: #c9d1db; --hud-ink: #1f2328; --hud-muted: #57606a; --hud-tile: #f3f5f8;
             --hud-tile-line: #b8c2ce; --hud-hover: #e2eaf4; --hud-accent: #1f4e8c; --hud-on-accent: #ffffff; }
.hud-themed { --hud-bg: color-mix(in srgb, var(--st-background-color, #ffffff) 90%, transparent);
              --hud-line: var(--st-border-color, #c9d1db); --hud-ink: var(--st-text-color, #1f2328);
              --hud-muted: color-mix(in srgb, var(--st-text-color, #1f2328) 65%, transparent);
              --hud-tile: var(--st-secondary-background-color, #f3f5f8); --hud-tile-line: var(--st-border-color, #b8c2ce);
              --hud-hover: color-mix(in srgb, var(--st-primary-color, #1f4e8c) 16%, var(--st-secondary-background-color, #f3f5f8));
              --hud-accent: var(--st-primary-color, #1f4e8c); --hud-on-accent: var(--st-background-color, #ffffff); }
.hud-head { display: flex; align-items: center; justify-content: space-between; gap: 10px; font-size: 11px;
            font-weight: 600; letter-spacing: .05em; text-transform: uppercase; color: var(--hud-muted); }
.hud-fold { border: 0; background: none; padding: 0 2px; font: inherit; font-size: 13px; line-height: 1;
            color: var(--hud-muted); cursor: pointer; }
.hud-net { display: grid; grid-template-columns: repeat(4, 46px); grid-template-rows: repeat(3, 26px); gap: 2px;
           margin-top: 5px; }
.hud-net[hidden] { display: none; }
.hud-tile { display: flex; align-items: center; justify-content: center; gap: 3px; padding: 0; font: inherit;
            font-size: 12px; color: var(--hud-ink); background: var(--hud-tile); border: 1px solid var(--hud-tile-line);
            border-radius: 3px; cursor: pointer; }
.hud-tile:hover:not(:disabled) { background: var(--hud-hover); border-color: var(--hud-accent); }
.hud-tile:focus-visible { outline: 2px solid var(--hud-accent); outline-offset: 1px; }
.hud-tile[aria-pressed="true"] { background: var(--hud-accent); border-color: var(--hud-accent); color: var(--hud-on-accent); }
.hud-tile:disabled { opacity: .35; cursor: not-allowed; }
.hud-iso { border-radius: 13px; }
"""

JS = r"""
// [view, grid column, grid row]: the unfolded cube, isometric in the free top-left corner.
const HUD_LAYOUT = [["isometric", 1, 1], ["top", 2, 1], ["left", 1, 2], ["front", 2, 2], ["right", 3, 2],
                    ["back", 4, 2], ["bottom", 2, 3]];
const HUD_KEYS = ["front", "back", "left", "right", "top", "bottom", "isometric"];   // keys 1-7
// Camera direction and screen-up axis of each view, as rendered by render_paraview.py (VIEWS).
const HUD_LOOK = {front: "looking along −z · y up", back: "looking along +z · y up", left: "looking along +x · y up",
                  right: "looking along −x · y up", top: "looking along −y · −z up", bottom: "looking along +y · +z up",
                  isometric: "looking along (−1, −1, −1) · y up"};
const HUD_ISO = `<svg viewBox="0 0 24 24" width="15" height="15" aria-hidden="true" fill="none" stroke="currentColor"
  stroke-width="1.7" stroke-linejoin="round"><path d="M12 3 20 7.5v9L12 21l-8-4.5v-9z"/><path d="M4 7.5 12 12l8-4.5M12 12v9"/></svg>`;
const hudTitle = v => v[0].toUpperCase() + v.slice(1);

// Builds the overlay in box, the focusable frame it sits in, and returns update(), to call whenever the
// shown view or the available views change. opts: state (keeps .open across reruns; unset, the buttons start
// folded in a frame narrower than 520 px, where they would cover most of it), paper (light styling
// on a white render), available(name), current() -> the shown standard view or null, select(name),
// missing(name) -> tooltip of a view that cannot be shown. The fold toggle sets the class "open" on box.
function viewHud(box, opts) {
  const hud = document.createElement("div");
  hud.className = "hud " + (opts.paper ? "hud-paper" : "hud-themed");
  hud.innerHTML = `<div class="hud-head"><span>View</span><button class="hud-fold" type="button"></button></div>`
    + `<div class="hud-net" role="toolbar" aria-label="Standard views"></div>`;
  box.appendChild(hud);
  if (opts.state.open === undefined) opts.state.open = !(box.clientWidth > 0 && box.clientWidth < 520);
  const fold = hud.querySelector(".hud-fold"), net = hud.querySelector(".hud-net"), tiles = {};
  for (const [name, col, row] of HUD_LAYOUT) {
    const tile = document.createElement("button");
    tile.type = "button"; tile.className = name === "isometric" ? "hud-tile hud-iso" : "hud-tile";
    tile.style.gridColumn = col; tile.style.gridRow = row;
    tile.innerHTML = name === "isometric" ? `${HUD_ISO}Iso` : hudTitle(name);
    tile.addEventListener("click", () => opts.select(name));
    net.appendChild(tile); tiles[name] = tile;
  }
  fold.addEventListener("click", () => { opts.state.open = !opts.state.open; update(); });
  box.addEventListener("keydown", ev => {
    const name = HUD_KEYS[+ev.key - 1];
    if (!name || ev.ctrlKey || ev.metaKey || ev.altKey || !opts.available(name)) return;
    ev.preventDefault(); opts.select(name);
  });

  function update() {
    const view = opts.current();
    for (const [name, tile] of Object.entries(tiles)) {
      tile.disabled = !opts.available(name);
      tile.title = tile.disabled ? opts.missing(name) : `${hudTitle(name)} view (${HUD_KEYS.indexOf(name) + 1}) · ${HUD_LOOK[name]}`;
      tile.setAttribute("aria-pressed", String(name === view));
    }
    net.hidden = !opts.state.open;
    box.classList.toggle("open", opts.state.open);
    fold.textContent = opts.state.open ? "▾" : "▸";
    fold.title = opts.state.open ? "Hide the view buttons" : "Show the view buttons";
    fold.setAttribute("aria-label", fold.title);
  }
  update();
  return update;
}
"""
