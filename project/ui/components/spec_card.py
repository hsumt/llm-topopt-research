"""The specification as role-coded cards: one card per section, each row label | value | role chip.

Draws the sections ui/spec_model.py builds. Colours come only from the theme's --st-* variables,
so the card follows a light/dark switch live. Every text is set as text, never parsed as HTML.
Below about 460 px the chip moves under its value.
"""
import streamlit as st

_JS = r"""
const CSS = `
.spec { container-type: inline-size; font-family: var(--st-font, 'Source Sans 3', sans-serif); color: var(--st-text-color);
        font-size: 15px; line-height: 1.4; }
.legend { display: flex; flex-wrap: wrap; gap: 6px; margin: 0 0 12px; }
.card { border: 1px solid var(--st-border-color-light, var(--st-border-color)); border-radius: var(--st-base-radius, 8px);
        margin: 0 0 12px; overflow: hidden; }
.head { display: flex; align-items: center; gap: 8px; padding: 8px 12px; font-weight: 600;
        background: var(--st-secondary-background-color); }
.head > .icon { font-size: 18px; opacity: .7; }
.counts { margin-left: auto; display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 0 10px; font-size: 12px;
          font-weight: 400; }
.counts span { opacity: .68; white-space: nowrap; }
.counts .missing { color: var(--st-orange-text-color); opacity: 1; }
.counts .unsupported { color: var(--st-red-text-color); opacity: 1; }
.row { display: grid; grid-template-columns: minmax(96px, 28%) minmax(0, 1fr) auto; align-items: center; gap: 2px 12px;
       padding: 6px 12px; border-top: 1px solid var(--st-border-color-light, var(--st-border-color)); }
.row.bare .value { grid-column: 2 / 4; }
.label { font-size: 13px; opacity: .68; overflow-wrap: anywhere; }
.value { min-width: 0; overflow-wrap: anywhere; font-variant-numeric: tabular-nums; }
.unit { margin-left: 4px; font-size: 13px; opacity: .62; }
.absent { font-style: italic; opacity: .7; }
.row.missing .absent { color: var(--st-orange-text-color); opacity: 1; }
.row.wide .value { font-size: 14px; font-style: italic; opacity: .75; }
.chip { justify-self: end; display: inline-flex; align-items: center; gap: 4px; max-width: 100%; padding: 1px 8px 1px 6px;
        border-radius: 999px; font-size: 12px; line-height: 18px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.chip.code { font-family: var(--st-code-font, monospace); font-size: 11.5px; }
.chip > span:last-child { overflow: hidden; text-overflow: ellipsis; }
.chip > .icon { font-size: 14px; }
.legend .chip { cursor: help; }
.icon { flex: none; display: inline-block; width: 1em; overflow: hidden; font-family: 'Material Symbols Rounded';
        font-weight: normal; font-style: normal; line-height: 1; letter-spacing: normal; text-transform: none;
        white-space: nowrap; direction: ltr; font-feature-settings: 'liga'; -webkit-font-smoothing: antialiased;
        user-select: none; }
.chip.input { color: var(--st-blue-text-color); background: var(--st-blue-background-color); }
.chip.checked { color: var(--st-gray-text-color); background: var(--st-gray-background-color); }
.chip.default { color: var(--st-yellow-text-color); background: var(--st-yellow-background-color); }
.chip.missing { color: var(--st-orange-text-color); background: var(--st-orange-background-color); }
.chip.unsupported { color: var(--st-red-text-color); background: var(--st-red-background-color); }
@container (max-width: 460px) {
  .row { grid-template-columns: minmax(84px, 34%) minmax(0, 1fr); }
  .row .chip { grid-column: 2; justify-self: start; }
  .row.bare .value { grid-column: 2; }
  .head { flex-wrap: wrap; }
}
`;
// role: [Material icon, legend text, legend tooltip, count word (one, many)]; one colour per role, set in CSS.
const ROLES = {
  input: ["input", "solver input", "The value goes into the solver configuration; the chip names the config field.", "input", "inputs"],
  checked: ["check", "checked", "A choice the connected solver requires; the code check compares it. No number goes in.", "checked", "checked"],
  default: ["tune", "solver default", "Fixed by the codebase, not by the specification.", "default", "defaults"],
  missing: ["help", "needs decision", "Missing or unresolved: a decision or data is needed before the solver can run.", "to resolve", "to resolve"],
  unsupported: ["block", "unsupported", "No solver capability expresses this yet.", "unsupported", "unsupported"],
};

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

function icon(name) {
  const node = el("span", "icon", name);
  node.setAttribute("aria-hidden", "true");
  return node;
}

function chip(role, text, code) {
  const node = el("span", `chip ${role}${code ? " code" : ""}`);
  node.append(icon(ROLES[role][0]), el("span", "", text));
  return node;
}

export default function (component) {
  const { data, parentElement } = component;
  const sections = (data && data.sections) || [];
  const style = el("style");
  style.textContent = CSS;   // in the module, not registered: the root is rebuilt on every call
  const root = el("div", "spec");

  const legend = el("div", "legend");
  for (const [role, [, text, tip]] of Object.entries(ROLES)) {
    const item = chip(role, text, false);
    item.title = tip;
    legend.appendChild(item);
  }
  root.appendChild(legend);

  for (const section of sections) {
    const card = el("section", "card");
    const head = el("div", "head");
    head.append(icon(section.icon || "notes"), el("span", "", section.title));
    const counts = el("span", "counts");
    for (const [role, n] of Object.entries(section.counts || {})) {
      if (!ROLES[role] || !n) continue;
      counts.appendChild(el("span", role, `${n} ${n === 1 ? ROLES[role][3] : ROLES[role][4]}`));
    }
    head.appendChild(counts);
    card.appendChild(head);
    for (const row of section.rows || []) {
      const hasChip = Boolean(ROLES[row.role] && row.chip);
      const line = el("div", `row ${row.role}${row.wide ? " wide" : ""}${hasChip ? "" : " bare"}`);
      if (row.tip) line.title = row.tip;
      const value = el("div", "value");
      // spec_model writes an absent value as exactly "not stated".
      value.appendChild(el("span", row.value === "not stated" ? "absent" : "", row.value));
      if (row.unit) value.appendChild(el("span", "unit", row.unit));
      line.append(el("div", "label", row.label), value);
      if (hasChip) line.appendChild(chip(row.role, row.chip, row.role === "input" || row.role === "default"));
      card.appendChild(line);
    }
    root.appendChild(card);
  }
  parentElement.replaceChildren(style, root);
}
"""


def _component(**kwargs):
    # Registered on each render: registration belongs to the running app, which tests and reruns may replace.
    return st.components.v2.component("ato_spec_card", js=_JS)(**kwargs)


def render(sections: list[dict], *, key: str) -> None:
    """sections: from spec_model.build. key: unique on the page. The height follows the content."""
    _component(data={"sections": sections}, key=f"spec_card_{key}")
