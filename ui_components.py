"""
ui_components.py
================
Small helpers that build the custom HTML pieces of the interface (hero banner,
stat cards, status indicator, legend, concept cards...). Styling lives in
assets/style.css.

Note: Streamlit renders these with st.markdown(..., unsafe_allow_html=True).
Markdown treats indented lines as code blocks, so every helper returns HTML
with no leading indentation (see ``_clean``).
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

STATUS_CLASS = {
    "Not Started": "not-started",
    "Training": "training",
    "Paused": "paused",
    "Completed": "completed",
}


def _clean(markup: str) -> str:
    """Strip indentation and blank lines so Markdown never sees a code block."""
    return "".join(line.strip() for line in markup.splitlines() if line.strip())


def esc(text) -> str:
    return html.escape(str(text), quote=True)


def load_css_text(path: Path) -> str:
    try:
        return f"<style>{Path(path).read_text(encoding='utf-8')}</style>"
    except OSError:
        return ""


def status_pill(status: str) -> str:
    cls = STATUS_CLASS.get(status, "not-started")
    return f'<span class="pill {cls}"><span class="dot"></span>{esc(status)}</span>'


def brand(name: str, sub: str) -> str:
    return _clean(f"""
    <div class="brand">
      <div class="brand-logo">🧠</div>
      <div><div class="brand-name">{esc(name)}</div><div class="brand-sub">{esc(sub)}</div></div>
    </div>""")


def hero(title: str, tagline: str, status: str, chips: Sequence[Tuple[str, str]] = (),
         kicker: str = "Reinforcement Learning Laboratory") -> str:
    chip_html = "".join(f'<span class="chip">{esc(k)} <b>{esc(v)}</b></span>' for k, v in chips)
    return _clean(f"""
    <div class="hero">
      <div class="hero-row">
        <div>
          <div class="kicker">{esc(kicker)}</div>
          <h1>{esc(title)}</h1>
          <div class="tagline">{esc(tagline)}</div>
        </div>
        <div>{status_pill(status)}</div>
      </div>
      <div class="chips">{chip_html}</div>
    </div>""")


def stat_card(label: str, value: str, sub: str = "", icon: str = "", accent: str = "#22D3EE",
              meter: Optional[float] = None, tip: str = "") -> str:
    tip_html = f'<span class="tip" data-tip="{esc(tip)}">ⓘ</span>' if tip else ""
    meter_html = ""
    if meter is not None:
        pct = max(0.0, min(100.0, float(meter) * 100.0))
        meter_html = f'<div class="meter"><span style="width:{pct:.1f}%"></span></div>'
    return _clean(f"""
    <div class="stat-card" style="--accent:{accent}">
      <div class="label"><span class="icon">{icon}</span>{esc(label)} {tip_html}</div>
      <div class="value">{esc(value)}</div>
      <div class="sub">{esc(sub)}</div>
      {meter_html}
    </div>""")


def stat_grid(cards: Iterable[str]) -> str:
    return f'<div class="stat-grid">{"".join(cards)}</div>'


def section_title(text: str, sub: str = "") -> str:
    sub_html = f'<div class="section-sub">{esc(sub)}</div>' if sub else ""
    return f'<div class="section-title">{esc(text)}</div>{sub_html}'


def callout(text: str, kind: str = "info", allow_html: bool = False) -> str:
    body = text if allow_html else esc(text)
    return f'<div class="callout {kind}">{body}</div>'


def legend() -> str:
    items = [("start", "Start position"), ("goal", "Goal / destination"), ("wall", "Wall / obstacle"),
             ("empty", "Empty traversable cell"), ("agent", "AI agent"), ("trail", "Agent's path")]
    inner = "".join(f'<span class="item"><span class="sw {k}"></span>{esc(v)}</span>' for k, v in items)
    return f'<div class="legend">{inner}</div>'


def vs_card(kind: str, title: str, metrics: Sequence[Tuple[str, str]], note: str = "") -> str:
    m = "".join(f'<div class="vs-metric"><div class="k">{esc(k)}</div><div class="v">{esc(v)}</div></div>'
                for k, v in metrics)
    note_html = f'<div class="section-sub" style="margin:8px 0 0 0">{esc(note)}</div>' if note else ""
    return _clean(f"""
    <div class="vs-card {kind}">
      <div class="vs-title">{esc(title)}</div>
      <div class="vs-metrics">{m}</div>
      {note_html}
    </div>""")


def delta_row(items: Sequence[Tuple[str, str, bool]]) -> str:
    """items: (label, value, positive?)"""
    inner = "".join(f'<div class="delta{"" if good else " neutral"}"><div class="k">{esc(k)}</div>'
                    f'<div class="v">{esc(v)}</div></div>' for k, v, good in items)
    return f'<div class="delta-row">{inner}</div>'


def concept_grid(concepts) -> str:
    cards = []
    for i, (icon, title, text, example) in enumerate(concepts, start=1):
        cards.append(
            f'<div class="concept"><div class="num">{i:02d} {icon}</div><h4>{esc(title)}</h4>'
            f'<p>{esc(text)}</p><div class="eg">In our maze: {esc(example)}</div></div>')
    return f'<div class="concept-grid">{"".join(cards)}</div>'


def variable_table(rows) -> str:
    body = "".join(f"<tr><td>{esc(k)}</td><td>{esc(v)}</td></tr>" for k, v in rows)
    return f'<div class="panel"><table class="var-table">{body}</table></div>'


def timeline(steps) -> str:
    inner = "".join(f'<div class="tl-step"><div class="n">{i}</div><h5>{esc(t)}</h5><p>{esc(d)}</p></div>'
                    for i, (t, d) in enumerate(steps, start=1))
    return f'<div class="timeline">{inner}</div>'


def stepper(labels: Sequence[str], current: int) -> str:
    """current is 1-based; 0 means not started."""
    parts = []
    for i, label in enumerate(labels, start=1):
        cls = "active" if i == current else ("done" if i < current else "")
        mark = "✓ " if i < current else f"{i}. "
        parts.append(f'<div class="s {cls}">{mark}{esc(label)}</div>')
    return f'<div class="stepper">{"".join(parts)}</div>'


def rl_loop_diagram_html(height: int = 300) -> str:
    """Animated agent ↔ environment loop (SVG), rendered inside an iframe component."""
    return """
<!DOCTYPE html><html><head><meta charset="utf-8"><style>
body{margin:0;background:transparent;font-family:Inter,'Segoe UI',Roboto,sans-serif}
.lbl{fill:#E2E8F0;font-size:15px;font-weight:700}
.sm{fill:#94A3B8;font-size:12px}
.arrow{fill:none;stroke-width:3}
</style></head><body>
<svg viewBox="0 0 760 290" width="100%" height="%HEIGHT%px">
<defs>
 <linearGradient id="ga" x1="0" x2="1"><stop offset="0" stop-color="#2563EB"/><stop offset="1" stop-color="#7C3AED"/></linearGradient>
 <linearGradient id="ge" x1="0" x2="1"><stop offset="0" stop-color="#0E7490"/><stop offset="1" stop-color="#1D4ED8"/></linearGradient>
 <marker id="m1" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#22D3EE"/></marker>
 <marker id="m2" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#FACC15"/></marker>
 <filter id="glow"><feGaussianBlur stdDeviation="4" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
</defs>
<rect x="40" y="95" width="210" height="100" rx="18" fill="url(#ga)" filter="url(#glow)"/>
<text x="145" y="138" text-anchor="middle" class="lbl">🤖 AI AGENT</text>
<text x="145" y="162" text-anchor="middle" class="sm" style="fill:#E0E7FF">Q-table = its memory</text>
<rect x="510" y="95" width="210" height="100" rx="18" fill="url(#ge)" filter="url(#glow)"/>
<text x="615" y="138" text-anchor="middle" class="lbl">🗺️ MAZE ENVIRONMENT</text>
<text x="615" y="162" text-anchor="middle" class="sm" style="fill:#CFFAFE">walls · start · goal</text>
<path id="p1" d="M250,115 C340,40 420,40 510,115" class="arrow" stroke="#22D3EE" marker-end="url(#m1)"/>
<path id="p2" d="M510,175 C420,250 340,250 250,175" class="arrow" stroke="#FACC15" marker-end="url(#m2)"/>
<text x="380" y="40" text-anchor="middle" class="lbl" style="fill:#22D3EE">ACTION</text>
<text x="380" y="58" text-anchor="middle" class="sm">Up · Down · Left · Right</text>
<text x="380" y="262" text-anchor="middle" class="lbl" style="fill:#FACC15">NEW STATE + REWARD</text>
<text x="380" y="280" text-anchor="middle" class="sm">+100 goal · −10 wall · −1 step</text>
<circle r="7" fill="#22D3EE" filter="url(#glow)"><animateMotion dur="2.4s" repeatCount="indefinite"><mpath href="#p1"/></animateMotion></circle>
<circle r="7" fill="#FACC15" filter="url(#glow)"><animateMotion dur="2.4s" begin="1.2s" repeatCount="indefinite"><mpath href="#p2"/></animateMotion></circle>
<text x="380" y="150" text-anchor="middle" class="sm" style="font-size:13px;fill:#A78BFA">repeat thousands of times →</text>
<text x="380" y="168" text-anchor="middle" class="sm" style="font-size:13px;fill:#A78BFA">the Q-table gets smarter</text>
</svg></body></html>""".replace("%HEIGHT%", str(height))
