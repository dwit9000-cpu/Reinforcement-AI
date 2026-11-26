"""
visualization.py
================
All drawing code for REINFORCE AI.

* Plotly figures: maze, policy arrows, Q-value maps, state visits and the
  training charts (rewards, success rate, steps, exploration).
* An animated maze "player" written in plain HTML/CSS/JavaScript, so the blue
  agent moves smoothly in the browser without blocking the Python app.

Every figure is drawn from real data passed in by the app — nothing here
invents numbers.
"""

from __future__ import annotations

import html as html_lib
import json
import math
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from environment import ACTION_ARROWS, ACTION_NAMES, WALL, MazeEnvironment

# ----------------------------------------------------------------------------
# Colour palette (dark AI-lab theme)
# ----------------------------------------------------------------------------
BG = "#060B18"
PANEL = "#0B1428"
EMPTY_CELL = "#13213F"
WALL_CELL = "#7F1D1D"
START_CELL = "#22C55E"
GOAL_CELL = "#FACC15"
AGENT = "#3B82F6"
CYAN = "#22D3EE"
VIOLET = "#A78BFA"
ELECTRIC = "#3B82F6"
PINK = "#F472B6"
TEXT = "#E2E8F0"
MUTED = "#94A3B8"
GRID = "rgba(148,163,184,0.12)"

VALUE_SCALE = [[0.0, "#1E1B4B"], [0.35, "#4C1D95"], [0.65, "#2563EB"], [1.0, "#22D3EE"]]
VISIT_SCALE = [[0.0, "#0B1428"], [0.3, "#312E81"], [0.6, "#7C3AED"], [1.0, "#F472B6"]]


# ----------------------------------------------------------------------------
# Shared styling
# ----------------------------------------------------------------------------
def style_figure(fig: go.Figure, height: int = 340, title: Optional[str] = None,
                 legend: bool = True) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(11,20,40,0.55)",
        font=dict(family="Inter, 'Segoe UI', Roboto, sans-serif", color="#CBD5E1", size=12),
        margin=dict(l=10, r=10, t=46 if title else 16, b=10),
        hoverlabel=dict(bgcolor="#0F172A", font=dict(color=TEXT), bordercolor=CYAN),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)"),
    )
    if title:
        fig.update_layout(title=dict(text=title, x=0.01, y=0.98, font=dict(size=14, color=TEXT)))
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def empty_figure(message: str, height: int = 300) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper",
                       showarrow=False, font=dict(size=14, color=MUTED))
    style_figure(fig, height=height, legend=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def _maze_height(env: MazeEnvironment, max_height: int = 620) -> int:
    return int(min(max_height, max(320, 34 * env.rows + 70)))


def _cell_px(env: MazeEnvironment, height: int) -> float:
    return max(8.0, (height - 60) / env.rows)


def _maze_axes(fig: go.Figure, env: MazeEnvironment, has_title: bool = False) -> None:
    fig.update_layout(margin=dict(l=10, r=10, t=68 if has_title else 30, b=10))
    fig.update_xaxes(range=[-0.5, env.cols - 0.5], showgrid=False, zeroline=False,
                     tickmode="linear", dtick=1, tickfont=dict(size=9, color="#475569"),
                     constrain="domain", fixedrange=True, side="top")
    fig.update_yaxes(range=[env.rows - 0.5, -0.5], showgrid=False, zeroline=False,
                     tickmode="linear", dtick=1, tickfont=dict(size=9, color="#475569"),
                     scaleanchor="x", scaleratio=1, constrain="domain", fixedrange=True)


def _cell_type_matrix(env: MazeEnvironment):
    """0 = empty, 1 = wall, 2 = start, 3 = goal."""
    z = env.grid.astype(int).copy()
    z[env.start] = 2
    z[env.goal] = 3
    names = np.array(["Empty cell", "Wall / obstacle", "Start", "Goal"])[z]
    return z, names


def _start_goal_labels(fig: go.Figure, env: MazeEnvironment, size: float) -> None:
    fig.add_trace(go.Scatter(
        x=[env.start[1], env.goal[1]], y=[env.start[0], env.goal[0]],
        mode="text", text=["S", "★"],
        textfont=dict(size=max(9, size * 0.5), color="#052e16", family="Arial Black"),
        hoverinfo="skip", showlegend=False))


# ----------------------------------------------------------------------------
# Maze figure (static view, live training view and click-to-edit editor)
# ----------------------------------------------------------------------------
def maze_figure(env: MazeEnvironment, path: Optional[List] = None,
                agent_pos=None, clickable: bool = False, height: Optional[int] = None,
                title: Optional[str] = None, path_color: str = CYAN,
                highlight_cells: Optional[List] = None) -> go.Figure:
    height = height or _maze_height(env)
    cell = _cell_px(env, height)
    z, names = _cell_type_matrix(env)
    c0, c1, c2, c3 = EMPTY_CELL, WALL_CELL, START_CELL, GOAL_CELL
    colorscale = [[0, c0], [1 / 6, c0], [1 / 6, c1], [0.5, c1],
                  [0.5, c2], [5 / 6, c2], [5 / 6, c3], [1, c3]]

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=z.tolist(), zmin=0, zmax=3, colorscale=colorscale, showscale=False,
        xgap=2, ygap=2, text=names.tolist(),
        hoverinfo="skip" if clickable else None,
        hovertemplate=None if clickable else "Row %{y}, Col %{x}<br>%{text}<extra></extra>"))

    if highlight_cells:
        fig.add_trace(go.Scatter(
            x=[c for r, c in highlight_cells], y=[r for r, c in highlight_cells],
            mode="markers", marker=dict(symbol="square", size=cell * 0.55,
                                        color="rgba(250,204,21,0.18)",
                                        line=dict(color="rgba(250,204,21,0.6)", width=1)),
            hoverinfo="skip", name="Shortest possible route", showlegend=True))

    if path and len(path) > 1:
        xs = [p[1] for p in path]
        ys = [p[0] for p in path]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers", name="Agent path",
            line=dict(color=path_color, width=3, shape="linear"),
            marker=dict(size=max(4, cell * 0.16), color=path_color),
            opacity=0.85, customdata=list(range(len(path))),
            hovertemplate="Step %{customdata}<extra></extra>",
            showlegend=False))

    _start_goal_labels(fig, env, cell)

    if agent_pos is not None:
        fig.add_trace(go.Scatter(
            x=[agent_pos[1]], y=[agent_pos[0]], mode="markers", name="AI agent",
            marker=dict(size=cell * 0.62, color=AGENT, symbol="circle",
                        line=dict(color="#BFDBFE", width=3)),
            hovertemplate="AI agent<extra></extra>", showlegend=False))

    if clickable:
        xs, ys = np.meshgrid(np.arange(env.cols), np.arange(env.rows))
        fig.add_trace(go.Scatter(
            x=xs.ravel().tolist(), y=ys.ravel().tolist(), mode="markers",
            marker=dict(symbol="square", size=cell * 0.9, color="rgba(255,255,255,0)"),
            text=names.ravel().tolist(),
            hovertemplate="Row %{y}, Col %{x} · %{text}<br>Click to apply the edit tool<extra></extra>",
            name="cells", showlegend=False))

    style_figure(fig, height=height, title=title, legend=bool(highlight_cells))
    _maze_axes(fig, env, bool(title))
    return fig


# ----------------------------------------------------------------------------
# Policy / value visualisations
# ----------------------------------------------------------------------------
def _to_nullable(matrix: np.ndarray) -> list:
    return [[None if (v is None or (isinstance(v, float) and math.isnan(v))) else float(v)
             for v in row] for row in matrix.tolist()]


def policy_figure(env: MazeEnvironment, agent, height: Optional[int] = None,
                  title: Optional[str] = None, route: Optional[List] = None) -> go.Figure:
    """Value heatmap V(s) = max Q(s,a) with the preferred action drawn as an arrow in each cell."""
    height = height or _maze_height(env)
    cell = _cell_px(env, height)
    values = agent.value_grid(env)
    learned = np.any(agent.q_table != 0, axis=1).reshape(env.rows, env.cols)
    shown = values.copy()
    shown[~learned] = np.nan

    walls = np.where(env.grid == WALL, 1.0, 0.0)
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=walls.tolist(), zmin=0, zmax=1, showscale=False, xgap=2, ygap=2,
                             colorscale=[[0, EMPTY_CELL], [1, WALL_CELL]], hoverinfo="skip"))
    finite = shown[np.isfinite(shown)]
    if finite.size:
        fig.add_trace(go.Heatmap(
            z=_to_nullable(shown), xgap=2, ygap=2, colorscale=VALUE_SCALE,
            zmin=float(finite.min()), zmax=float(finite.max()),
            colorbar=dict(title=dict(text="V(s)", side="right"), thickness=12, len=0.8,
                          tickfont=dict(size=10)),
            hovertemplate="Row %{y}, Col %{x}<br>State value V(s) = %{z:.2f}<extra></extra>"))

    policy = agent.policy_grid(env)
    xs, ys, arrows, custom = [], [], [], []
    for r, c in env.free_cells():
        a = policy[r, c]
        if a < 0:
            continue
        q = agent.q_table[env.state_of((r, c))]
        xs.append(c)
        ys.append(r)
        arrows.append(ACTION_ARROWS[a])
        custom.append([float(q[0]), float(q[1]), float(q[2]), float(q[3]), ACTION_NAMES[a]])
    if xs:
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="text", text=arrows, customdata=custom,
            textfont=dict(size=max(10, cell * 0.55), color="#F8FAFC", family="Arial"),
            hovertemplate=("Row %{y}, Col %{x}<br><b>Best action: %{customdata[4]}</b><br>"
                           "Q(Up)=%{customdata[0]:.2f}<br>Q(Down)=%{customdata[1]:.2f}<br>"
                           "Q(Left)=%{customdata[2]:.2f}<br>Q(Right)=%{customdata[3]:.2f}"
                           "<extra></extra>"),
            showlegend=False))

    if route and len(route) > 1:
        fig.add_trace(go.Scatter(x=[p[1] for p in route], y=[p[0] for p in route],
                                 mode="lines", line=dict(color=GOAL_CELL, width=2, dash="dot"),
                                 hoverinfo="skip", showlegend=False, opacity=0.8))
    _start_goal_labels(fig, env, cell)
    style_figure(fig, height=height, title=title, legend=False)
    _maze_axes(fig, env, bool(title))
    return fig


def action_value_maps(env: MazeEnvironment, agent, height: int = 300) -> go.Figure:
    """Four small maps: Q(s, Up), Q(s, Down), Q(s, Left), Q(s, Right) for every cell."""
    fig = make_subplots(rows=1, cols=4, horizontal_spacing=0.03,
                        subplot_titles=[f"Q(s, {n}) {a}" for n, a in zip(ACTION_NAMES, ACTION_ARROWS)])
    q = agent.q_table
    finite = q[np.any(q != 0, axis=1)]
    zmin = float(finite.min()) if finite.size else -1.0
    zmax = float(finite.max()) if finite.size else 1.0
    for a in range(4):
        grid = q[:, a].reshape(env.rows, env.cols).astype(float)
        grid[env.grid == WALL] = np.nan
        fig.add_trace(go.Heatmap(
            z=_to_nullable(grid), zmin=zmin, zmax=zmax, colorscale=VALUE_SCALE,
            xgap=1, ygap=1, showscale=(a == 3),
            colorbar=dict(thickness=10, len=0.9, tickfont=dict(size=9)),
            hovertemplate=f"Row %{{y}}, Col %{{x}}<br>Q(s, {ACTION_NAMES[a]}) = %{{z:.2f}}<extra></extra>"),
            row=1, col=a + 1)
        fig.update_xaxes(visible=False, row=1, col=a + 1, constrain="domain")
        fig.update_yaxes(visible=False, autorange="reversed", row=1, col=a + 1,
                         scaleanchor=f"x{'' if a == 0 else a + 1}", constrain="domain")
    style_figure(fig, height=height, legend=False)
    fig.update_annotations(font=dict(size=12, color=TEXT))
    return fig


def q_values_bar(q_values, title: str = "") -> go.Figure:
    q_values = [float(v) for v in q_values]
    best = int(np.argmax(q_values)) if any(v != 0 for v in q_values) else -1
    colors = [CYAN if i == best else "#334155" for i in range(4)]
    fig = go.Figure(go.Bar(
        x=[f"{a} {n}" for n, a in zip(ACTION_NAMES, ACTION_ARROWS)], y=q_values,
        marker=dict(color=colors, line=dict(color="#0F172A", width=1)),
        text=[f"{v:.2f}" for v in q_values], textposition="outside",
        hovertemplate="%{x}: %{y:.3f}<extra></extra>"))
    style_figure(fig, height=280, title=title, legend=False)
    fig.update_yaxes(title_text="Q-value")
    return fig


def visit_heatmap(env: MazeEnvironment, agent, height: Optional[int] = None,
                  title: Optional[str] = None) -> go.Figure:
    height = height or _maze_height(env)
    visits = agent.visit_counts.reshape(env.rows, env.cols).astype(float)
    shown = np.log10(1 + visits)
    shown[env.grid == WALL] = np.nan
    walls = np.where(env.grid == WALL, 1.0, 0.0)
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=walls.tolist(), zmin=0, zmax=1, showscale=False, xgap=2, ygap=2,
                             colorscale=[[0, EMPTY_CELL], [1, WALL_CELL]], hoverinfo="skip"))
    fig.add_trace(go.Heatmap(
        z=_to_nullable(shown), customdata=visits.astype(int).tolist(), colorscale=VISIT_SCALE,
        xgap=2, ygap=2,
        colorbar=dict(title=dict(text="log₁₀ visits", side="right"), thickness=12, len=0.8),
        hovertemplate="Row %{y}, Col %{x}<br>Visited %{customdata:,} times<extra></extra>"))
    _start_goal_labels(fig, env, _cell_px(env, height))
    style_figure(fig, height=height, title=title, legend=False)
    _maze_axes(fig, env, bool(title))
    return fig


# ----------------------------------------------------------------------------
# Training charts (all from the recorded history DataFrame)
# ----------------------------------------------------------------------------
def reward_chart(df: pd.DataFrame, height: int = 320, title: Optional[str] = "Reward per episode") -> go.Figure:
    if df.empty:
        return empty_figure("Train the agent to see rewards", height)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["episode"], y=df["reward"], mode="lines", name="Episode reward",
                             line=dict(color="rgba(59,130,246,0.35)", width=1)))
    fig.add_trace(go.Scatter(x=df["episode"], y=df["avg_reward"], mode="lines", name="Moving average",
                             line=dict(color=CYAN, width=3)))
    style_figure(fig, height=height, title=title)
    fig.update_xaxes(title_text="Episode")
    fig.update_yaxes(title_text="Total reward")
    return fig


def cumulative_reward_chart(df: pd.DataFrame, height: int = 300,
                            title: Optional[str] = "Cumulative reward") -> go.Figure:
    if df.empty:
        return empty_figure("No training data yet", height)
    fig = go.Figure(go.Scatter(x=df["episode"], y=df["cumulative_reward"], mode="lines",
                               fill="tozeroy", fillcolor="rgba(167,139,250,0.15)",
                               line=dict(color=VIOLET, width=2.5), name="Cumulative reward"))
    style_figure(fig, height=height, title=title, legend=False)
    fig.update_xaxes(title_text="Episode")
    fig.update_yaxes(title_text="Sum of all rewards")
    return fig


def success_chart(df: pd.DataFrame, height: int = 320,
                  title: Optional[str] = "Success rate") -> go.Figure:
    if df.empty:
        return empty_figure("Train the agent to see the success rate", height)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["episode"], y=df["success_rate"], mode="lines",
                             name="Recent episodes (moving window)", line=dict(color="#34D399", width=3),
                             fill="tozeroy", fillcolor="rgba(52,211,153,0.10)"))
    fig.add_trace(go.Scatter(x=df["episode"], y=df["cumulative_success_rate"], mode="lines",
                             name="All episodes so far", line=dict(color=VIOLET, width=2, dash="dash")))
    style_figure(fig, height=height, title=title)
    fig.update_xaxes(title_text="Episode")
    fig.update_yaxes(title_text="Reached goal (%)", range=[0, 105])
    return fig


def steps_chart(df: pd.DataFrame, optimal: Optional[int] = None, height: int = 320,
                title: Optional[str] = "Steps per episode") -> go.Figure:
    if df.empty:
        return empty_figure("Train the agent to see step counts", height)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["episode"], y=df["steps"], mode="lines", name="Steps (all episodes)",
                             line=dict(color="rgba(244,114,182,0.35)", width=1)))
    fig.add_trace(go.Scatter(x=df["episode"], y=df["avg_steps_success"], mode="lines",
                             name="Avg steps when goal reached", line=dict(color=PINK, width=3),
                             connectgaps=True))
    if optimal:
        fig.add_hline(y=optimal, line=dict(color=GOAL_CELL, width=1.5, dash="dot"),
                      annotation_text=f"Shortest possible = {optimal}",
                      annotation_font=dict(color=GOAL_CELL, size=11),
                      annotation_position="top right")
    style_figure(fig, height=height, title=title)
    fig.update_xaxes(title_text="Episode")
    fig.update_yaxes(title_text="Steps")
    return fig


def epsilon_chart(df: pd.DataFrame, height: int = 320,
                  title: Optional[str] = "Exploration vs exploitation") -> go.Figure:
    if df.empty:
        return empty_figure("Train the agent to see exploration decay", height)
    smooth = df["explore_ratio"].rolling(20, min_periods=1).mean() / 100.0
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["episode"], y=df["epsilon"], mode="lines",
                             name="ε (chance of a random move)", line=dict(color=VIOLET, width=3)))
    fig.add_trace(go.Scatter(x=df["episode"], y=smooth, mode="lines",
                             name="Measured share of random moves", line=dict(color=CYAN, width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=df["episode"], y=1 - smooth, mode="lines",
                             name="Measured share of best-known moves",
                             line=dict(color="#34D399", width=2, dash="dot")))
    style_figure(fig, height=height, title=title)
    fig.update_xaxes(title_text="Episode")
    fig.update_yaxes(title_text="Probability / share", range=[0, 1.05])
    return fig


def explore_exploit_area(df: pd.DataFrame, height: int = 300,
                         title: Optional[str] = "Random vs learned moves per episode") -> go.Figure:
    if df.empty:
        return empty_figure("No training data yet", height)
    explore = df["explore_actions"].rolling(10, min_periods=1).mean()
    exploit = df["exploit_actions"].rolling(10, min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["episode"], y=explore, mode="lines", stackgroup="one",
                             name="Exploration (random)", line=dict(color=VIOLET, width=0.5)))
    fig.add_trace(go.Scatter(x=df["episode"], y=exploit, mode="lines", stackgroup="one",
                             name="Exploitation (best Q)", line=dict(color=CYAN, width=0.5)))
    style_figure(fig, height=height, title=title)
    fig.update_xaxes(title_text="Episode")
    fig.update_yaxes(title_text="Moves (10-episode average)")
    return fig


def wall_hits_chart(df: pd.DataFrame, height: int = 300,
                    title: Optional[str] = "Wall collisions per episode") -> go.Figure:
    if df.empty:
        return empty_figure("No training data yet", height)
    fig = go.Figure(go.Scatter(x=df["episode"], y=df["wall_hits"].rolling(20, min_periods=1).mean(),
                               mode="lines", line=dict(color="#F87171", width=2.5),
                               fill="tozeroy", fillcolor="rgba(248,113,113,0.12)", name="Wall hits"))
    style_figure(fig, height=height, title=title, legend=False)
    fig.update_xaxes(title_text="Episode")
    fig.update_yaxes(title_text="Wall hits (20-episode avg)")
    return fig


def comparison_bars(before: dict, after: dict, optimal: Optional[int] = None,
                    height: int = 320) -> go.Figure:
    """Side-by-side bars of the recorded before/after test results."""
    metrics = [
        ("Success rate (%)", before["success_rate"], after["success_rate"]),
        ("Average steps", before["avg_steps"], after["avg_steps"]),
        ("Average reward", before["avg_reward"], after["avg_reward"]),
        ("Wall hits / run", before["avg_wall_hits"], after["avg_wall_hits"]),
    ]
    fig = make_subplots(rows=1, cols=4, subplot_titles=[m[0] for m in metrics], horizontal_spacing=0.07)
    for i, (name, b, a) in enumerate(metrics, start=1):
        fig.add_trace(go.Bar(x=["Before", "After"], y=[b, a],
                             marker=dict(color=["#F87171", CYAN]),
                             text=[f"{b:,.1f}", f"{a:,.1f}"], textposition="outside",
                             hovertemplate=f"{name}<br>%{{x}}: %{{y:,.2f}}<extra></extra>",
                             showlegend=False), row=1, col=i)
        if name == "Average steps" and optimal:
            fig.add_hline(y=optimal, line=dict(color=GOAL_CELL, dash="dot", width=1.5), row=1, col=i)
    style_figure(fig, height=height, legend=False)
    fig.update_annotations(font=dict(size=12, color=TEXT))
    return fig


# ----------------------------------------------------------------------------
# Animated maze player (HTML + CSS + JavaScript)
# ----------------------------------------------------------------------------
def maze_player_html(env: MazeEnvironment, result=None, title: str = "", accent: str = CYAN,
                     autoplay: bool = True, speed_ms: int = 140, max_width: int = 520,
                     subtitle: str = ""):
    """
    Returns (html, height). The page draws the maze and animates the agent along
    ``result.path`` (an EpisodeResult). Wall bumps flash red. Runs entirely in the
    browser, so Streamlit stays responsive.
    """
    gap = 3
    cell = int(max(14, min(42, (max_width - 24) // env.cols - gap)))
    board_w = env.cols * (cell + gap) - gap
    board_h = env.rows * (cell + gap) - gap

    path = [list(map(int, p)) for p in (result.path if result is not None else [env.start])]
    rewards = [float(r) for r in (result.rewards if result is not None else [])]
    hits = [bool(h) for h in (result.hits if result is not None else [])]
    success = bool(result.success) if result is not None else False

    cells_html = []
    for r in range(env.rows):
        for c in range(env.cols):
            kind = "wall" if env.grid[r, c] == WALL else "empty"
            label = ""
            if (r, c) == env.start:
                kind, label = "start", "S"
            elif (r, c) == env.goal:
                kind, label = "goal", "★"
            trail = '' if kind in ("start", "goal") else '<div class="trail"></div>'
            cells_html.append(f'<div class="cell {kind}" id="c{r}_{c}">{trail}<span>{label}</span></div>')

    title = html_lib.escape(title)
    subtitle = html_lib.escape(subtitle)
    data = json.dumps({"path": path, "rewards": rewards, "hits": hits, "success": success,
                       "cell": cell, "gap": gap, "autoplay": bool(autoplay) and len(path) > 1,
                       "speed": int(speed_ms), "goal": list(env.goal)})
    height = board_h + 150

    html = f"""
<!DOCTYPE html><html><head><meta charset="utf-8"><style>
  * {{ box-sizing: border-box; }}
  body {{ margin:0; background:transparent; font-family: Inter, 'Segoe UI', Roboto, sans-serif; color:#E2E8F0; }}
  .wrap {{ display:flex; flex-direction:column; align-items:center; gap:10px; padding:6px 4px; }}
  .hud {{ width:{max(board_w, 300)}px; display:flex; justify-content:space-between; align-items:center;
          background:linear-gradient(90deg, rgba(15,23,42,.9), rgba(30,27,75,.6)); border:1px solid rgba(34,211,238,.25);
          border-radius:12px; padding:8px 12px; font-size:12px; }}
  .hud .title {{ font-weight:700; color:{accent}; letter-spacing:.3px; }}
  .hud .sub {{ color:#94A3B8; font-size:11px; }}
  .stats {{ display:flex; gap:14px; font-variant-numeric: tabular-nums; }}
  .stats b {{ color:#F8FAFC; }}
  .board {{ position:relative; display:grid; grid-template-columns:repeat({env.cols}, {cell}px);
            grid-auto-rows:{cell}px; gap:{gap}px; padding:8px; border-radius:14px;
            background:#070D1D; border:1px solid rgba(59,130,246,.25);
            box-shadow:0 0 30px rgba(59,130,246,.12), inset 0 0 20px rgba(0,0,0,.6); }}
  .cell {{ position:relative; border-radius:5px; display:flex; align-items:center; justify-content:center;
           font-weight:800; font-size:{max(9, cell // 2)}px; overflow:hidden; }}
  .cell span {{ position:relative; z-index:2; }}
  .empty {{ background:#13213F; }}
  .wall {{ background:linear-gradient(135deg,#991B1B,#450A0A); box-shadow: inset 0 0 0 1px rgba(248,113,113,.25); }}
  .start {{ background:#16A34A; color:#052E16; box-shadow:0 0 12px rgba(34,197,94,.6); }}
  .goal {{ background:#EAB308; color:#422006; box-shadow:0 0 14px rgba(250,204,21,.7); }}
  .goal.win {{ animation: pulse 0.8s ease-in-out 3; }}
  .trail {{ position:absolute; inset:0; background:{accent}; opacity:0; transition:opacity .25s; z-index:1; }}
  .agent {{ position:absolute; left:8px; top:8px; width:{cell}px; height:{cell}px; display:flex;
            align-items:center; justify-content:center; z-index:5; pointer-events:none;
            transition: transform var(--dur,120ms) ease-in-out; }}
  .agent .orb {{ width:72%; height:72%; border-radius:50%;
                 background: radial-gradient(circle at 35% 30%, #DBEAFE, #3B82F6 45%, #1D4ED8);
                 box-shadow:0 0 14px 4px rgba(59,130,246,.75); position:relative; }}
  .agent .orb::before, .agent .orb::after {{ content:""; position:absolute; top:34%; width:16%; height:20%;
                 background:#0B1020; border-radius:50%; }}
  .agent .orb::before {{ left:26%; }} .agent .orb::after {{ right:26%; }}
  .agent.bump .orb {{ background: radial-gradient(circle at 35% 30%, #FEE2E2, #EF4444 50%, #991B1B);
                      box-shadow:0 0 16px 6px rgba(239,68,68,.8); animation: shake .18s; }}
  .controls {{ display:flex; gap:8px; align-items:center; font-size:12px; color:#94A3B8; }}
  button {{ background:linear-gradient(135deg,#1D4ED8,#7C3AED); color:white; border:none; border-radius:8px;
            padding:6px 12px; font-weight:700; cursor:pointer; font-size:12px; }}
  button:hover {{ filter:brightness(1.15); }}
  select {{ background:#0F172A; color:#E2E8F0; border:1px solid #334155; border-radius:6px; padding:4px; }}
  .status {{ font-weight:700; }}
  .ok {{ color:#34D399; }} .bad {{ color:#F87171; }}
  @keyframes pulse {{ 0%,100% {{ transform:scale(1); }} 50% {{ transform:scale(1.18); }} }}
  @keyframes shake {{ 0%,100% {{ margin-left:0 }} 25% {{ margin-left:-3px }} 75% {{ margin-left:3px }} }}
</style></head><body>
<div class="wrap">
  <div class="hud">
    <div><div class="title">{title}</div><div class="sub">{subtitle}</div></div>
    <div class="stats">
      <div>Step <b id="step">0</b>/<span id="total">0</span></div>
      <div>Reward <b id="reward">0</b></div>
      <div>Wall hits <b id="hits">0</b></div>
    </div>
  </div>
  <div class="board" id="board">{''.join(cells_html)}<div class="agent" id="agent"><div class="orb"></div></div></div>
  <div class="controls">
    <button id="play">⏸ Pause</button><button id="restart">⟲ Replay</button>
    <label>Speed <select id="speed"><option value="0.5">0.5×</option><option value="1" selected>1×</option>
      <option value="2">2×</option><option value="4">4×</option><option value="10">10×</option></select></label>
    <span class="status" id="status"></span>
  </div>
</div>
<script>
const D = {data};
const agent = document.getElementById('agent');
const stepEl = document.getElementById('step'), rewardEl = document.getElementById('reward');
const hitsEl = document.getElementById('hits'), statusEl = document.getElementById('status');
const playBtn = document.getElementById('play'), speedSel = document.getElementById('speed');
document.getElementById('total').textContent = Math.max(0, D.path.length - 1);
let i = 0, timer = null, reward = 0, nhits = 0, visits = {{}};
function interval() {{ return D.speed / parseFloat(speedSel.value); }}
function place(idx, animate) {{
  const p = D.path[idx];
  agent.style.setProperty('--dur', animate ? Math.min(220, interval() * 0.9) + 'ms' : '0ms');
  agent.style.transform = 'translate(' + (p[1] * (D.cell + D.gap)) + 'px,' + (p[0] * (D.cell + D.gap)) + 'px)';
  const key = p[0] + '_' + p[1];
  visits[key] = (visits[key] || 0) + 1;
  const cellEl = document.getElementById('c' + key);
  const tr = cellEl ? cellEl.querySelector('.trail') : null;
  if (tr && idx > 0) tr.style.opacity = Math.min(0.12 + 0.12 * visits[key], 0.7);
}}
function reset() {{
  i = 0; reward = 0; nhits = 0; visits = {{}};
  document.querySelectorAll('.trail').forEach(t => t.style.opacity = 0);
  document.querySelectorAll('.goal').forEach(g => g.classList.remove('win'));
  statusEl.textContent = ''; statusEl.className = 'status';
  stepEl.textContent = 0; rewardEl.textContent = 0; hitsEl.textContent = 0;
  place(0, false);
}}
function finish() {{
  stop();
  if (D.path.length <= 1) {{ statusEl.textContent = 'Ready'; return; }}
  if (D.success) {{
    statusEl.textContent = '🏁 Goal reached in ' + (D.path.length - 1) + ' steps'; statusEl.className = 'status ok';
    document.querySelectorAll('.goal').forEach(g => g.classList.add('win'));
  }} else {{
    statusEl.textContent = '⏱ Step limit reached — goal not found'; statusEl.className = 'status bad';
  }}
  playBtn.textContent = '▶ Play';
}}
function tick() {{
  if (i >= D.path.length - 1) {{ finish(); return; }}
  i += 1;
  reward += D.rewards[i - 1] || 0;
  if (D.hits[i - 1]) {{ nhits += 1; agent.classList.add('bump'); setTimeout(() => agent.classList.remove('bump'), 160); }}
  place(i, true);
  stepEl.textContent = i; rewardEl.textContent = Math.round(reward * 10) / 10; hitsEl.textContent = nhits;
  timer = setTimeout(tick, interval());
}}
function start() {{ if (i >= D.path.length - 1) reset(); playBtn.textContent = '⏸ Pause'; timer = setTimeout(tick, interval()); }}
function stop() {{ if (timer) clearTimeout(timer); timer = null; playBtn.textContent = '▶ Play'; }}
playBtn.onclick = () => {{ if (timer) stop(); else start(); }};
document.getElementById('restart').onclick = () => {{ stop(); reset(); start(); }};
reset();
if (D.autoplay) {{ start(); }} else {{ if (D.path.length > 1) {{ stop(); }} else {{ finish(); }} }}
</script></body></html>
"""
    return html, height


def show_full_path_html(env: MazeEnvironment, result, title: str = "", accent: str = CYAN,
                        subtitle: str = ""):
    """Same player, but starts paused so the viewer can press Play."""
    return maze_player_html(env, result, title=title, accent=accent, autoplay=False, subtitle=subtitle)
