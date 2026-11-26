"""
app.py
======
REINFORCE AI – AI Through Experience
"Learning Through Experience, One Decision at a Time."

Main Streamlit application. Run it with:

    streamlit run app.py

Architecture
------------
environment.py   – the maze (states, actions, rewards, walls, maze generation)
q_learning.py    – the Q-Learning agent (Q-table, ε-greedy, Bellman update, evaluation)
analytics.py     – statistics computed from the recorded training history
visualization.py – Plotly charts + the animated HTML maze player
education.py     – teaching text, tooltips and demo-mode content
ui_components.py – HTML building blocks (cards, hero, legend...) styled by assets/style.css

How training stays responsive
-----------------------------
Streamlit re-runs this script after every interaction. While the status is
"Training", each run trains a small batch of episodes (a time budget that
depends on the chosen speed), redraws the page with fresh numbers and then
immediately schedules the next run with st.rerun(). Buttons use on_click
callbacks, which Streamlit executes *before* the next run — so Pause, Reset
etc. take effect between two batches and the interface never freezes.
"""

from __future__ import annotations

import inspect
import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import analytics as an
import education as edu
import ui_components as ui
import visualization as viz
from environment import (ACTION_ARROWS, ACTION_NAMES, MAX_SIZE, MAZE_STYLES, MIN_SIZE,
                         REWARD_GOAL, REWARD_STEP, REWARD_WALL, MazeEnvironment, MazeError)
from q_learning import EpisodeResult, Hyperparameters, QLearningAgent, evaluate_agent

APP_DIR = Path(__file__).resolve().parent
APP_VERSION = "1.0"

st.set_page_config(
    page_title="REINFORCE AI – AI Through Experience",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(ui.load_css_text(APP_DIR / "assets" / "style.css"), unsafe_allow_html=True)

# ============================================================================
# Constants
# ============================================================================
NOT_STARTED, TRAINING, PAUSED, COMPLETED = "Not Started", "Training", "Paused", "Completed"

PAGES = [
    "🏠 Dashboard",
    "🧩 Maze Lab",
    "⚖️ Before vs After",
    "🧠 AI Visualization",
    "📊 Analytics",
    "📘 Learn RL",
    "🎓 Demo Mode",
    "💾 Save & Load",
]

EPISODE_CHOICES = [100, 500, 1000, 2000, 5000]
DEMO_EPISODE_CHOICES = [100, 300, 500, 1000]
SPEED_OPTIONS = ["Watch", "Normal", "Turbo"]
SPEED_LABELS = {
    "Watch": "👁 Watch — 1 episode per refresh",
    "Normal": "▶ Normal — smooth live updates",
    "Turbo": "⚡ Turbo — fastest",
}
SPEED_BUDGET = {"Watch": 0.0, "Normal": 0.30, "Turbo": 1.2}      # seconds of training per refresh
REFRESH_PAUSE = {"Watch": 0.35, "Normal": 0.05, "Turbo": 0.02}   # pause before the next refresh
EVAL_TRIALS = 30
DEFAULT_MAZE = dict(rows=10, cols=10, style="maze", seed=42)
EDIT_TOOLS = ["Toggle wall", "Set start", "Set goal"]
PATH_SOURCES = ["Learned route (greedy)", "Last training episode", "Last test run", "Before-training sample"]
DEMO_MAZES = ["Keep current maze", "Fresh 8×8 maze", "Fresh 10×10 maze", "Fresh 12×12 maze"]

_DEFAULT_HP = Hyperparameters()
PARAM_DEFAULTS = {
    "p_alpha": _DEFAULT_HP.alpha,
    "p_gamma": _DEFAULT_HP.gamma,
    "p_epsilon": _DEFAULT_HP.epsilon,
    "p_decay": _DEFAULT_HP.epsilon_decay,
    "p_min_eps": _DEFAULT_HP.min_epsilon,
    "p_max_steps": _DEFAULT_HP.max_steps,
}
TRAINING_DEFAULTS = {"t_episodes": 500, "t_speed": "Normal"}
WIDGET_DEFAULTS = {
    **PARAM_DEFAULTS,
    **TRAINING_DEFAULTS,
    "nav": PAGES[0],
    "m_rows": DEFAULT_MAZE["rows"],
    "m_cols": DEFAULT_MAZE["cols"],
    "m_style": DEFAULT_MAZE["style"],
    "m_density": 0.25,
    "m_seed": 0,
    "edit_tool": EDIT_TOOLS[0],
    "keep_knowledge": False,
    "show_bfs": False,
    "an_window": 50,
    "path_source": PATH_SOURCES[0],
    "demo_episodes": 300,
    "demo_maze": DEMO_MAZES[0],
}

ss = st.session_state


# ============================================================================
# Streamlit version compatibility helpers
# ============================================================================
def _params_of(func) -> dict:
    try:
        return dict(inspect.signature(func).parameters)
    except (TypeError, ValueError):
        return {}


def _stretch_kwargs(func) -> dict:
    """Full-width keyword for this Streamlit version (width='stretch' or use_container_width=True)."""
    params = _params_of(func)
    width = params.get("width")
    if width is not None and isinstance(width.default, str):
        return {"width": "stretch"}
    if "use_container_width" in params:
        return {"use_container_width": True}
    return {}


BUTTON_KW = _stretch_kwargs(st.button)
DOWNLOAD_KW = _stretch_kwargs(st.download_button)
DATAFRAME_KW = _stretch_kwargs(st.dataframe)
_CHART_PARAMS = _params_of(st.plotly_chart)
CHART_KW = _stretch_kwargs(st.plotly_chart)
if "theme" in _CHART_PARAMS:
    CHART_KW["theme"] = None          # keep our own dark Plotly styling
SUPPORTS_CLICK_EDIT = "on_select" in _CHART_PARAMS
SUPPORTS_BORDER = "border" in _params_of(st.container)


def plot(fig, key=None, modebar: bool = False, **extra):
    kwargs = dict(CHART_KW)
    kwargs["config"] = {"displaylogo": False, "displayModeBar": modebar,
                        "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}
    if key is not None:
        kwargs["key"] = key
    kwargs.update(extra)
    return st.plotly_chart(fig, **kwargs)


def button(label: str, key: str, container=None, **kwargs) -> bool:
    target = container if container is not None else st
    return target.button(label, key=key, **BUTTON_KW, **kwargs)


def bordered():
    return st.container(border=True) if SUPPORTS_BORDER else st.container()


def html_block(markup: str, container=None) -> None:
    (container or st).markdown(markup, unsafe_allow_html=True)


def show_player(env, result, title, subtitle="", accent=viz.CYAN, autoplay=True, speed_ms=140,
                max_width=520):
    markup, height = viz.maze_player_html(env, result, title=title, subtitle=subtitle, accent=accent,
                                          autoplay=autoplay, speed_ms=speed_ms, max_width=max_width)
    components.html(markup, height=height, scrolling=False)


def notify(kind: str, message: str) -> None:
    ss.flash.append((kind, message))


def show_flash() -> None:
    icons = {"success": "✅", "info": "ℹ️", "warning": "⚠️", "error": "❌"}
    for kind, message in ss.flash:
        if kind in ("warning", "error"):
            getattr(st, kind)(message)
        elif hasattr(st, "toast"):
            st.toast(message, icon=icons.get(kind, "ℹ️"))
        else:
            st.info(message)
    ss.flash = []


# ============================================================================
# Session state (persists the maze, Q-table and history across interactions)
# ============================================================================
def current_params() -> Hyperparameters:
    eps = float(ss.p_epsilon)
    return Hyperparameters(
        alpha=float(ss.p_alpha), gamma=float(ss.p_gamma), epsilon=eps,
        epsilon_decay=float(ss.p_decay), min_epsilon=min(float(ss.p_min_eps), eps),
        max_steps=int(ss.p_max_steps),
    ).validate()


def reset_learning_state() -> None:
    ss.history = []
    ss.status = NOT_STARTED
    ss.run_target = 0
    ss.run_done = 0
    ss.baseline = None          # recorded "before training" test results
    ss.baseline_stale = False   # True if the maze was edited after the baseline was recorded
    ss.after = None             # latest "after training" test results
    ss.playback = None          # episode shown in the animated player
    ss.last_episode = None      # most recent training episode (with its path)
    ss.lab_run = None


def init_state() -> None:
    for key, value in WIDGET_DEFAULTS.items():
        if key not in ss:
            ss[key] = value
    if ss.get("initialized"):
        return
    env = MazeEnvironment.generate(**DEFAULT_MAZE)
    ss.env = env
    ss.env_original = env.copy()
    ss.agent = QLearningAgent(env.n_states, params=current_params())
    reset_learning_state()
    ss.maze_version = 0
    ss.flash = []
    ss.editor_nonce = 0
    ss.demo_step = 0
    ss.initialized = True


def persist_widget_state() -> None:
    """Keep widget values alive when their page is not displayed (Streamlit would otherwise drop them)."""
    for key in WIDGET_DEFAULTS:
        if key in ss:
            ss[key] = ss[key]


def sync_agent_params() -> None:
    """Apply sidebar hyper-parameters to the live agent."""
    agent = ss.agent
    agent.params = current_params()
    if agent.episodes_trained == 0:
        agent.epsilon = agent.params.epsilon


def history_df() -> pd.DataFrame:
    key = (id(ss.history), len(ss.history), int(ss.an_window))
    cache = ss.get("_df_cache")
    if cache is not None and cache[0] == key:
        return cache[1]
    df = an.history_to_dataframe(ss.history, window=int(ss.an_window))
    ss._df_cache = (key, df)
    return df


# ============================================================================
# Training core
# ============================================================================
def maze_is_solvable(report: bool = True) -> bool:
    if ss.env.is_solvable():
        return True
    if report:
        notify("error", "The goal cannot be reached from the start. Remove some walls in the Maze Lab first.")
    return False


def run_evaluation() -> dict:
    """30 test runs with learning switched off (ε = 0). Never changes the Q-table."""
    result = evaluate_agent(ss.agent, ss.env, n_trials=EVAL_TRIALS, epsilon=0.0,
                            max_steps=ss.agent.params.max_steps)
    result["maze_version"] = ss.maze_version
    result["time"] = datetime.now().strftime("%H:%M:%S")
    return result


def ensure_baseline() -> None:
    """Record the untrained AI's performance right before its first training episode."""
    if ss.baseline is None and ss.agent.episodes_trained == 0 and ss.env.is_solvable():
        ss.baseline = run_evaluation()
        ss.baseline_stale = False


def after_is_current() -> bool:
    a = ss.after
    return (a is not None and a["episodes_trained"] == ss.agent.episodes_trained
            and a["maze_version"] == ss.maze_version and a["max_steps"] == ss.agent.params.max_steps)


def get_after() -> dict:
    if not after_is_current():
        ss.after = run_evaluation()
    return ss.after


def train_one_episode() -> EpisodeResult:
    ensure_baseline()
    result = ss.agent.run_episode(ss.env, learn=True)
    ss.history.append(result.to_record())
    ss.last_episode = result
    return result


def make_playback(result: EpisodeResult, title: str, subtitle: str = "", accent: str = viz.CYAN) -> dict:
    return {"result": result, "title": title, "subtitle": subtitle, "accent": accent,
            "maze_version": ss.maze_version}


def finish_run() -> None:
    ss.status = COMPLETED
    ss.after = run_evaluation()
    sample = ss.after["sample"]
    ss.playback = make_playback(sample, "Trained AI — test run",
                                f"ε = 0 · learning off · {ss.agent.episodes_trained:,} episodes of experience")
    notify("success", f"Training complete ({ss.run_target:,} episodes). "
                      f"Test success rate: {ss.after['success_rate']:.0f}% · route: {sample.steps} steps.")


def training_tick() -> None:
    """Train one batch of episodes (called once per script run while status == Training)."""
    if ss.status != TRAINING:
        return
    if not maze_is_solvable(report=False):
        ss.status = PAUSED
        notify("warning", "Training paused: the goal is unreachable in the current maze.")
        return
    budget = SPEED_BUDGET.get(ss.t_speed, 0.3)
    started = time.perf_counter()
    while ss.run_done < ss.run_target:
        train_one_episode()
        ss.run_done += 1
        if ss.t_speed == "Watch" or time.perf_counter() - started >= budget:
            break
    if ss.run_done >= ss.run_target:
        finish_run()


def instant_train(n_episodes: int) -> None:
    """Train immediately with a progress bar (no live redraws)."""
    if ss.status == TRAINING or not maze_is_solvable():
        return
    ensure_baseline()
    bar = st.progress(0.0, text="Training…")
    update_every = max(1, n_episodes // 50)
    for i in range(n_episodes):
        train_one_episode()
        if (i + 1) % update_every == 0 or i + 1 == n_episodes:
            bar.progress((i + 1) / n_episodes, text=f"Training… episode {i + 1:,} / {n_episodes:,}")
    ss.run_target = ss.run_done = n_episodes
    finish_run()
    st.rerun()


# ---------------------------------------------------------------- callbacks --
def cb_start(n_key: str = "t_episodes") -> None:
    if ss.status == TRAINING or not maze_is_solvable():
        return
    ss.run_target = int(ss[n_key])
    ss.run_done = 0
    ensure_baseline()
    ss.status = TRAINING
    notify("info", f"Training started: {ss.run_target:,} episodes.")


def cb_pause() -> None:
    if ss.status == TRAINING:
        ss.status = PAUSED
        notify("info", f"Training paused at episode {ss.run_done:,} of {ss.run_target:,}.")


def cb_resume() -> None:
    if ss.status == PAUSED and ss.run_done < ss.run_target and maze_is_solvable():
        ss.status = TRAINING


def cb_reset_training(silent: bool = False) -> None:
    ss.agent = QLearningAgent(ss.env.n_states, params=current_params())
    reset_learning_state()
    if not silent:
        notify("success", "Training reset — the Q-table is all zeros again. The AI has forgotten everything.")


def cb_train_one() -> None:
    if ss.status == TRAINING or not maze_is_solvable():
        return
    result = train_one_episode()
    if ss.status != PAUSED:
        ss.status = COMPLETED
    outcome = "reached the goal" if result.success else "did not reach the goal"
    ss.playback = make_playback(result, f"Training episode #{result.episode}",
                                f"ε = {result.epsilon:.2f} · learning on · {outcome}", accent=viz.VIOLET)
    notify("success", f"Episode #{result.episode}: {outcome} in {result.steps} steps "
                      f"(reward {result.total_reward:+.0f}).")


def cb_test() -> None:
    if ss.status == TRAINING or not maze_is_solvable():
        return
    ss.after = run_evaluation()
    sample = ss.after["sample"]
    trained = ss.agent.episodes_trained
    label = "Trained AI — test run" if trained else "Untrained AI — test run"
    ss.playback = make_playback(sample, label, f"ε = 0 · learning off · {trained:,} episodes of experience")
    notify("success", f"Test complete: {ss.after['success_rate']:.0f}% success over {EVAL_TRIALS} runs.")


def set_environment(env: MazeEnvironment) -> None:
    ss.env = env
    ss.maze_version += 1
    ss.editor_nonce += 1
    ss.agent = QLearningAgent(env.n_states, params=current_params())
    reset_learning_state()


def cb_new_maze() -> None:
    seed = int(ss.m_seed) or None
    try:
        env = MazeEnvironment.generate(int(ss.m_rows), int(ss.m_cols), style=ss.m_style,
                                       density=float(ss.m_density), seed=seed)
    except MazeError as exc:
        notify("error", str(exc))
        return
    set_environment(env)
    ss.env_original = env.copy()
    notify("success", f"New {env.rows}×{env.cols} maze generated — shortest route is "
                      f"{env.shortest_path_length()} steps. Training was reset.")


def cb_reset_maze() -> None:
    set_environment(ss.env_original.copy())
    notify("success", "Maze restored to its original layout. Training was reset.")


def on_maze_edited(message: str) -> None:
    ss.maze_version += 1
    if ss.keep_knowledge and ss.agent.n_states == ss.env.n_states:
        if ss.baseline is not None:
            ss.baseline_stale = True
        ss.after = None
        ss.playback = None
        ss.lab_run = None
        note = " Learned knowledge kept — keep training to watch the AI adapt."
    else:
        had_progress = ss.agent.episodes_trained > 0
        ss.agent = QLearningAgent(ss.env.n_states, params=current_params())
        reset_learning_state()
        note = " Training was reset for the new layout." if had_progress else ""
    if not ss.env.is_solvable():
        if ss.status == TRAINING:
            ss.status = PAUSED
        notify("warning", f"{message} The goal is now unreachable — training is blocked until a path exists.")
    else:
        notify("success", message + note)


def apply_edit(tool: str, r: int, c: int) -> None:
    try:
        if tool == "Set start":
            message = ss.env.set_start(r, c)
        elif tool == "Set goal":
            message = ss.env.set_goal(r, c)
        else:
            message = ss.env.toggle_wall(r, c)
    except MazeError as exc:
        notify("warning", str(exc))
        return
    on_maze_edited(message)


def cb_clear_walls() -> None:
    on_maze_edited(ss.env.clear_walls())


def cb_record_baseline() -> None:
    if maze_is_solvable():
        ensure_baseline()
        notify("success", "'Before training' performance recorded.")


def cb_defaults() -> None:
    for key, value in {**PARAM_DEFAULTS, **TRAINING_DEFAULTS}.items():
        ss[key] = value
    notify("success", "Learning parameters and training settings restored to defaults.")


def cb_factory_reset() -> None:
    for key, value in WIDGET_DEFAULTS.items():
        ss[key] = value
    env = MazeEnvironment.generate(**DEFAULT_MAZE)
    set_environment(env)
    ss.env_original = env.copy()
    ss.demo_step = 0
    ss.nav = PAGES[0]
    notify("success", "Everything reset: default maze, default settings, empty Q-table.")


def cb_goto(page: str) -> None:
    ss.nav = page


# ---- demo mode ---------------------------------------------------------------
def cb_demo_start() -> None:
    choice = ss.demo_maze
    if choice != DEMO_MAZES[0]:
        size = int(choice.split()[1].split("×")[0])
        env = MazeEnvironment.generate(size, size, style="maze")
        set_environment(env)
        ss.env_original = env.copy()
    else:
        cb_reset_training(silent=True)
    if not maze_is_solvable():
        return
    ss.demo_step = 1
    notify("info", "Live demonstration started — the AI begins with an empty Q-table.")


def cb_demo_move(delta: int) -> None:
    ss.demo_step = int(np.clip(ss.demo_step + delta, 1, len(edu.DEMO_STEPS)))


def cb_demo_exit() -> None:
    ss.demo_step = 0


# ---- save / load -------------------------------------------------------------
def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"Cannot serialise {type(obj).__name__}")


def episode_to_dict(result: EpisodeResult) -> dict:
    data = asdict(result)
    data["path"] = [list(map(int, p)) for p in result.path]
    return data


def episode_from_dict(data: dict) -> EpisodeResult:
    data = dict(data)
    data["path"] = [tuple(map(int, p)) for p in data.get("path", [])]
    return EpisodeResult(**data)


def evaluation_to_dict(ev):
    if ev is None:
        return None
    out = dict(ev)
    out["sample"] = episode_to_dict(ev["sample"])
    return out


def evaluation_from_dict(data):
    if not data:
        return None
    out = dict(data)
    out["sample"] = episode_from_dict(data["sample"])
    return out


def session_json() -> bytes:
    payload = {
        "app": edu.APP_NAME,
        "format_version": 1,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "maze": ss.env.to_dict(),
        "maze_original": ss.env_original.to_dict(),
        "agent": ss.agent.to_dict(),
        "history": ss.history,
        "status": ss.status,
        "run_target": int(ss.run_target),
        "run_done": int(ss.run_done),
        "baseline": evaluation_to_dict(ss.baseline),
        "baseline_stale": bool(ss.baseline_stale),
        "settings": {k: ss[k] for k in list(PARAM_DEFAULTS) + list(TRAINING_DEFAULTS)},
    }
    return json.dumps(payload, default=_json_default).encode("utf-8")


def cb_load_session() -> None:
    upload = ss.get("upload_session")
    if upload is None:
        notify("warning", "Choose a saved .json session file first.")
        return
    try:
        data = json.loads(upload.getvalue().decode("utf-8"))
        if data.get("app") != edu.APP_NAME:
            raise ValueError("this is not a REINFORCE AI save file")
        env = MazeEnvironment.from_dict(data["maze"])
        original = MazeEnvironment.from_dict(data.get("maze_original", data["maze"]))
        agent = QLearningAgent.from_dict(data["agent"])
        if agent.n_states != env.n_states:
            raise ValueError("the Q-table does not match the maze size")
        history = []
        for rec in data.get("history", []):
            history.append({col: rec[col] for col in an.BASE_COLUMNS})
        baseline = evaluation_from_dict(data.get("baseline"))
    except (ValueError, KeyError, TypeError, UnicodeDecodeError, MazeError) as exc:
        notify("error", f"Could not load the file: {exc}")
        return

    ss.env, ss.env_original, ss.agent = env, original, agent
    reset_learning_state()
    ss.history = history
    ss.baseline = baseline
    ss.baseline_stale = bool(data.get("baseline_stale", False))
    ss.run_target = int(data.get("run_target", 0))
    ss.run_done = min(int(data.get("run_done", 0)), ss.run_target)
    status = data.get("status", NOT_STARTED)
    if status == TRAINING:
        status = PAUSED
    if status not in (NOT_STARTED, PAUSED, COMPLETED):
        status = COMPLETED if agent.episodes_trained else NOT_STARTED
    ss.status = status
    params = agent.params
    ss.p_alpha, ss.p_gamma, ss.p_epsilon = params.alpha, params.gamma, params.epsilon
    ss.p_decay, ss.p_min_eps, ss.p_max_steps = params.epsilon_decay, params.min_epsilon, params.max_steps
    settings = data.get("settings", {})
    if settings.get("t_episodes") in EPISODE_CHOICES:
        ss.t_episodes = settings["t_episodes"]
    if settings.get("t_speed") in SPEED_OPTIONS:
        ss.t_speed = settings["t_speed"]
    ss.maze_version += 1
    ss.editor_nonce += 1
    notify("success", f"Session loaded: {env.rows}×{env.cols} maze, {agent.episodes_trained:,} episodes of training.")


# ============================================================================
# Shared UI pieces
# ============================================================================
def fmt_pct(value) -> str:
    return "—" if value is None else f"{value:.0f}%"


def greedy_route():
    agent = ss.agent
    if agent.episodes_trained == 0:
        return None
    return agent.greedy_route(ss.env)


def render_stat_cards(df: pd.DataFrame, summ: dict, route, optimal) -> None:
    agent = ss.agent
    total = agent.episodes_trained
    last = df.iloc[-1] if not df.empty else None

    if ss.run_target:
        current, current_sub = f"{ss.run_done:,} / {ss.run_target:,}", "episode of the current run"
        progress = ss.run_done / ss.run_target
    else:
        current = f"#{total:,}" if total else "—"
        current_sub = "no training run started" if not total else "single-episode training"
        progress = 1.0 if total else 0.0

    if route is None:
        route_val, route_sub, route_meter = "—", "train the AI first", 0.0
    elif route.success and optimal:
        route_val = f"{route.steps} steps"
        route_meter = optimal / route.steps
        route_sub = f"{route_meter * 100:.0f}% efficient · shortest = {optimal}"
    else:
        route_val, route_sub, route_meter = "Not found", "greedy policy can't reach the goal yet", 0.0

    recent_steps = summ["recent_avg_steps_success"]
    cards = [
        ui.stat_card("Total Episodes", f"{total:,}", f"{agent.total_updates:,} Q-table updates", "🧪",
                     "#3B82F6", tip=edu.TOOLTIPS["episodes"]),
        ui.stat_card("Current Episode", current, current_sub, "⏱️", "#22D3EE"),
        ui.stat_card("Total Reward", an.fmt_num(summ["total_reward"], 0),
                     f"last episode: {last['reward']:+.0f}" if last is not None else "sum over all episodes",
                     "💎", "#A78BFA"),
        ui.stat_card("Success Rate", fmt_pct(summ["success_rate"] if total else None),
                     f"last 100 episodes: {summ['recent_success_rate']:.0f}%" if total else "no episodes yet",
                     "🎯", "#34D399", meter=summ["recent_success_rate"] / 100 if total else 0,
                     tip=edu.TOOLTIPS["success_rate"]),
        ui.stat_card("Avg Steps to Goal", an.fmt_num(recent_steps, 1),
                     f"last 100 successful · shortest = {optimal if optimal else '—'}", "👣", "#F472B6"),
        ui.stat_card("Exploration ε", f"{agent.epsilon:.3f}",
                     f"{agent.epsilon * 100:.0f}% random · {(1 - agent.epsilon) * 100:.0f}% best-known moves",
                     "🧭", "#A78BFA", meter=agent.epsilon, tip=edu.TOOLTIPS["epsilon"]),
        ui.stat_card("Learning Progress", f"{progress * 100:.0f}%",
                     "of the current training run" if ss.run_target else "start a training run",
                     "📈", "#22D3EE", meter=progress),
        ui.stat_card("Learned Route", route_val, route_sub, "🗺️", "#FACC15", meter=route_meter,
                     tip=edu.TOOLTIPS["policy"]),
    ]
    html_block(ui.stat_grid(cards))


def render_training_controls(prefix: str) -> None:
    training = ss.status == TRAINING
    n = int(ss.t_episodes)
    row1 = st.columns(4)
    button("▶ Start Training", f"{prefix}_start", row1[0], on_click=cb_start, disabled=training,
           type="primary", help=f"Train for {n:,} episodes with live updates.")
    button("⏸ Pause Training", f"{prefix}_pause", row1[1], on_click=cb_pause, disabled=not training)
    button("⏯ Resume Training", f"{prefix}_resume", row1[2], on_click=cb_resume,
           disabled=ss.status != PAUSED or ss.run_done >= ss.run_target)
    button("⟲ Reset Training", f"{prefix}_reset", row1[3], on_click=cb_reset_training,
           help="Erase the Q-table and all statistics. The maze stays the same.")
    row2 = st.columns(4)
    button("➕ Train One Episode", f"{prefix}_one", row2[0], on_click=cb_train_one, disabled=training,
           help="Run a single learning episode and replay it.")
    instant = button(f"⚡ Instant Train {n:,}", f"{prefix}_instant", row2[1], disabled=training,
                     help="Train the selected number of episodes immediately (no live animation).")
    button("🧪 Test Trained AI", f"{prefix}_test", row2[2], on_click=cb_test, disabled=training,
           help=f"{EVAL_TRIALS} test runs with ε = 0 and learning switched off.")
    button("🎲 Generate New Maze", f"{prefix}_maze", row2[3], on_click=cb_new_maze,
           help="Uses the size and style chosen in Maze Lab. Resets training.")
    if ss.run_target and ss.status in (TRAINING, PAUSED):
        st.progress(min(1.0, ss.run_done / ss.run_target),
                    text=f"Training run: episode {ss.run_done:,} of {ss.run_target:,}")
    st.caption(f"Run length **{n:,} episodes** · speed **{ss.t_speed}** — change both in the sidebar under "
               f"*Training setup*.")
    if instant:
        instant_train(n)


def render_live_maze(height: int = 470) -> None:
    env = ss.env
    if ss.status == TRAINING and ss.last_episode is not None:
        ep = ss.last_episode
        title = (f"LIVE · episode #{ep.episode} · {'✅ goal' if ep.success else '❌ no goal'} · "
                 f"{ep.steps} steps · reward {ep.total_reward:+.0f} · ε {ep.epsilon:.2f}")
        color = viz.CYAN if ep.success else "#F87171"
        plot(viz.maze_figure(env, path=ep.path, agent_pos=ep.path[-1], height=height, title=title,
                             path_color=color))
        return
    pb = ss.playback
    if pb is not None and pb.get("maze_version") == ss.maze_version:
        show_player(env, pb["result"], pb["title"], pb["subtitle"], accent=pb["accent"])
    else:
        hint = ("Press ▶ Start Training to begin" if ss.agent.episodes_trained == 0
                else "Press 🧪 Test Trained AI to watch the learned route")
        show_player(env, None, "Maze ready", hint)


def selected_cells(event) -> list:
    """Extract clicked cells (row, col) from a Plotly selection event."""
    try:
        selection = event["selection"] if hasattr(event, "__getitem__") else event.selection
        points = selection["points"]
    except (KeyError, TypeError, AttributeError):
        return []
    cells = []
    for point in points or []:
        try:
            x, y = point.get("x"), point.get("y")
        except AttributeError:
            continue
        if x is None or y is None:
            continue
        cells.append((int(round(float(y))), int(round(float(x)))))
    return cells


def eval_metrics(res: dict):
    return [
        ("Success rate", f"{res['success_rate']:.0f}%"),
        ("Avg steps / run", f"{res['avg_steps']:.1f}"),
        ("Avg reward / run", f"{res['avg_reward']:+.1f}"),
        ("Wall hits / run", f"{res['avg_wall_hits']:.1f}"),
        ("Steps when successful", an.fmt_num(res["avg_steps_success"], 1)),
        ("Experience", f"{res['episodes_trained']:,} episodes"),
    ]


def improvement_items(before: dict, after: dict, optimal):
    items = [
        ("Success rate", f"{before['success_rate']:.0f}% → {after['success_rate']:.0f}%",
         after["success_rate"] > before["success_rate"]),
    ]
    b_steps, a_steps = before["avg_steps"], after["avg_steps"]
    change = (a_steps - b_steps) / b_steps * 100 if b_steps else 0.0
    items.append(("Average steps", f"{b_steps:.0f} → {a_steps:.0f} ({change:+.0f}%)", a_steps < b_steps))
    items.append(("Average reward", f"{before['avg_reward']:+.0f} → {after['avg_reward']:+.0f}",
                  after["avg_reward"] > before["avg_reward"]))
    items.append(("Wall hits / run", f"{before['avg_wall_hits']:.1f} → {after['avg_wall_hits']:.1f}",
                  after["avg_wall_hits"] < before["avg_wall_hits"]))
    if after["best_steps"] is not None and optimal:
        items.append(("Learned vs shortest route", f"{after['best_steps']} vs {optimal} steps",
                      after["best_steps"] == optimal))
    return items


def performance_story(df: pd.DataFrame, summ: dict, optimal, route) -> list:
    """Plain-English summary built only from recorded numbers."""
    if df.empty:
        return ["No training episodes have been recorded yet."]
    wins = int(df["success"].sum())
    lines = [f"The agent trained for **{len(df):,} episodes** and reached the goal in **{wins:,}** of them "
             f"(**{summ['success_rate']:.1f}%**)."]
    if summ["first_success"] is not None:
        lines.append(f"Its first successful episode was **episode {summ['first_success']:,}**.")
    phases = an.phase_comparison(df)
    if not phases.empty:
        first, last = phases.columns[0], phases.columns[1]
        lines.append(
            f"{first}: success {phases.loc['Success rate (%)', first]:.0f}%, average reward "
            f"{phases.loc['Average reward', first]:+.1f}, {phases.loc['Average steps', first]:.0f} steps per episode. "
            f"{last}: success {phases.loc['Success rate (%)', last]:.0f}%, average reward "
            f"{phases.loc['Average reward', last]:+.1f}, {phases.loc['Average steps', last]:.0f} steps per episode.")
    if summ["convergence_episode"] is not None:
        lines.append(f"From **episode {summ['convergence_episode']:,}** the agent reached the goal 20 times in a row "
                     f"using no more than 10% extra steps compared with the shortest route ({optimal} steps).")
    elif optimal:
        lines.append("The agent has not yet reached the goal 20 times in a row within 10% of the shortest route.")
    if route is not None:
        if route.success:
            lines.append(f"Its current learned (greedy) route takes **{route.steps} steps**; "
                         f"the shortest possible route is **{optimal} steps**.")
        else:
            lines.append("Its greedy policy cannot reach the goal yet — more training is needed.")
    return lines


# ============================================================================
# Sidebar
# ============================================================================
def render_sidebar():
    with st.sidebar:
        html_block(ui.brand(edu.APP_NAME, edu.APP_SUBTITLE))
        status_box = st.container()
        st.radio("Navigation", PAGES, key="nav", label_visibility="collapsed")
        st.divider()
        with st.expander("⚙️ Learning parameters", expanded=True):
            st.slider("Learning rate (α)", 0.01, 1.0, step=0.01, key="p_alpha", help=edu.TOOLTIPS["alpha"])
            st.slider("Discount factor (γ)", 0.0, 0.99, step=0.01, key="p_gamma", help=edu.TOOLTIPS["gamma"])
            st.slider("Exploration rate (ε) — initial", 0.0, 1.0, step=0.01, key="p_epsilon",
                      help=edu.TOOLTIPS["epsilon"])
            st.slider("Epsilon decay (per episode)", 0.90, 1.0, step=0.001, key="p_decay", format="%.3f",
                      help=edu.TOOLTIPS["epsilon_decay"])
            st.slider("Minimum epsilon", 0.0, 0.5, step=0.01, key="p_min_eps", help=edu.TOOLTIPS["min_epsilon"])
            st.slider("Max steps per episode", 10, 2000, step=10, key="p_max_steps",
                      help=edu.TOOLTIPS["max_steps"])
            if ss.p_min_eps > ss.p_epsilon:
                st.caption("⚠️ Minimum ε is above the initial ε, so it is capped at the initial ε.")
            st.caption("α, γ, decay, minimum ε and the step limit apply immediately. "
                       "The initial ε applies after **Reset Training**.")
        with st.expander("🏋️ Training setup", expanded=True):
            st.selectbox("Episodes per training run", EPISODE_CHOICES, key="t_episodes",
                         format_func=lambda v: f"{v:,} episodes", help=edu.TOOLTIPS["episodes"])
            st.radio("Training speed", SPEED_OPTIONS, key="t_speed", format_func=SPEED_LABELS.get,
                     help=edu.TOOLTIPS["speed"])
        button("↺ Reset to default settings", "sb_defaults", on_click=cb_defaults)
        st.caption(f"{edu.APP_NAME} v{APP_VERSION} · Q-Learning maze laboratory")
    return status_box


def fill_status_box(box) -> None:
    with box:
        html_block(f'<div style="margin:0 0 10px">{ui.status_pill(ss.status)}</div>')
        if ss.run_target and ss.status in (TRAINING, PAUSED):
            st.progress(min(1.0, ss.run_done / ss.run_target),
                        text=f"Episode {ss.run_done:,} / {ss.run_target:,}")
        if ss.status == TRAINING:
            button("⏸ Pause training", "sb_pause", on_click=cb_pause)
        elif ss.status == PAUSED and ss.run_done < ss.run_target:
            button("⏯ Resume training", "sb_resume", on_click=cb_resume)


# ============================================================================
# Pages
# ============================================================================
def page_dashboard() -> None:
    env, agent = ss.env, ss.agent
    optimal = env.shortest_path_length()
    df = history_df()
    summ = an.training_summary(df, optimal)
    route = greedy_route()

    html_block(ui.hero(edu.APP_NAME, edu.TAGLINE, ss.status, chips=[
        ("Maze", f"{env.rows}×{env.cols}"),
        ("States", f"{len(env.free_cells())} open cells"),
        ("Actions", "↑ ↓ ← →"),
        ("Shortest route", f"{optimal} steps" if optimal else "none"),
        ("Algorithm", "Q-Learning · ε-greedy"),
    ]))
    render_stat_cards(df, summ, route, optimal)

    html_block(ui.section_title("Mission Control", "Train, pause, test and reset the AI"))
    render_training_controls("dash")

    left, right = st.columns([1.1, 1])
    with left:
        html_block(ui.section_title("Maze Environment"))
        render_live_maze()
        html_block(ui.legend())
    with right:
        html_block(ui.section_title("Live Learning Curves"))
        small = an.downsample(df)
        plot(viz.reward_chart(small, height=255, title="Reward per episode"))
        plot(viz.success_chart(small, height=255, title="Success rate"))
        q1, q2, q3 = st.columns(3)
        button("⚖️ Before vs After", "dash_go_cmp", q1, on_click=cb_goto, args=(PAGES[2],))
        button("🧠 See the Q-table", "dash_go_viz", q2, on_click=cb_goto, args=(PAGES[3],))
        button("🎓 Demo Mode", "dash_go_demo", q3, on_click=cb_goto, args=(PAGES[6],))


def page_maze_lab() -> None:
    env = ss.env
    html_block(ui.section_title("Maze Lab", "Design the environment the AI has to learn"))
    left, right = st.columns([1.3, 1])

    with right:
        with bordered():
            st.markdown("**🎲 Generate a maze**")
            c1, c2 = st.columns(2)
            c1.slider("Rows", MIN_SIZE, MAX_SIZE, key="m_rows")
            c2.slider("Columns", MIN_SIZE, MAX_SIZE, key="m_cols")
            st.selectbox("Maze style", list(MAZE_STYLES), key="m_style", format_func=MAZE_STYLES.get)
            if ss.m_style == "obstacles":
                st.slider("Obstacle density", 0.05, 0.45, step=0.05, key="m_density")
            st.number_input("Random seed (0 = a different maze every time)", min_value=0, max_value=999_999,
                            step=1, key="m_seed")
            b1, b2, b3 = st.columns(3)
            button("🎲 Generate", "lab_new", b1, on_click=cb_new_maze, type="primary",
                   help="Create a new random maze (always solvable). Resets training.")
            button("⟲ Reset Maze", "lab_reset", b2, on_click=cb_reset_maze,
                   help="Restore the layout to how it was when it was generated or loaded.")
            button("🧹 Clear Walls", "lab_clear", b3, on_click=cb_clear_walls)

        with bordered():
            st.markdown("**✏️ Edit the maze**")
            st.radio("Edit tool", EDIT_TOOLS, key="edit_tool", horizontal=True)
            st.checkbox("Keep learned knowledge when editing (watch the AI adapt)", key="keep_knowledge",
                        help=edu.TOOLTIPS["keep_knowledge"])
            if SUPPORTS_CLICK_EDIT:
                st.caption("👆 Click any cell in the maze to apply the selected tool — or use the form below.")
            with st.form("manual_edit", clear_on_submit=False):
                f1, f2 = st.columns(2)
                row = f1.number_input("Row", min_value=0, max_value=env.rows - 1, value=0, step=1)
                col = f2.number_input("Column", min_value=0, max_value=env.cols - 1, value=0, step=1)
                submitted = st.form_submit_button("Apply tool to this cell")
            if submitted:
                apply_edit(ss.edit_tool, int(row), int(col))
                st.rerun()

        optimal = env.shortest_path_length()
        checks = [
            ("Solvable", "✅ yes" if optimal is not None else "❌ goal unreachable"),
            ("Shortest route (BFS)", f"{optimal} steps" if optimal is not None else "—"),
            ("Open cells (states)", f"{len(env.free_cells())}"),
            ("Walls / obstacles", f"{env.wall_count}"),
            ("Start → Goal", f"{env.start} → {env.goal}"),
        ]
        html_block(ui.vs_card("after", "Environment check", checks))
        st.checkbox("Show the shortest possible route (computed by BFS — never shown to the AI)", key="show_bfs")

    with left:
        title = f"{env.rows}×{env.cols} maze · tool: {ss.edit_tool}"
        fig = viz.maze_figure(env, agent_pos=env.start, clickable=SUPPORTS_CLICK_EDIT, title=title,
                              highlight_cells=env.shortest_path() if ss.show_bfs else None)
        if SUPPORTS_CLICK_EDIT:
            event = plot(fig, key=f"maze_editor_{ss.editor_nonce}", on_select="rerun", selection_mode="points")
            cells = selected_cells(event)
            if cells:
                apply_edit(ss.edit_tool, *cells[0])
                ss.editor_nonce += 1       # fresh chart = cleared selection
                st.rerun()
        else:
            plot(fig)
        html_block(ui.legend())

    html_block(ui.section_title("Watch the AI in this maze",
                                "One test run with learning switched off — untrained AI wanders, trained AI follows its route"))
    c1, c2 = st.columns([1, 3])
    with c1:
        if button("▶ Run the AI once", "lab_watch", disabled=ss.status == TRAINING or optimal is None):
            run = ss.agent.run_episode(env, learn=False, epsilon=0.0)
            ss.lab_run = {"result": run, "maze_version": ss.maze_version,
                          "trained": ss.agent.episodes_trained}
        st.caption("Uses the AI's current Q-table. Train it on the Dashboard and come back to compare.")
    with c2:
        lab = ss.get("lab_run")
        if lab and lab["maze_version"] == ss.maze_version:
            label = "Trained AI" if lab["trained"] else "Untrained AI (random moves)"
            show_player(env, lab["result"], label, f"{lab['trained']:,} episodes of experience",
                        accent=viz.CYAN if lab["trained"] else "#F87171")


def page_before_after() -> None:
    env, agent = ss.env, ss.agent
    optimal = env.shortest_path_length()
    html_block(ui.section_title("Before vs After Learning",
                                "The same AI on the same maze — first with an empty memory, then after training"))
    html_block(ui.callout(
        f"<b>How this is measured:</b> just before the first training episode, the untrained AI (all Q-values = 0) "
        f"makes {EVAL_TRIALS} test runs. Afterwards the trained AI makes {EVAL_TRIALS} test runs with ε = 0 and learning "
        f"switched off. All numbers below are recorded from those runs; the replays show the median-reward run.",
        "info", allow_html=True))

    base = ss.baseline
    if base is None:
        if agent.episodes_trained == 0:
            html_block(ui.callout("No 'before' snapshot yet. It is recorded automatically when training starts, "
                                  "or you can record it now.", "warn"))
            button("📸 Record 'Before' performance now", "cmp_record", on_click=cb_record_baseline, type="primary")
        else:
            html_block(ui.callout("This AI was already trained (or loaded) without a 'before' snapshot. "
                                  "Reset training to record a fresh comparison.", "warn"))
            button("⟲ Reset Training", "cmp_reset", on_click=cb_reset_training)
        return
    if ss.baseline_stale:
        html_block(ui.callout("The maze was edited after the 'before' snapshot was recorded, so the two sides "
                              "were measured on different layouts. Reset training for a fair comparison.", "warn"))

    col_b, col_a = st.columns(2)
    with col_b:
        html_block(ui.vs_card("before", "Before training · untrained AI", eval_metrics(base),
                              note=f"Recorded at {base['time']} · {base['trials']} runs · "
                                   f"step limit {base['max_steps']}"))
        show_player(env, base["sample"], "BEFORE — random exploration", "empty Q-table · median run",
                    accent="#F87171", speed_ms=45, max_width=480)
    with col_a:
        if agent.episodes_trained == 0:
            html_block(ui.callout("Now train the AI (Dashboard → ▶ Start Training) and come back here to see "
                                  "the 'after' results appear next to these.", "info"))
            button("🏠 Go to Dashboard", "cmp_go_dash", on_click=cb_goto, args=(PAGES[0],))
            return
        if ss.status == TRAINING:
            st.caption("Training is running — the 'after' results refresh with every update.")
        after = get_after()
        html_block(ui.vs_card("after", "After training · trained AI", eval_metrics(after),
                              note=f"Measured at {after['time']} · {after['trials']} runs · "
                                   f"step limit {after['max_steps']}"))
        show_player(env, after["sample"], "AFTER — learned route", "ε = 0 · learning off",
                    accent=viz.CYAN, speed_ms=140, max_width=480)

    html_block(ui.section_title("Measured improvement"))
    html_block(ui.delta_row(improvement_items(base, after, optimal)))
    plot(viz.comparison_bars(base, after, optimal, height=320))

    html_block(ui.section_title("Training history: early vs late episodes",
                                "From the recorded training episodes (exploration still switched on)"))
    df = history_df()
    phases = an.phase_comparison(df)
    if phases.empty:
        st.caption("Train for at least 10 episodes to see this table.")
    else:
        st.dataframe(phases, **DATAFRAME_KW)


def q_table_frame(env: MazeEnvironment, agent: QLearningAgent) -> pd.DataFrame:
    rows = []
    for r, c in env.free_cells():
        s = env.state_of((r, c))
        q = agent.q_table[s]
        learned = bool(np.any(q != 0))
        if (r, c) == env.goal:
            best = "🏁 goal (episode ends)"
        elif learned:
            a = int(np.argmax(q))
            best = f"{ACTION_ARROWS[a]} {ACTION_NAMES[a]}"
        else:
            best = "— not explored yet"
        rows.append({"State": s, "Cell (row, col)": f"({r}, {c})",
                     "Q ↑ Up": q[0], "Q ↓ Down": q[1], "Q ← Left": q[2], "Q → Right": q[3],
                     "Best action": best, "Visits": int(agent.visit_counts[s])})
    return pd.DataFrame(rows)


def page_visualization() -> None:
    env, agent = ss.env, ss.agent
    optimal = env.shortest_path_length()
    html_block(ui.section_title("AI Visualization", "Look inside the AI's memory and behaviour"))
    if agent.episodes_trained == 0:
        html_block(ui.callout("The AI has not trained yet, so every Q-value is 0. Train it on the Dashboard "
                              "and these views come alive.", "info"))
    route = greedy_route()
    df = history_df()
    small = an.downsample(df)
    height = viz._maze_height(env, 560)
    tabs = st.tabs(["📋 Q-Table", "🧭 Policy Map", "👣 Agent Path", "🔀 Exploration vs Exploitation",
                    "💎 Reward Progress", "🎯 Success Rate", "🔥 State Visits"])

    with tabs[0]:
        st.markdown(f"**{agent.learned_state_count()} of {len(env.free_cells())} states updated** · "
                    f"{agent.total_updates:,} updates so far. " + edu.TOOLTIPS["q_table"])
        plot(viz.action_value_maps(env, agent, height=max(260, min(360, 22 * env.rows + 60))))
        c1, c2 = st.columns([1.6, 1])
        with c1:
            table = q_table_frame(env, agent)
            number = st.column_config.NumberColumn(format="%.2f")
            st.dataframe(table, hide_index=True, height=380,
                         column_config={k: number for k in ["Q ↑ Up", "Q ↓ Down", "Q ← Left", "Q → Right"]},
                         **DATAFRAME_KW)
        with c2:
            cells = env.free_cells()
            default = cells.index(env.start) if env.start in cells else 0
            idx = st.selectbox("Inspect one state", range(len(cells)), index=default,
                               format_func=lambda i: f"Row {cells[i][0]}, Col {cells[i][1]}"
                               + ("  (start)" if cells[i] == env.start else "")
                               + ("  (goal)" if cells[i] == env.goal else ""))
            state = env.state_of(cells[idx])
            plot(viz.q_values_bar(agent.q_table[state], title=f"Q-values in state {state}"))
            if agent.state_is_learned(state):
                best = int(np.argmax(agent.q_table[state]))
                st.caption(f"Best action here: **{ACTION_ARROWS[best]} {ACTION_NAMES[best]}** · "
                           f"visited {int(agent.visit_counts[state]):,} times.")
            else:
                st.caption("This state has never been updated — all four moves still look equally good.")

    with tabs[1]:
        st.caption("Background colour = how valuable the AI thinks each cell is, V(s) = max Q(s,a). "
                   "Arrow = the move it prefers there. The dotted gold line is its current greedy route. "
                   "Hover over a cell to see all four Q-values.")
        plot(viz.policy_figure(env, agent, height=height, title="Policy heatmap — preferred action in every cell",
                               route=route.path if route is not None and route.success else None))

    with tabs[2]:
        st.radio("Which path?", PATH_SOURCES, key="path_source", horizontal=True)
        source = ss.path_source
        result, label = None, source
        if source == PATH_SOURCES[0] and route is not None:
            result = route
        elif source == PATH_SOURCES[1] and ss.last_episode is not None:
            result = ss.last_episode
            label = f"Training episode #{result.episode}"
        elif source == PATH_SOURCES[2] and ss.after is not None and ss.after["maze_version"] == ss.maze_version:
            result = ss.after["sample"]
        elif source == PATH_SOURCES[3] and ss.baseline is not None:
            result = ss.baseline["sample"]
        if result is None:
            st.info("No path of this kind has been recorded yet.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                outcome = "reached the goal" if result.success else "did not reach the goal"
                plot(viz.maze_figure(env, path=result.path, agent_pos=result.path[-1], height=height,
                                     title=f"{label}: {outcome} · {result.steps} steps · "
                                           f"reward {result.total_reward:+.0f}"))
            with c2:
                show_player(env, result, label, f"{result.steps} steps · {result.wall_hits} wall hits",
                            autoplay=False, max_width=480)

    with tabs[3]:
        plot(viz.epsilon_chart(small, height=330))
        plot(viz.explore_exploit_area(small, height=300))
        st.caption("Early on most moves are random (exploration). As ε decays, the AI increasingly picks the "
                   "move with the highest Q-value (exploitation).")

    with tabs[4]:
        plot(viz.reward_chart(small, height=330, title="Reward per episode (with moving average)"))
        plot(viz.cumulative_reward_chart(small, height=280))

    with tabs[5]:
        plot(viz.success_chart(small, height=330, title="Success rate over time"))
        plot(viz.steps_chart(small, optimal, height=300, title="Steps per episode"))

    with tabs[6]:
        st.caption("How often the AI has stood on each cell during training (log scale). "
                   "Early exploration spreads everywhere; later visits concentrate on the learned route.")
        plot(viz.visit_heatmap(env, agent, height=height, title="State visit counts"))


def page_analytics() -> None:
    env = ss.env
    optimal = env.shortest_path_length()
    html_block(ui.section_title("Learning Analytics", "Every number below is computed from recorded training episodes"))
    st.slider("Moving-average window (episodes)", 5, 500, step=5, key="an_window")
    df = history_df()
    if df.empty:
        html_block(ui.callout("No training data yet. Train the AI on the Dashboard to populate this page.", "info"))
        return
    summ = an.training_summary(df, optimal)
    route = greedy_route()
    cards = [
        ui.stat_card("Episodes", f"{summ['episodes']:,}", f"{summ['total_steps']:,} moves in total", "🧪", "#3B82F6"),
        ui.stat_card("Overall success", f"{summ['success_rate']:.1f}%", "all episodes", "🎯", "#34D399",
                     meter=summ["success_rate"] / 100),
        ui.stat_card("Recent success", f"{summ['recent_success_rate']:.1f}%", "last 100 episodes", "📈", "#22D3EE",
                     meter=summ["recent_success_rate"] / 100),
        ui.stat_card("Avg reward (recent)", f"{summ['recent_avg_reward']:+.1f}",
                     f"overall {summ['avg_reward']:+.1f} · best {summ['best_reward']:+.0f}", "💎", "#A78BFA"),
        ui.stat_card("Avg steps (successful)", an.fmt_num(summ["recent_avg_steps_success"], 1),
                     f"shortest possible = {optimal}", "👣", "#F472B6"),
        ui.stat_card("First success", f"#{summ['first_success']:,}" if summ["first_success"] else "—",
                     "episode number", "🏁", "#FACC15"),
        ui.stat_card("Converged at", f"#{summ['convergence_episode']:,}" if summ["convergence_episode"] else "—",
                     "20 wins in a row, ≤10% extra steps", "✅", "#34D399"),
        ui.stat_card("Wall hits", f"{summ['total_wall_hits']:,}", "total during training", "🧱", "#F87171"),
    ]
    html_block(ui.stat_grid(cards))

    small = an.downsample(df)
    c1, c2 = st.columns(2)
    with c1:
        plot(viz.reward_chart(small, height=320, title="Episode-wise reward (with moving average)"))
    with c2:
        plot(viz.success_chart(small, height=320, title="Success rate"))
    c1, c2 = st.columns(2)
    with c1:
        plot(viz.steps_chart(small, optimal, height=320, title="Average steps per successful episode"))
    with c2:
        plot(viz.epsilon_chart(small, height=320, title="Exploration decay"))
    c1, c2 = st.columns(2)
    with c1:
        plot(viz.cumulative_reward_chart(small, height=280))
    with c2:
        plot(viz.wall_hits_chart(small, height=280))

    html_block(ui.section_title("Training performance summary"))
    story = performance_story(df, summ, optimal, route)
    st.markdown("\n".join(f"- {line}" for line in story))
    phases = an.phase_comparison(df)
    if not phases.empty:
        st.dataframe(phases, **DATAFRAME_KW)

    html_block(ui.section_title("Training statistics", "One row per training episode"))
    st.dataframe(df.iloc[::-1].head(500), hide_index=True, height=300, **DATAFRAME_KW)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Download training statistics (CSV)", data=an.dataframe_to_csv(df),
                       file_name=f"reinforce_ai_training_{stamp}.csv", mime="text/csv", key="dl_csv", **DOWNLOAD_KW)
    d2.download_button("⬇️ Download results report (Markdown)", data=results_report(df, summ, optimal, route),
                       file_name=f"reinforce_ai_report_{stamp}.md", mime="text/markdown", key="dl_report",
                       **DOWNLOAD_KW)


def results_report(df, summ, optimal, route) -> bytes:
    env, agent, p = ss.env, ss.agent, ss.agent.params
    lines = [
        f"# {edu.APP_NAME} — Training Results",
        f"_Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}_",
        "",
        "## Environment",
        f"- Maze: {env.rows}×{env.cols}, {len(env.free_cells())} open cells, {env.wall_count} walls",
        f"- Start {env.start} → Goal {env.goal}; shortest possible route: {optimal} steps",
        f"- Rewards: goal {REWARD_GOAL:+.0f}, wall {REWARD_WALL:+.0f}, step {REWARD_STEP:+.0f}",
        "",
        "## Hyper-parameters",
        f"- α = {p.alpha}, γ = {p.gamma}, initial ε = {p.epsilon}, ε-decay = {p.epsilon_decay}, "
        f"min ε = {p.min_epsilon}, max steps = {p.max_steps}",
        "",
        "## Summary",
    ] + [f"- {line.replace('**', '')}" for line in performance_story(df, summ, optimal, route)]
    if ss.baseline is not None and agent.episodes_trained:
        after = get_after()
        lines += ["", f"## Before vs after ({EVAL_TRIALS} test runs each, learning off)",
                  "| Metric | Before | After |", "|---|---|---|"]
        for (k, b), (_, a) in zip(eval_metrics(ss.baseline), eval_metrics(after)):
            lines.append(f"| {k} | {b} | {a} |")
    return "\n".join(lines).encode("utf-8")


def q_update_calculator() -> None:
    c1, c2, c3 = st.columns(3)
    q_old = c1.number_input("Current Q(s,a)", value=0.0, step=1.0, key="calc_q")
    reward = c2.selectbox("Reward r", [REWARD_STEP, REWARD_WALL, REWARD_GOAL], key="calc_r",
                          format_func=lambda v: {REWARD_STEP: "−1 (normal move)", REWARD_WALL: "−10 (hit a wall)",
                                                 REWARD_GOAL: "+100 (reached goal)"}[v])
    terminal = reward == REWARD_GOAL
    max_next = c3.number_input("Best next value max Q(s′,a′)", value=10.0, step=1.0, key="calc_next",
                               disabled=terminal)
    c4, c5 = st.columns(2)
    alpha = c4.slider("α", 0.01, 1.0, 0.2, 0.01, key="calc_alpha")
    gamma = c5.slider("γ", 0.0, 0.99, 0.95, 0.01, key="calc_gamma")
    future = 0.0 if terminal else max_next
    target = reward + gamma * future
    error = target - q_old
    q_new = q_old + alpha * error
    st.latex(rf"Q_{{new}} = {q_old:.2f} + {alpha:.2f}\,\big[\,{reward:+.0f} + {gamma:.2f}\times{future:.2f} - {q_old:.2f}\,\big]"
             rf" = \mathbf{{{q_new:.2f}}}")
    note = " The goal ends the episode, so there is no future term." if terminal else ""
    st.caption(f"TD target = {target:.2f} · TD error (surprise) = {error:+.2f} → the estimate moves "
               f"{alpha * 100:.0f}% of the way from {q_old:.2f} towards {target:.2f}.{note}")


def page_learn() -> None:
    html_block(ui.section_title("Understand Reinforcement Learning", "The ideas behind REINFORCE AI, in plain language"))
    components.html(ui.rl_loop_diagram_html(280), height=300)
    html_block(ui.concept_grid(edu.CONCEPTS))

    html_block(ui.section_title("The Q-Learning equation"))
    with bordered():
        st.latex(edu.EQUATION_LATEX)
    html_block(ui.variable_table(edu.VARIABLES))

    html_block(ui.section_title("Try it yourself: one Q-Learning update",
                                "Change the numbers and see exactly what the AI does after a single move"))
    with bordered():
        q_update_calculator()

    html_block(ui.section_title("How Our AI Learns", "Everything that happens from an empty memory to a solved maze"))
    html_block(ui.timeline(edu.HOW_IT_LEARNS))

    html_block(ui.section_title("Where to see each idea in this app"))
    st.markdown(
        "- **States, actions, environment** → 🧩 Maze Lab\n"
        "- **Q-table and policy** → 🧠 AI Visualization (Q-Table and Policy Map tabs)\n"
        "- **Exploration vs exploitation** → 🧠 AI Visualization and 📊 Analytics (ε decay charts)\n"
        "- **Rewards** → 📊 Analytics (reward curves)\n"
        "- **Learning through experience** → ⚖️ Before vs After")


# ---------------------------------------------------------------- demo mode --
def demo_live_numbers() -> None:
    df = history_df()
    summ = an.training_summary(df, ss.env.shortest_path_length(), recent=50)
    cards = [
        ui.stat_card("Episodes trained", f"{ss.agent.episodes_trained:,}", "", "🧪", "#3B82F6"),
        ui.stat_card("Success (last 50)", f"{summ['recent_success_rate']:.0f}%" if not df.empty else "—", "",
                     "🎯", "#34D399", meter=summ["recent_success_rate"] / 100 if not df.empty else 0),
        ui.stat_card("Avg reward (last 50)",
                     f"{summ['recent_avg_reward']:+.0f}" if summ["recent_avg_reward"] is not None else "—",
                     "", "💎", "#A78BFA"),
        ui.stat_card("Exploration ε", f"{ss.agent.epsilon:.2f}", "", "🧭", "#22D3EE", meter=ss.agent.epsilon),
    ]
    html_block(ui.stat_grid(cards))


def demo_intro() -> None:
    env = ss.env
    st.markdown(f"## 🧠 {edu.APP_NAME}")
    html_block(f'<div class="big-quote">{ui.esc(edu.PROJECT_INTRO)}</div>')
    st.markdown("\n".join(f"- {p}" for p in edu.PROJECT_POINTS))
    c1, c2 = st.columns([1, 1])
    with c1:
        plot(viz.maze_figure(env, agent_pos=env.start, height=420, title="Today's maze"))
    with c2:
        html_block(ui.vs_card("after", "The challenge", [
            ("Maze size", f"{env.rows}×{env.cols}"),
            ("Open cells (states)", f"{len(env.free_cells())}"),
            ("Possible actions", "4 (↑ ↓ ← →)"),
            ("Shortest route", f"{env.shortest_path_length()} steps"),
            ("AI's knowledge", "none (Q = 0)"),
            ("Tech stack", "Python · NumPy · Streamlit · Plotly"),
        ]))
        html_block(ui.legend())


def demo_rl() -> None:
    st.markdown("## How does an AI learn from experience?")
    components.html(ui.rl_loop_diagram_html(280), height=300)
    c1, c2, c3 = st.columns(3)
    with c1:
        html_block(ui.vs_card("after", "📍 State", [("What", "the cell the agent is in")]))
    with c2:
        html_block(ui.vs_card("after", "🕹️ Action", [("What", "move up, down, left or right")]))
    with c3:
        html_block(ui.vs_card("after", "🏆 Reward", [("Goal", "+100"), ("Wall", "−10"), ("Step", "−1")]))
    st.markdown("After **every move** the AI updates one entry of its Q-table:")
    st.latex(edu.EQUATION_LATEX)
    st.markdown("It picks moves with **ε-greedy**: a random move with probability ε (*exploration*), otherwise "
                "the move with the highest Q-value (*exploitation*). ε starts at 1.0 and shrinks every episode.")


def demo_before() -> None:
    env = ss.env
    st.markdown("## Before training: an AI with no experience")
    if ss.baseline is None:
        ensure_baseline()
    base = ss.baseline
    if base is None:
        html_block(ui.callout("A 'before' snapshot can only be taken while the AI is untrained. "
                              "Use Reset Training (Dashboard) or restart the demo.", "warn"))
        return
    c1, c2 = st.columns([1.2, 1])
    with c1:
        show_player(env, base["sample"], "BEFORE — untrained AI", "all Q-values are 0 → random moves",
                    accent="#F87171", speed_ms=45)
    with c2:
        html_block(ui.vs_card("before", f"{base['trials']} test runs · untrained", eval_metrics(base)[:5]))
        wins = round(base["success_rate"] * base["trials"] / 100)
        st.markdown(f"In **{base['trials']}** attempts the untrained AI reached the goal **{wins}** times. "
                    f"On average it made **{base['avg_steps']:.0f} moves**, hit a wall **{base['avg_wall_hits']:.0f} "
                    f"times** and scored **{base['avg_reward']:+.0f}** reward. It has no idea where it is going.")


def demo_training() -> None:
    st.markdown("## Live training: watch it learn")
    training = ss.status == TRAINING
    c1, c2, c3, c4 = st.columns([1.3, 1, 1, 1])
    c1.selectbox("Episodes to train", DEMO_EPISODE_CHOICES, key="demo_episodes",
                 format_func=lambda v: f"{v:,} episodes", disabled=training)
    with c2:
        st.write("")
        button("▶ Train the AI", "demo_start", on_click=cb_start, kwargs={"n_key": "demo_episodes"},
               disabled=training, type="primary")
    with c3:
        st.write("")
        if training:
            button("⏸ Pause", "demo_pause", on_click=cb_pause)
        else:
            button("⏯ Resume", "demo_resume", on_click=cb_resume,
                   disabled=ss.status != PAUSED or ss.run_done >= ss.run_target)
    with c4:
        st.write("")
        st.caption(f"Speed: **{ss.t_speed}** (sidebar). Tip: *Watch* shows every single attempt.")
    if ss.run_target and ss.status in (TRAINING, PAUSED):
        st.progress(min(1.0, ss.run_done / ss.run_target),
                    text=f"Episode {ss.run_done:,} of {ss.run_target:,}")
    demo_live_numbers()
    left, right = st.columns([1.1, 1])
    with left:
        render_live_maze(height=440)
    with right:
        small = an.downsample(history_df())
        plot(viz.reward_chart(small, height=240, title="Reward per episode"))
        plot(viz.success_chart(small, height=240, title="Success rate"))


def demo_after() -> None:
    env = ss.env
    st.markdown("## After training: the same AI, the same maze")
    if ss.agent.episodes_trained == 0 or ss.baseline is None:
        html_block(ui.callout("Train the AI in the previous step first.", "warn"))
        return
    if ss.status == TRAINING:
        html_block(ui.callout("Training is still running — pause it or wait until it finishes.", "info"))
    after = get_after()
    base = ss.baseline
    c1, c2 = st.columns(2)
    with c1:
        html_block(ui.vs_card("before", "Before", eval_metrics(base)[:4]))
        show_player(env, base["sample"], "BEFORE", "untrained", accent="#F87171", speed_ms=45, max_width=470)
    with c2:
        html_block(ui.vs_card("after", "After", eval_metrics(after)[:4]))
        show_player(env, after["sample"], "AFTER", f"{ss.agent.episodes_trained:,} episodes of experience",
                    accent=viz.CYAN, max_width=470)
    html_block(ui.delta_row(improvement_items(base, after, env.shortest_path_length())))


def demo_conclusion() -> None:
    env, agent = ss.env, ss.agent
    st.markdown("## Conclusion: how the AI improved through experience")
    if agent.episodes_trained == 0 or ss.baseline is None:
        html_block(ui.callout("Complete the training step to generate the conclusion.", "warn"))
        return
    after, base = get_after(), ss.baseline
    optimal = env.shortest_path_length()
    df = history_df()
    summ = an.training_summary(df, optimal)
    route = greedy_route()
    route_text = (f"follows a **{route.steps}-step** route (the shortest possible is **{optimal}**)"
                  if route is not None and route.success else "is still searching for a complete route")
    first = (f"It found the goal for the first time in episode **{summ['first_success']:,}**"
             if summ["first_success"] else "It has not found the goal during training yet")
    html_block(ui.callout(
        f"Before training, the AI reached the goal in <b>{base['success_rate']:.0f}%</b> of test runs with an average "
        f"reward of <b>{base['avg_reward']:+.0f}</b>. After <b>{agent.episodes_trained:,}</b> episodes of trial and "
        f"error it reaches the goal in <b>{after['success_rate']:.0f}%</b> of test runs with an average reward of "
        f"<b>{after['avg_reward']:+.0f}</b>.", "good", allow_html=True))
    st.markdown(f"- {first}.\n"
                f"- Today the AI {route_text}, without ever being shown a map.\n"
                f"- It made **{summ['total_wall_hits']:,}** wall collisions while learning — each one taught it "
                f"something (−10 reward).\n"
                "- Nobody programmed the route: it emerged from rewards flowing backwards through the Q-table.")
    plot(viz.policy_figure(env, agent, height=viz._maze_height(env, 500),
                           title="What the AI learned: its preferred move in every cell",
                           route=route.path if route is not None and route.success else None))
    st.markdown("**Key takeaway:** reinforcement learning turns *experience* into *knowledge*. "
                "Trial → error → reward → update, repeated thousands of times, produces intelligent behaviour.")


DEMO_RENDERERS = [demo_intro, demo_rl, demo_before, demo_training, demo_after, demo_conclusion]


def page_demo() -> None:
    step = int(ss.demo_step)
    if step == 0:
        html_block(ui.hero("Demonstration Mode", "A guided, live presentation of REINFORCE AI for evaluation",
                           ss.status, kicker="College presentation",
                           chips=[(f"{i}.", label) for i, label in enumerate(edu.DEMO_STEPS, start=1)]))
        c1, c2 = st.columns([1.2, 1])
        with c1:
            st.markdown("This guided mode walks your audience through the project in six stages: the idea, how "
                        "reinforcement learning works, the untrained AI, **live training**, the before-vs-after "
                        "comparison and a conclusion generated from the real results.")
            st.selectbox("Maze for the demonstration", DEMO_MAZES, key="demo_maze")
            button("🚀 Start Live Demonstration", "demo_go", on_click=cb_demo_start, type="primary")
            st.caption("Starting clears the current training so the audience sees the AI begin from zero. "
                       "Save your progress first (💾 Save & Load) if you need it.")
        with c2:
            html_block(ui.callout("<b>Presenter tips</b><br>• Use <b>Normal</b> speed for 300–500 episodes, or "
                                  "<b>Watch</b> for the first 20–30 episodes to show individual attempts.<br>"
                                  "• Point at the reward curve rising and ε falling.<br>"
                                  "• Finish on the policy map: every arrow was learned, not programmed.",
                                  "info", allow_html=True))
        return

    html_block(ui.stepper(edu.DEMO_STEPS, step))
    DEMO_RENDERERS[step - 1]()
    st.divider()
    n1, n2, n3 = st.columns([1, 1, 1])
    button("◀ Previous", "demo_prev", n1, on_click=cb_demo_move, args=(-1,), disabled=step <= 1)
    needs_training = step == 4 and (ss.agent.episodes_trained == 0 or ss.status == TRAINING)
    if step < len(edu.DEMO_STEPS):
        button("Next ▶", "demo_next", n2, on_click=cb_demo_move, args=(1,), type="primary",
               disabled=needs_training,
               help="Train the AI (and let the run finish or pause it) to continue." if needs_training else None)
    else:
        button("🔁 Restart demonstration", "demo_restart", n2, on_click=cb_demo_exit)
    button("✖ Exit demo", "demo_exit", n3, on_click=cb_demo_exit)


def page_save_load() -> None:
    html_block(ui.section_title("Save & Load", "Keep your trained AI, share results, restore settings"))
    c1, c2 = st.columns(2)
    with c1:
        with bordered():
            st.markdown("**💾 Save training progress**")
            st.caption("Saves the maze, the full Q-table, ε, hyper-parameters, the training history and the "
                       "'before' snapshot into one JSON file.")
            st.download_button("⬇️ Download session (.json)", data=session_json(),
                               file_name=f"reinforce_ai_session_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                               mime="application/json", key="dl_session", type="primary", **DOWNLOAD_KW)
    with c2:
        with bordered():
            st.markdown("**📂 Load training progress**")
            st.file_uploader("Choose a REINFORCE AI session file", type=["json"], key="upload_session")
            button("Load session", "load_session", on_click=cb_load_session,
                   disabled=ss.get("upload_session") is None)

    html_block(ui.section_title("Download results"))
    df = history_df()
    optimal = ss.env.shortest_path_length()
    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Training statistics (CSV)", data=an.dataframe_to_csv(df),
                       file_name="reinforce_ai_training.csv", mime="text/csv", key="sl_csv",
                       disabled=df.empty, **DOWNLOAD_KW)
    d2.download_button("⬇️ Results report (Markdown)",
                       data=results_report(df, an.training_summary(df, optimal), optimal, greedy_route()),
                       file_name="reinforce_ai_report.md", mime="text/markdown", key="sl_report",
                       disabled=df.empty, **DOWNLOAD_KW)

    html_block(ui.section_title("Reset"))
    r1, r2 = st.columns(2)
    button("↺ Reset learning parameters to defaults", "sl_defaults", r1, on_click=cb_defaults)
    button("🧨 Factory reset (default maze, settings and empty Q-table)", "sl_factory", r2,
           on_click=cb_factory_reset)


PAGE_RENDERERS = {
    PAGES[0]: page_dashboard,
    PAGES[1]: page_maze_lab,
    PAGES[2]: page_before_after,
    PAGES[3]: page_visualization,
    PAGES[4]: page_analytics,
    PAGES[5]: page_learn,
    PAGES[6]: page_demo,
    PAGES[7]: page_save_load,
}


# ============================================================================
# Main
# ============================================================================
def main() -> None:
    init_state()
    persist_widget_state()
    status_box = render_sidebar()
    sync_agent_params()
    training_tick()
    fill_status_box(status_box)
    show_flash()
    PAGE_RENDERERS.get(ss.nav, page_dashboard)()

    if ss.status == TRAINING:
        # Keep the training loop going: short pause, then re-run the script.
        time.sleep(REFRESH_PAUSE.get(ss.t_speed, 0.05))
        st.rerun()


main()
