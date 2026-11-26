"""
environment.py
==============
The maze world that the AI agent lives in (the "Environment" in Reinforcement Learning).

Responsibilities
----------------
* Store the maze grid (empty cells and walls), the start cell and the goal cell.
* Turn grid positions into numbered *states* (one state per cell).
* Apply an *action* (Up / Down / Left / Right) and return the next state,
  the reward and whether the episode has finished.
* Generate random (always solvable) mazes and support editing.
* Compute the true shortest path with Breadth-First Search (BFS). BFS is never
  shown to the agent; it is only used to validate mazes and to measure how close
  the learned route is to the best possible route.

Reward scheme (from the project specification)
----------------------------------------------
* Reaching the goal          : +100  (episode ends)
* Hitting a wall / boundary  :  -10  (agent bounces back and stays where it was)
* Any other normal movement  :   -1  (encourages short routes)
"""

from __future__ import annotations

import random
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

Position = Tuple[int, int]

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
EMPTY = 0
WALL = 1

# Action index -> (row change, column change).  Row 0 is the TOP of the maze.
ACTIONS: Dict[int, Position] = {
    0: (-1, 0),  # Up
    1: (1, 0),   # Down
    2: (0, -1),  # Left
    3: (0, 1),   # Right
}
ACTION_NAMES = ["Up", "Down", "Left", "Right"]
ACTION_ARROWS = ["↑", "↓", "←", "→"]
N_ACTIONS = len(ACTIONS)

REWARD_GOAL = 100.0
REWARD_WALL = -10.0
REWARD_STEP = -1.0

MIN_SIZE = 5
MAX_SIZE = 20

MAZE_STYLES = {
    "maze": "Classic maze (corridors with a few loops)",
    "obstacles": "Scattered obstacles",
    "empty": "Open room (no obstacles)",
}


class MazeError(ValueError):
    """Raised for invalid maze configurations or edits."""


class MazeEnvironment:
    """A deterministic grid-world maze."""

    def __init__(self, grid, start: Position, goal: Position):
        grid = np.asarray(grid, dtype=np.int8)
        if grid.ndim != 2:
            raise MazeError("The maze grid must be two-dimensional.")
        rows, cols = grid.shape
        if not (MIN_SIZE <= rows <= MAX_SIZE and MIN_SIZE <= cols <= MAX_SIZE):
            raise MazeError(f"Maze size must be between {MIN_SIZE} and {MAX_SIZE} in each direction.")
        if not np.isin(grid, (EMPTY, WALL)).all():
            raise MazeError("The maze grid may only contain 0 (empty) and 1 (wall).")

        self.grid: np.ndarray = grid.copy()
        self.rows: int = int(rows)
        self.cols: int = int(cols)

        start = (int(start[0]), int(start[1]))
        goal = (int(goal[0]), int(goal[1]))
        for name, pos in (("Start", start), ("Goal", goal)):
            if not self.in_bounds(*pos):
                raise MazeError(f"{name} position {pos} is outside the maze.")
        if start == goal:
            raise MazeError("Start and goal must be different cells.")

        self.start: Position = start
        self.goal: Position = goal
        # Start and goal are always traversable.
        self.grid[start] = EMPTY
        self.grid[goal] = EMPTY
        self.agent_pos: Position = start

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------
    @property
    def n_states(self) -> int:
        """One state per grid cell (wall cells simply never get visited)."""
        return self.rows * self.cols

    def state_of(self, pos: Position) -> int:
        return pos[0] * self.cols + pos[1]

    def pos_of(self, state: int) -> Position:
        r, c = divmod(int(state), self.cols)
        return (r, c)

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_wall(self, r: int, c: int) -> bool:
        """Outside the grid counts as a wall (the maze has solid borders)."""
        return (not self.in_bounds(r, c)) or self.grid[r, c] == WALL

    def free_cells(self) -> List[Position]:
        return [(int(r), int(c)) for r, c in zip(*np.nonzero(self.grid == EMPTY))]

    @property
    def wall_count(self) -> int:
        return int((self.grid == WALL).sum())

    # ------------------------------------------------------------------
    # The RL interface: reset() and step()
    # ------------------------------------------------------------------
    def reset(self) -> int:
        """Put the agent back on the start cell and return the start state."""
        self.agent_pos = self.start
        return self.state_of(self.agent_pos)

    def step(self, action: int):
        """
        Apply one action.

        Returns ``(next_state, reward, done, info)``.

        * Invalid move (wall or boundary): the agent does NOT move, receives -10,
          and ``info["hit_wall"]`` is True. The learner therefore sees
          s' == s with a penalty — it is never treated as a successful move.
        * Reaching the goal: +100 and ``done = True``.
        * Normal move: -1.
        """
        if action not in ACTIONS:
            raise MazeError(f"Unknown action {action!r}; expected 0-3.")
        dr, dc = ACTIONS[action]
        r, c = self.agent_pos
        nr, nc = r + dr, c + dc

        if self.is_wall(nr, nc):
            return self.state_of(self.agent_pos), REWARD_WALL, False, {"hit_wall": True}

        self.agent_pos = (nr, nc)
        if self.agent_pos == self.goal:
            return self.state_of(self.agent_pos), REWARD_GOAL, True, {"hit_wall": False}
        return self.state_of(self.agent_pos), REWARD_STEP, False, {"hit_wall": False}

    # ------------------------------------------------------------------
    # Analysis helpers (never used by the learning agent itself)
    # ------------------------------------------------------------------
    def shortest_path(self) -> Optional[List[Position]]:
        """Breadth-First Search from start to goal. Returns the cell list or None."""
        return _bfs_path(self.grid, self.start, self.goal)

    def shortest_path_length(self) -> Optional[int]:
        path = self.shortest_path()
        return None if path is None else len(path) - 1

    def is_solvable(self) -> bool:
        return self.shortest_path() is not None

    def reachable_cells(self) -> set:
        return _reachable(self.grid, self.start)

    # ------------------------------------------------------------------
    # Editing
    # ------------------------------------------------------------------
    def toggle_wall(self, r: int, c: int) -> str:
        self._check_cell(r, c)
        if (r, c) in (self.start, self.goal):
            raise MazeError("You cannot place a wall on the start or goal cell.")
        self.grid[r, c] = WALL if self.grid[r, c] == EMPTY else EMPTY
        return f"{'Added' if self.grid[r, c] == WALL else 'Removed'} wall at row {r}, column {c}."

    def set_wall(self, r: int, c: int, wall: bool) -> str:
        self._check_cell(r, c)
        if wall and (r, c) in (self.start, self.goal):
            raise MazeError("You cannot place a wall on the start or goal cell.")
        self.grid[r, c] = WALL if wall else EMPTY
        return f"{'Added' if wall else 'Removed'} wall at row {r}, column {c}."

    def set_start(self, r: int, c: int) -> str:
        self._check_cell(r, c)
        if (r, c) == self.goal:
            raise MazeError("The start cannot be on the goal cell.")
        self.grid[r, c] = EMPTY
        self.start = (r, c)
        self.agent_pos = self.start
        return f"Start moved to row {r}, column {c}."

    def set_goal(self, r: int, c: int) -> str:
        self._check_cell(r, c)
        if (r, c) == self.start:
            raise MazeError("The goal cannot be on the start cell.")
        self.grid[r, c] = EMPTY
        self.goal = (r, c)
        return f"Goal moved to row {r}, column {c}."

    def clear_walls(self) -> str:
        self.grid[:, :] = EMPTY
        return "All walls removed."

    def _check_cell(self, r: int, c: int) -> None:
        if not self.in_bounds(r, c):
            raise MazeError(f"Cell (row {r}, column {c}) is outside the {self.rows}×{self.cols} maze.")

    # ------------------------------------------------------------------
    # Construction / serialisation
    # ------------------------------------------------------------------
    def copy(self) -> "MazeEnvironment":
        return MazeEnvironment(self.grid.copy(), self.start, self.goal)

    def to_dict(self) -> dict:
        return {
            "rows": self.rows,
            "cols": self.cols,
            "grid": self.grid.astype(int).tolist(),
            "start": list(self.start),
            "goal": list(self.goal),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MazeEnvironment":
        try:
            return cls(data["grid"], tuple(data["start"]), tuple(data["goal"]))
        except (KeyError, TypeError) as exc:
            raise MazeError(f"Saved maze data is incomplete: {exc}") from exc

    @classmethod
    def generate(
        cls,
        rows: int = 10,
        cols: int = 10,
        style: str = "maze",
        density: float = 0.25,
        loop_factor: float = 0.15,
        seed: Optional[int] = None,
    ) -> "MazeEnvironment":
        """
        Create a random maze that is guaranteed to be solvable.

        style = "maze"      -> depth-first-search corridors + some loops (multiple routes)
        style = "obstacles" -> each cell becomes an obstacle with probability ``density``
        style = "empty"     -> no obstacles at all
        """
        if not (MIN_SIZE <= rows <= MAX_SIZE and MIN_SIZE <= cols <= MAX_SIZE):
            raise MazeError(f"Maze size must be between {MIN_SIZE} and {MAX_SIZE}.")
        if style not in MAZE_STYLES:
            raise MazeError(f"Unknown maze style {style!r}.")
        rng = random.Random(seed)
        start, goal = (0, 0), (rows - 1, cols - 1)

        if style == "maze":
            grid = _carve_maze(rows, cols, rng, loop_factor)
        elif style == "obstacles":
            density = min(max(float(density), 0.0), 0.6)
            grid = np.zeros((rows, cols), dtype=np.int8)
            for r in range(rows):
                for c in range(cols):
                    if rng.random() < density:
                        grid[r, c] = WALL
        else:
            grid = np.zeros((rows, cols), dtype=np.int8)

        grid[start] = EMPTY
        grid[goal] = EMPTY
        _connect(grid, start, goal)
        _fill_unreachable(grid, start)
        return cls(grid, start, goal)


# ----------------------------------------------------------------------------
# Internal helpers for maze generation and search
# ----------------------------------------------------------------------------
def _neighbours(r: int, c: int, rows: int, cols: int):
    for dr, dc in ACTIONS.values():
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            yield nr, nc


def _reachable(grid: np.ndarray, start: Position) -> set:
    rows, cols = grid.shape
    seen = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbours(r, c, rows, cols):
            if (nr, nc) not in seen and grid[nr, nc] == EMPTY:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return seen


def _bfs_path(grid: np.ndarray, start: Position, goal: Position) -> Optional[List[Position]]:
    rows, cols = grid.shape
    parent = {start: None}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        if cur == goal:
            path = []
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            return path[::-1]
        for nxt in _neighbours(cur[0], cur[1], rows, cols):
            if nxt not in parent and grid[nxt] == EMPTY:
                parent[nxt] = cur
                queue.append(nxt)
    return None


def _carve_maze(rows: int, cols: int, rng: random.Random, loop_factor: float) -> np.ndarray:
    """Iterative depth-first 'recursive backtracker' maze, then knock out a few walls for loops."""
    grid = np.ones((rows, cols), dtype=np.int8)
    grid[0, 0] = EMPTY
    stack = [(0, 0)]
    while stack:
        r, c = stack[-1]
        options = []
        for dr, dc in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == WALL:
                options.append((nr, nc))
        if options:
            nr, nc = rng.choice(options)
            grid[(r + nr) // 2, (c + nc) // 2] = EMPTY  # knock down the wall in between
            grid[nr, nc] = EMPTY
            stack.append((nr, nc))
        else:
            stack.pop()

    # Even-sized mazes leave a solid last row/column; open parts of it up.
    if rows % 2 == 0:
        for c in range(cols):
            if grid[rows - 2, c] == EMPTY and rng.random() < 0.55:
                grid[rows - 1, c] = EMPTY
    if cols % 2 == 0:
        for r in range(rows):
            if grid[r, cols - 2] == EMPTY and rng.random() < 0.55:
                grid[r, cols - 1] = EMPTY

    # Remove some internal walls so there is more than one possible route.
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != WALL or rng.random() >= loop_factor:
                continue
            horizontal = 0 < c < cols - 1 and grid[r, c - 1] == EMPTY and grid[r, c + 1] == EMPTY
            vertical = 0 < r < rows - 1 and grid[r - 1, c] == EMPTY and grid[r + 1, c] == EMPTY
            if horizontal or vertical:
                grid[r, c] = EMPTY
    return grid


def _connect(grid: np.ndarray, start: Position, goal: Position) -> None:
    """If the goal is cut off, carve the shortest tunnel from the goal to the reachable region."""
    reachable = _reachable(grid, start)
    if goal in reachable:
        return
    rows, cols = grid.shape
    parent = {goal: None}
    queue = deque([goal])
    while queue:
        cur = queue.popleft()
        if cur in reachable:
            while cur is not None:
                grid[cur] = EMPTY
                cur = parent[cur]
            return
        for nxt in _neighbours(cur[0], cur[1], rows, cols):
            if nxt not in parent:
                parent[nxt] = cur
                queue.append(nxt)


def _fill_unreachable(grid: np.ndarray, start: Position) -> None:
    """Turn isolated pockets (cells the agent can never reach) into walls."""
    reachable = _reachable(grid, start)
    rows, cols = grid.shape
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == EMPTY and (r, c) not in reachable:
                grid[r, c] = WALL
