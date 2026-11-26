"""
q_learning.py
=============
A genuine tabular Q-Learning agent.

Nothing here is simulated: every number the app shows comes from this agent
actually moving through the maze, receiving rewards and updating its Q-table.

Core idea
---------
The agent keeps a table Q[state, action] = "how good is it to take this action
in this state, counting all future rewards".  After every single move it nudges
that estimate towards what it just experienced (the Bellman / TD update):

    Q(s,a) ← Q(s,a) + α · [ r + γ · max_a' Q(s',a') − Q(s,a) ]

* α (alpha)  – learning rate: how far to move the estimate each time.
* γ (gamma)  – discount factor: how much future rewards matter.
* r          – the immediate reward just received.
* s'         – the state reached after the move.
* max Q(s',a') – the best value the agent currently believes it can get from s'.
  When s' is the goal (terminal state) there is no future, so this term is 0.

Actions are chosen with ε-greedy:
* with probability ε pick a random action      (EXPLORATION)
* otherwise pick the action with the highest Q (EXPLOITATION)
ε starts high and decays after every episode, so the agent explores first and
exploits what it has learned later.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from typing import Callable, List, Optional

import numpy as np

from environment import N_ACTIONS, MazeEnvironment, Position


# ----------------------------------------------------------------------------
# Hyper-parameters
# ----------------------------------------------------------------------------
@dataclass
class Hyperparameters:
    alpha: float = 0.2            # learning rate
    gamma: float = 0.95           # discount factor
    epsilon: float = 1.0          # initial exploration rate
    epsilon_decay: float = 0.99   # multiply ε by this after every episode
    min_epsilon: float = 0.05     # ε never drops below this during training
    max_steps: int = 200          # step limit per episode (prevents endless episodes)

    # Allowed ranges (used by the UI sliders and by validation)
    LIMITS = {
        "alpha": (0.01, 1.0),
        "gamma": (0.0, 0.99),
        "epsilon": (0.0, 1.0),
        "epsilon_decay": (0.9, 1.0),
        "min_epsilon": (0.0, 0.5),
        "max_steps": (10, 2000),
    }

    def validate(self) -> "Hyperparameters":
        for name, (lo, hi) in self.LIMITS.items():
            value = getattr(self, name)
            if not (lo <= value <= hi):
                raise ValueError(f"{name} must be between {lo} and {hi} (got {value}).")
        if self.min_epsilon > self.epsilon:
            raise ValueError("Minimum epsilon cannot be larger than the initial epsilon.")
        self.max_steps = int(self.max_steps)
        return self

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, data: dict) -> "Hyperparameters":
        defaults = cls()
        kwargs = {k: type(getattr(defaults, k))(data.get(k, getattr(defaults, k)))
                  for k in defaults.to_dict()}
        return cls(**kwargs).validate()


# ----------------------------------------------------------------------------
# Result of one episode
# ----------------------------------------------------------------------------
@dataclass
class EpisodeResult:
    episode: int                  # training episode number (0 for test runs)
    total_reward: float
    steps: int
    success: bool                 # reached the goal before the step limit
    wall_hits: int
    explore_actions: int          # random actions taken
    exploit_actions: int          # greedy (best-known) actions taken
    epsilon: float                # ε used during this episode
    learned: bool                 # True if the Q-table was updated during this episode
    path: List[Position] = field(default_factory=list)     # cells visited (starts with start cell)
    rewards: List[float] = field(default_factory=list)     # reward after each step
    hits: List[bool] = field(default_factory=list)         # wall bump at each step
    actions: List[int] = field(default_factory=list)       # action taken at each step

    def to_record(self) -> dict:
        """Compact row for the training history table (no path data)."""
        return {
            "episode": self.episode,
            "reward": round(float(self.total_reward), 3),
            "steps": int(self.steps),
            "success": bool(self.success),
            "wall_hits": int(self.wall_hits),
            "epsilon": round(float(self.epsilon), 5),
            "explore_actions": int(self.explore_actions),
            "exploit_actions": int(self.exploit_actions),
        }


# ----------------------------------------------------------------------------
# The agent
# ----------------------------------------------------------------------------
class QLearningAgent:
    def __init__(self, n_states: int, n_actions: int = N_ACTIONS,
                 params: Optional[Hyperparameters] = None, seed: Optional[int] = None):
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.params = (params or Hyperparameters()).validate()
        self.seed = seed
        self._rng = random.Random(seed)
        self.reset()

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Forget everything: Q-table back to zeros, ε back to its initial value."""
        self.q_table = np.zeros((self.n_states, self.n_actions), dtype=float)
        self.visit_counts = np.zeros(self.n_states, dtype=np.int64)
        self.epsilon = float(self.params.epsilon)
        self.episodes_trained = 0
        self.total_updates = 0

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------
    def greedy_action(self, state: int, rng: Optional[random.Random] = None) -> int:
        """Best action for this state. Ties are broken randomly, so an untrained
        agent (all Q-values equal to 0) behaves like a purely random explorer."""
        row = self.q_table[state]
        best = row.max()
        candidates = np.flatnonzero(row == best)
        if len(candidates) == 1:
            return int(candidates[0])
        return int((rng or self._rng).choice(candidates.tolist()))

    def choose_action(self, state: int, epsilon: float, rng: Optional[random.Random] = None):
        """ε-greedy. Returns (action, explored_flag)."""
        rng = rng or self._rng
        if rng.random() < epsilon:
            return rng.randrange(self.n_actions), True     # EXPLORE
        return self.greedy_action(state, rng), False         # EXPLOIT

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------
    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> float:
        """The Q-Learning (Bellman) update. Returns the TD error."""
        alpha, gamma = self.params.alpha, self.params.gamma
        current_q = self.q_table[state, action]
        best_future_q = 0.0 if done else float(self.q_table[next_state].max())
        td_target = reward + gamma * best_future_q
        td_error = td_target - current_q
        self.q_table[state, action] = current_q + alpha * td_error
        self.total_updates += 1
        return float(td_error)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.params.min_epsilon, self.epsilon * self.params.epsilon_decay)

    # ------------------------------------------------------------------
    # Running episodes
    # ------------------------------------------------------------------
    def run_episode(self, env: MazeEnvironment, learn: bool = True,
                    epsilon: Optional[float] = None, max_steps: Optional[int] = None,
                    rng: Optional[random.Random] = None) -> EpisodeResult:
        """
        Play one episode from the start cell.

        learn=True  : a real training episode – Q-table updated after every move,
                      ε decays at the end, episode counter increases.
        learn=False : a test run – the Q-table is NOT touched.
        """
        eps = self.epsilon if epsilon is None else float(epsilon)
        limit = int(max_steps or self.params.max_steps)
        rng = rng or self._rng

        state = env.reset()
        path = [env.agent_pos]
        rewards: List[float] = []
        hits: List[bool] = []
        actions: List[int] = []
        total_reward = 0.0
        wall_hits = explored = exploited = 0
        success = False

        for _ in range(limit):
            action, was_random = self.choose_action(state, eps, rng)
            next_state, reward, done, info = env.step(action)

            if learn:
                self.update(state, action, reward, next_state, done)
                self.visit_counts[next_state] += 1

            explored += was_random
            exploited += (not was_random)
            wall_hits += info["hit_wall"]
            total_reward += reward
            path.append(env.agent_pos)
            rewards.append(float(reward))
            hits.append(bool(info["hit_wall"]))
            actions.append(int(action))

            state = next_state
            if done:
                success = True
                break

        if learn:
            self.episodes_trained += 1
            self.visit_counts[env.state_of(env.start)] += 1
            self.decay_epsilon()

        return EpisodeResult(
            episode=self.episodes_trained if learn else 0,
            total_reward=total_reward,
            steps=len(actions),
            success=success,
            wall_hits=wall_hits,
            explore_actions=explored,
            exploit_actions=exploited,
            epsilon=eps,
            learned=learn,
            path=path,
            rewards=rewards,
            hits=hits,
            actions=actions,
        )

    def train(self, env: MazeEnvironment, n_episodes: int,
              callback: Optional[Callable[[EpisodeResult], None]] = None) -> List[EpisodeResult]:
        """Train for ``n_episodes`` episodes (used by tests and the instant-train button)."""
        results = []
        for _ in range(int(n_episodes)):
            result = self.run_episode(env, learn=True)
            results.append(result)
            if callback:
                callback(result)
        return results

    def greedy_route(self, env: MazeEnvironment, max_steps: Optional[int] = None) -> EpisodeResult:
        """The route the agent currently believes is best (ε = 0, no learning).
        A fixed random seed makes the display stable when Q-values tie."""
        return self.run_episode(env, learn=False, epsilon=0.0, max_steps=max_steps,
                                rng=random.Random(12345))

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def state_is_learned(self, state: int) -> bool:
        return bool(np.any(self.q_table[state] != 0.0))

    def policy_grid(self, env: MazeEnvironment) -> np.ndarray:
        """Best action for every cell (-1 for walls, the goal and never-updated cells)."""
        grid = np.full((env.rows, env.cols), -1, dtype=int)
        for r, c in env.free_cells():
            s = env.state_of((r, c))
            if (r, c) == env.goal or not self.state_is_learned(s):
                continue
            row = self.q_table[s]
            grid[r, c] = int(np.argmax(row))
        return grid

    def value_grid(self, env: MazeEnvironment) -> np.ndarray:
        """V(s) = max_a Q(s,a) for every free cell, NaN for walls."""
        values = self.q_table.max(axis=1).reshape(env.rows, env.cols).astype(float)
        values[env.grid == 1] = np.nan
        return values

    def learned_state_count(self) -> int:
        return int(np.any(self.q_table != 0.0, axis=1).sum())

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "n_states": self.n_states,
            "n_actions": self.n_actions,
            "params": self.params.to_dict(),
            "q_table": np.round(self.q_table, 6).tolist(),
            "visit_counts": self.visit_counts.astype(int).tolist(),
            "epsilon": float(self.epsilon),
            "episodes_trained": int(self.episodes_trained),
            "total_updates": int(self.total_updates),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "QLearningAgent":
        agent = cls(int(data["n_states"]), int(data.get("n_actions", N_ACTIONS)),
                    Hyperparameters.from_dict(data.get("params", {})))
        q = np.asarray(data["q_table"], dtype=float)
        if q.shape != (agent.n_states, agent.n_actions) or not np.all(np.isfinite(q)):
            raise ValueError("Saved Q-table has the wrong shape or invalid numbers.")
        agent.q_table = q
        visits = np.asarray(data.get("visit_counts", np.zeros(agent.n_states)), dtype=np.int64)
        if visits.shape == (agent.n_states,):
            agent.visit_counts = visits
        agent.epsilon = float(np.clip(data.get("epsilon", agent.params.epsilon), 0.0, 1.0))
        agent.episodes_trained = int(data.get("episodes_trained", 0))
        agent.total_updates = int(data.get("total_updates", 0))
        return agent


# ----------------------------------------------------------------------------
# Evaluation (test runs that never change the Q-table)
# ----------------------------------------------------------------------------
def evaluate_agent(agent: QLearningAgent, env: MazeEnvironment, n_trials: int = 30,
                   epsilon: float = 0.0, max_steps: Optional[int] = None,
                   seed: int = 2024) -> dict:
    """
    Run ``n_trials`` test episodes with learning switched off and summarise them.

    For an UNTRAINED agent every Q-value is 0, so the greedy choice is a random
    tie-break: the agent wanders randomly. That is exactly the "before" behaviour.
    """
    rng = random.Random(seed)
    runs = [agent.run_episode(env, learn=False, epsilon=epsilon, max_steps=max_steps, rng=rng)
            for _ in range(int(n_trials))]
    wins = [r for r in runs if r.success]
    # Representative run = the one with the median reward (not cherry-picked).
    ordered = sorted(runs, key=lambda r: r.total_reward)
    sample = ordered[len(ordered) // 2]
    return {
        "trials": len(runs),
        "success_rate": 100.0 * len(wins) / len(runs),
        "avg_reward": float(np.mean([r.total_reward for r in runs])),
        "avg_steps": float(np.mean([r.steps for r in runs])),
        "avg_steps_success": float(np.mean([r.steps for r in wins])) if wins else None,
        "best_steps": int(min(r.steps for r in wins)) if wins else None,
        "avg_wall_hits": float(np.mean([r.wall_hits for r in runs])),
        "episodes_trained": int(agent.episodes_trained),
        "max_steps": int(max_steps or agent.params.max_steps),
        "sample": sample,
    }
