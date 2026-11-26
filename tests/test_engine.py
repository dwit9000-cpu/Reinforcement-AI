"""
tests/test_engine.py
====================
Proof that the AI really learns (no Streamlit needed).

Run from the project folder:
    python tests/test_engine.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from environment import REWARD_GOAL, REWARD_STEP, REWARD_WALL, MazeEnvironment  # noqa: E402
from q_learning import QLearningAgent, evaluate_agent  # noqa: E402


def test_rewards_and_walls():
    grid = np.zeros((5, 5), dtype=int)
    grid[0, 1] = 1                                   # wall to the right of the start
    env = MazeEnvironment(grid, (0, 0), (0, 2))
    s = env.reset()
    s2, r, done, info = env.step(3)                  # Right -> wall
    assert (s2, r, done, info["hit_wall"]) == (s, REWARD_WALL, False, True), "wall must block and give -10"
    s2, r, done, info = env.step(0)                  # Up -> outside the grid
    assert s2 == s and r == REWARD_WALL, "border counts as a wall"
    s2, r, done, _ = env.step(1)                     # Down -> normal move
    assert r == REWARD_STEP and env.agent_pos == (1, 0) and not done
    env.agent_pos = (0, 3)
    _, r, done, _ = env.step(2)                      # Left -> goal
    assert r == REWARD_GOAL and done
    print("✓ rewards: goal +100, wall -10 (agent stays put), step -1")


def test_bellman_update():
    agent = QLearningAgent(4)
    agent.params.alpha, agent.params.gamma = 0.5, 0.9
    agent.q_table[1] = [2.0, 8.0, 0.0, 1.0]
    agent.update(0, 3, -1.0, 1, done=False)          # Q = 0 + 0.5 * (-1 + 0.9*8 - 0)
    assert abs(agent.q_table[0, 3] - 3.1) < 1e-9
    agent.update(2, 0, 100.0, 3, done=True)          # terminal: no future term
    assert abs(agent.q_table[2, 0] - 50.0) < 1e-9
    print("✓ Q(s,a) ← Q(s,a) + α[r + γ·max Q(s′,a′) − Q(s,a)] computed exactly")


def test_agent_learns_shortest_route():
    for seed, size in [(7, 10), (3, 12), (5, 15)]:
        env = MazeEnvironment.generate(size, size, style="maze", seed=seed)
        agent = QLearningAgent(env.n_states, seed=0)
        before = evaluate_agent(agent, env, 30)
        agent.train(env, 1000)
        after = evaluate_agent(agent, env, 30)
        optimal = env.shortest_path_length()
        assert after["success_rate"] == 100.0
        assert after["best_steps"] == optimal
        assert after["avg_reward"] > before["avg_reward"]
        print(f"✓ {size}×{size} maze: before {before['success_rate']:.0f}% success, reward "
              f"{before['avg_reward']:+.0f} → after {after['success_rate']:.0f}% success, "
              f"reward {after['avg_reward']:+.0f}, route {after['best_steps']} steps (shortest {optimal})")


def test_random_mazes_always_solvable():
    rng = np.random.default_rng(0)
    for i in range(300):
        rows, cols = rng.integers(5, 21, size=2)
        style = ["maze", "obstacles", "empty"][i % 3]
        env = MazeEnvironment.generate(int(rows), int(cols), style=style, density=0.4, seed=i)
        assert env.is_solvable()
    print("✓ 300 random mazes generated — every one solvable")


if __name__ == "__main__":
    test_rewards_and_walls()
    test_bellman_update()
    test_agent_learns_shortest_route()
    test_random_mazes_always_solvable()
    print("\nAll engine tests passed.")
