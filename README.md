# 🧠 REINFORCE AI – AI Through Experience

> *"Learning Through Experience, One Decision at a Time."*

REINFORCE AI is an interactive web laboratory that shows how an AI agent teaches itself to solve a maze using **Reinforcement Learning (Q-Learning)**. Nobody gives the agent a map or the route. It starts with zero knowledge, moves around, bumps into walls, earns rewards and penalties, and gradually becomes intelligent.

Everything on screen comes from real learning. There are **no fake or hard-coded statistics**: every number and chart is computed from episodes the agent actually played.

---

## ✨ Features

| Area | What you can do |
|---|---|
| 🏠 **Dashboard** | Live stat cards (total episodes, current episode, total reward, success rate, average steps, ε, learning progress, learned route), a training status indicator (Not Started / Training / Paused / Completed), all training controls, the animated maze and live learning curves |
| 🧩 **Maze Lab** | Generate random mazes (5×5 up to 20×20, three styles, always solvable). Click cells to add or remove walls and to move the start or goal. Show the true shortest route (BFS). Optionally keep the learned knowledge while editing, to watch the AI adapt |
| ⚖️ **Before vs After** | The same AI on the same maze: 30 recorded test runs before training and 30 after, side by side with animated replays and the measured improvement |
| 🧠 **AI Visualization** | Q-table (table and four action maps), policy heatmap with arrows, agent paths, exploration vs exploitation, reward progress, success rate, state-visit heatmap |
| 📊 **Analytics** | Episode rewards, moving averages, success rate, average steps per successful episode, ε decay, wall hits, an automatically written performance summary, and CSV / report downloads |
| 📘 **Learn RL** | 12 beginner-friendly concept cards, the Q-Learning equation with every variable explained, an interactive "one update" calculator, and "How Our AI Learns" step by step |
| 🎓 **Demo Mode** | A guided 6-stage presentation for college evaluation, started with **🚀 Start Live Demonstration** |
| 💾 **Save & Load** | Save the whole session (maze, Q-table, ε, history, before-snapshot) as JSON and load it later. Reset to default settings or factory reset |

**Training controls:** Start Training · Pause · Resume · Reset Training · Train One Episode · Instant Train N · Test Trained AI · Generate New Maze. Run lengths: 100 / 500 / 1000 / 2000 / 5000 episodes. Speeds: Watch / Normal / Turbo.

---

## 📁 Project structure

```
reinforce-ai/
│
├── app.py              # Streamlit app: pages, controls, session state, training loop
├── environment.py      # Maze environment: states, actions, rewards, walls, generation, BFS
├── q_learning.py       # Q-Learning agent: Q-table, ε-greedy, Bellman update, evaluation
├── visualization.py    # Plotly charts + animated HTML/JS maze player
├── analytics.py        # Statistics from the recorded training history, CSV export
├── education.py        # Teaching content, tooltips, demo-mode text
├── ui_components.py    # HTML building blocks (cards, hero, legend, timeline…)
├── requirements.txt
├── README.md
├── assets/
│   └── style.css       # Dark navy theme with electric blue / cyan / violet accents
├── .streamlit/
│   └── config.toml     # Dark Streamlit theme
└── tests/
    └── test_engine.py  # Proves the agent really learns (runs without Streamlit)
```

---

## ⚙️ Installation

You need **Python 3.9 or newer** (3.10–3.12 recommended). Check with `python --version`.

```bash
# 1. Go into the project folder
cd reinforce-ai

# 2. (Recommended) create a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# 3. Install the dependencies
pip install -r requirements.txt
```

## ▶️ Running the website locally

```bash
streamlit run app.py
```

Your browser opens **http://localhost:8501** automatically. If it doesn't, open that address yourself. To stop the server, press `Ctrl + C` in the terminal.

**Optional, to prove the engine works without the UI:**

```bash
python tests/test_engine.py
```

Expected output (the numbers come from real training):

```
✓ rewards: goal +100, wall -10 (agent stays put), step -1
✓ Q(s,a) ← Q(s,a) + α[r + γ·max Q(s′,a′) − Q(s,a)] computed exactly
✓ 10×10 maze: before 3% success, reward -1106 → after 100% success, reward +75, route 26 steps (shortest 26)
...
All engine tests passed.
```

---

## 🧮 How the Q-Learning implementation works

### The environment (`environment.py`)
- **States:** every grid cell is a state, numbered `state = row × columns + column`. A 10×10 maze has 100 states. Wall cells are never visited.
- **Actions:** `0 = Up`, `1 = Down`, `2 = Left`, `3 = Right`.
- **Rewards:**

  | Event | Reward | Effect |
  |---|---|---|
  | Reach the goal | **+100** | Episode ends (terminal state) |
  | Hit a wall or the border | **−10** | **The agent does not move.** It stays in the same state |
  | Normal move | **−1** | Encourages short routes |
  | Step limit reached | — | Episode stops (default 200 steps) |

- **Invalid moves are never treated as successful moves.** A wall bump returns `next_state == state` with −10, so the agent learns that action is bad in that cell.
- Random mazes use a depth-first "recursive backtracker" plus a few removed walls (so there are loops and several routes). Every maze is checked with Breadth-First Search and repaired if needed, so it is **always solvable**. BFS is only used for validation and for comparing the learned route with the shortest one. **The agent never sees it.**

### The agent (`q_learning.py`)
- **Q-table:** a NumPy array of shape `(number_of_states, 4)`, initialised to **zeros**. The agent starts knowing nothing.
- **ε-greedy action selection:**
  - with probability **ε** → a random action (**exploration**)
  - otherwise → the action with the highest Q-value (**exploitation**); ties are broken randomly, so an untrained agent wanders randomly
- **The update.** After *every* move, `QLearningAgent.update()` applies:

  ```
  Q(s,a) ← Q(s,a) + α [ r + γ · max Q(s′,a′) − Q(s,a) ]
  ```

  ```python
  current_q     = Q[s, a]
  best_future_q = 0 if done else max(Q[s'])      # no future after the goal
  td_target     = r + gamma * best_future_q
  Q[s, a]       = current_q + alpha * (td_target - current_q)
  ```
- **ε decay:** after each episode `ε = max(min_ε, ε × decay)`. The agent explores a lot at first, then increasingly uses what it learned.
- **Step-limit truncation** is *not* treated as terminal. The future term is still used, which is the correct Q-Learning behaviour.

### Default hyper-parameters (all adjustable in the sidebar)

| Parameter | Default | Meaning |
|---|---|---|
| Learning rate α | 0.2 | How strongly each experience changes the Q-value |
| Discount factor γ | 0.95 | How much future rewards matter |
| Initial ε | 1.0 | 100% random moves at the start |
| ε decay | 0.99 | ε × 0.99 after every episode |
| Minimum ε | 0.05 | Always keep 5% curiosity |
| Max steps | 200 | Episode limit |

### How "Before vs After" is measured
1. Just before the **first** training episode, the untrained agent (all Q-values = 0) makes **30 test runs** with learning switched off. This is the *before* snapshot.
2. After training, the trained agent makes **30 test runs** with **ε = 0 and learning switched off**. This is the *after* result.
3. Both use the same maze and step limit. The replay shows the run with the median reward, so it is not cherry-picked. Test runs never change the Q-table.

### Keeping the interface responsive
Streamlit re-runs the script after every interaction. While training, each run trains a small batch of episodes (one episode in *Watch* mode, about 0.3 s of training in *Normal*, about 1.2 s in *Turbo*), redraws the page, and calls `st.rerun()`. Buttons use `on_click` callbacks, which run before the next batch, so **Pause takes effect between batches and the page never freezes**. The maze, agent, Q-table and history live in `st.session_state`, so nothing is lost between interactions.

---

## 🎓 How to present the project (8–10 minute script)

**Before the presentation**
1. Run `streamlit run app.py` and open the browser in full screen (`F11`).
2. Optional: run `python tests/test_engine.py` in a terminal you can show if asked "is the learning real?".
3. In the sidebar, set **Training speed → Normal**.

**Live demonstration: open 🎓 Demo Mode → 🚀 Start Live Demonstration**

| Stage | What to show | What to say |
|---|---|---|
| 1. Introduction | The maze and "The challenge" card | "The AI gets no map and no instructions. Only rewards and penalties." |
| 2. How RL works | The animated agent ↔ environment loop and the equation | "After every move it updates one number in its Q-table using this equation." |
| 3. Before training | The red replay: the robot wanders and bumps into walls | "With an empty memory it reached the goal in only X of 30 attempts." |
| 4. Live training | Click **▶ Train the AI** (300–500 episodes). Point at the reward curve rising and the success rate climbing | "Each point on this curve is a real attempt. Watch ε fall: it explores less and uses its knowledge more." Tip: switch to **Watch** speed for a few episodes to show single attempts |
| 5. After training | Before vs after side by side, plus the measured improvement | "Same AI, same maze: from X% to 100%, from about 200 steps to the shortest possible route." |
| 6. Conclusion | The policy map: an arrow in every cell | "Nobody programmed these arrows. The +100 goal reward flowed backwards through the Q-table." |

**Bonus "wow" moment:** go to **🧩 Maze Lab**, tick **Keep learned knowledge when editing**, click a cell on the learned route to block it, then train 100 more episodes. The AI discovers a new route by itself.

**Likely viva questions**
- *Why −1 per step?* So shorter routes have higher total reward. Without it, any route to the goal would look equally good.
- *Why does the agent explore?* Without exploration it could get stuck on the first route it found and never discover a better one.
- *What happens when it hits a wall?* It stays in place and gets −10, so that Q(s, a) drops and the move is avoided later.
- *Why is the learned route optimal?* With γ < 1 and a −1 step cost, the highest-value route is the shortest one. The app checks it against BFS.
- *Is the data real?* Yes. Training statistics can be downloaded as CSV from Analytics, and `tests/test_engine.py` verifies the learning independently.

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| `streamlit: command not found` | Use `python -m streamlit run app.py` |
| `ModuleNotFoundError` | Activate the virtual environment and run `pip install -r requirements.txt` again |
| Port 8501 already in use | `streamlit run app.py --server.port 8502` |
| Clicking maze cells doesn't edit them | Update Streamlit (`pip install -U streamlit`), or use the *Row / Column* form under the editor |
| "Goal cannot be reached" | You walled off the goal. Remove a wall, or use **⟲ Reset Maze** |
| Training feels slow | Switch the sidebar speed to **Turbo**, or use **⚡ Instant Train** |

---

**Tech stack:** Python · NumPy (Q-table and maths) · Streamlit (web app) · Plotly (interactive charts) · Pandas (training statistics)
