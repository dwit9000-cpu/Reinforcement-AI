"""
education.py
============
Beginner-friendly teaching content for the "Understand Reinforcement Learning"
section, tooltips for technical terms, and text used by Demonstration Mode.
"""

APP_NAME = "REINFORCE AI"
APP_SUBTITLE = "AI Through Experience"
TAGLINE = "Learning Through Experience, One Decision at a Time."

# ----------------------------------------------------------------------------
# Tooltips (shown as ⓘ help icons next to controls)
# ----------------------------------------------------------------------------
TOOLTIPS = {
    "alpha": "Learning rate α (0–1). How strongly each new experience changes the AI's memory. "
             "High = learns fast but can be unstable; low = slow but steady.",
    "gamma": "Discount factor γ (0–1). How much the AI cares about future rewards. "
             "Close to 1 = plans ahead for the goal; close to 0 = only cares about the next step.",
    "epsilon": "Exploration rate ε (0–1). The chance of making a random move instead of the best-known move. "
               "Starts high so the AI explores. Takes effect after 'Reset Training'.",
    "epsilon_decay": "After every episode ε is multiplied by this number, so the AI gradually explores less "
                     "and uses what it has learned more.",
    "min_epsilon": "ε never falls below this value during training, so the AI always keeps a little curiosity.",
    "max_steps": "If the AI has not reached the goal after this many moves the episode ends. "
                 "Prevents endless wandering.",
    "episodes": "An episode is one complete attempt: from the start cell until the goal is reached "
                "or the step limit runs out.",
    "speed": "Watch = one episode per screen refresh, so you can see every attempt. "
             "Normal/Turbo train many episodes between refreshes.",
    "q_table": "A table with one row per cell (state) and one column per move (action). "
               "Each number is the AI's estimate of how good that move is from that cell.",
    "success_rate": "Percentage of episodes in which the agent reached the goal before the step limit.",
    "policy": "The policy is the AI's strategy: for every cell, the move with the highest Q-value.",
    "keep_knowledge": "If ticked, editing walls keeps the learned Q-table so you can watch the AI adapt to "
                      "the changed maze. If not ticked, any edit resets learning.",
}

# ----------------------------------------------------------------------------
# The 12 core concepts
# ----------------------------------------------------------------------------
CONCEPTS = [
    ("🎮", "What is Reinforcement Learning?",
     "A way for a computer to learn by trial and error. Nobody gives it the right answers. "
     "It tries actions, gets rewards or penalties, and gradually prefers what works — "
     "just like learning to ride a bicycle.",
     "Our AI is never told the route. It discovers it by bumping around the maze."),
    ("🤖", "What is an AI Agent?",
     "The learner and decision-maker. At every moment it looks at its situation and picks an action.",
     "The glowing blue robot in the maze is the agent."),
    ("🗺️", "What is an Environment?",
     "Everything the agent interacts with. It reacts to each action by moving the agent "
     "and giving back a reward.",
     "The maze grid — walls, empty cells, start and goal — is the environment."),
    ("📍", "What are States?",
     "A state is a snapshot of the current situation — all the agent needs to decide what to do next.",
     "Each cell is a state. A 10×10 maze has 100 possible states."),
    ("🕹️", "What are Actions?",
     "The choices available to the agent in each state.",
     "Four actions: Move Up, Move Down, Move Left, Move Right."),
    ("🏆", "What are Rewards and Penalties?",
     "Numbers that tell the agent how good its last action was. Positive = good, negative = bad. "
     "The agent tries to collect as much total reward as possible.",
     "+100 for the goal, −10 for hitting a wall, −1 for every normal move (so shorter is better)."),
    ("📋", "What is a Q-Table?",
     "The agent's memory: a table with a row for every state and a column for every action. "
     "Each entry Q(s, a) estimates the total future reward of doing action a in state s.",
     "Our Q-table has (rows × columns) rows and 4 columns — see it live on the AI Visualization page."),
    ("🧮", "What is Q-Learning?",
     "An algorithm that fills in the Q-table from experience. After every move it updates one entry "
     "using the reward it got plus the best value it expects from the next state.",
     "Each step in the maze changes exactly one number in the Q-table."),
    ("🧭", "What is Exploration?",
     "Trying random actions to discover new information — even if they might be worse.",
     "Early on the agent wanders randomly and finds out where the walls are."),
    ("🎯", "What is Exploitation?",
     "Using what has already been learned: choosing the action with the highest Q-value.",
     "Later the agent follows its best-known route straight to the goal."),
    ("⚡", "What is the Learning Rate (α)?",
     "How much a single new experience changes the agent's existing belief. "
     "α = 1 replaces the old value completely; α = 0.1 changes it only a little.",
     "Default α = 0.2: every move moves the estimate 20% of the way to the new target."),
    ("🔭", "What is the Discount Factor (γ)?",
     "How much future rewards matter compared with immediate ones. A reward k steps away is worth γᵏ today.",
     "With γ = 0.95 the +100 goal reward still 'shines back' many cells along the route."),
]

# ----------------------------------------------------------------------------
# The Q-Learning equation and its variables
# ----------------------------------------------------------------------------
EQUATION_LATEX = r"Q(s,a) \leftarrow Q(s,a) + \alpha \Big[\, r + \gamma \max_{a'} Q(s',a') - Q(s,a) \Big]"

VARIABLES = [
    ("Q(s, a)", "Current Q-value: the agent's present estimate of how good action a is in state s."),
    ("α (alpha)", "Learning rate: how big a step to take towards the new estimate."),
    ("r", "Immediate reward received for this move (+100, −10 or −1)."),
    ("γ (gamma)", "Discount factor: how much the future is worth."),
    ("s′", "Next state: the cell the agent ended up in after the move."),
    ("max Q(s′, a′)", "The best Q-value available from the next state — the agent's best guess of the future. "
                      "It is 0 when s′ is the goal, because the episode is over."),
    ("r + γ·max Q(s′, a′)", "The TD target: what the move turned out to be worth."),
    ("[ … − Q(s, a) ]", "The TD error: the surprise — how wrong the old estimate was."),
]

# ----------------------------------------------------------------------------
# How our AI learns (step by step)
# ----------------------------------------------------------------------------
HOW_IT_LEARNS = [
    ("Start with an empty memory",
     "Every Q-value is 0. The agent knows nothing, so all four moves look equally good."),
    ("Begin an episode",
     "The agent is placed on the green start cell. The episode lasts until it reaches the gold goal "
     "or runs out of steps."),
    ("Choose an action (ε-greedy)",
     "With probability ε it explores (random move); otherwise it exploits (move with the highest Q-value)."),
    ("Act and observe",
     "The maze moves the agent (or blocks it if there is a wall) and returns a reward: −1, −10 or +100."),
    ("Update the Q-table",
     "Using the Q-Learning equation, the agent adjusts Q(s, a) towards r + γ·max Q(s′, a′). "
     "Walls give −10 and keep the agent in place, so it learns those moves are bad."),
    ("Repeat until the episode ends",
     "Steps 3–5 repeat move after move. Reaching the goal ends the episode with +100."),
    ("Explore a little less",
     "After each episode ε is multiplied by the decay factor, so the agent relies more on its knowledge."),
    ("Values flow backwards from the goal",
     "Over many episodes the +100 reward spreads backwards through γ: cells near the goal get high values "
     "first, then cells further away. Following the highest values becomes the shortest route."),
]

# ----------------------------------------------------------------------------
# Demonstration mode
# ----------------------------------------------------------------------------
DEMO_STEPS = [
    "Introduction",
    "How RL Works",
    "Before Training",
    "Live Training",
    "After Training",
    "Conclusion",
]

PROJECT_INTRO = (
    "REINFORCE AI is an interactive laboratory that shows how an AI can teach itself to solve a "
    "maze with no instructions, no map and no examples — only rewards and penalties. It uses "
    "Q-Learning, a classic reinforcement-learning algorithm, implemented from scratch in Python with NumPy."
)

PROJECT_POINTS = [
    "The agent starts with zero knowledge (an all-zero Q-table).",
    "It learns only by moving, bumping into walls and occasionally finding the goal.",
    "Every chart and number on screen comes from real recorded episodes.",
    "We compare the same AI before and after training on the same maze.",
]
