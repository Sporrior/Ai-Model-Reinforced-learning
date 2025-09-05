# test_loop_dqn_visualized.py
import numpy as np
import cv2
from simple_color_env import SimpleColorEnv
from collections import deque
import random


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Flatten
        from tensorflow.keras.optimizers import Adam

        model = Sequential()
        model.add(Flatten(input_shape=(84, 84, 3)))  # Flatten the image
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, return_q_values=False):
        if np.random.rand() <= self.epsilon:
            action = random.randrange(self.action_size)
            q_values = None
        else:
            # Add batch dimension for prediction
            state_batch = np.expand_dims(state, axis=0)
            q_values = self.model.predict(state_batch, verbose=0)[0]
            action = np.argmax(q_values)

        if return_q_values:
            return action, q_values
        return action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            # Add batch dimension for prediction
            state = np.expand_dims(state, axis=0)
            next_state = np.expand_dims(next_state, axis=0)

            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# Initialize environment and agent
env = SimpleColorEnv()
state_size = (84, 84, 3)
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)
batch_size = 32

rolling_window = 50
rewards_history = []
target_average = 10

action_names = ["LEFT (for RED)", "RIGHT (for GREEN)"]


def process_state(obs):
    return obs / 255.0


def create_debug_display(obs, action, reward, q_values, info, epsilon):
    # Create a debug panel
    debug_panel = np.ones((300, 600, 3), dtype=np.uint8) * 255

    # Display the observation (what the agent sees)
    obs_display = cv2.resize(obs, (150, 150))
    debug_panel[10:160, 10:160] = obs_display

    # Display Q-values as a bar chart
    if q_values is not None:
        max_q = max(q_values) if max(q_values) > 0 else 1
        for i, q_val in enumerate(q_values):
            bar_height = int((q_val / max_q) * 100) if max_q != 0 else 0
            color = (0, 255, 0) if i == action else (0, 0, 255)
            cv2.rectangle(debug_panel, (180 + i * 60, 200), (180 + i * 60 + 40, 200 - bar_height), color, -1)
            cv2.putText(debug_panel, f"{q_val:.2f}", (180 + i * 60, 220),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            cv2.putText(debug_panel, action_names[i], (180 + i * 60, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    # Add text information
    cv2.putText(debug_panel, f"Target: {info['target']}", (180, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(debug_panel, f"Action: {action_names[action]}", (180, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(debug_panel, f"Reward: {reward:.2f}", (180, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(debug_panel, f"Epsilon: {epsilon:.3f}", (180, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(debug_panel, f"Exploration: {'RANDOM' if np.random.rand() <= epsilon else 'LEARNED'}",
                (180, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    return debug_panel


for e in range(1000):
    obs, _ = env.reset()
    state = process_state(obs)
    total_reward = 0
    done = False

    while not done:
        # Get action and Q-values for visualization
        action, q_values = agent.act(state, return_q_values=True)
        next_obs, reward, done, truncated, info = env.step(action)
        next_state = process_state(next_obs)

        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        # Create visualization
        debug_display = create_debug_display(
            obs, action, reward, q_values, info, agent.epsilon
        )

        # Show both the environment and debug info
        env_frame = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
        cv2.imshow("Environment", env_frame)
        cv2.imshow("Agent Debug Info", debug_display)

        # Check for ESC key press to exit
        key = cv2.waitKey(100) & 0xFF
        if key == 27:  # ESC key
            break
        elif key == ord('p'):  # Pause with 'p' key
            cv2.waitKey(0)

    rewards_history.append(total_reward)

    # Train the agent
    if len(agent.memory) > batch_size:
        agent.replay(batch_size)

    # Compute rolling average
    if len(rewards_history) >= rolling_window:
        avg_reward = np.mean(rewards_history[-rolling_window:])
    else:
        avg_reward = np.mean(rewards_history)

    print(f"Episode {e + 1}/1000 | Reward: {total_reward:+.2f} | Avg: {avg_reward:+.2f} | Epsilon: {agent.epsilon:.3f}")

    if avg_reward >= target_average:
        print(f"Target average reward {target_average} reached! Stopping...")
        break

env.close()
cv2.destroyAllWindows()