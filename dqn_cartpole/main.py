import gymnasium as gym
from dqnAgent import DQNAgent

def train_dqn(episodes=500):
    env = gym.make("CartPole-v1", render_mode="human")
    agent = DQNAgent(state_dim=4, action_dim=2)

    for e in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            state = next_state
            total_reward += reward

        agent.update_target_model()
        agent.decay_epsilon()
        print(f"Episode {e+1}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

if __name__ == "__main__":
    train_dqn()
