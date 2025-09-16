# train_color_agent.py
import time
import json
import numpy as np
from DnQAgent import SmartAdaptiveDQNAgent
from color_env import ColorRecognitionEnvironment
import os

# Training configuration you can modify
TRAIN_CONFIG = {
    'difficulty': 'basic',           # 'basic' or 'advanced'
    'episodes': 2000,
    'max_steps_per_episode': 1,      # env is single-step per sample
    'early_stop_patience': 150,      # stop if no improvement on validation window
    'validation_interval': 100,      # evaluate every N episodes
    'validation_episodes': 200,      # how many samples in validation
    'save_path': 'color_agent_best'
}


def evaluate(agent: SmartAdaptiveDQNAgent, env: ColorRecognitionEnvironment, n: int = 200):
    """Run deterministic evaluation (epsilon=0) for n episodes and return accuracy"""
    orig_epsilon = agent.epsilon
    agent.epsilon = 0.0
    correct = 0
    for _ in range(n):
        s = env.reset()
        action = agent.act(s, training=False)
        _, reward, done, info = env.step(action)
        if info.get('correct', False):
            correct += 1
    acc = correct / n
    agent.epsilon = orig_epsilon
    return acc


def train():
    os.makedirs("models", exist_ok=True)
    env = ColorRecognitionEnvironment(difficulty_level=TRAIN_CONFIG['difficulty'], return_full_state=True)
    state_size = env.get_state_size()
    action_size = env.get_action_size()

    # create agent with safe defaults; you can tweak in config
    agent = SmartAdaptiveDQNAgent(state_size=state_size, action_size=action_size,
                                  config={'use_tensorboard': False})

    print(f"Training Agent on {action_size} colors")
    print(f"State size: {state_size}, Action size: {action_size}")

    best_val = -1.0
    best_epoch = 0
    history = {'episode': [], 'train_reward': [], 'val_acc': []}

    start_time = time.time()
    for episode in range(1, TRAIN_CONFIG['episodes'] + 1):
        # For this env a single step per episode: agent sees one sample and predicts
        state = env.reset()
        action = agent.act(state, training=True)
        next_state, reward, done, info = env.step(action)

        # store and learn
        agent.remember(state, action, reward, next_state, done)
        loss = agent.replay()

        total_reward = reward  # single-step
        agent.end_episode(total_reward, 1)

        # record training reward
        history['episode'].append(episode)
        history['train_reward'].append(float(total_reward))

        # validation
        if episode % TRAIN_CONFIG['validation_interval'] == 0:
            val_acc = evaluate(agent, env, n=TRAIN_CONFIG['validation_episodes'])
            history['val_acc'].append(float(val_acc))
            print(f"[Val] Episode {episode}: val_acc={val_acc:.4f}")
            # save if better
            if val_acc > best_val + 1e-6:
                best_val = val_acc
                best_epoch = episode
                agent.save(os.path.join("models", TRAIN_CONFIG['save_path']))
                print(f"New best val_acc={best_val:.4f} saved at episode {episode}")

        # early stopping based on validation performance not improving
        if (episode - best_epoch) >= TRAIN_CONFIG['early_stop_patience'] and episode > TRAIN_CONFIG['validation_interval']:
            print(f"Early stopping: no improvement in last {TRAIN_CONFIG['early_stop_patience']} episodes.")
            break

        # protect against NaNs or exploding behavior
        if len(agent.loss_window) > 0 and (np.isnan(np.mean(agent.loss_window)) or np.isinf(np.mean(agent.loss_window))):
            print("Training stopped due to NaN/inf loss.")
            break

    elapsed = time.time() - start_time
    print("Training finished in {:.1f}s".format(elapsed))
    # final evaluation
    final_val = evaluate(agent, env, n=TRAIN_CONFIG['validation_episodes'])
    print(f"Final validation accuracy: {final_val:.4f}")
    # save history
    with open(os.path.join("models", TRAIN_CONFIG['save_path'] + "_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    return agent, env, history


if __name__ == "__main__":
    agent, env, history = train()
    print("Training complete. Models/history saved to ./models/")
    env.print_stats()
