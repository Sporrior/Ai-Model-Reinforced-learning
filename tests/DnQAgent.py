# smart_agent.py
import numpy as np
from collections import deque
import random
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LayerNormalization, Add, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
import json
from typing import Dict, Any, Optional


class SmartAdaptiveDQNAgent:
    """
    Improved DQN agent:
    - smaller default networks to avoid OOM
    - prioritized replay (simple proportional)
    - double-dqn + dueling + target soft-updates
    - gradient clipping, Huber loss
    - TensorBoard logging (optional)
    - save/load + training stats + early stopping support from trainer
    """

    def __init__(self, state_size: int, action_size: int, config: Optional[Dict[str, Any]] = None):
        self.state_size = state_size
        self.action_size = action_size

        # safer defaults for memory/conservative model to avoid OOM
        self.config = {
            'gamma': 0.99,
            'learning_rate': 3e-4,
            'epsilon_start': 1.0,
            'epsilon_min': 0.02,
            'epsilon_decay': 0.995,
            'batch_size': 32,
            'memory_size': 20000,
            'target_update_freq': 50,
            'hidden_layers': [128, 64],
            'dueling': True,
            'double_dqn': True,
            'prioritized_replay': True,
            'soft_update_tau': 0.01,
            'use_tensorboard': False
        }
        if config:
            self.config.update(config)

        # parameters
        self.gamma = self.config['gamma']
        self.learning_rate = self.config['learning_rate']
        self.epsilon = self.config['epsilon_start']
        self.epsilon_min = self.config['epsilon_min']
        self.epsilon_decay = self.config['epsilon_decay']
        self.batch_size = self.config['batch_size']

        # replay
        self.memory = deque(maxlen=self.config['memory_size'])
        self.priorities = deque(maxlen=self.config['memory_size'])
        self.max_priority = 1.0

        # networks
        self.q_network = self._build_network()
        self.target_network = self._build_network()
        self.update_target_network(hard=True)  # ✅ now works

        # tracking
        self.training_step = 0
        self.episode_count = 0
        self.performance_window = deque(maxlen=50)
        self.loss_window = deque(maxlen=200)
        self.q_values_window = deque(maxlen=200)
        self.learning_efficiency = deque(maxlen=50)
        self.adaptation_cooldown = 0
        self.last_adaptation_performance = -1e9

        # tensorboard (optional)
        self.summary_writer = None
        if self.config['use_tensorboard']:
            self.summary_writer = tf.summary.create_file_writer("logs/agent")

    def update_target_network(self, hard: bool = True, tau: Optional[float] = None):
        """
        Update the target network.
        - If hard=True: copy all weights directly
        - If hard=False: do a soft update with factor tau
        """
        if hard:
            self.target_network.set_weights(self.q_network.get_weights())
        else:
            if tau is None:
                tau = self.config.get('soft_update_tau', 0.01)
            qw = self.q_network.get_weights()
            tw = self.target_network.get_weights()
            new = [tau * q + (1.0 - tau) * t for q, t in zip(qw, tw)]
            self.target_network.set_weights(new)

    def _build_network(self) -> Model:
        inputs = Input(shape=(self.state_size,))
        x = inputs
        for units in self.config['hidden_layers']:
            x = Dense(units, activation='relu', kernel_initializer='he_uniform')(x)
            x = LayerNormalization()(x)

        if self.config['dueling']:
            v = Dense(64, activation='relu')(x)
            v = Dense(1, activation='linear')(v)
            a = Dense(64, activation='relu')(x)
            a = Dense(self.action_size, activation='linear')(a)
            a_mean = Lambda(lambda y: tf.reduce_mean(y, axis=1, keepdims=True))(a)
            a_cent = Lambda(lambda z: z[0] - z[1])([a, a_mean])
            q_out = Add()([v, a_cent])
        else:
            q_out = Dense(self.action_size, activation='linear')(x)

        model = Model(inputs=inputs, outputs=q_out)
        model.compile(optimizer=Adam(learning_rate=self.learning_rate, clipnorm=1.0),
                      loss=Huber(), metrics=['mae'])
        return model

    # ---------- replay memory ----------
    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        experience = (np.array(state, dtype=np.float32), int(action), float(reward),
                      np.array(next_state, dtype=np.float32), bool(done))
        self.memory.append(experience)
        # initial priority: use max_priority so new experiences are likely sampled
        p = self.max_priority if self.config['prioritized_replay'] else 1.0
        self.priorities.append(p)

    def _calculate_priority(self, state, action, reward, next_state, done) -> float:
        # compute TD error approximately to set priority
        try:
            s = state.reshape(1, -1)
            ns = next_state.reshape(1, -1)
            q = self.q_network.predict(s, verbose=0)[0]
            q_next_target = self.target_network.predict(ns, verbose=0)[0]
            if done:
                td_target = reward
            else:
                if self.config['double_dqn']:
                    next_action = int(np.argmax(self.q_network.predict(ns, verbose=0)[0]))
                    td_target = reward + self.gamma * q_next_target[next_action]
                else:
                    td_target = reward + self.gamma * np.max(q_next_target)
            td_error = abs(td_target - q[action])
            return float(td_error) + 1e-6
        except Exception:
            return float(self.max_priority)

    # ---------- action ----------
    def act(self, state: np.ndarray, training: bool = True) -> int:
        if training and np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        q = self.q_network.predict(state.reshape(1, -1), verbose=0)[0]
        if training:
            self.q_values_window.append(float(np.max(q)))
        return int(np.argmax(q))

    # ---------- replay / train ----------
    def replay(self) -> float:
        if len(self.memory) < self.batch_size:
            return float('nan')

        # sample
        if self.config['prioritized_replay']:
            batch, indices, is_weights = self._sample_prioritized_batch()
        else:
            indices = np.random.randint(0, len(self.memory), size=self.batch_size)
            batch = [self.memory[i] for i in indices]
            is_weights = np.ones(self.batch_size, dtype=np.float32)

        # prepare arrays
        states = np.vstack([b[0] for b in batch])
        actions = np.array([b[1] for b in batch], dtype=np.int32)
        rewards = np.array([b[2] for b in batch], dtype=np.float32)
        next_states = np.vstack([b[3] for b in batch])
        dones = np.array([b[4] for b in batch], dtype=np.bool_)

        # predictions
        q_current = self.q_network.predict(states, verbose=0)
        q_next_target = self.target_network.predict(next_states, verbose=0)

        if self.config['double_dqn']:
            next_actions = np.argmax(self.q_network.predict(next_states, verbose=0), axis=1)
            targets = rewards + self.gamma * q_next_target[np.arange(self.batch_size), next_actions] * (~dones)
        else:
            targets = rewards + self.gamma * np.max(q_next_target, axis=1) * (~dones)

        # compose target q-values
        target_q = q_current.copy()
        target_q[np.arange(self.batch_size), actions] = targets

        # train: use sample weights from prioritized replay importance sampling
        history = self.q_network.fit(states, target_q, sample_weight=is_weights,
                                     epochs=1, verbose=0, batch_size=self.batch_size)
        loss = float(history.history['loss'][0])
        if np.isnan(loss) or np.isinf(loss):
            # skip updating priorities if loss is invalid
            self.loss_window.append(1e6)
            return loss

        self.loss_window.append(loss)

        # update priorities if used
        if self.config['prioritized_replay']:
            # compute new priorities (vectorized-ish)
            new_ps = []
            for i, idx in enumerate(indices):
                s, a, r, ns, d = batch[i]
                p = self._calculate_priority(s, a, r, ns, d)
                new_ps.append(p)
                # also keep track of max
                if p > self.max_priority:
                    self.max_priority = p
            # write back
            for idx, p in zip(indices, new_ps):
                if idx < len(self.priorities):
                    self.priorities[idx] = p

        # soft update target network
        self._soft_update_target(self.config['soft_update_tau'])

        # tensorboard
        if self.summary_writer is not None:
            with self.summary_writer.as_default():
                tf.summary.scalar('loss', loss, step=self.training_step)
                tf.summary.scalar('epsilon', self.epsilon, step=self.training_step)
                tf.summary.scalar('avg_q', np.mean(self.q_values_window) if self.q_values_window else 0, step=self.training_step)

        self.training_step += 1
        return loss

    def _sample_prioritized_batch(self):
        # simple proportional sampling
        pr = np.array(self.priorities, dtype=np.float64)
        # small stability add
        pr = pr + 1e-6
        probs = pr / pr.sum()
        indices = np.random.choice(len(self.memory), size=self.batch_size, p=probs)
        # simple importance-sampling weights (normalized)
        beta = min(1.0, 0.4 + self.training_step * 1e-4)
        weights = (len(self.memory) * probs[indices]) ** (-beta)
        weights = weights / (weights.max() + 1e-12)
        batch = [self.memory[i] for i in indices]
        return batch, indices, weights.astype(np.float32)

    def _soft_update_target(self, tau: float):
        qw = self.q_network.get_weights()
        tw = self.target_network.get_weights()
        new = [tau * q + (1.0 - tau) * t for q, t in zip(qw, tw)]
        self.target_network.set_weights(new)

    # ---------- end episode / adapt ----------
    def end_episode(self, total_reward: float, steps: int):
        self.episode_count += 1
        self.performance_window.append(total_reward)

        if len(self.loss_window) >= 5:
            recent_loss = float(np.mean(list(self.loss_window)[-5:]))
            eff = total_reward / (recent_loss + 1e-6)
            self.learning_efficiency.append(eff)

        # epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)

        # periodic logging
        if self.episode_count % 10 == 0:
            avg10 = float(np.mean(list(self.performance_window)[-10:]))
            prev10 = float(np.mean(list(self.performance_window)[-20:-10])) if len(self.performance_window) >= 20 else avg10
            trend = "up" if avg10 > prev10 else "down"
            eff_mean = float(np.mean(self.learning_efficiency)) if self.learning_efficiency else 0.0
            print(f"[Episode {self.episode_count}] Reward={total_reward:.3f}, Avg10={avg10:.3f} ({trend}), ε={self.epsilon:.3f}, Eff={eff_mean:.3f}")

        # simple adaptation if stuck (conservative)
        if self.episode_count % 50 == 0 and self.adaptation_cooldown <= 0:
            self._smart_adaptation()
            self.adaptation_cooldown = 5

        if self.adaptation_cooldown > 0:
            self.adaptation_cooldown -= 1

    def _smart_adaptation(self):
        if len(self.performance_window) < 20:
            return
        curr_perf = float(np.mean(list(self.performance_window)[-10:]))
        prev_perf = float(np.mean(list(self.performance_window)[-20:-10]))
        adaptations = []
        if curr_perf < prev_perf - 0.05:
            # small LR reduction
            self.learning_rate *= 0.9
            self.q_network.optimizer.learning_rate.assign(self.learning_rate)
            adaptations.append(f"lr->{self.learning_rate:.6f}")
            # nudge epsilon up a bit to explore
            self.epsilon = min(1.0, self.epsilon * 1.1)
            adaptations.append(f"epsilon->{self.epsilon:.3f}")
        if adaptations:
            print(f"Adaptations at ep {self.episode_count}: {', '.join(adaptations)}")

    # ---------- utils ----------
    def get_stats(self) -> Dict[str, Any]:
        return {
            'episode_count': self.episode_count,
            'training_step': self.training_step,
            'epsilon': float(self.epsilon),
            'learning_rate': float(self.learning_rate),
            'memory_size': len(self.memory),
            'avg_loss': float(np.mean(self.loss_window)) if self.loss_window else 0.0,
            'avg_q': float(np.mean(self.q_values_window)) if self.q_values_window else 0.0,
            'avg_perf_10': float(np.mean(list(self.performance_window)[-10:])) if len(self.performance_window) >= 1 else 0.0
        }

    def save(self, path: str):
        # save weights + state json
        self.q_network.save_weights(f"{path}_q.h5")
        self.target_network.save_weights(f"{path}_t.h5")
        state = {
            'config': self.config,
            'state_size': self.state_size,
            'action_size': self.action_size,
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'epsilon': float(self.epsilon),
            'learning_rate': float(self.learning_rate),
            'performance_window': list(self.performance_window),
            'loss_window': list(self.loss_window),
        }
        with open(f"{path}_state.json", "w") as f:
            json.dump(state, f, indent=2)
        print(f"Agent saved to {path}")

    def load(self, path: str) -> bool:
        try:
            with open(f"{path}_state.json", "r") as f:
                state = json.load(f)
            self.config.update(state.get('config', {}))
            # rebuild networks to match config
            self.q_network = self._build_network()
            self.target_network = self._build_network()
            self.q_network.load_weights(f"{path}_q.h5")
            self.target_network.load_weights(f"{path}_t.h5")
            self.training_step = state.get('training_step', 0)
            self.episode_count = state.get('episode_count', 0)
            self.epsilon = state.get('epsilon', self.epsilon)
            self.learning_rate = state.get('learning_rate', self.learning_rate)
            print(f"Agent loaded from {path}")
            return True
        except Exception as e:
            print("Failed to load agent:", e)
            return False
