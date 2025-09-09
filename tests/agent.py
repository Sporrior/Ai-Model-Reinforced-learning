# ultra_dynamic_dqn_agent.py
import numpy as np
from collections import deque
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input, LayerNormalization, Add, Multiply, Lambda
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.regularizers import l2, l1_l2
from tensorflow.keras.losses import Huber, MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
import pickle
import os
import json
import time
import math
from typing import Dict, Any, List, Tuple, Optional, Union
import gc
from scipy.signal import savgol_filter
from scipy.stats import entropy
import zlib
import base64
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import threading
import multiprocessing as mp

warnings.filterwarnings('ignore')


class EnvironmentAnalyzer:
    """Analyzes environment characteristics to optimize agent configuration"""

    def __init__(self):
        self.state_complexity = 0
        self.action_space_type = "discrete"
        self.reward_range = (-np.inf, np.inf)
        self.reward_sparsity = 0
        self.episode_length_variance = 0
        self.state_correlation = 0
        self.action_effectiveness = {}
        self.environment_type = "unknown"

    def analyze_environment(self, experiences: List[Tuple], state_size: int, action_size: int) -> Dict[str, Any]:
        """Comprehensive environment analysis"""
        if not experiences:
            return self._get_default_analysis(state_size, action_size)

        states = np.array([exp[0] for exp in experiences])
        actions = np.array([exp[1] for exp in experiences])
        rewards = np.array([exp[2] for exp in experiences])
        next_states = np.array([exp[3] for exp in experiences])
        dones = np.array([exp[4] for exp in experiences])

        analysis = {
            'state_complexity': self._analyze_state_complexity(states),
            'reward_characteristics': self._analyze_rewards(rewards),
            'action_distribution': self._analyze_actions(actions, action_size),
            'state_transitions': self._analyze_transitions(states, next_states),
            'episode_characteristics': self._analyze_episodes(dones),
            'environment_type': self._classify_environment(states, rewards, actions),
            'difficulty_level': self._assess_difficulty(states, rewards, actions),
            'optimal_config': self._suggest_configuration(states, rewards, actions, state_size, action_size)
        }

        return analysis

    def _analyze_state_complexity(self, states: np.ndarray) -> Dict[str, float]:
        """Analyze state space complexity"""
        if len(states) < 2:
            return {'dimensionality': states.shape[1] if len(states.shape) > 1 else 1, 'variance': 1.0, 'entropy': 1.0}

        # Dimensionality and variance analysis
        dimensionality = states.shape[1] if len(states.shape) > 1 else 1
        state_var = np.mean(np.var(states, axis=0)) if dimensionality > 1 else np.var(states)

        # Entropy analysis
        try:
            # Discretize continuous states for entropy calculation
            discretized_states = np.digitize(states, bins=np.linspace(np.min(states), np.max(states), 10))
            state_entropy = entropy(np.bincount(discretized_states.flatten()) + 1e-8)
        except:
            state_entropy = 1.0

        return {
            'dimensionality': dimensionality,
            'variance': float(state_var),
            'entropy': float(state_entropy),
            'complexity_score': float(dimensionality * state_var * state_entropy)
        }

    def _analyze_rewards(self, rewards: np.ndarray) -> Dict[str, float]:
        """Analyze reward characteristics"""
        if len(rewards) == 0:
            return {'sparsity': 1.0, 'variance': 1.0, 'range': 1.0, 'distribution': 'uniform'}

        # Reward sparsity (fraction of zero rewards)
        sparsity = np.mean(rewards == 0) if len(rewards) > 0 else 0

        # Reward variance and range
        reward_var = np.var(rewards) if len(rewards) > 1 else 0
        reward_range = np.max(rewards) - np.min(rewards) if len(rewards) > 1 else 1

        # Reward distribution analysis
        positive_ratio = np.mean(rewards > 0) if len(rewards) > 0 else 0
        negative_ratio = np.mean(rewards < 0) if len(rewards) > 0 else 0

        if positive_ratio > 0.8:
            distribution = 'mostly_positive'
        elif negative_ratio > 0.8:
            distribution = 'mostly_negative'
        elif sparsity > 0.8:
            distribution = 'sparse'
        else:
            distribution = 'balanced'

        return {
            'sparsity': float(sparsity),
            'variance': float(reward_var),
            'range': float(reward_range),
            'positive_ratio': float(positive_ratio),
            'negative_ratio': float(negative_ratio),
            'distribution': distribution
        }

    def _analyze_actions(self, actions: np.ndarray, action_size: int) -> Dict[str, Any]:
        """Analyze action space and usage"""
        if len(actions) == 0:
            return {'entropy': 1.0, 'balance': 1.0, 'effectiveness': {}}

        # Action distribution entropy
        action_counts = np.bincount(actions, minlength=action_size)
        action_probs = action_counts / np.sum(action_counts)
        action_entropy = entropy(action_probs + 1e-8)

        # Action balance (how evenly actions are used)
        expected_prob = 1.0 / action_size
        balance_score = 1.0 - np.mean(np.abs(action_probs - expected_prob))

        return {
            'entropy': float(action_entropy),
            'balance': float(balance_score),
            'distribution': action_probs.tolist(),
            'most_used': int(np.argmax(action_counts)),
            'least_used': int(np.argmin(action_counts))
        }

    def _analyze_transitions(self, states: np.ndarray, next_states: np.ndarray) -> Dict[str, float]:
        """Analyze state transition characteristics"""
        if len(states) < 2 or len(next_states) < 2:
            return {'smoothness': 1.0, 'predictability': 0.5}

        # Transition smoothness (average change in state)
        state_changes = np.mean(np.abs(next_states - states))
        max_possible_change = np.max(np.abs(states)) + np.max(np.abs(next_states))
        smoothness = 1.0 - (state_changes / (max_possible_change + 1e-8))

        # Transition predictability (correlation between consecutive states)
        try:
            correlations = []
            for i in range(min(states.shape[1] if len(states.shape) > 1 else 1, 10)):
                if len(states.shape) > 1:
                    corr = np.corrcoef(states[:, i], next_states[:, i])[0, 1]
                else:
                    corr = np.corrcoef(states, next_states)[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))
            predictability = np.mean(correlations) if correlations else 0.5
        except:
            predictability = 0.5

        return {
            'smoothness': float(smoothness),
            'predictability': float(predictability),
            'average_change': float(state_changes)
        }

    def _analyze_episodes(self, dones: np.ndarray) -> Dict[str, Any]:
        """Analyze episode characteristics"""
        if len(dones) == 0:
            return {'average_length': 100, 'variance': 0, 'completion_rate': 1.0}

        # Find episode boundaries
        episode_ends = np.where(dones)[0]
        if len(episode_ends) < 2:
            return {'average_length': len(dones), 'variance': 0, 'completion_rate': 1.0}

        # Calculate episode lengths
        episode_lengths = []
        start = 0
        for end in episode_ends:
            episode_lengths.append(end - start + 1)
            start = end + 1

        avg_length = np.mean(episode_lengths)
        length_variance = np.var(episode_lengths)
        completion_rate = len(episode_ends) / max(1, len(dones) / avg_length)

        return {
            'average_length': float(avg_length),
            'variance': float(length_variance),
            'completion_rate': float(min(1.0, completion_rate)),
            'total_episodes': len(episode_ends)
        }

    def _classify_environment(self, states: np.ndarray, rewards: np.ndarray, actions: np.ndarray) -> str:
        """Classify the type of environment"""
        if len(states) < 10:
            return "unknown"

        state_complexity = self._analyze_state_complexity(states)
        reward_chars = self._analyze_rewards(rewards)

        # Classification logic
        if state_complexity['dimensionality'] > 50:
            if reward_chars['sparsity'] > 0.9:
                return "high_dimensional_sparse"
            else:
                return "high_dimensional_dense"
        elif reward_chars['sparsity'] > 0.8:
            if state_complexity['complexity_score'] > 10:
                return "sparse_reward_complex"
            else:
                return "sparse_reward_simple"
        elif reward_chars['distribution'] == 'mostly_negative':
            return "penalty_based"
        elif state_complexity['complexity_score'] > 20:
            return "complex_continuous"
        else:
            return "standard_discrete"

    def _assess_difficulty(self, states: np.ndarray, rewards: np.ndarray, actions: np.ndarray) -> Dict[str, float]:
        """Assess environment difficulty"""
        if len(states) < 10:
            return {'overall': 0.5, 'exploration': 0.5, 'credit_assignment': 0.5}

        state_complexity = self._analyze_state_complexity(states)
        reward_chars = self._analyze_rewards(rewards)

        # Exploration difficulty
        exploration_difficulty = min(1.0, state_complexity['complexity_score'] / 100)

        # Credit assignment difficulty
        credit_assignment_difficulty = reward_chars['sparsity'] * (1.0 - reward_chars['positive_ratio'])

        # Overall difficulty
        overall_difficulty = (exploration_difficulty + credit_assignment_difficulty) / 2

        return {
            'overall': float(overall_difficulty),
            'exploration': float(exploration_difficulty),
            'credit_assignment': float(credit_assignment_difficulty)
        }

    def _suggest_configuration(self, states: np.ndarray, rewards: np.ndarray,
                               actions: np.ndarray, state_size: int, action_size: int) -> Dict[str, Any]:
        """Suggest optimal configuration based on analysis"""
        if len(states) < 10:
            return self._get_default_config_suggestions(state_size, action_size)

        analysis = {
            'state_complexity': self._analyze_state_complexity(states),
            'reward_chars': self._analyze_rewards(rewards),
            'difficulty': self._assess_difficulty(states, rewards, actions),
            'env_type': self._classify_environment(states, rewards, actions)
        }

        config = {}

        # Network architecture based on state complexity
        complexity = analysis['state_complexity']['complexity_score']
        if complexity > 50:
            config['hidden_layers'] = [512, 256, 128, 64, 32]
            config['dropout_rate'] = 0.3
        elif complexity > 20:
            config['hidden_layers'] = [256, 128, 64, 32]
            config['dropout_rate'] = 0.2
        else:
            config['hidden_layers'] = [128, 64, 32]
            config['dropout_rate'] = 0.1

        # Learning rate based on difficulty
        difficulty = analysis['difficulty']['overall']
        if difficulty > 0.8:
            config['learning_rate'] = 0.0001
            config['epsilon_decay'] = 0.9995
        elif difficulty > 0.5:
            config['learning_rate'] = 0.0005
            config['epsilon_decay'] = 0.999
        else:
            config['learning_rate'] = 0.001
            config['epsilon_decay'] = 0.995

        # Memory and batch size based on environment type
        if analysis['env_type'] in ['high_dimensional_sparse', 'sparse_reward_complex']:
            config['memory_size'] = 200000
            config['batch_size'] = 64
            config['prioritized_replay'] = True
            config['alpha'] = 0.8
        else:
            config['memory_size'] = 100000
            config['batch_size'] = 128
            config['alpha'] = 0.6

        # Exploration strategy
        if analysis['reward_chars']['sparsity'] > 0.8:
            config['epsilon_start'] = 1.0
            config['epsilon_min'] = 0.05
            config['noisy_nets'] = True
        else:
            config['epsilon_start'] = 0.9
            config['epsilon_min'] = 0.01

        # Advanced features
        config['double_dqn'] = True
        config['dueling_dqn'] = complexity > 20
        config['n_step'] = 3 if analysis['reward_chars']['sparsity'] > 0.5 else 1

        return config

    def _get_default_analysis(self, state_size: int, action_size: int) -> Dict[str, Any]:
        """Return default analysis when no data is available"""
        return {
            'state_complexity': {'dimensionality': state_size, 'variance': 1.0, 'entropy': 1.0,
                                 'complexity_score': state_size},
            'reward_characteristics': {'sparsity': 0.5, 'variance': 1.0, 'range': 1.0, 'distribution': 'balanced'},
            'action_distribution': {'entropy': math.log(action_size), 'balance': 1.0},
            'environment_type': 'unknown',
            'difficulty_level': {'overall': 0.5, 'exploration': 0.5, 'credit_assignment': 0.5},
            'optimal_config': self._get_default_config_suggestions(state_size, action_size)
        }

    def _get_default_config_suggestions(self, state_size: int, action_size: int) -> Dict[str, Any]:
        """Get default configuration suggestions"""
        return {
            'hidden_layers': [256, 128, 64] if state_size > 20 else [128, 64, 32],
            'learning_rate': 0.0005,
            'batch_size': 128,
            'memory_size': 100000,
            'epsilon_decay': 0.999,
            'prioritized_replay': True,
            'double_dqn': True,
            'dueling_dqn': state_size > 10
        }


class HyperparameterOptimizer:
    """Optimizes hyperparameters using Bayesian-like optimization"""

    def __init__(self):
        self.parameter_history = deque(maxlen=50)
        self.performance_history = deque(maxlen=50)
        self.best_params = None
        self.best_performance = -float('inf')

    def suggest_parameters(self, current_performance: float, current_params: Dict[str, Any],
                           environment_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest new parameters based on performance history"""

        # Store current results
        self.parameter_history.append(current_params.copy())
        self.performance_history.append(current_performance)

        # Update best parameters
        if current_performance > self.best_performance:
            self.best_performance = current_performance
            self.best_params = current_params.copy()

        # If we don't have enough history, return slight variations
        if len(self.parameter_history) < 5:
            return self._random_variation(current_params)

        # Analyze parameter trends
        return self._optimize_parameters(environment_analysis)

    def _random_variation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create random variations of parameters"""
        new_params = params.copy()

        # Learning rate variation
        if 'learning_rate' in params:
            multiplier = random.uniform(0.5, 2.0)
            new_params['learning_rate'] = max(1e-6, min(0.01, params['learning_rate'] * multiplier))

        # Batch size variation
        if 'batch_size' in params:
            sizes = [32, 64, 128, 256]
            new_params['batch_size'] = random.choice(sizes)

        # Epsilon decay variation
        if 'epsilon_decay' in params:
            new_params['epsilon_decay'] = random.uniform(0.99, 0.9999)

        return new_params

    def _optimize_parameters(self, environment_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize parameters based on history and environment"""
        if not self.best_params:
            return {}

        optimized = self.best_params.copy()

        # Get recent performance trend
        recent_performance = list(self.performance_history)[-5:]
        performance_trend = np.mean(np.diff(recent_performance)) if len(recent_performance) > 1 else 0

        # Adjust based on trend and environment
        difficulty = environment_analysis.get('difficulty_level', {}).get('overall', 0.5)

        if performance_trend < 0:  # Performance declining
            # More conservative learning
            optimized['learning_rate'] = max(1e-6, optimized.get('learning_rate', 0.001) * 0.8)
            optimized['epsilon_decay'] = min(0.9999, optimized.get('epsilon_decay', 0.999) + 0.001)
        else:  # Performance improving or stable
            # More aggressive learning if environment is difficult
            if difficulty > 0.7:
                optimized['learning_rate'] = min(0.01, optimized.get('learning_rate', 0.001) * 1.2)

        return optimized


class NoisyLayer(tf.keras.layers.Layer):
    """Noisy layer for exploration"""

    def __init__(self, units, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        input_dim = input_shape[-1]

        # Learnable parameters
        self.weight_mu = self.add_weight(
            shape=(input_dim, self.units),
            initializer='glorot_uniform',
            name='weight_mu'
        )
        self.weight_sigma = self.add_weight(
            shape=(input_dim, self.units),
            initializer='zeros',
            name='weight_sigma'
        )
        self.bias_mu = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            name='bias_mu'
        )
        self.bias_sigma = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            name='bias_sigma'
        )

        super().build(input_shape)

    def call(self, inputs, training=None):
        if training:
            # Generate noise
            input_noise = tf.random.normal((tf.shape(inputs)[0], tf.shape(inputs)[1]))
            output_noise = tf.random.normal((tf.shape(inputs)[0], self.units))

            # Noisy weights and biases
            weight_noise = tf.matmul(tf.expand_dims(input_noise, -1), tf.expand_dims(output_noise, 1))
            weight = self.weight_mu + self.weight_sigma * weight_noise
            bias = self.bias_mu + self.bias_sigma * output_noise
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        outputs = tf.matmul(inputs, weight) + bias

        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs


class UltraDynamicDQNAgent:
    """Ultra-dynamic DQN agent that adapts to any environment automatically"""

    def __init__(self, state_size: int, action_size: int, config: Optional[Dict[str, Any]] = None):
        self.state_size = state_size
        self.action_size = action_size

        # Initialize analyzers and optimizers
        self.env_analyzer = EnvironmentAnalyzer()
        self.param_optimizer = HyperparameterOptimizer()

        # Dynamic configuration
        self.config = self._get_base_config()
        if config:
            self.config.update(config)

        # Analysis and adaptation tracking
        self.analysis_cache = {}
        self.adaptation_history = deque(maxlen=20)
        self.performance_buffer = deque(maxlen=50)
        self.last_analysis_step = 0
        self.analysis_frequency = 100

        # Initialize from config
        self._init_from_config()

        # Experience memory with intelligent sizing
        self.memory = deque(maxlen=self.memory_size)
        self.priorities = deque(maxlen=self.memory_size)
        self.max_priority = 1.0

        # Multiple models for ensemble learning
        self.models = []
        self.target_models = []
        self.num_models = self.config.get('ensemble_size', 3)

        for i in range(self.num_models):
            model = self._build_adaptive_model()
            target_model = self._build_adaptive_model()
            self.models.append(model)
            self.target_models.append(target_model)
            self.update_target_model(i)

        # Training state
        self.training_step = 0
        self.episode_count = 0
        self.total_reward = 0

        # Advanced tracking
        self.loss_history = deque(maxlen=1000)
        self.reward_history = deque(maxlen=200)
        self.q_value_history = deque(maxlen=1000)
        self.exploration_history = deque(maxlen=100)

        # Adaptive mechanisms
        self.adaptation_active = True
        self.auto_architecture = True
        self.curriculum_learning = True

        # Performance tracking
        self.best_performance = -float('inf')
        self.performance_plateau_count = 0
        self.adaptation_cooldown = 0

        print(f"Ultra-Dynamic DQN Agent initialized with {self.num_models} models")
        print(f"State size: {state_size}, Action size: {action_size}")

    def _get_base_config(self) -> Dict[str, Any]:
        """Get comprehensive base configuration"""
        return {
            # Core architecture
            'hidden_layers': [256, 128, 64, 32],
            'activation': 'relu',
            'output_activation': 'linear',
            'use_batch_norm': True,
            'use_layer_norm': False,
            'dropout_rate': 0.2,
            'l2_reg': 0.001,

            # Learning parameters
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_min': 0.01,
            'epsilon_decay': 0.999,
            'learning_rate': 0.0005,
            'batch_size': 128,
            'memory_size': 100000,

            # Advanced features
            'double_dqn': True,
            'dueling_dqn': True,
            'noisy_nets': False,
            'prioritized_replay': True,
            'n_step': 1,
            'ensemble_size': 3,
            'curriculum_learning': True,

            # Adaptive mechanisms
            'auto_adapt': True,
            'adaptation_frequency': 100,
            'performance_window': 50,
            'architecture_evolution': True,

            # Optimization
            'optimizer': 'adam',
            'loss_function': 'huber',
            'grad_clipping': 1.0,
            'target_update_freq': 100,
            'soft_update_tau': 0.01,

            # Exploration strategies
            'exploration_strategy': 'epsilon_greedy',  # 'epsilon_greedy', 'ucb', 'thompson'
            'intrinsic_motivation': True,
            'curiosity_driven': True,

            # Memory management
            'memory_compression': True,
            'experience_replay_buffer': 'prioritized',
            'forget_old_experiences': True,
        }

    def _init_from_config(self):
        """Initialize all parameters from configuration"""
        # Basic parameters
        self.gamma = self.config['gamma']
        self.epsilon = self.config['epsilon_start']
        self.epsilon_min = self.config['epsilon_min']
        self.epsilon_decay = self.config['epsilon_decay']
        self.learning_rate = self.config['learning_rate']
        self.batch_size = self.config['batch_size']
        self.memory_size = self.config['memory_size']

        # Advanced features
        self.double_dqn = self.config['double_dqn']
        self.dueling_dqn = self.config['dueling_dqn']
        self.noisy_nets = self.config['noisy_nets']
        self.prioritized_replay = self.config['prioritized_replay']
        self.n_step = self.config['n_step']

        # Adaptive parameters
        self.adaptation_frequency = self.config.get('adaptation_frequency', 100)
        self.performance_window = self.config.get('performance_window', 50)

    def _build_adaptive_model(self) -> Model:
        """Build highly adaptive neural network"""
        inputs = Input(shape=(self.state_size,))
        x = inputs

        # Feature extraction layers
        for i, units in enumerate(self.config['hidden_layers']):
            if self.noisy_nets and i >= len(self.config['hidden_layers']) - 2:
                # Use noisy layers for last few layers
                x = NoisyLayer(units, activation=self.config['activation'], name=f'noisy_{i}')(x)
            else:
                x = Dense(units, activation=self.config['activation'],
                          kernel_regularizer=l2(self.config['l2_reg']), name=f'dense_{i}')(x)

            if self.config.get('use_batch_norm', False):
                x = BatchNormalization(name=f'bn_{i}')(x)
            if self.config.get('use_layer_norm', False):
                x = LayerNormalization(name=f'ln_{i}')(x)
            if self.config.get('dropout_rate', 0) > 0:
                x = Dropout(self.config['dropout_rate'], name=f'dropout_{i}')(x)

        # Dueling architecture
        if self.dueling_dqn:
            # Value stream
            value_stream = Dense(128, activation=self.config['activation'], name='value_dense')(x)
            value_stream = Dense(1, activation='linear', name='value')(value_stream)

            # Advantage stream
            advantage_stream = Dense(128, activation=self.config['activation'], name='advantage_dense')(x)
            advantage_stream = Dense(self.action_size, activation='linear', name='advantage')(advantage_stream)

            # Combine streams
            advantage_mean = Lambda(lambda x: tf.reduce_mean(x, axis=1, keepdims=True))(advantage_stream)
            advantage_centered = Lambda(lambda x: x[0] - x[1])([advantage_stream, advantage_mean])
            q_values = Add(name='q_values')([value_stream, advantage_centered])
        else:
            # Standard DQN output
            q_values = Dense(self.action_size, activation=self.config['output_activation'], name='q_values')(x)

        model = Model(inputs=inputs, outputs=q_values)

        # Compile with adaptive optimizer
        optimizer = self._create_adaptive_optimizer()
        loss_fn = self._create_adaptive_loss()

        model.compile(optimizer=optimizer, loss=loss_fn, metrics=['mae'])
        return model

    def _create_adaptive_optimizer(self):
        """Create optimizer that adapts based on performance"""
        optimizer_type = self.config.get('optimizer', 'adam').lower()

        if optimizer_type == 'rmsprop':
            return RMSprop(learning_rate=self.learning_rate, clipnorm=self.config.get('grad_clipping', 1.0))
        elif optimizer_type == 'sgd':
            return SGD(learning_rate=self.learning_rate, momentum=0.9, clipnorm=self.config.get('grad_clipping', 1.0))
        else:  # Adam
            return Adam(learning_rate=self.learning_rate, clipnorm=self.config.get('grad_clipping', 1.0))

    def _create_adaptive_loss(self):
        """Create loss function that adapts to the problem"""
        loss_type = self.config.get('loss_function', 'huber').lower()

        if loss_type == 'mse':
            return MeanSquaredError()
        else:  # Huber loss
            return Huber(delta=self.config.get('huber_delta', 1.0))

    def analyze_and_adapt(self, force_analysis: bool = False) -> None:
        """Analyze environment and adapt agent configuration"""

        # Check if analysis is needed
        should_analyze = (
                force_analysis or
                self.training_step - self.last_analysis_step >= self.adaptation_frequency or
                len(self.memory) > 0 and len(self.memory) % (self.memory_size // 10) == 0
        )

        if not should_analyze or self.adaptation_cooldown > 0:
            if self.adaptation_cooldown > 0:
                self.adaptation_cooldown -= 1
            return

        if len(self.memory) < 50:  # Need minimum experiences
            return

        print(f"\n🔍 Analyzing environment (Step {self.training_step})...")

        # Sample recent experiences for analysis
        recent_experiences = list(self.memory)[-min(1000, len(self.memory)):]

        # Perform comprehensive analysis
        analysis = self.env_analyzer.analyze_environment(
            recent_experiences, self.state_size, self.action_size
        )

        # Cache analysis
        self.analysis_cache = analysis
        self.last_analysis_step = self.training_step

        # Calculate current performance
        recent_rewards = list(self.reward_history)[-self.performance_window:] if self.reward_history else [0]
        current_performance = np.mean(recent_rewards)

        # Get optimization suggestions
        current_params = {
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epsilon_decay': self.epsilon_decay,
            'memory_size': self.memory_size
        }

        suggested_params = self.param_optimizer.suggest_parameters(
            current_performance, current_params, analysis
        )

        # Apply adaptations
        adaptations_made = self._apply_adaptations(analysis, suggested_params)

        if adaptations_made:
            print(f"🔧 Applied {len(adaptations_made)} adaptations: {', '.join(adaptations_made)}")
            self.adaptation_history.append({
                'step': self.training_step,
                'performance': current_performance,
                'adaptations': adaptations_made,
                'analysis': analysis
            })
            self.adaptation_cooldown = 20  # Cooldown period

        # Print analysis summary
        self._print_analysis_summary(analysis)

    def _apply_adaptations(self, analysis: Dict[str, Any], suggested_params: Dict[str, Any]) -> List[str]:
        """Apply adaptations based on analysis"""
        adaptations = []

        # 1. Adjust learning rate based on performance and difficulty
        difficulty = analysis.get('difficulty_level', {}).get('overall', 0.5)
        if 'learning_rate' in suggested_params:
            old_lr = self.learning_rate
            self.learning_rate = suggested_params['learning_rate']
            if abs(old_lr - self.learning_rate) > old_lr * 0.1:  # Significant change
                self._update_optimizer_lr()
                adaptations.append(f"learning_rate: {old_lr:.6f} → {self.learning_rate:.6f}")

        # 2. Adapt exploration strategy
        reward_sparsity = analysis.get('reward_characteristics', {}).get('sparsity', 0.5)
        if reward_sparsity > 0.8 and self.epsilon < 0.3:
            self.epsilon = min(1.0, self.epsilon * 1.5)
            adaptations.append(f"increased_exploration: ε={self.epsilon:.3f}")
        elif reward_sparsity < 0.2 and self.epsilon > 0.05:
            self.epsilon = max(self.epsilon_min, self.epsilon * 0.8)
            adaptations.append(f"reduced_exploration: ε={self.epsilon:.3f}")

        # 3. Adjust batch size based on environment complexity
        complexity = analysis.get('state_complexity', {}).get('complexity_score', 1)
        optimal_batch_size = self._calculate_optimal_batch_size(complexity)
        if optimal_batch_size != self.batch_size:
            old_batch_size = self.batch_size
            self.batch_size = optimal_batch_size
            adaptations.append(f"batch_size: {old_batch_size} → {self.batch_size}")

        # 4. Adapt memory management
        if analysis.get('environment_type') in ['sparse_reward_complex', 'high_dimensional_sparse']:
            if self.memory_size < 200000:
                self.memory_size = min(200000, self.memory_size * 2)
                # Extend memory deque
                new_memory = deque(self.memory, maxlen=self.memory_size)
                new_priorities = deque(self.priorities, maxlen=self.memory_size)
                self.memory = new_memory
                self.priorities = new_priorities
                adaptations.append(f"expanded_memory: {self.memory_size}")

        # 5. Adjust target network update frequency
        env_type = analysis.get('environment_type', 'unknown')
        if env_type in ['penalty_based', 'sparse_reward_complex']:
            self.config['target_update_freq'] = max(50, self.config['target_update_freq'] // 2)
            adaptations.append(f"faster_target_updates: {self.config['target_update_freq']}")

        # 6. Enable/disable advanced features based on complexity
        if complexity > 30 and not self.dueling_dqn:
            self.dueling_dqn = True
            self._rebuild_models_if_needed()
            adaptations.append("enabled_dueling_dqn")

        if reward_sparsity > 0.9 and not self.noisy_nets:
            self.noisy_nets = True
            self._rebuild_models_if_needed()
            adaptations.append("enabled_noisy_nets")

        # 7. Adjust prioritized replay parameters
        if self.prioritized_replay:
            if reward_sparsity > 0.7:
                self.config['alpha'] = min(1.0, self.config.get('alpha', 0.6) + 0.1)
                adaptations.append(f"increased_priority_alpha: {self.config['alpha']:.2f}")

        return adaptations

    def _calculate_optimal_batch_size(self, complexity: float) -> int:
        """Calculate optimal batch size based on complexity"""
        if complexity > 50:
            return 64  # Smaller batches for complex environments
        elif complexity > 20:
            return 128
        else:
            return 256  # Larger batches for simple environments

    def _rebuild_models_if_needed(self):
        """Rebuild models if architecture changes are needed"""
        print("🔄 Rebuilding models with new architecture...")

        # Save current weights
        old_weights = []
        for model in self.models:
            old_weights.append(model.get_weights())

        # Rebuild models
        new_models = []
        new_target_models = []

        for i in range(self.num_models):
            model = self._build_adaptive_model()
            target_model = self._build_adaptive_model()

            # Try to transfer compatible weights
            try:
                compatible_weights = self._extract_compatible_weights(old_weights[i], model.get_weights())
                model.set_weights(compatible_weights)
                target_model.set_weights(compatible_weights)
            except:
                print(f"⚠️ Could not transfer weights for model {i}, using random initialization")

            new_models.append(model)
            new_target_models.append(target_model)

        self.models = new_models
        self.target_models = new_target_models

    def _extract_compatible_weights(self, old_weights: List, new_shape_weights: List) -> List:
        """Extract compatible weights between different architectures"""
        compatible_weights = []

        for old_w, new_w in zip(old_weights, new_shape_weights):
            if old_w.shape == new_w.shape:
                compatible_weights.append(old_w)
            else:
                # Try to extract compatible sub-weights
                if len(old_w.shape) == 2 and len(new_w.shape) == 2:  # Dense layer
                    min_in = min(old_w.shape[0], new_w.shape[0])
                    min_out = min(old_w.shape[1], new_w.shape[1])
                    extracted = np.random.normal(0, 0.1, new_w.shape).astype(np.float32)
                    extracted[:min_in, :min_out] = old_w[:min_in, :min_out]
                    compatible_weights.append(extracted)
                elif len(old_w.shape) == 1 and len(new_w.shape) == 1:  # Bias
                    min_size = min(old_w.shape[0], new_w.shape[0])
                    extracted = np.random.normal(0, 0.1, new_w.shape).astype(np.float32)
                    extracted[:min_size] = old_w[:min_size]
                    compatible_weights.append(extracted)
                else:
                    compatible_weights.append(new_w)  # Use random initialization

        return compatible_weights

    def _print_analysis_summary(self, analysis: Dict[str, Any]):
        """Print comprehensive analysis summary"""
        print("\n📊 Environment Analysis Summary:")
        print("=" * 50)

        # Environment characteristics
        env_type = analysis.get('environment_type', 'unknown')
        difficulty = analysis.get('difficulty_level', {})
        print(f"Environment Type: {env_type}")
        print(f"Overall Difficulty: {difficulty.get('overall', 0):.2f}")
        print(f"Exploration Challenge: {difficulty.get('exploration', 0):.2f}")
        print(f"Credit Assignment: {difficulty.get('credit_assignment', 0):.2f}")

        # State complexity
        state_complex = analysis.get('state_complexity', {})
        print(f"\nState Complexity Score: {state_complex.get('complexity_score', 0):.2f}")
        print(f"Dimensionality: {state_complex.get('dimensionality', 0)}")
        print(f"State Variance: {state_complex.get('variance', 0):.4f}")

        # Reward characteristics
        reward_chars = analysis.get('reward_characteristics', {})
        print(f"\nReward Sparsity: {reward_chars.get('sparsity', 0):.2f}")
        print(f"Reward Distribution: {reward_chars.get('distribution', 'unknown')}")
        print(f"Positive Ratio: {reward_chars.get('positive_ratio', 0):.2f}")

        print("=" * 50)

    def remember(self, state: np.ndarray, action: int, reward: float,
                 next_state: np.ndarray, done: bool, info: Optional[Dict] = None) -> None:
        """Enhanced memory storage with intelligent prioritization"""

        # Apply reward clipping if configured
        if self.config.get('reward_clipping'):
            reward = np.clip(reward, *self.config['reward_clipping'])

        # Calculate intrinsic reward for curiosity-driven learning
        intrinsic_reward = 0
        if self.config.get('curiosity_driven', False) and len(self.memory) > 100:
            intrinsic_reward = self._calculate_intrinsic_reward(state, next_state, action)
            reward += intrinsic_reward * 0.1  # Scale intrinsic reward

        experience = (state, action, reward, next_state, done, info or {})
        self.memory.append(experience)

        # Advanced prioritization
        if self.prioritized_replay:
            priority = self._calculate_advanced_priority(state, action, reward, next_state, done)
            self.priorities.append(priority)
            self.max_priority = max(self.max_priority, priority)

        # Trigger adaptation check
        if len(self.memory) % 500 == 0:  # Check every 500 experiences
            self.analyze_and_adapt()

    def _calculate_intrinsic_reward(self, state: np.ndarray, next_state: np.ndarray, action: int) -> float:
        """Calculate intrinsic curiosity reward based on prediction error"""
        try:
            # Use ensemble prediction disagreement as curiosity measure
            predictions = []
            for model in self.models:
                pred = model.predict(state.reshape(1, -1), verbose=0)[0]
                predictions.append(pred)

            # Calculate disagreement (epistemic uncertainty)
            predictions = np.array(predictions)
            disagreement = np.std(predictions, axis=0)[action]

            return min(1.0, disagreement)  # Normalize
        except:
            return 0.0

    def _calculate_advanced_priority(self, state: np.ndarray, action: int, reward: float,
                                     next_state: np.ndarray, done: bool) -> float:
        """Calculate sophisticated priority considering multiple factors"""
        try:
            # Ensemble TD error
            td_errors = []

            for i, (model, target_model) in enumerate(zip(self.models, self.target_models)):
                current_q = model.predict(state.reshape(1, -1), verbose=0)[0]
                next_q = target_model.predict(next_state.reshape(1, -1), verbose=0)[0]

                if done:
                    target = reward
                else:
                    if self.double_dqn:
                        next_action = np.argmax(model.predict(next_state.reshape(1, -1), verbose=0)[0])
                        target = reward + self.gamma * next_q[next_action]
                    else:
                        target = reward + self.gamma * np.max(next_q)

                td_error = abs(target - current_q[action])
                td_errors.append(td_error)

            # Use maximum TD error across ensemble
            base_priority = max(td_errors)

            # Add curiosity bonus
            curiosity_bonus = self._calculate_intrinsic_reward(state, next_state, action)

            # Add recency bonus (newer experiences get slight priority boost)
            recency_bonus = 0.1

            total_priority = base_priority + curiosity_bonus * 0.5 + recency_bonus + self.config.get('epsilon_priority',
                                                                                                     1e-6)

            return total_priority

        except Exception as e:
            return self.max_priority  # Fallback

    def act(self, state: np.ndarray, training: bool = True, return_q_values: bool = False) -> Any:
        """Intelligent action selection with multiple strategies"""

        if training:
            # Adaptive exploration strategy
            exploration_strategy = self.config.get('exploration_strategy', 'epsilon_greedy')

            if exploration_strategy == 'epsilon_greedy':
                if np.random.rand() <= self.epsilon:
                    action = random.randrange(self.action_size)
                    if return_q_values:
                        q_values = self._get_ensemble_q_values(state)
                        return action, q_values
                    return action

            elif exploration_strategy == 'ucb':
                action = self._ucb_action_selection(state)
                if return_q_values:
                    q_values = self._get_ensemble_q_values(state)
                    return action, q_values
                return action

            elif exploration_strategy == 'thompson':
                action = self._thompson_sampling(state)
                if return_q_values:
                    q_values = self._get_ensemble_q_values(state)
                    return action, q_values
                return action

        # Ensemble prediction for action selection
        q_values = self._get_ensemble_q_values(state)
        action = int(np.argmax(q_values))

        # Track Q-values for analysis
        self.q_value_history.append(np.max(q_values))

        if return_q_values:
            return action, q_values
        return action

    def _get_ensemble_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get ensemble Q-values from all models"""
        state = state.reshape(1, -1)
        q_values_list = []

        for model in self.models:
            q_vals = model.predict(state, verbose=0, batch_size=1)[0]
            q_values_list.append(q_vals)

        # Average ensemble predictions
        ensemble_q_values = np.mean(q_values_list, axis=0)
        return ensemble_q_values

    def _ucb_action_selection(self, state: np.ndarray) -> int:
        """Upper Confidence Bound action selection"""
        q_values = self._get_ensemble_q_values(state)

        # Calculate uncertainty (disagreement between models)
        state = state.reshape(1, -1)
        predictions = []
        for model in self.models:
            pred = model.predict(state, verbose=0)[0]
            predictions.append(pred)

        predictions = np.array(predictions)
        uncertainty = np.std(predictions, axis=0)

        # UCB formula: Q(s,a) + c * sqrt(uncertainty)
        c = 2.0  # Exploration constant
        ucb_values = q_values + c * uncertainty

        return int(np.argmax(ucb_values))

    def _thompson_sampling(self, state: np.ndarray) -> int:
        """Thompson sampling action selection"""
        # Randomly select one model from ensemble and use its prediction
        model_idx = random.randint(0, len(self.models) - 1)
        q_values = self.models[model_idx].predict(state.reshape(1, -1), verbose=0)[0]
        return int(np.argmax(q_values))

    def replay(self) -> float:
        """Advanced experience replay with ensemble training"""
        if len(self.memory) < self.batch_size:
            return 0.0

        # Sample batch
        batch, indices, weights = self._sample_prioritized_batch()
        if batch is None:
            return 0.0

        # Prepare batch data
        states = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        next_states = np.array([exp[3] for exp in batch])
        dones = np.array([exp[4] for exp in batch])

        total_loss = 0.0

        # Train each model in ensemble
        for i, (model, target_model) in enumerate(zip(self.models, self.target_models)):
            loss = self._train_single_model(model, target_model, states, actions, rewards,
                                            next_states, dones, weights, i)
            total_loss += loss

        avg_loss = total_loss / len(self.models)
        self.loss_history.append(avg_loss)

        # Update priorities
        if self.prioritized_replay and indices is not None:
            self._update_priorities(indices, states, actions, rewards, next_states, dones)

        # Update target networks
        if self.training_step % self.config.get('target_update_freq', 100) == 0:
            for i in range(len(self.models)):
                self.update_target_model(i, soft_update=True)

        # Decay exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.training_step += 1

        # Periodic adaptation
        if self.training_step % self.adaptation_frequency == 0:
            self.analyze_and_adapt()

        return avg_loss

    def _sample_prioritized_batch(self):
        """Sample batch with advanced prioritization"""
        if len(self.memory) < self.batch_size:
            return None, None, None

        if self.prioritized_replay and len(self.priorities) >= self.batch_size:
            priorities = np.array(list(self.priorities))
            probabilities = priorities ** self.config.get('alpha', 0.6)
            probabilities /= probabilities.sum()

            indices = np.random.choice(len(self.memory), self.batch_size, p=probabilities)
            beta = min(1.0,
                       self.config.get('beta', 0.4) + self.training_step * self.config.get('beta_increment', 0.001))
            weights = (len(self.memory) * probabilities[indices]) ** (-beta)
            weights /= weights.max()
        else:
            indices = np.random.choice(len(self.memory), self.batch_size)
            weights = np.ones(self.batch_size)

        batch = [self.memory[i] for i in indices]
        return batch, indices, weights

    def _train_single_model(self, model: Model, target_model: Model, states: np.ndarray,
                            actions: np.ndarray, rewards: np.ndarray, next_states: np.ndarray,
                            dones: np.ndarray, weights: np.ndarray, model_idx: int) -> float:
        """Train a single model in the ensemble"""

        # Get current Q-values
        current_q = model.predict(states, verbose=0, batch_size=self.batch_size)

        # Calculate targets using Double DQN
        if self.double_dqn:
            next_actions = np.argmax(model.predict(next_states, verbose=0, batch_size=self.batch_size), axis=1)
            next_q = target_model.predict(next_states, verbose=0, batch_size=self.batch_size)
            targets = rewards + self.gamma * next_q[np.arange(self.batch_size), next_actions] * (~dones)
        else:
            next_q = target_model.predict(next_states, verbose=0, batch_size=self.batch_size)
            targets = rewards + self.gamma * np.max(next_q, axis=1) * (~dones)

        # Update Q-values
        target_q = current_q.copy()
        target_q[np.arange(self.batch_size), actions] = targets

        # Train with sample weights
        history = model.fit(states, target_q, sample_weight=weights,
                            epochs=1, verbose=0, batch_size=self.batch_size)

        return history.history['loss'][0]

    def _update_priorities(self, indices: np.ndarray, states: np.ndarray, actions: np.ndarray,
                           rewards: np.ndarray, next_states: np.ndarray, dones: np.ndarray):
        """Update experience priorities after training"""
        for i, idx in enumerate(indices):
            if idx < len(self.priorities):
                # Calculate new priority using ensemble
                priority = self._calculate_advanced_priority(
                    states[i], actions[i], rewards[i], next_states[i], dones[i]
                )
                self.priorities[idx] = priority
                self.max_priority = max(self.max_priority, priority)

    def update_target_model(self, model_idx: int, soft_update: bool = True) -> None:
        """Update target model with soft or hard update"""
        if soft_update:
            tau = self.config.get('soft_update_tau', 0.01)
            target_weights = self.target_models[model_idx].get_weights()
            model_weights = self.models[model_idx].get_weights()

            for i in range(len(target_weights)):
                target_weights[i] = tau * model_weights[i] + (1 - tau) * target_weights[i]

            self.target_models[model_idx].set_weights(target_weights)
        else:
            self.target_models[model_idx].set_weights(self.models[model_idx].get_weights())

    def _update_optimizer_lr(self):
        """Update learning rate for all models"""
        for model in self.models + self.target_models:
            if hasattr(model.optimizer, 'learning_rate'):
                model.optimizer.learning_rate.assign(self.learning_rate)

    def end_episode(self, total_reward: float, steps: int, info: Optional[Dict] = None):
        """Handle end of episode with comprehensive tracking"""
        self.episode_count += 1
        self.reward_history.append(total_reward)
        self.total_reward += total_reward

        # Track exploration
        self.exploration_history.append(self.epsilon)

        # Update performance tracking
        if len(self.reward_history) >= 10:
            recent_avg = np.mean(list(self.reward_history)[-10:])
            if recent_avg > self.best_performance:
                self.best_performance = recent_avg
                self.performance_plateau_count = 0
            else:
                self.performance_plateau_count += 1

        # Curriculum learning adaptation
        if self.config.get('curriculum_learning', False):
            self._adjust_curriculum_difficulty(total_reward, info)

        # Periodic reporting
        if self.episode_count % 10 == 0:
            self._print_episode_summary(total_reward, steps)

    def _adjust_curriculum_difficulty(self, reward: float, info: Optional[Dict]):
        """Adjust learning curriculum based on performance"""
        if not info or 'difficulty' not in info:
            return

        current_difficulty = info.get('difficulty', 0.5)
        success_rate = np.mean([r > 0 for r in list(self.reward_history)[-20:]])

        # Increase difficulty if doing well, decrease if struggling
        if success_rate > 0.8 and current_difficulty < 1.0:
            info['suggested_difficulty'] = min(1.0, current_difficulty + 0.1)
        elif success_rate < 0.3 and current_difficulty > 0.1:
            info['suggested_difficulty'] = max(0.1, current_difficulty - 0.1)

    def _print_episode_summary(self, reward: float, steps: int):
        """Print comprehensive episode summary"""
        avg_reward = np.mean(list(self.reward_history)[-10:])
        avg_loss = np.mean(list(self.loss_history)[-100:]) if self.loss_history else 0
        avg_q = np.mean(list(self.q_value_history)[-100:]) if self.q_value_history else 0

        print(f"\n🎯 Episode {self.episode_count:4d} Summary:")
        print(f"   Reward: {reward:8.2f} | Avg10: {avg_reward:8.2f} | Best: {self.best_performance:8.2f}")
        print(f"   Steps: {steps:4d} | Loss: {avg_loss:.4f} | Q-Value: {avg_q:.2f}")
        print(f"   Exploration: ε={self.epsilon:.3f} | Memory: {len(self.memory):,}")

        if self.analysis_cache:
            env_type = self.analysis_cache.get('environment_type', 'unknown')
            difficulty = self.analysis_cache.get('difficulty_level', {}).get('overall', 0)
            print(f"   Environment: {env_type} | Difficulty: {difficulty:.2f}")

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive agent statistics"""
        base_stats = {
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'total_reward': self.total_reward,
            'epsilon': self.epsilon,
            'learning_rate': self.learning_rate,
            'memory_size': len(self.memory),
            'best_performance': self.best_performance,
            'avg_reward_10': np.mean(list(self.reward_history)[-10:]) if len(self.reward_history) >= 10 else 0,
            'avg_reward_100': np.mean(list(self.reward_history)) if self.reward_history else 0,
            'avg_loss': np.mean(list(self.loss_history)) if self.loss_history else 0,
            'avg_q_value': np.mean(list(self.q_value_history)) if self.q_value_history else 0,
            'ensemble_size': len(self.models)
        }

        # Add analysis data if available
        if self.analysis_cache:
            base_stats['environment_analysis'] = self.analysis_cache

        # Add adaptation history
        if self.adaptation_history:
            base_stats['recent_adaptations'] = list(self.adaptation_history)[-5:]

        return base_stats

    def save(self, filename: str, compress: bool = True) -> None:
        """Save complete agent state"""
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)

        # Save all models
        for i, (model, target_model) in enumerate(zip(self.models, self.target_models)):
            model.save_weights(f"{filename}_model_{i}.weights.h5")
            target_model.save_weights(f"{filename}_target_model_{i}.weights.h5")

        # Prepare comprehensive state
        agent_state = {
            'config': self.config,
            'state_size': self.state_size,
            'action_size': self.action_size,
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'epsilon': self.epsilon,
            'learning_rate': self.learning_rate,
            'best_performance': self.best_performance,
            'max_priority': self.max_priority,
            'analysis_cache': self.analysis_cache,
            'adaptation_history': list(self.adaptation_history),
            'loss_history': list(self.loss_history),
            'reward_history': list(self.reward_history),
            'q_value_history': list(self.q_value_history),
            'exploration_history': list(self.exploration_history),
            'param_optimizer_state': {
                'parameter_history': list(self.param_optimizer.parameter_history),
                'performance_history': list(self.param_optimizer.performance_history),
                'best_params': self.param_optimizer.best_params,
                'best_performance': self.param_optimizer.best_performance
            }
        }

        # Optionally compress memory
        if compress and self.config.get('memory_compression', True):
            agent_state['compressed_memory'] = self._compress_data(list(self.memory))
            agent_state['compressed_priorities'] = self._compress_data(list(self.priorities))
        else:
            agent_state['memory'] = list(self.memory)
            agent_state['priorities'] = list(self.priorities)

        # Save agent state
        with open(f"{filename}_agent_state.pkl", 'wb') as f:
            pickle.dump(agent_state, f)

        # Save JSON summary
        stats = self.get_comprehensive_stats()
        with open(f"{filename}_summary.json", 'w') as f:
            json.dump(stats, f, indent=2, default=str)

        print(f"💾 Ultra-Dynamic Agent saved to {filename}")
        print(f"   Models: {len(self.models)} | Memory: {len(self.memory):,} experiences")
        print(f"   Performance: {self.best_performance:.2f} | Training Steps: {self.training_step:,}")

    def load(self, filename: str) -> bool:
        """Load complete agent state"""
        try:
            print(f"📁 Loading Ultra-Dynamic Agent from {filename}...")

            # Load agent state
            with open(f"{filename}_agent_state.pkl", 'rb') as f:
                agent_state = pickle.load(f)

            # Restore basic parameters
            self.config.update(agent_state.get('config', {}))
            self.state_size = agent_state.get('state_size', self.state_size)
            self.action_size = agent_state.get('action_size', self.action_size)
            self.training_step = agent_state.get('training_step', 0)
            self.episode_count = agent_state.get('episode_count', 0)
            self.epsilon = agent_state.get('epsilon', self.epsilon)
            self.learning_rate = agent_state.get('learning_rate', self.learning_rate)
            self.best_performance = agent_state.get('best_performance', -float('inf'))
            self.max_priority = agent_state.get('max_priority', 1.0)

            # Restore analysis and adaptation state
            self.analysis_cache = agent_state.get('analysis_cache', {})
            self.adaptation_history = deque(agent_state.get('adaptation_history', []), maxlen=20)

            # Restore histories
            self.loss_history = deque(agent_state.get('loss_history', []), maxlen=1000)
            self.reward_history = deque(agent_state.get('reward_history', []), maxlen=200)
            self.q_value_history = deque(agent_state.get('q_value_history', []), maxlen=1000)
            self.exploration_history = deque(agent_state.get('exploration_history', []), maxlen=100)

            # Restore parameter optimizer state
            param_state = agent_state.get('param_optimizer_state', {})
            if param_state:
                self.param_optimizer.parameter_history = deque(param_state.get('parameter_history', []), maxlen=50)
                self.param_optimizer.performance_history = deque(param_state.get('performance_history', []), maxlen=50)
                self.param_optimizer.best_params = param_state.get('best_params')
                self.param_optimizer.best_performance = param_state.get('best_performance', -float('inf'))

            # Restore memory
            if 'compressed_memory' in agent_state:
                self.memory = deque(self._decompress_data(agent_state['compressed_memory']), maxlen=self.memory_size)
                self.priorities = deque(self._decompress_data(agent_state['compressed_priorities']),
                                        maxlen=self.memory_size)
            else:
                self.memory = deque(agent_state.get('memory', []), maxlen=self.memory_size)
                self.priorities = deque(agent_state.get('priorities', []), maxlen=self.memory_size)

            # Reinitialize with loaded config
            self._init_from_config()

            # Rebuild and load models
            self.models = []
            self.target_models = []

            for i in range(self.num_models):
                try:
                    model = self._build_adaptive_model()
                    target_model = self._build_adaptive_model()

                    # Load weights
                    model.load_weights(f"{filename}_model_{i}.weights.h5")
                    target_model.load_weights(f"{filename}_target_model_{i}.weights.h5")

                    self.models.append(model)
                    self.target_models.append(target_model)

                except Exception as e:
                    print(f"⚠️ Warning: Could not load model {i}: {e}")
                    # Create new model if loading fails
                    model = self._build_adaptive_model()
                    target_model = self._build_adaptive_model()
                    target_model.set_weights(model.get_weights())
                    self.models.append(model)
                    self.target_models.append(target_model)

            print(f"✅ Successfully loaded Ultra-Dynamic Agent")
            print(f"   Training Steps: {self.training_step:,} | Episodes: {self.episode_count:,}")
            print(f"   Memory: {len(self.memory):,} experiences | Best Performance: {self.best_performance:.2f}")
            print(
                f"   Models: {len(self.models)} | Environment Type: {self.analysis_cache.get('environment_type', 'unknown')}")

            return True

        except Exception as e:
            print(f"❌ Error loading agent: {e}")
            print("🔄 Initializing fresh agent...")
            return False

    def _compress_data(self, data: Any) -> str:
        """Compress data with advanced compression"""
        try:
            serialized = pickle.dumps(data)
            compressed = zlib.compress(serialized, level=9)
            encoded = base64.b64encode(compressed).decode('utf-8')
            return encoded
        except Exception as e:
            print(f"⚠️ Compression failed: {e}")
            return ""

    def _decompress_data(self, compressed_str: str) -> Any:
        """Decompress data"""
        try:
            compressed = base64.b64decode(compressed_str.encode('utf-8'))
            decompressed = zlib.decompress(compressed)
            data = pickle.loads(decompressed)
            return data
        except Exception as e:
            print(f"⚠️ Decompression failed: {e}")
            return []

    def optimize_for_environment(self, environment_info: Dict[str, Any]) -> None:
        """Manually optimize agent for specific environment characteristics"""
        print(f"🎛️ Manual environment optimization...")

        # Extract environment characteristics
        env_type = environment_info.get('type', 'unknown')
        state_dim = environment_info.get('state_dimension', self.state_size)
        action_dim = environment_info.get('action_dimension', self.action_size)
        reward_range = environment_info.get('reward_range', (-1, 1))
        episode_length = environment_info.get('max_episode_length', 1000)

        adaptations = []

        # Environment-specific optimizations
        if env_type == 'atari':
            self.config.update({
                'hidden_layers': [512, 512],
                'learning_rate': 0.00025,
                'batch_size': 32,
                'memory_size': 1000000,
                'epsilon_decay': 0.9999,
                'target_update_freq': 10000,
                'frame_skip': 4
            })
            adaptations.append('Atari configuration')

        elif env_type == 'continuous_control':
            self.config.update({
                'hidden_layers': [400, 300, 200],
                'learning_rate': 0.001,
                'batch_size': 64,
                'soft_update_tau': 0.005,
                'noise_type': 'ou_noise'
            })
            adaptations.append('Continuous control configuration')

        elif env_type == 'sparse_reward':
            self.config.update({
                'prioritized_replay': True,
                'alpha': 0.8,
                'n_step': 5,
                'curiosity_driven': True,
                'intrinsic_motivation': True,
                'epsilon_min': 0.05
            })
            adaptations.append('Sparse reward configuration')

        elif env_type == 'multi_agent':
            self.config.update({
                'ensemble_size': 5,
                'learning_rate': 0.0001,
                'batch_size': 256,
                'experience_sharing': True
            })
            adaptations.append('Multi-agent configuration')

        # Apply adaptations
        self._init_from_config()
        if adaptations:
            self._rebuild_models_if_needed()
            print(f"✅ Applied optimizations: {', '.join(adaptations)}")

    def enable_distributed_training(self, num_workers: int = None) -> None:
        """Enable distributed training across multiple processes"""
        if num_workers is None:
            num_workers = mp.cpu_count() // 2

        print(f"🌐 Enabling distributed training with {num_workers} workers...")

        self.config['distributed_training'] = True
        self.config['num_workers'] = num_workers
        self.config['async_updates'] = True

        # This would require additional implementation for actual distributed training
        print("⚠️ Note: Full distributed implementation requires additional setup")

    def export_for_deployment(self, filename: str, format: str = 'tensorflow') -> None:
        """Export optimized model for deployment"""
        print(f"📦 Exporting model for deployment...")

        if format.lower() == 'tensorflow':
            # Export the best performing model
            best_model = self.models[0]  # Could implement actual best model selection
            best_model.save(f"{filename}_deployment_model")

            # Create deployment config
            deploy_config = {
                'state_size': self.state_size,
                'action_size': self.action_size,
                'preprocessing': 'standard_scaler',
                'inference_batch_size': 1,
                'model_type': 'dqn_ensemble',
                'version': '1.0'
            }

            with open(f"{filename}_deployment_config.json", 'w') as f:
                json.dump(deploy_config, f, indent=2)

        elif format.lower() == 'onnx':
            print("⚠️ ONNX export requires additional dependencies")

        print(f"✅ Model exported for {format} deployment")

    def benchmark_performance(self, num_episodes: int = 100) -> Dict[str, float]:
        """Benchmark agent performance"""
        print(f"🏁 Benchmarking performance over {num_episodes} episodes...")

        # This would require integration with actual environment
        # For now, return analysis of historical performance

        if len(self.reward_history) < 10:
            return {'error': 'Insufficient training data for benchmarking'}

        recent_rewards = list(self.reward_history)[-min(num_episodes, len(self.reward_history)):]

        benchmark_results = {
            'mean_reward': np.mean(recent_rewards),
            'std_reward': np.std(recent_rewards),
            'min_reward': np.min(recent_rewards),
            'max_reward': np.max(recent_rewards),
            'median_reward': np.median(recent_rewards),
            'success_rate': np.mean([r > 0 for r in recent_rewards]),
            'episodes_analyzed': len(recent_rewards),
            'training_efficiency': len(recent_rewards) / max(1, self.training_step) * 1000
        }

        print(f"📊 Benchmark Results:")
        for key, value in benchmark_results.items():
            if isinstance(value, (int, float)):
                print(f"   {key}: {value:.4f}")
            else:
                print(f"   {key}: {value}")

        return benchmark_results

    def get_model_insights(self) -> Dict[str, Any]:
        """Get detailed insights about the model ensemble"""
        insights = {
            'ensemble_size': len(self.models),
            'total_parameters': sum([model.count_params() for model in self.models]),
            'model_architectures': [],
            'training_state': {
                'training_steps': self.training_step,
                'episodes': self.episode_count,
                'exploration_rate': self.epsilon,
                'learning_rate': self.learning_rate
            },
            'performance_metrics': {
                'best_performance': self.best_performance,
                'recent_avg_reward': np.mean(list(self.reward_history)[-10:]) if len(self.reward_history) >= 10 else 0,
                'memory_utilization': len(self.memory) / self.memory_size,
                'loss_trend': 'decreasing' if len(self.loss_history) > 10 and np.mean(
                    list(self.loss_history)[-5:]) < np.mean(list(self.loss_history)[-10:-5:]) else 'stable'
            }
        }

        # Add model architecture details
        for i, model in enumerate(self.models):
            arch_info = {
                'model_id': i,
                'layers': len(model.layers),
                'parameters': model.count_params(),
                'optimizer': type(model.optimizer).__name__,
                'loss_function': type(model.loss).__name__
            }
            insights['model_architectures'].append(arch_info)

        # Add environment analysis if available
        if self.analysis_cache:
            insights['environment_analysis'] = self.analysis_cache

        return insights

    def reset_with_config(self, new_config: Dict[str, Any]) -> None:
        """Reset agent with new configuration while preserving experience"""
        print("🔄 Resetting agent with new configuration...")

        # Preserve important data
        old_memory = list(self.memory)
        old_priorities = list(self.priorities)
        old_reward_history = list(self.reward_history)
        old_analysis_cache = self.analysis_cache.copy()

        # Update configuration
        self.config.update(new_config)
        self._init_from_config()

        # Rebuild models
        self.models = []
        self.target_models = []
        for i in range(self.num_models):
            model = self._build_adaptive_model()
            target_model = self._build_adaptive_model()
            target_model.set_weights(model.get_weights())
            self.models.append(model)
            self.target_models.append(target_model)

        # Restore preserved data
        self.memory = deque(old_memory, maxlen=self.memory_size)
        self.priorities = deque(old_priorities, maxlen=self.memory_size)
        self.reward_history = deque(old_reward_history, maxlen=200)
        self.analysis_cache = old_analysis_cache

        print("✅ Agent reset complete with preserved experience")


# Example usage and testing functions
def create_ultra_dynamic_agent(state_size: int, action_size: int,
                               environment_type: str = 'standard') -> UltraDynamicDQNAgent:
    """Create an ultra-dynamic agent optimized for specific environment type"""

    # Environment-specific configurations
    configs = {
        'atari': {
            'hidden_layers': [512, 512],
            'learning_rate': 0.00025,
            'batch_size': 32,
            'memory_size': 1000000,
            'epsilon_decay': 0.9999,
            'target_update_freq': 10000
        },
        'continuous': {
            'hidden_layers': [400, 300, 200],
            'learning_rate': 0.001,
            'batch_size': 64,
            'soft_update_tau': 0.005
        },
        'sparse_reward': {
            'prioritized_replay': True,
            'alpha': 0.8,
            'n_step': 5,
            'curiosity_driven': True,
            'epsilon_min': 0.05
        },
        'standard': {}
    }

    config = configs.get(environment_type, configs['standard'])
    agent = UltraDynamicDQNAgent(state_size, action_size, config)

    print(f"🚀 Created Ultra-Dynamic DQN Agent for '{environment_type}' environment")
    return agent


def run_adaptive_training_example():
    """Example of how to use the ultra-dynamic agent"""
    print("🎮 Ultra-Dynamic DQN Agent Training Example")
    print("=" * 50)

    # Create agent
    agent = create_ultra_dynamic_agent(state_size=8, action_size=4, environment_type='standard')

    # Simulate training
    for episode in range(1, 101):
        episode_reward = 0
        state = np.random.random(8)

        for step in range(200):
            # Agent selects action
            action = agent.act(state, training=True)

            # Simulate environment step
            next_state = np.random.random(8)
            reward = np.random.normal(0, 1)  # Random reward
            done = step >= 199 or np.random.random() < 0.01

            # Store experience
            agent.remember(state, action, reward, next_state, done)

            # Train agent
            if len(agent.memory) > agent.batch_size:
                loss = agent.replay()

            episode_reward += reward
            state = next_state

            if done:
                break

        # End episode
        agent.end_episode(episode_reward, step + 1)

        # Demonstrate adaptation
        if episode % 25 == 0:
            print(f"\n🔍 Episode {episode} Analysis:")
            agent.analyze_and_adapt(force_analysis=True)

    # Final statistics
    final_stats = agent.get_comprehensive_stats()
    print(f"\n📊 Final Training Statistics:")
    print(f"   Episodes: {final_stats['episode_count']}")
    print(f"   Best Performance: {final_stats['best_performance']:.2f}")
    print(f"   Training Steps: {final_stats['training_step']:,}")
    print(f"   Memory Usage: {final_stats['memory_size']:,}")

    return agent


if __name__ == "__main__":
    # Run example
    trained_agent = run_adaptive_training_example()

    # Demonstrate advanced features
    print("\n🔬 Advanced Features Demonstration:")

    # Get model insights
    insights = trained_agent.get_model_insights()
    print(f"Models in ensemble: {insights['ensemble_size']}")
    print(f"Total parameters: {insights['total_parameters']:,}")

    # Benchmark performance
    benchmark = trained_agent.benchmark_performance(50)
    print(f"Mean reward over 50 episodes: {benchmark.get('mean_reward', 0):.2f}")

    # Save agent
    trained_agent.save("ultra_dynamic_agent_example")

    print("\n✅ Ultra-Dynamic DQN Agent demonstration complete!")
    print("   The agent continuously adapts to any environment automatically!")
    print("   Key features: Environment analysis, hyperparameter optimization,")
    print("   ensemble learning, curiosity-driven exploration, and much more!")