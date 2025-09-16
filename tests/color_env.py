# color_env.py
import numpy as np
import random
from collections import defaultdict
from typing import Tuple, Dict


class ColorRecognitionEnvironment:
    """
    Color recognition environment.
    - Produces a state vector (default 13 features: rgb norm, brightness, ratios, hsv, dominant one-hot)
    - Reward: +1 for correct, -1 for incorrect (stable scaling)
    - Optionally basic (11 colors) or advanced (21 colors)
    """

    def __init__(self, difficulty_level: str = "basic", return_full_state: bool = True, seed: int = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Canonical color palettes (sample RGB base values)
        self.color_definitions = {
            'red': [(255, 0, 0), (220, 20, 60), (178, 34, 34), (255, 99, 71)],
            'green': [(0, 255, 0), (0, 128, 0), (34, 139, 34), (50, 205, 50)],
            'blue': [(0, 0, 255), (0, 0, 139), (30, 144, 255), (70, 130, 180)],
            'yellow': [(255, 255, 0), (255, 215, 0), (255, 255, 224), (240, 230, 140)],
            'orange': [(255, 165, 0), (255, 140, 0), (255, 69, 0), (255, 99, 71)],
            'purple': [(128, 0, 128), (75, 0, 130), (138, 43, 226), (147, 112, 219)],
            'pink': [(255, 192, 203), (255, 20, 147), (255, 105, 180), (219, 112, 147)],
            'brown': [(139, 69, 19), (160, 82, 45), (210, 180, 140), (222, 184, 135)],
            'gray': [(128, 128, 128), (105, 105, 105), (169, 169, 169), (192, 192, 192)],
            'black': [(0, 0, 0), (47, 79, 79), (25, 25, 25), (64, 64, 64)],
            'white': [(255, 255, 255), (255, 250, 250), (248, 248, 255), (245, 245, 245)],
            'cyan': [(0, 255, 255), (0, 206, 209), (72, 209, 204), (175, 238, 238)],
            'magenta': [(255, 0, 255), (199, 21, 133), (218, 112, 214), (221, 160, 221)],
            'lime': [(50, 205, 50), (124, 252, 0), (127, 255, 0), (173, 255, 47)],
            'navy': [(0, 0, 128), (25, 25, 112), (72, 61, 139), (106, 90, 205)],
            'maroon': [(128, 0, 0), (139, 0, 0), (165, 42, 42), (220, 20, 60)],
            'olive': [(128, 128, 0), (85, 107, 47), (107, 142, 35), (154, 205, 50)],
            'teal': [(0, 128, 128), (0, 139, 139), (72, 209, 204), (95, 158, 160)],
            'silver': [(192, 192, 192), (169, 169, 169), (211, 211, 211), (220, 220, 220)],
            'gold': [(255, 215, 0), (255, 193, 37), (218, 165, 32), (184, 134, 11)],
            'indigo': [(75, 0, 130), (72, 61, 139), (123, 104, 238), (138, 43, 226)]
        }

        self.difficulty_level = difficulty_level
        if difficulty_level == "basic":
            # first 11 colors
            items = list(self.color_definitions.items())[:11]
            self.active_colors = dict(items)
        else:
            self.active_colors = self.color_definitions.copy()

        # mapping
        self.color_names = list(self.active_colors.keys())
        self.num_colors = len(self.color_names)
        self.color_to_id = {c: i for i, c in enumerate(self.color_names)}
        self.id_to_color = {i: c for c, i in self.color_to_id.items()}

        # noise / variation
        self.noise_level = 0.08 if difficulty_level == "basic" else 0.16
        self.variation = 0.12 if difficulty_level == "basic" else 0.25

        # state representation choices
        self.return_full_state = return_full_state  # if False, returns just RGB normalized

        # stats
        self.correct_predictions = defaultdict(int)
        self.total_predictions = defaultdict(int)
        self.confusion_matrix = np.zeros((self.num_colors, self.num_colors), dtype=np.int32)

        # current
        self.current_color = None
        self.current_rgb = None

    # --- Public API (gym-like) ---
    def reset(self) -> np.ndarray:
        self.current_color, self.current_rgb = self._generate_random_color()
        return self._rgb_to_state(self.current_rgb)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        # invalid action handling
        if action < 0 or action >= self.num_colors:
            info = {'correct_color': self.current_color, 'predicted_color': 'invalid', 'correct': False, 'rgb': self.current_rgb}
            return self._rgb_to_state(self.current_rgb), -1.0, True, info

        predicted = self.id_to_color[action]
        correct = predicted == self.current_color

        # update stats
        self.total_predictions[self.current_color] += 1
        if correct:
            self.correct_predictions[self.current_color] += 1

        tid = self.color_to_id[self.current_color]
        self.confusion_matrix[tid, action] += 1

        # reward scaled to [-1, +1]
        reward = 1.0 if correct else -1.0

        info = {
            'correct_color': self.current_color,
            'predicted_color': predicted,
            'correct': correct,
            'rgb': self.current_rgb
        }

        # single-step episode
        next_state = self.reset()
        return next_state, reward, True, info

    # --- Helpers ---
    def _generate_random_color(self) -> Tuple[str, Tuple[int, int, int]]:
        color = random.choice(self.color_names)
        base = random.choice(self.active_colors[color])

        # variation per channel
        varied = []
        for ch in base:
            var = random.uniform(-self.variation, self.variation)
            v = int(ch + ch * var)
            v = max(0, min(255, v))
            varied.append(v)

        # add noise
        noisy = []
        for ch in varied:
            noise = random.uniform(-self.noise_level * 255, self.noise_level * 255)
            v = int(ch + noise)
            noisy.append(max(0, min(255, v)))

        return color, tuple(noisy)

    def _rgb_to_state(self, rgb: Tuple[int, int, int]) -> np.ndarray:
        r, g, b = rgb
        r_norm = r / 255.0
        g_norm = g / 255.0
        b_norm = b / 255.0

        if not self.return_full_state:
            return np.array([r_norm, g_norm, b_norm], dtype=np.float32)

        # additional features
        brightness = (r + g + b) / (3 * 255.0)
        total = r + g + b + 1e-6
        r_ratio = r / total
        g_ratio = g / total
        b_ratio = b / total

        # HSV
        rn, gn, bn = r_norm, g_norm, b_norm
        maxv = max(rn, gn, bn)
        minv = min(rn, gn, bn)
        diff = maxv - minv
        if diff == 0:
            hue = 0.0
        elif maxv == rn:
            hue = (60 * ((gn - bn) / diff) + 360) % 360
        elif maxv == gn:
            hue = (60 * ((bn - rn) / diff) + 120) % 360
        else:
            hue = (60 * ((rn - gn) / diff) + 240) % 360
        sat = 0.0 if maxv == 0 else diff / maxv
        val = maxv

        hue_norm = hue / 360.0

        # dominant channel one-hot
        dominant = int(np.argmax([r, g, b]))
        dom_one_hot = [0.0, 0.0, 0.0]
        dom_one_hot[dominant] = 1.0

        state = np.array([
            r_norm, g_norm, b_norm,
            brightness,
            r_ratio, g_ratio, b_ratio,
            hue_norm, sat, val,
            dom_one_hot[0], dom_one_hot[1], dom_one_hot[2]
        ], dtype=np.float32)

        return state

    def get_state_size(self) -> int:
        return 13 if self.return_full_state else 3

    def get_action_size(self) -> int:
        return self.num_colors

    def get_accuracy(self) -> Dict:
        overall_correct = sum(self.correct_predictions.values())
        overall_total = sum(self.total_predictions.values())
        overall = overall_correct / overall_total if overall_total > 0 else 0.0
        per_color = {c: (self.correct_predictions[c] / self.total_predictions[c] if self.total_predictions[c] > 0 else 0.0)
                     for c in self.color_names}
        return {'overall': overall, 'per_color': per_color}

    def print_stats(self):
        acc = self.get_accuracy()
        print(f"Overall accuracy: {acc['overall']:.3f}")
        for c in self.color_names:
            print(f"  {c:12} -> {acc['per_color'][c]:.3f} ({self.total_predictions[c]} samples)")
