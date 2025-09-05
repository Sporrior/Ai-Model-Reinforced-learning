import gym
from gym import spaces
import numpy as np
from PIL import Image, ImageDraw

class SimpleColorEnv(gym.Env):
    """
    Observation: 84x84 RGB image (red or green square)
    Actions: 0=LEFT, 1=RIGHT
    Rule: LEFT for RED, RIGHT for GREEN
    Reward: +1 if correct, -1 if wrong
    Episode: single step
    """
    
    def __init__(self):
        super(SimpleColorEnv, self).__init__()
        self.observation_space = spaces.Box(0, 255, (84, 84, 3), dtype=np.uint8)
        self.action_space = spaces.Discrete(2)
        self.target_color = None

    def _generate_image(self, color):
        img = Image.new("RGB", (84, 84), (200, 200, 200))
        draw = ImageDraw.Draw(img)
        draw.rectangle([20, 20, 64, 64], fill=color)
        return np.array(img, dtype=np.uint8)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.target_color = np.random.choice(["red", "green"])
        color = (255, 0, 0) if self.target_color == "red" else (0, 255, 0)
        return self._generate_image(color), {}

    def step(self, action):
        correct_action = 0 if self.target_color == "red" else 1
        reward = 1.0 if action == correct_action else -1.0
        done = True  # one-step episode
        info = {"target": self.target_color}
        obs, _ = self.reset()
        return obs, reward, done, False, info
