# test_loop_visual.py
import numpy as np
import cv2
from simple_color_env import SimpleColorEnv

env = SimpleColorEnv()
rolling_window = 50  # average over last 50 episodes
rewards_history = []

target_average = 10  # stop condition

while True:
    obs, _ = env.reset()
    action = env.action_space.sample()
    obs, reward, done, _, info = env.step(action)
    rewards_history.append(reward)

    # show the current frame
    frame = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)  # Pillow uses RGB, OpenCV wants BGR
    cv2.putText(frame, f"Color: {info['target']}, Action: {action}, Reward: {reward:.2f}",
                (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA)
    cv2.imshow("Simple RL Test", frame)
    cv2.waitKey(100)  # ~10 FPS

    # compute rolling average
    if len(rewards_history) >= rolling_window:
        avg_reward = np.mean(rewards_history[-rolling_window:])
    else:
        avg_reward = np.mean(rewards_history)

    print(f"Episode {len(rewards_history)} | Reward: {reward:+.2f} | Rolling Avg: {avg_reward:+.2f}")

    # stop condition
    if avg_reward >= target_average:
        print(f"Target average reward {target_average} reached! Stopping...")
        break

env.close()
cv2.destroyAllWindows()
