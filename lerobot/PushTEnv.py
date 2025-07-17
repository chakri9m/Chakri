# lerobot/envs/PushTEnv.py


import numpy as np
import cv2  # Optional, for human rendering

class PushTEnv:
    metadata = {
        "render_modes": ["rgb_array", "human"],
        "semantics.async": False
    }

    def __init__(self, render=False):
        self.render_mode = 'rgb_array' if render else None
        self.last_frame = self._generate_dummy_frame()
        print(f"PushTEnv created with render={render}")

    def _generate_dummy_frame(self):
        # Use a non-black frame to confirm rendering works
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 127  # Gray image
        cv2.putText(frame, 'PushTEnv Frame', (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return frame

    def reset(self):
        print("Environment reset")
        obs = np.zeros(3, dtype=np.float32)
        self.last_frame = self._generate_dummy_frame()
        if self.render_mode == 'rgb_array':
            frame = self.render(mode='rgb_array')
            return obs, frame
        else:
            return obs

    def step(self, action):
        print(f"Action taken: {action}")
        obs = np.zeros(3, dtype=np.float32)
        reward = 0.0
        done = False
        info = {}
        self.last_frame = self._generate_dummy_frame()
        frame = self.render(mode='rgb_array') if self.render_mode == 'rgb_array' else None
        return obs, reward, done, info, frame

    def render(self, mode='rgb_array'):
        if mode == 'rgb_array':
            return self.last_frame
        elif mode == 'human':
            cv2.imshow("PushTEnv", self.last_frame)
            cv2.waitKey(1)
        else:
            raise NotImplementedError(f"Render mode '{mode}' not supported")

    def close(self):
        print("Environment closed")
        cv2.destroyAllWindows()
