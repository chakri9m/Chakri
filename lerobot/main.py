import threading
import tqdm

# --- Fix for AttributeError: type object 'tqdm' has no attribute '_lock' ---
# This must come before any libraries that might use tqdm (like moviepy or proglog)
if not hasattr(tqdm.tqdm, "_lock"):
    tqdm.tqdm._lock = threading.RLock()

# Standard and project imports
import numpy as np
import torch
from gym.wrappers import RecordVideo
from lerobot.envs.PushTEnv import PushTEnv
from lerobot.diffusion_pusht import DiffusionPushT

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pretrained diffusion policy
    model = DiffusionPushT.from_pretrained("lerobot/diffusion_pusht")
    model.eval()

    # Create environment with rendering enabled
    env = PushTEnv(render=True)

    # Wrap the environment with video recording
    env = RecordVideo(env, video_folder="./videos", name_prefix="run")

    # Reset environment
    result = env.reset()
    if isinstance(result, tuple):
        obs = result[0]
    else:
        obs = result
    done = False
    step = 0

    while not done and step < 100:
        # Example action (replace with actual model output if needed)
        action = [0.0, 0.0, 0.0]

        # Take a step in the environment
        result = env.step(action)

        if len(result) == 5:
            obs, reward, done, info, _ = result
        else:
            obs, reward, done, info = result

        print(f"Step {step}, Reward: {reward}")
        step += 1

    env.close()
    print("Simulation finished.")

if __name__ == "__main__":
    main()
