import sys
import os
# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
from stable_baselines3.dsact import DSACT

# Simple test with fewer timesteps
env = gym.make("Pendulum-v1")

model = DSACT(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=1e-4,  # Smaller learning rate for stability
    buffer_size=10000,   # Smaller buffer
    learning_starts=1000,
    batch_size=64,       # Smaller batch size
    tau=0.01,           # Larger tau for faster target updates
    gamma=0.99,
    delay_update=2,
    tau_b=0.01,
)

print("Starting DSACT training...")
model.learn(total_timesteps=5000,
            tb_log_name="DSACT_Pendulum"
            )  # Much fewer timesteps for quick test
print("Training completed!")

# Test the trained model
obs, _ = env.reset()
for i in range(100):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()

print("DSACT test completed successfully!")
model.save("dsact_simple_test") 