import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
import os

env = make_vec_env("CartPole-v1", n_envs=4, seed=0)
eval_env = make_vec_env("CartPole-v1", n_envs=1, seed=42)
log_dir = "./logs/"
os.makedirs(log_dir, exist_ok=True)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=log_dir,
    log_path=log_dir,
    eval_freq=1000,
    deterministic=True,
    render=False
)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=log_dir,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
)

model.learn(
    total_timesteps=50_000,
    callback=eval_callback,
    tb_log_name="PPO_CartPole"
)

model.save("ppo_cartpole")
