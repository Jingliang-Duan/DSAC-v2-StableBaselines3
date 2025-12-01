import gymnasium as gym
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
import os

# Import DSACT from the new module
from stable_baselines3.dsact.dsact import DSACT

# Use a continuous action space environment, suitable for DSACT
env = make_vec_env("Pendulum-v1", n_envs=1, seed=0)
eval_env = make_vec_env("Pendulum-v1", n_envs=1, seed=42)
log_dir = "./logs_dsact/"
os.makedirs(log_dir, exist_ok=True)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=log_dir,
    log_path=log_dir,
    eval_freq=1000,
    deterministic=True,
    render=False
)

# Instantiate the DSACT model
# DSACT has additional parameters like delay_update and tau_b, we can use the defaults for now
model = DSACT(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=log_dir,
    learning_rate=3e-4,
    buffer_size=1_000_000,
    learning_starts=100,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
    # DSACT specific parameters (using defaults)
    # delay_update=2,
    # tau_b=0.005,
)

model.learn(
    total_timesteps=100_000,
    callback=eval_callback,
    tb_log_name="DSACT_Pendulum"
)

model.save("dsact_pendulum") 