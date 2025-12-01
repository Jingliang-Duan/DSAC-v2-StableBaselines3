import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
import os

# 使用连续动作空间的环境，SAC更适合这类环境
env = make_vec_env("Pendulum-v1", n_envs=1, seed=0)  # SAC通常用单环境训练
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

model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=log_dir,
    learning_rate=3e-4,
    buffer_size=1_000_000,  # SAC特有的回放缓冲区大小
    learning_starts=100,    # 开始学习前的随机步数
    batch_size=256,         # SAC通常使用更大的批次
    tau=0.005,             # 软更新系数
    gamma=0.99,            # 折扣因子
    train_freq=1,          # 每步都训练
    gradient_steps=1,      # 每次训练的梯度步数
)

model.learn(
    total_timesteps=100_000,  # SAC通常需要更多步数
    callback=eval_callback,
    tb_log_name="SAC_Pendulum"
)

model.save("sac_pendulum") 