from typing import Any, ClassVar, Optional, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F
from torch.distributions import Normal

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update
from stable_baselines3.dsact.policies import Actor, CnnPolicy, MlpPolicy, MultiInputPolicy, DSACTPolicy, DistributionalCritic

SelfDSACT = TypeVar("SelfDSACT", bound="DSACT")


class DSACT(OffPolicyAlgorithm):
    """
    Distributional Soft Actor-Critic (DSACT)
    This implementation is based on the paper: https://arxiv.org/abs/2310.05858
    and adapts the stable-baselines3 SAC implementation.

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps.
    :param gradient_steps: How many gradient steps to do after each rollout.
    :param action_noise: the action noise type (None by default).
    :param replay_buffer_class: Replay buffer class to use. If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer.
    :param ent_coef: Entropy regularization coefficient.
    :param target_update_interval: update the target network every ``target_network_update_freq`` gradient steps.
    :param target_entropy: target entropy when learning ``ent_coef`` (``ent_coef = 'auto'``)
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
    :param use_sde_at_warmup: Whether to use gSDE instead of uniform sampling during the warm up phase.
    :param delay_update: The number of critic updates before one policy update.
    :param tau_b: The soft update coefficient for the running mean of standard deviations.
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }
    policy: DSACTPolicy
    actor: Actor
    critic: DistributionalCritic
    critic_target: DistributionalCritic

    def __init__(
        self,
        policy: Union[str, type[DSACTPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        n_steps: int = 1,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        delay_update: int = 2,
        tau_b: float = 0.005,
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            n_steps=n_steps,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            supported_action_spaces=(spaces.Box,),
            support_multi_env=True,
        )

        self.target_entropy = target_entropy
        self.log_ent_coef = None  # type: Optional[th.Tensor]
        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.ent_coef_optimizer: Optional[th.optim.Adam] = None

        self.delay_update = delay_update
        self.tau_b = tau_b
        self.mean_std1 = None
        self.mean_std2 = None

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        # Running mean and running var
        self.batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])
        # Target entropy is used when learning the entropy coefficient
        if self.target_entropy == "auto":
            # automatically set target entropy if needed
            self.target_entropy = float(-np.prod(self.env.action_space.shape).astype(np.float32))  # type: ignore
        else:
            # Force conversion
            # this will also throw an error for unexpected string
            self.target_entropy = float(self.target_entropy)

        # The entropy coefficient or entropy can be learned automatically
        # see Automating Entropy Adjustment for Maximum Entropy RL section
        # of https://arxiv.org/abs/1812.05905
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            # Default initial value of ent_coef when learned
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"

            # Note: we optimize the log of the entropy coeff which is slightly different from the paper
            # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
            self.log_ent_coef = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.ent_coef_optimizer = th.optim.Adam([self.log_ent_coef], lr=3e-4) # FIXME: should be a parameter
        else:
            # Force conversion to float
            # this will throw an error if a malformed string (different from 'auto')
            # is passed
            self.ent_coef_tensor = th.tensor(float(self.ent_coef), device=self.device)

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target
        self.actor_target = self.policy.actor_target

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []
        q1_means, q2_means, q1_stds, q2_stds = [], [], [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            # For n-step replay, discount factor is gamma**n_steps (when no early termination)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                assert isinstance(self.target_entropy, float)
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                next_actions, next_log_prob = self.actor_target.action_log_prob(replay_data.next_observations)
                
                # Q-value distribution for next observations
                q_target_dist_1, q_target_dist_2 = self.critic_target(replay_data.next_observations, next_actions)

                # Sample from the Q-distributions

                next_q1_sample,next_q1_mean,_  = self.critic_target.sample(q_target_dist_1)
                next_q2_sample,next_q2_mean,_  = self.critic_target.sample(q_target_dist_2)
                
                next_q_mean = th.min(next_q1_mean, next_q2_mean)
                next_q_sample = th.where(next_q1_mean < next_q2_mean, next_q1_sample, next_q2_sample)
                
                next_soft_q_mean = next_q_mean - ent_coef * next_log_prob.reshape(-1, 1)
                next_soft_q_sample = next_q_sample - ent_coef * next_log_prob.reshape(-1, 1)
                
                target_q_values_mean = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_soft_q_mean
                target_q_values_sample = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_soft_q_sample
            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_dist1, current_q_dist2 = self.critic(replay_data.observations, replay_data.actions)
            q1_mean, q1_std = th.chunk(current_q_dist1, 2, dim=-1)
            q2_mean, q2_std = th.chunk(current_q_dist2, 2, dim=-1)

            # q1_std = F.softplus(q1_std) + 1e-6  # Ensure std is positive
            # q2_std = F.softplus(q2_std) + 1e-6  # Ensure std is positive

            # Update running mean of stds
            if self.mean_std1 is None:
                self.mean_std1 = q1_std.detach().mean()
                self.mean_std2 = q2_std.detach().mean()
            else:
                self.mean_std1 = (1 - self.tau_b) * self.mean_std1 + self.tau_b * q1_std.detach().mean()
                self.mean_std2 = (1 - self.tau_b) * self.mean_std2 + self.tau_b * q2_std.detach().mean()

            # Bounded TD error
            td_bound1 = 3 * self.mean_std1.detach()
            difference1 = th.clamp(target_q_values_sample - q1_mean, -td_bound1, td_bound1)
            target_q_bound1 = q1_mean.detach() + difference1.detach()

            td_bound2 = 3 * self.mean_std2.detach()
            difference2 = th.clamp(target_q_values_sample - q2_mean, -td_bound2, td_bound2)
            target_q_bound2 = q2_mean.detach() + difference2.detach()

            bias = 0.1
            delta = 50
            ratio1 = (th.pow(self.mean_std1, 2) / (th.pow(q1_std.detach(), 2) + bias)).clamp(min=0.1,max=10)
            ratio2 = (th.pow(self.mean_std2, 2) / (th.pow(q2_std.detach(), 2) + bias)).clamp(min=0.1,max=10)



            # q1_loss = th.mean(
            #     ratio1 * F.huber_loss(q1_mean, target_q_bound1, delta=delta, reduction='none') +
            #     q1_std * (q1_std.pow(2) - 2*F.huber_loss(q1_mean.detach(), target_q_values_sample, delta=delta, reduction='none')) / (q1_std.pow(3) + bias)
            # )
            # q2_loss = th.mean(
            #     ratio2 * F.huber_loss(q2_mean, target_q_bound2, delta=delta, reduction='none') +
            #     q2_std * (q2_std.pow(2) - 2*F.huber_loss(q2_mean.detach(), target_q_values_sample, delta=delta, reduction='none')) / (q2_std.pow(3) + bias)
            # )   
            # q1_loss = torch.mean(ratio1 *(huber_loss(q1, target_q1, delta = 50, reduction='none') 
            #                           + q1_std *(q1_std_detach.pow(2) - huber_loss(q1.detach(), target_q1_bound, delta = 50, reduction='none'))/(q1_std_detach +bias)
            #                 ))
            # q2_loss = torch.mean(ratio2 *(huber_loss(q2, target_q2, delta = 50, reduction='none')
            #                           + q2_std *(q2_std_detach.pow(2) - huber_loss(q2.detach(), target_q2_bound, delta = 50, reduction='none'))/(q2_std_detach +bias)
            #                           ))
            q1_loss = th.mean(
                ratio1 * (F.huber_loss(q1_mean, target_q_values_mean, delta=delta, reduction='none') +
                q1_std * (q1_std.detach().pow(2) - F.huber_loss(q1_mean.detach(), target_q_bound1, delta=delta, reduction='none')) / (q1_std.detach() + bias))
            )
            q2_loss = th.mean(
                ratio2 * (F.huber_loss(q2_mean, target_q_values_mean, delta=delta, reduction='none') +
                q2_std * (q2_std.detach().pow(2) - F.huber_loss(q2_mean.detach(), target_q_bound2, delta=delta, reduction='none')) / (q2_std.detach() + bias))
            )
            critic_loss = q1_loss + q2_loss
            critic_losses.append(critic_loss.item())
            q1_means.append(q1_mean.mean().item())
            q2_means.append(q2_mean.mean().item())
            q1_stds.append(q1_std.mean().item())
            q2_stds.append(q2_std.mean().item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # --- Actor and alpha loss ---
            if self._n_updates % self.delay_update == 0:
                q_dists_pi = self.critic(replay_data.observations, actions_pi)
                q1_pi_mean, q2_pi_mean = q_dists_pi[0][:, 0:1], q_dists_pi[1][:, 0:1]
                min_qf_pi = th.min(q1_pi_mean, q2_pi_mean)
                actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
                actor_losses.append(actor_loss.item())

                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()
                

                # Update target networks
                with th.no_grad():
                    polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                    polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
                    polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)
                    
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
        self.logger.record("train/q1_mean", np.mean(q1_means))
        self.logger.record("train/q2_mean", np.mean(q2_means))
        self.logger.record("train/q1_std", np.mean(q1_stds))
        self.logger.record("train/q2_std", np.mean(q2_stds))
        if self.mean_std1 is not None:
            self.logger.record("train/mean_std1", self.mean_std1.item())
            self.logger.record("train/mean_std2", self.mean_std2.item())

    def learn(
        self: SelfDSACT,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "DSACT",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfDSACT:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["actor", "critic", "critic_target", "actor_target"]

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        if self.ent_coef_optimizer is not None:
            saved_pytorch_variables = ["log_ent_coef"]
            state_dicts.append("ent_coef_optimizer")
        else:
            saved_pytorch_variables = ["ent_coef_tensor"]
        return state_dicts, saved_pytorch_variables
