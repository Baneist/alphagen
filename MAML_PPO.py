import random
import math
import time
from alphagen.rl.env.wrapper import AlphaEnv
from copy import deepcopy

import cherry as ch
import gym
import numpy as np
import torch
from cherry.algorithms import a2c, ppo
from cherry.models.robotics import LinearValue
from tqdm import tqdm

import learn2learn as l2l

import torch as th
import torch.nn as nn
from torch import autograd
from torch.distributions.kl import kl_divergence
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.distributions import Normal, Categorical
from torch.utils.tensorboard import SummaryWriter

from stable_baselines3.ppo.policies import MlpPolicy
from alphagen.rl.policy import LSTMSharedNet

EPSILON = 1e-6

def linear_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.zero_()
    return module


class MetaMlpPolicy(nn.Module):
    def __init__(self, input_size, output_size, observation_space, ext_device, d_model,
                 hiddens=None, activation='relu', device='cpu'):
        super(MetaMlpPolicy, self).__init__()
        self.device = device
        if hiddens is None:
            hiddens = [100, 100]
        if activation == 'relu':
            activation = nn.ReLU
        elif activation == 'tanh':
            activation = nn.Tanh
        self.feature_extractor = LSTMSharedNet(observation_space, 1, d_model, .1, ext_device)
        layers = [linear_init(nn.Linear(d_model, hiddens[0])), activation()]
        for i, o in zip(hiddens[:-1], hiddens[1:]):
            layers.append(linear_init(nn.Linear(i, o)))
            layers.append(activation())
        layers.append(linear_init(nn.Linear(hiddens[-1], output_size)))
        self.mlp = nn.Sequential(*layers)
        self.sigma = nn.Parameter(torch.Tensor(output_size))
        self.sigma.data.fill_(math.log(1))

    def density(self, state):
        state = state.to(self.device, non_blocking=True)
        feature = self.feature_extractor(state)
        feature = feature.reshape(feature.shape[0], -1)
        loc = self.mlp(feature)
        scale = torch.exp(torch.clamp(self.sigma, min=math.log(EPSILON)))
        return Normal(loc=loc, scale=scale)

    def log_prob(self, state, action):
        density = self.density(state)
        return density.log_prob(action).mlp(dim=1, keepdim=True)
    def forward(self, state):
        feature = self.feature_extractor(state)
        assert feature.shape[0] == 2 and feature.shape[1] == 1
        feature = feature.reshape(feature.shape[0], -1)
        action = self.mlp(feature)
        return action

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class MAMLPPO():
    def __init__(self, env_pool, env_device,
                 adapt_lr=1e-1, meta_lr=1e-2,
                 adapt_steps=3, ppo_steps=5,
                 adapt_batch_size=1024, meta_batch_size=1024,
                 gamma=0.99, tau=1.0,
                 policy_clip=0.2, value_clip=None,
                 num_workers=2,
                 seed=42,
                 device=None, name="MAMLPPO", tensorboard_log="./logs"):

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            self.device = torch.device("cpu")#cuda
            torch.cuda.manual_seed(seed)
        else:
            self.device = torch.device("cpu")
        if device:
            self.device = torch.device(device)
        print("Running on: " + str(self.device))

        def make_env():
            env = AlphaEnv(pool=env_pool, device=env_device, print_expr=True)
            #env = ch.envs.ActionSpaceScaler(env)
            return env
        env = l2l.gym.AsyncVectorEnv([make_env for _ in range(num_workers)])
        #env = make_env()
        env.seed(seed)
        env.set_task(env.sample_tasks(1)[0])
        self.env = ch.envs.Torch(env)

        self.gamma = gamma
        self.tau = tau
        self.adapt_lr = adapt_lr
        self.meta_lr = meta_lr
        self.adapt_steps = adapt_steps
        self.adapt_batch_size = adapt_batch_size
        self.meta_batch_size = meta_batch_size
        self.policy_clip = policy_clip
        self.value_clip = value_clip
        self.ppo_steps = ppo_steps
        self.global_iteration = 0
        '''
        self.policy = MlpPolicy(observation_space=env.observation_space, action_space=env.action_space,lr_schedule=lambda x:x, features_extractor_class=LSTMSharedNet,
                features_extractor_kwargs=dict(
                    n_layers=2,
                    d_model=64,#128
                    dropout=0.1,
                    device=device,
                ),)'''

        self.policy = MetaMlpPolicy(20, 48, device='cpu', observation_space=self.env.observation_space,
                                    ext_device=self.device, d_model=64)
        self.baseline = LinearValue(20, 1)
        #self.policy = DiagNormalPolicy(self.env.state_size, self.env.action_size, device=self.device)
        #self.baseline = LinearValue(self.env.state_size, self.env.action_size)

        self.policy.to(self.device)
        self.baseline.to(self.device)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), meta_lr)

        if tensorboard_log is not None:
            self.run_name = name + "_" + str(int(time.time()))
            self.writer = SummaryWriter(f"{tensorboard_log}/{self.run_name}")
        else:
            self.writer = None

    def save(self, path="./"):
        torch.save(self.baseline.state_dict(), path + "/baseline.pt")
        torch.save(self.policy.state_dict(), path + "/policy.pt")

    def load(self, path="./"):
        self.baseline.load_state_dict(torch.load(path + "/baseline.pt"))
        self.policy.load_state_dict(torch.load(path + "/policy.pt"))

    def collect_steps(self, policy, n_episodes):
        self.env.reset()
        task = ch.envs.Runner(self.env)
        replay = task.run(lambda x: np.argmax(policy(x).detach(), 1), episodes=n_episodes).to(self.device)
        #lambda x: tuple(policy(x)[0])
        with torch.no_grad():
            next_state_value = self.baseline(replay[-1].next_state)
            mass = policy.density(replay.state())

        log_probs = mass.log_prob(replay.action()).mean(dim=1, keepdim=True)
        values = self.baseline(replay.state())

        advantages = ch.generalized_advantage(self.gamma,
                                              self.tau,
                                              replay.reward(),
                                              replay.done(),
                                              values.detach(),
                                              next_state_value)
        returns = advantages + values.detach()
        advantages = ch.normalize(advantages, epsilon=1e-8)

        for i, sars in enumerate(replay):
            sars.returns = returns[i]
            sars.advantage = advantages[i]
            sars.log_prob = log_probs[i]

        self.baseline.fit(replay.state(), returns)
        return replay

    def maml_a2c_loss(self, train_episodes, learner):
        # Update policy and baseline
        states = train_episodes.state()
        actions = train_episodes.action()
        density = learner.density(states)
        log_probs = density.log_prob(actions).mean(dim=1, keepdim=True)

        advantages = train_episodes.advantage()
        return a2c.policy_loss(log_probs, train_episodes.advantage())

    def fast_adapt(self, clone, train_episodes, first_order=False):
        second_order = not first_order
        loss = self.maml_a2c_loss(train_episodes, clone)
        gradients = autograd.grad(loss,
                                  clone.parameters(),
                                  retain_graph=second_order,
                                  create_graph=second_order)
        return l2l.algorithms.maml.maml_update(clone, self.adapt_lr, gradients)

    def meta_loss(self, iteration_replays, iteration_policies, policy):
        mean_loss = 0.0
        for task_replays, old_policy in tqdm(zip(iteration_replays, iteration_policies),
                                             total=len(iteration_replays),
                                             desc='Surrogate Loss',
                                             leave=False):
            train_replays = task_replays[:-1]
            valid_episodes = task_replays[-1]
            new_policy = l2l.clone_module(policy)

            # Fast Adapt
            for train_episodes in train_replays:
                new_policy = self.fast_adapt(new_policy, train_episodes, first_order=False)

            # Useful values
            states = valid_episodes.state()
            actions = valid_episodes.action()

            # Compute KL
            old_densities = old_policy.density(states)
            new_densities = new_policy.density(states)

            # Compute Surrogate Loss
            advantages = valid_episodes.advantage()
            old_log_probs = old_densities.log_prob(actions).mean(dim=1, keepdim=True).detach()
            new_log_probs = new_densities.log_prob(actions).mean(dim=1, keepdim=True)
            mean_loss += ppo.policy_loss(new_log_probs,
                                         old_log_probs,
                                         advantages,
                                         clip=self.policy_clip)
        mean_loss /= len(iteration_replays)
        return mean_loss

    def meta_optimize(self, iteration_replays, iteration_policies):
        for ppo_epoch in range(self.ppo_steps):
            loss = self.meta_loss(iteration_replays, iteration_policies, self.policy)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.writer is not None:
            self.writer.add_scalar("loss", loss, self.global_iteration)

    def train(self, callback, num_iterations=100):
        for iteration in range(num_iterations):
            self.global_iteration += 1
            iteration_reward = 0.0
            iteration_replays = []
            iteration_policies = []
            iter_loss = 0.0

            for task_config in tqdm(self.env.sample_tasks(self.meta_batch_size), leave=False, desc='Data'):
                clone = deepcopy(self.policy)
                self.env.set_task(task_config)
                task_replay = []

                # Fast Adapt
                for step in range(self.adapt_steps):
                    train_episodes = self.collect_steps(clone, n_episodes=self.adapt_batch_size)
                    self.fast_adapt(clone, train_episodes, first_order=True)
                    task_replay.append(train_episodes)

                # Compute Validation Loss
                valid_episodes = self.collect_steps(clone, n_episodes=self.adapt_batch_size)
                task_replay.append(valid_episodes)
                iteration_reward += valid_episodes.reward().sum().item() / self.adapt_batch_size
                iteration_replays.append(task_replay)
                iteration_policies.append(clone)

            # Print statistics
            print('\nIteration', self.global_iteration)
            adaptation_reward = iteration_reward / self.meta_batch_size
            print('adaptation_reward', adaptation_reward)

            if self.writer is not None:
                self.writer.add_scalar("adaptation_reward", adaptation_reward, self.global_iteration)

            self.meta_optimize(iteration_replays, iteration_policies)
