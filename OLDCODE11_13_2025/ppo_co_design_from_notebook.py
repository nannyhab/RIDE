
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PPO co-design trainer extracted from the RL_PPO_susp_CCD_demo.ipynb structure.

This script provides:
- DynamicSystemEnv (quarter-car suspension) as in the notebook
- MLP-based actor & critic (NeuralNetwork, NeuralNetworkValue)
- PPO_num class with familiar methods: put_data, make_batch, calc_advantage, train_net_CCD, pi, v
- Design-parameter conditioning: appends normalized (ks, cs) to observations
- Optional CSV pretraining if Data_susp_CCD2.csv / Data_value_susp_CCD2.csv are present
- Domain randomization over (ks_norm, cs_norm) or fixed design configuration
"""
import os
import time
import argparse
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import carla

from dynamic_sys_env import DynamicSystemEnv
from carla_env import CarlaPointToPointEnv

class NeuralNetwork(nn.Module):
    def __init__(self, in_dim: int, hidden=(128, 128), out_dim: int = 1):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.Tanh()]
            last = h
        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers)
        self.log_std = nn.Parameter(torch.zeros(out_dim))
    def forward(self, x):
        mu = self.net(x)
        std = torch.exp(self.log_std).expand_as(mu)
        return mu, std

class NeuralNetworkValue(nn.Module):
    def __init__(self, in_dim: int, hidden=(128, 128)):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.Tanh()]
            last = h
        layers += [nn.Linear(last, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x).squeeze(-1)

@dataclass
class PPOConfig:
    total_steps: int = 50_000
    update_every: int = 2048
    epochs: int = 10
    minibatch_size: int = 256
    gamma: float = 0.99
    lam: float = 0.95
    clip_ratio: float = 0.2
    pi_lr: float = 3e-4
    vf_lr: float = 1e-3
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    max_grad_norm: float = 0.5
    device: str = "cpu"
    mode: str = "DOMAIN_RANDOMIZED"
    ks_norm: float = 0.5
    cs_norm: float = 0.5
    ks_norm_min: float = 0.2
    ks_norm_max: float = 0.8
    cs_norm_min: float = 0.2
    cs_norm_max: float = 0.8

class PPO_num:
    def __init__(self, env: DynamicSystemEnv, cfg: PPOConfig):
        self.env = env
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.obs_dim = int(env.observation_space.shape[0]) + 2  # +2 design vars
        self.act_dim = int(env.action_space.shape[0])
        self.actor = NeuralNetwork(self.obs_dim, out_dim=self.act_dim).to(self.device)
        self.critic = NeuralNetworkValue(self.obs_dim).to(self.device)
        self.pi_opt = optim.Adam(self.actor.parameters(), lr=cfg.pi_lr)
        self.vf_opt = optim.Adam(self.critic.parameters(), lr=cfg.vf_lr)
        N = cfg.update_every
        self.buf_obs = torch.zeros((N, self.obs_dim), dtype=torch.float32, device=self.device)
        self.buf_act = torch.zeros((N, self.act_dim), dtype=torch.float32, device=self.device)
        self.buf_logp = torch.zeros(N, dtype=torch.float32, device=self.device)
        self.buf_rew = torch.zeros(N, dtype=torch.float32, device=self.device)
        self.buf_done = torch.zeros(N, dtype=torch.float32, device=self.device)
        self.buf_val = torch.zeros(N, dtype=torch.float32, device=self.device)
        self.ptr = 0
        # gear ratio and tire friction as learnable parameters
        self.design_params = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32, device=self.device))
        self.pi_opt = optim.Adam(
            list(self.actor.parameters()) + [self.design_params],
            lr=cfg.pi_lr
)

    @torch.no_grad()
    def pi(self, obs_aug: torch.Tensor):
        mu, std = self.actor(obs_aug)
        dist = Normal(mu, std)
        act = dist.sample()
        logp = dist.log_prob(act).sum(-1)
        return act, logp, mu

    @torch.no_grad()
    def v(self, obs_aug: torch.Tensor):
        return self.critic(obs_aug)

    def put_data(self, obs_aug, act, logp, rew, done, val):
        self.buf_obs[self.ptr] = obs_aug
        self.buf_act[self.ptr] = act
        self.buf_logp[self.ptr] = logp
        self.buf_rew[self.ptr] = rew
        self.buf_done[self.ptr] = done
        self.buf_val[self.ptr] = val
        self.ptr += 1

    def make_batch(self):
        obs = self.buf_obs[:self.ptr]
        act = self.buf_act[:self.ptr]
        logp = self.buf_logp[:self.ptr]
        rew = self.buf_rew[:self.ptr]
        done = self.buf_done[:self.ptr]
        val = self.buf_val[:self.ptr]
        self.ptr = 0
        return obs, act, logp, rew, done, val

    def calc_advantage(self, rews, vals, dones, last_val):
        rews = rews.cpu().numpy()
        vals = vals.cpu().numpy()
        dones = dones.cpu().numpy()
        advs = np.zeros_like(rews)
        lastgaelam = 0.0
        for t in reversed(range(len(rews))):
            nonterminal = 1.0 - dones[t]
            next_val = last_val if t == len(rews) - 1 else vals[t + 1]
            delta = rews[t] + self.cfg.gamma * next_val * nonterminal - vals[t]
            lastgaelam = delta + self.cfg.gamma * self.cfg.lam * nonterminal * lastgaelam
            advs[t] = lastgaelam
        returns = advs + vals
        advs_t = torch.as_tensor(advs, dtype=torch.float32, device=self.device)
        rets_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        advs_t = (advs_t - advs_t.mean()) / (advs_t.std() + 1e-8)
        return advs_t, rets_t

    def _sample_design(self) -> Tuple[float, float]:
        # if self.cfg.mode == "DOMAIN_RANDOMIZED":
        #     ks = float(np.random.uniform(self.cfg.ks_norm_min, self.cfg.ks_norm_max))
        #     cs = float(np.random.uniform(self.cfg.cs_norm_min, self.cfg.cs_norm_max))
        # else:
        #     ks, cs = float(self.cfg.ks_norm), float(self.cfg.cs_norm)
        # return ks, cs
        with torch.no_grad():
            ks_norm = torch.clamp(self.design_params[0], 0, 1).item()
            cs_norm = torch.clamp(self.design_params[1], 0, 1).item()
        return ks_norm, cs_norm
    

    def _augment(self, obs: np.ndarray, ks_norm: float, cs_norm: float) -> torch.Tensor:
        obs_aug = np.concatenate([obs.astype(np.float32), np.array([ks_norm, cs_norm], dtype=np.float32)], axis=-1)
        return torch.as_tensor(obs_aug, dtype=torch.float32, device=self.device)

    def train_net_CCD(self):
        steps = 0
        while steps < self.cfg.total_steps:
            ks_norm, cs_norm = self._sample_design()
            obs = self.env.reset(ks_norm, cs_norm)
            done = False
            ep_steps = 0
            while not done and ep_steps < self.env.max_steps:
                obs_aug_t = self._augment(obs, ks_norm, cs_norm)
                with torch.no_grad():
                    val_t = self.v(obs_aug_t)
                    act_t, logp_t, _ = self.pi(obs_aug_t)
                act_np = act_t.cpu().numpy()
                next_obs, rew, done, _ = self.env.step(act_np, ks_norm, cs_norm)
                self.put_data(obs_aug_t, torch.as_tensor(act_np, device=self.device, dtype=torch.float32),
                              logp_t, torch.tensor(rew, device=self.device),
                              torch.tensor(float(done), device=self.device), val_t)
                obs = next_obs
                ep_steps += 1
                steps += 1
                if self.ptr >= self.cfg.update_every:
                    last_obs_aug = self._augment(obs, ks_norm, cs_norm)
                    with torch.no_grad():
                        last_val = float(self.v(last_obs_aug).cpu())
                    self._ppo_update(last_val)
                if steps >= self.cfg.total_steps:
                    break

    def _ppo_update(self, last_val: float):
        obs, act, old_logp, rew, done, val = self.make_batch()
        adv, ret = self.calc_advantage(rew, val, done, last_val)
        n = obs.shape[0]
        idx = np.arange(n)
        for _ in range(self.cfg.epochs):
            np.random.shuffle(idx)
            for start in range(0, n, self.cfg.minibatch_size):
                mb = idx[start:start + self.cfg.minibatch_size]
                obs_b = obs[mb]
                act_b = act[mb]
                old_logp_b = old_logp[mb]
                adv_b = adv[mb]
                ret_b = ret[mb]
                mu, std = self.actor(obs_b)
                dist = Normal(mu, std)
                logp = dist.log_prob(act_b).sum(-1)
                ratio = torch.exp(logp - old_logp_b)
                unclipped = ratio * adv_b
                clipped = torch.clamp(ratio, 1.0 - self.cfg.clip_ratio, 1.0 + self.cfg.clip_ratio) * adv_b
                pi_loss = -torch.min(unclipped, clipped).mean()
                v = self.critic(obs_b)
                vf_loss = 0.5 * (v - ret_b).pow(2).mean()
                ent = dist.entropy().sum(-1).mean()
                loss = pi_loss + self.cfg.vf_coef * vf_loss - self.cfg.ent_coef * ent
                self.pi_opt.zero_grad(set_to_none=True)
                loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
                self.pi_opt.step()
                self.vf_opt.zero_grad(set_to_none=True)
                vf_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
                self.vf_opt.step()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--mode", type=str, default="DOMAIN_RANDOMIZED", choices=["DOMAIN_RANDOMIZED", "FIXED_DESIGN"])
    parser.add_argument("--ks", type=float, default=0.5, help="normalized [0,1]")
    parser.add_argument("--cs", type=float, default=0.5, help="normalized [0,1]")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()


    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)

    env = CarlaPointToPointEnv(
        client=client,
        town="Town03",
        dt=0.05,
        sync_mode=True,
        vehicle_filter="vehicle.tesla.model3",
    )

    #env = DynamicSystemEnv()
    env = CarlaPointToPointEnv()

    # obs = env.reset(gear_norm=0.5, mu_norm=0.5)
    # for t in range(1000):
    #     action = policy(obs)  # e.g., np.array([steer, throttle, brake])
    #     obs, reward, done, info = env.step(action, gear_norm=0.5, mu_norm=0.5)
    #     if done:
    #         break
    # env.close()

    cfg = PPOConfig(total_steps=args.steps, mode=args.mode, ks_norm=args.ks, cs_norm=args.cs, device=args.device)
    agent = PPO_num(env, cfg)
    t0 = time.time()
    agent.train_net_CCD()
    print(f"Training finished in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
