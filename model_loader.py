import os

import gym
import torch
from torchrl.algos.sac import CriticQ

import cifar10.model_loader


def load(dataset, model_name, model_file, data_parallel=False):
    if dataset == 'cifar10':
        net = cifar10.model_loader.load(model_name, model_file, data_parallel)
    return net


def load_rl_model(n_layer, n_unit, env_name):
    env = gym.make(env_name)
    net = CriticQ(state_shape=env.observation_space.shape,
                  action_dim=env.action_space.shape[0],
                  critic_units=[n_unit for _ in range(n_layer)])
    exp_name = f"L{n_layer}-U{n_unit}"
    net.load_state_dict(torch.load(os.path.join("data", env_name, f"SAC_{exp_name}",
                                                f"critic_q_final_{exp_name}.pth")))
    return net


if __name__ == "__main__":
    import numpy as np
    n_layer, n_unit, env_name = 2, 256, "Pendulum-v0"
    env = gym.make(env_name)
    obs = env.reset()

    net_restored = load_rl_model(n_layer, n_unit, env_name)
    net_raw = CriticQ(state_shape=env.observation_space.shape,
                      action_dim=env.action_space.shape[0],
                      critic_units=[n_unit for _ in range(n_layer)])

    input = np.ones(shape=(4,), dtype=np.float32)
    input = torch.from_numpy(input).to(torch.device("cpu"))

    print(net_restored(input))
    print(net_raw(input))
