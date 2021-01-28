import os

import joblib
import numpy as np
import torch


def load_trajectories(filenames):
    assert len(filenames) > 0
    paths = []
    for filename in filenames:
        try:
            paths.append(joblib.load(filename))
        except FileNotFoundError:
            print(f"Cannot find {filename}... skip this.")

    for i, path in enumerate(paths):
        # if i == 0:
        #     obses, next_obses, acts, rews, dones = path['obs'], path['next_obs'], path['act'], path['rew'], path[
        #         'done']
        # else:
        #     _obses, _next_obses, _acts, _rews, _dones = path['obs'], path['next_obs'], path['act'], path['rew'], \
        #                                                 path['done']
        #     obses = np.vstack((_obses, obses))
        #     next_obses = np.vstack((_next_obses, next_obses))
        #     acts = np.vstack((_acts, acts))
        #     rews = np.vstack((_rews, rews))
        #     dones = np.vstack((_dones, dones))
        if i == 0:
            target_vals = path['target_val_q']
            n_data = target_vals.shape[0]
            obses, acts = path['obs'][:n_data], path['act'][:n_data]
            # obses, acts = path['obs'][:n_data], path['sampled_act'][:n_data]
        else:
            _target_vals = path['target_val_q']
            n_data = target_vals.shape[0]
            _obses, _acts = path['obs'][:n_data], path['act'][:n_data]
            # _obses, _acts = path['obs'][:n_data], path['sampled_act'][:n_data]
            obses = np.vstack((_obses, obses))
            acts = np.vstack((_acts, acts))
            target_vals = np.vstack((_target_vals, target_vals))
    return obses, acts, target_vals


class PendulumDataset(torch.utils.data.Dataset):

    def __init__(self, exp_name, filenames=None, transform=None):
        self.transform = transform

        self.data = []
        self.label = []

        if filenames is None:
            filenames = [
                os.path.join("data", "Pendulum-v0", f"SAC_{exp_name}", f"all_transition_{exp_name}.pkl"),
                # os.path.join("data", "Pendulum-v0", "SAC_L2-U2048", "all_transition_L2-U2048.pkl"),
                # os.path.join("data", "Pendulum-v0", "SAC_L16-U256", "all_transition_L16-U256.pkl")
            ]
        obses, acts, target_vals = load_trajectories(filenames)
        for obs, act, target_val in zip(obses, acts, target_vals):
            self.data.append((obs, act))
            self.label.append(target_val)

        self.data_num = target_vals.shape[0]

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        out_data = self.data[idx]
        out_label = self.label[idx]

        if self.transform:
            out_data = self.transform(out_data)

        return out_data, out_label


def load_rl_dataset(env_name, exp_name, split_ratio, batch_size):
    if env_name == "Pendulum-v0":
        dataset = PendulumDataset(exp_name)
    else:
        raise ValueError(f"Dataset of {env_name} is not supported.")

    n_samples = len(dataset)
    train_size = int(n_samples * split_ratio)
    val_size = n_samples - train_size

    # train_dataset, val_dataset = torch.utils.data.random_split(
    #     dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    test_loader = None
    # test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


if __name__ == "__main__":
    dataset = PendulumDataset()

    n_samples = len(dataset)
    train_size = int(n_samples * 0.8)
    val_size = n_samples - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    print(len(train_dataset), len(val_dataset))

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=False)
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        # print(batch_idx, inputs[0].shape, inputs[1].shape, targets.shape)
        pass
