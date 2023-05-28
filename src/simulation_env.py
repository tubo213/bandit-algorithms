import numpy as np
import torch
import torch.nn as nn

from src.utils import set_seed


class BanditEnv:
    def __init__(self, n_actions, dim_context, action_context=None, random_state=11111):
        self.n_actions = n_actions
        self.dim_context = dim_context
        self.action_context = action_context if action_context is not None else np.identity(n_actions)
        self.dim_action_context = self.action_context.shape[1]
        self.model = MLP(dim_context + self.dim_action_context)
        set_seed(random_state)

    def get_context(self, n):
        return np.random.uniform(-1, 1, size=(n, self.dim_context))

    def get_reward(self, context):
        n = context.shape[0]
        e = np.random.normal(0, 0.5, size=n)
        rewards = []
        for i in range(self.n_actions):
            x_i = concat_context_and_action_context(context, self.action_context[[i]])
            x_i = torch.from_numpy(x_i).float()
            reward = self.model(x_i).detach().numpy().flatten() + e
            rewards.append(reward)
        rewards = np.stack(rewards, axis=1)

        return rewards


class MLP(nn.Module):
    def __init__(self, dim, dim_hidden=16, n_hidden=2, dim_output=1):
        super().__init__()
        module_list = [nn.Linear(dim, dim_hidden), nn.SELU()]

        if n_hidden > 2:
            for _ in range(n_hidden - 2):
                module_list.append(nn.Linear(dim_hidden, dim_hidden))
                module_list.append(nn.SELU())

        module_list.append(nn.Linear(dim_hidden, dim_output))
        self.fc = nn.ModuleList(module_list)

    def forward(self, x):
        for layer in self.fc:
            x = layer(x)
        return x


def concat_context_and_action_context(context, action_context):
    n_contexts = context.shape[0]
    return np.concatenate([context, np.tile(action_context, (n_contexts, 1))], axis=1)
