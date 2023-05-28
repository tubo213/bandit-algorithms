import numpy as np


class AbstractPolicy:
    def select_action(self, context):
        raise NotImplementedError

    def update_params(self, action, reward, context):
        pass


class RandomPolicy(AbstractPolicy):
    def __init__(self, n_actions):
        self.n_actions = n_actions

    def select_action(self, context):
        return np.random.choice(self.n_actions, size=context.shape[0])


class EpsilonGreedyPolicy(AbstractPolicy):
    def __init__(self, n_actions, epsilon=0.01):
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.rewars = np.zeros(n_actions)
        self.cnts = np.zeros(n_actions)
        self.e = np.zeros(n_actions)

    def select_action(self, context):
        n = context.shape[0]
        p = np.random.uniform(size=n)
        action = np.where(p < self.epsilon, np.random.choice(self.n_actions, size=n), np.argmax(self.e))

        return action

    def update_params(self, action, reward, _):
        n = action.shape[0]
        for i in range(n):
            self.cnts[action[i]] += 1
            self.rewars[action[i]] += reward[i]
            self.e[action[i]] = self.rewars[action[i]] / self.cnts[action[i]]


class SoftMaxPolicy(AbstractPolicy):
    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.rewars = np.zeros(n_actions)
        self.cnts = np.zeros(n_actions)
        self.e = np.ones(n_actions)

    def select_action(self, context):
        prob = self.softmax(self.e)
        action = np.random.choice(self.n_actions, size=context.shape[0], p=prob)

        return action

    def update_params(self, action, reward, _):
        n = action.shape[0]
        for i in range(n):
            self.cnts[action[i]] += 1
            self.rewars[action[i]] += reward[i]
            self.e[action[i]] = self.rewars[action[i]] / self.cnts[action[i]]

    @staticmethod
    def softmax(x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)


class UCBPolicy(AbstractPolicy):
    def __init__(self, n_actions, alpha=1):
        self.n_actions = n_actions
        self.alpha = alpha
        self.rewars = np.zeros(n_actions)
        self.cnts = np.zeros(n_actions)
        self.e = np.zeros(n_actions)

    def select_action(self, context):
        n = context.shape[0]
        t = np.sum(self.cnts)
        ucb_scores = []
        for i in range(self.n_actions):
            mu = self.e[i]
            std = np.sqrt(2 * np.log(t) / self.cnts[i])
            ucb_scores.append(mu + self.alpha * std)

        return np.full(n, np.argmax(ucb_scores))

    def update_params(self, action, reward, _):
        n = action.shape[0]
        for i in range(n):
            self.cnts[action[i]] += 1
            self.rewars[action[i]] += reward[i]
            self.e[action[i]] = self.rewars[action[i]] / self.cnts[action[i]]


class LinUCBPolicy(AbstractPolicy):
    def __init__(self, n_actions, dim_context, alpha=1, action_context=None):
        self.n_actions = n_actions
        self.dim_context = dim_context
        self.action_context = action_context if action_context is not None else np.identity(n_actions)
        self.alpha = alpha

        dim = dim_context + self.action_context.shape[1]
        self.A_inv = np.identity(dim)
        self.b = np.zeros((dim, 1))

    def select_action(self, context):
        theta_hat = (self.A_inv @ self.b).flatten()
        ucb_list = []
        for action in range(self.n_actions):
            x = self.concat_context_and_action_context(context, self.action_context[[action]])
            e = (x * theta_hat).sum(axis=1)
            std = np.apply_along_axis(lambda x_i: np.sqrt(x_i.T @ self.A_inv @ x_i), 1, x)
            p = e + self.alpha * std
            ucb_list.append(p)

        ucb = np.stack(ucb_list, axis=1)
        return np.argmax(ucb, axis=1)

    def update_params(self, action, reward, context):
        n = action.shape[0]
        x = np.concatenate([context, self.action_context[action]], axis=1)

        for i in range(n):
            x_i = x[i].reshape(-1, 1)
            self.A_inv -= (self.A_inv @ x_i @ x_i.T @ self.A_inv) / (1 + x_i.T @ self.A_inv @ x_i)
            self.b += reward[i] * x_i.reshape(-1, 1)

    @staticmethod
    def concat_context_and_action_context(context, action_context):
        n_contexts = context.shape[0]
        return np.concatenate([context, np.tile(action_context, (n_contexts, 1))], axis=1)
