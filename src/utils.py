import numpy as np


def set_seed(seed):
    np.random.seed(seed)


def concat_context_and_action_context(context, action_context):
    n_contexts = context.shape[0]
    return np.concatenate([context, np.tile(action_context, (n_contexts, 1))], axis=1)
