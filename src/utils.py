import contextlib
from typing import Optional

import joblib
import numpy as np
from tqdm.auto import tqdm


def set_seed(seed):
    np.random.seed(seed)


def concat_context_and_action_context(context, action_context):
    n_contexts = context.shape[0]
    return np.concatenate([context, np.tile(action_context, (n_contexts, 1))], axis=1)


@contextlib.contextmanager
def tqdm_joblib(total: Optional[int] = None, **kwargs):
    pbar = tqdm(total=total, miniters=1, smoothing=0, **kwargs)

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            pbar.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback

    try:
        yield pbar
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        pbar.close()
