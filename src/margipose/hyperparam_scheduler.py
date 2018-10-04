# Implementation of the 1cycle policy of training: https://arxiv.org/abs/1803.09820

import numpy as np


def make_1cycle(optimizer, max_iters, lr_max, momentum=0):
    lr_min = lr_max * 1e-1
    lr_nihil = lr_min * 1e-3
    t3 = max_iters
    t2 = 0.9 * t3
    t1 = t2 / 2
    m_max = momentum
    m_min = min(m_max, 0.85)
    return HyperparameterScheduler(
        optimizer,
        ts=[1, t1, t2, t3],
        hyperparam_milestones={
            'lr': [lr_min, lr_max, lr_min, lr_nihil],
            'momentum': [m_max, m_min, m_max, m_max],
        }
    )


class HyperparameterScheduler():
    def __init__(self, optimizer, ts, hyperparam_milestones):
        for k, v in hyperparam_milestones.items():
            assert len(v) == len(ts),\
                'expected {} milestones for hyperparameter "{}"'.format(len(ts), k)
            for param_group in optimizer.param_groups:
                assert k in param_group,\
                    '"{}" is not an optimizer hyperparameter'.format(k)
        self.optimizer = optimizer
        self.ts = np.array(ts)
        self.hyperparam_milestones = {k: np.array(v) for k, v in hyperparam_milestones.items()}
        self.batch_count = 0

    def batch_step(self):
        self.batch_count += 1
        for hyperparam_name, milestones in self.hyperparam_milestones.items():
            value = float(np.interp(self.batch_count, self.ts, milestones))
            for param_group in self.optimizer.param_groups:
                param_group[hyperparam_name] = value
