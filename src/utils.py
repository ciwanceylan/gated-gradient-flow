from __future__ import annotations
from typing import *
from collections import namedtuple
import sys
import json
import warnings

import numpy as np
import torch
from scipy import stats as stats
from torch import nn as nn

GradientNoise = namedtuple("GradientNoise", "add_gradient_noise noise_interval std")
SheafFlowReg = namedtuple("SheafFlowReg", "loss_fun weight")


class Config(object):

    def __str__(self):
        return str(self.__dict__)

    def to_dict(self):
        json_dict = {'_class': self.__class__.__name__}
        content = vars(self)
        for key, val in content.items():
            if isinstance(val, GradientNoise):
                json_dict[key] = {'add_gradient_noise': val.add_gradient_noise,
                                  'noise_interval': val.noise_interval, 'std': val.std}
            elif isinstance(val, SheafFlowReg):
                json_dict[key] = {'loss_fun': val.loss_fun.__class__.__name__, 'weight': val.weight}
            else:
                json_dict[key] = val

        return json_dict


class EvalConfig(Config):

    def __init__(self):
        super(EvalConfig, self).__init__()
        self.gt_num_emb_modes: int = 1
        self.gt_num_gate_modes: int = 1
        self.model_chp_path = None
        self.best_val_eval_coeff = 0.97


class InitConfig(Config):
    def __init__(self):
        super(InitConfig, self).__init__()
        self.embedding_init = 'auto'
        self.auto_noise_std = 0.05
        self.gates_init = 'zeros'
        self.tol: float = 1e-4
        self.max_steps: int = 500
        self.beta = 1.
        self.embedding_dim = 2
        self.min_gates_auto_weight = 1e-8
        self.max_gates_auto_weight = 1e2
        self.use_simple_gates = False


class TrainingConfig(Config):

    def __init__(self):
        super(TrainingConfig, self).__init__()
        self.max_steps: int = 18000
        self.tol: float = 1e-4
        self.max_gates_steps: int = 300
        self.max_emb_steps: int = 300
        self.substep_tol: float = 1e-4
        self.lr: float = 0.01
        self.gates_lr: float = 0.01
        self.max_lbfgs_step: int = 20
        self.use_lbfgs: bool = False
        self.use_bootstrap = False
        # self.use_student_t_loss = True
        # self.nu = 1


class HyperParamConfig(Config):

    def __init__(self):
        super(HyperParamConfig, self).__init__()
        self.gates_grad_noise: GradientNoise = GradientNoise(False, np.inf, 0.)
        self.emb_grad_noise: GradientNoise = GradientNoise(False, np.inf, 0.)
        self.embeddings_reg: SheafFlowReg = SheafFlowReg(nn.L1Loss(), 0.0)
        self.gates_reg: SheafFlowReg = SheafFlowReg(nn.L1Loss(), 0.0)
        self.use_proportional_noise = True
        self.proportional_noise_cutoff = 1


class BaselineHyperParamConfig(Config):

    def __init__(self):
        super(BaselineHyperParamConfig, self).__init__()
        self.grad_noise: GradientNoise = GradientNoise(False, np.inf, 0.)
        self.reg: SheafFlowReg = SheafFlowReg(nn.L1Loss(), 0.0)
        self.use_proportional_noise = True
        self.proportional_noise_cutoff = 1


class LossConfig(Config):

    def __init__(self):
        super(LossConfig, self).__init__()
        self.use_student_t_loss = True
        self.nu = 1.  # type: Union[float, str]
        self.use_squared_weighting = False
        self.min_flow_weight = 1e-1
        self.max_flow_weight = np.inf


def configs2dict(**configs):
    dicts = {}
    for name, config in configs.items():
        # json_dict = {'_class': config.__class__.__name__}
        # content = vars(config)
        # for key, val in content.items():
        #     if isinstance(val, GradientNoise):
        #         json_dict[key] = {'add_gradient_noise': val.add_gradient_noise,
        #                           'noise_interval': val.noise_interval, 'std': val.std}
        #     elif isinstance(val, SheafFlowReg):
        #         json_dict[key] = {'loss_fun': val.loss_fun.__class__.__name__, 'weight': val.weight}
        #     else:
        #         json_dict[key] = val
        dicts[name] = config.to_dict()
    return dicts


def dicts2configs(dicts):
    configs = {}
    for name, dict_ in dicts.items():
        config = getattr(sys.modules[__name__], dict_['_class'])()
        content = vars(config)
        for key, val in content.items():
            if key == '_class':
                continue
            if key not in dict_:
                warnings.warn(f"key {key} not found when loading configs. Using default value {content[key]}")
            elif isinstance(content[key], GradientNoise):
                content[key] = GradientNoise(**dict_[key])
            elif isinstance(val, SheafFlowReg):
                loss_fun = getattr(nn, dict_[key]['loss_fun'])()
                weight = dict_[key]['weight']
                content[key] = SheafFlowReg(loss_fun=loss_fun, weight=weight)
            else:
                content[key] = dict_[key]
        configs[name] = config
    return configs


def load_configs(path):
    with open(path, 'r') as fp:
        config_dicts = json.load(fp)
    configs = dicts2configs(config_dicts)
    return configs


def save_configs(path, **configs):
    configs_ = {}
    for name, config in configs.items():
        configs_[name] = config.to_dict()
    with open(path, 'w') as fp:
        json.dump(configs_, fp, indent=2)


class EMAMeter:
    def __init__(self, alpha=0.9):
        self.n_total = 0  # The total number of samples seen
        self.mean = np.nan  # The exponential moving average
        self.var = np.nan  # The moving variance of batches
        self.std = np.nan  # The moving standard deviation
        self.val = 0  # The current value is not part of reset()
        self.alpha = alpha  # The EMA decay factor.
        # 1 -> mean is batch mean, 0.5 -> mean over all samples,
        # 0 -> mean is the value of the first batch

    def reset(self):
        self.n_total = 0  # The total number of samples seen
        self.mean = np.nan  # The exponential moving average
        self.var = np.nan  # The moving variance of batches
        self.std = np.nan  # The moving standard deviation

    def add(self, value, n=1, *args):
        self.val = value
        # Check if it is the first iteration
        if np.isnan(self.mean):
            self.mean = value

        # Check if it is the first iteration
        if np.isnan(self.var):
            self.var = 0.0

        # Update variance and
        delta = value - self.mean
        self.var = (1 - self.alpha) * (self.var + self.alpha * delta ** 2)
        self.std = np.sqrt(self.var)

        # Update the mean
        self.mean = (self.alpha * n * value +
                     (1 - self.alpha) * self.n_total * self.mean) / \
                    ((1 - self.alpha) * self.n_total + self.alpha * n)

        # Update the total number of samples
        self.n_total += n

    def value(self):
        # type: () -> Dict[str, Union[int, float]]
        return dict(mean=self.mean, std=self.std)

    def __str__(self):
        return "{:.4} +/- {:.4}".format(self.mean, self.std)


def sample_student_t(num_samples, dim, df, scale=1.):
    r = stats.t.rvs(df=df, size=(num_samples, dim)).astype(np.float32)
    return scale * torch.from_numpy(r)
