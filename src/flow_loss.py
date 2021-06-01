import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ScaledFlowLoss(nn.Module):

    def __init__(self, use_student_t_loss=False, nu=1., use_squared_weighting=False):
        super().__init__()
        self.use_cauchy_likelihood = use_student_t_loss
        self.nu = nu
        self.use_squared_weighting = use_squared_weighting

    def forward(self, pred_flow: torch.FloatTensor, gt_flow: torch.FloatTensor, weighting=None):
        if weighting is None:
            weighting = torch.ones_like(gt_flow)
        if self.use_squared_weighting:
            weighted_err = torch.pow((gt_flow - pred_flow) / weighting, 2)
        else:
            weighted_err = torch.pow(gt_flow - pred_flow, 2) / weighting
        if self.use_cauchy_likelihood:
            return torch.mean(0.5 * (self.nu + 1) * torch.log(1 + weighted_err / self.nu))
        else:
            return torch.mean(weighted_err)


class CauchyLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        err = torch.sum(torch.pow((x - y), 2), dim=-1)
        return torch.mean(torch.log(1 + err), dim=-1)


class SoftplusLoss(nn.Module):

    def __init__(self, beta=1.5):
        super().__init__()
        self.beta = beta

    def forward(self, x, y):
        return torch.mean(F.softplus(-x, beta=self.beta))


def compute_simple_weighting(flow_tensor: torch.FloatTensor, min_flow_weight=1e-6, max_flow_weight=1e6):
    return torch.minimum(
        torch.maximum(torch.abs(flow_tensor), torch.tensor(min_flow_weight, device=flow_tensor.device)),
        torch.tensor(max_flow_weight, device=flow_tensor.device)
    )


def compute_simple_weighting_np(flow_tensor: np.ndarray, min_flow_weight=1e-6, max_flow_weight=1e6):
    return np.minimum(
        np.maximum(np.abs(flow_tensor), min_flow_weight),
        max_flow_weight
    )


def np_relative_abs_error(pred_flow, gt_flow, rel_weights):
    err = np.abs((gt_flow - pred_flow)) / rel_weights
    return err


def np_relative_squared_error(pred_flow, gt_flow, rel_weights):
    err = np.power((gt_flow - pred_flow), 2) / rel_weights
    return err
