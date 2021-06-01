from itertools import permutations

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans
import torch

import src.dataprocessing as dataproc
from src.flow_loss import ScaledFlowLoss, compute_simple_weighting
from src.utils import LossConfig


def get_flow_prediction(test_graph: dataproc.Graph, model):
    test_source_nodes = torch.from_numpy(test_graph.src_nodes)
    test_target_nodes = torch.from_numpy(test_graph.dst_nodes)
    model.eval()

    with torch.no_grad():
        output = model(source_nodes=test_source_nodes, target_nodes=test_target_nodes)
        output = output.detach().cpu().numpy()
    return output


def eval_test_edges(test_graph: dataproc.Graph, device, model, nu, loss_config: LossConfig = LossConfig()):

    test_loss_weighting = compute_simple_weighting(test_graph.flow, min_flow_weight=loss_config.min_flow_weight,
                                                   max_flow_weight=loss_config.max_flow_weight).to(device)
    scaled_loss = ScaledFlowLoss(use_student_t_loss=loss_config.use_student_t_loss, nu=nu,
                                 use_squared_weighting=loss_config.use_squared_weighting)

    def loss_fun(output, gt_flow):
        return scaled_loss(output, gt_flow, test_loss_weighting)

    test_flow = test_graph.flow.to(device)
    test_source_nodes, test_target_nodes = test_graph.src_nodes.to(device), test_graph.dst_nodes.to(device)
    model.to(device)
    res = flow_prediction_evaluation(model, test_source_nodes, test_target_nodes, test_flow, loss_fun=loss_fun)
    return res


def sign_agreement(pred_flow: np.ndarray, gt_flow: np.ndarray):
    return ((np.sign(gt_flow) * np.sign(pred_flow)) > 0).sum() / len(gt_flow)


def old_magnitude_error(pred_flow: np.ndarray, gt_flow: np.ndarray, cutoff: float = 1e-5) -> np.ndarray:
    return (
            np.log10(np.maximum(np.abs(pred_flow), cutoff)) -
            np.log10(np.maximum(np.abs(gt_flow), cutoff))
    )


def magnitude_error(pred_flow: np.ndarray, gt_flow: np.ndarray, cutoff: float = 1e-6):
    abs_error = np.abs(pred_flow - gt_flow)
    mag_error = np.log10(np.maximum(abs_error, cutoff) / np.maximum(np.abs(gt_flow), cutoff))
    return mag_error


def flow_prediction_evaluation(model, source_nodes, target_nodes, gt_flow, loss_fun):
    model.eval()
    with torch.no_grad():
        output = model(source_nodes=source_nodes, target_nodes=target_nodes)
        loss = loss_fun(output, gt_flow).item()
        output = output.detach().cpu().numpy()
        gt_flow = gt_flow.detach().cpu().numpy()

    res = {'loss': loss}
    res.update(calc_flow_prediction_evaluation(output, gt_flow))
    return res


def calc_flow_prediction_evaluation(model_output, gt_flow, prefix: str = None):
    sign_agr = sign_agreement(model_output, gt_flow)
    mag_error = magnitude_error(model_output, gt_flow)
    within_1_mag = np.sum(mag_error < 1) / len(mag_error)
    within_scale = np.sum(mag_error < 0) / len(mag_error)
    within_scale_neg_1 = np.sum(mag_error < -1) / len(mag_error)
    mean_mag_error = np.mean(mag_error)
    median_new_mag_error = np.median(mag_error)
    rmse = np.sqrt(np.mean(np.power(model_output - gt_flow, 2)))
    MAE = np.mean(np.abs(model_output - gt_flow))
    MeAE = np.median(np.abs(model_output - gt_flow))
    res = {'median_mag_error': median_new_mag_error, 'within_scale_neg_1': within_scale_neg_1,
           'sign_agr': sign_agr,
           'within_1_mag': within_1_mag, 'within_scale': within_scale, 'mean_mag_error': mean_mag_error,
           'rmse': rmse, 'mae': MAE, 'MeAE': MeAE}
    if prefix is not None:
        res = {prefix + "_" + k: v for k, v in res.items()}
    return res


def get_embeddings(model, subtract_mean=True):
    with torch.no_grad():
        embeddings = model.node_embeddings.weight.detach()
        if subtract_mean:
            embeddings = subtract_embedding_mean(embeddings)
    return embeddings


def subtract_embedding_mean(embeddings):
    return embeddings - torch.mean(embeddings, dim=0)


def model_embedding_error(model, gt_embeddings, subtract_mean=True):
    model_embeddings = get_embeddings(model, subtract_mean=subtract_mean)
    return embedding_error(model_embeddings, gt_embeddings)


def embedding_error(model_embeddings, gt_embeddings, subtract_mean=True):
    with torch.no_grad():
        if subtract_mean:
            model_embeddings = subtract_embedding_mean(model_embeddings)
            gt_embeddings = subtract_embedding_mean(gt_embeddings)

        error = torch.sqrt(torch.pow(gt_embeddings - model_embeddings, 2).sum() / np.prod(gt_embeddings.shape))
    return error.item()


def inferred_variables_evaluation(model_embeddings, gt_embeddings, num_modes=1):
    with torch.no_grad():
        model_embeddings = subtract_embedding_mean(model_embeddings).cpu().numpy()
        gt_embeddings = subtract_embedding_mean(gt_embeddings).cpu().numpy()

    if model_embeddings.shape == gt_embeddings.shape:
        residuals = np.power(gt_embeddings - model_embeddings, 2).sum(axis=0)
        distance = np.sqrt(residuals.sum() / np.prod(gt_embeddings.shape))
    else:
        residuals = np.nan
        distance = np.nan

    model_std = np.std(model_embeddings)
    gt_std = np.std(gt_embeddings)

    if model_std > 0.0 and gt_std > 0.0:

        if model_embeddings.shape == gt_embeddings.shape:
            R2 = 1 - (residuals.sum() / np.sum(np.power(gt_embeddings, 2)))
        else:
            R2 = np.nan

        model_embeddings_plus = np.concatenate((model_embeddings, np.ones((model_embeddings.shape[0], 1))), axis=1)
        A, _, _, _ = np.linalg.lstsq(model_embeddings_plus, gt_embeddings, rcond=None)
        fit_residuals = np.power(gt_embeddings - model_embeddings_plus @ A, 2).sum(axis=0)
        distance_after_fit = np.sqrt(fit_residuals.sum() / np.prod(gt_embeddings.shape))
        R2_after_fit = 1 - (fit_residuals.sum() / np.sum(np.power(gt_embeddings, 2)))

        pearson_outlier_corr, spearman_outlier_corr, outlier_agreement_score75 = outlier_correlation(model_embeddings,
                                                                                                     gt_embeddings)
        unimodal_kl_div = unimodal_evaluation(model_embeddings, gt_embeddings)
        multimodal_score = multimodal_evaluation(model_embeddings, gt_embeddings, num_modes=num_modes)

        return {'model_std': model_std, 'gt_std': gt_std,
                'error': distance, 'R2': R2,
                'error_affine': distance_after_fit, 'R2_affine': R2_after_fit,
                'pearson_corr': pearson_outlier_corr, 'spearman_corr': spearman_outlier_corr,
                'outlier_agreement_score75': outlier_agreement_score75,
                'unimodal_kl': unimodal_kl_div, 'multimodal_score': multimodal_score
                }

    else:
        return {'model_std': model_std, 'gt_std': gt_std,
                'error': distance, 'R2': np.nan,
                'error_affine': np.nan, 'R2_affine': np.nan,
                'pearson_corr': np.nan, 'spearman_corr': np.nan,
                'outlier_agreement_score75': np.nan,
                'unimodal_kl': np.nan, 'multimodal_score': np.nan
                }


def baseline_features_eval(baseline_features: torch.Tensor, gt_embeddings: torch.Tensor, num_modes=1):
    with torch.no_grad():
        gt_embeddings = subtract_embedding_mean(gt_embeddings).cpu().numpy()
        baseline_features = baseline_features.cpu().numpy()

    model_std = np.std(baseline_features)
    gt_std = np.std(gt_embeddings)

    if model_std > 0.0 and gt_std > 0.0:

        pearson_outlier_corr, spearman_outlier_corr, outlier_agreement_score75 = outlier_correlation(baseline_features,
                                                                                                     gt_embeddings)
        multimodal_score = multimodal_evaluation(baseline_features, gt_embeddings, num_modes=num_modes)

        baseline_features_plus = np.concatenate((baseline_features, np.ones((baseline_features.shape[0], 1))), axis=1)
        A, _, _, _ = np.linalg.lstsq(baseline_features_plus, gt_embeddings, rcond=None)
        fit_residuals = np.power(gt_embeddings - baseline_features_plus @ A, 2).sum(axis=0)
        distance_after_fit = np.sqrt(fit_residuals.sum() / np.prod(gt_embeddings.shape))
        R2_after_fit = 1 - (fit_residuals.sum() / np.sum(np.power(gt_embeddings, 2)))
        return {'model_std': model_std, 'gt_std': gt_std,
                'error_affine': distance_after_fit, 'R2_affine': R2_after_fit,
                'pearson_corr': pearson_outlier_corr, 'spearman_corr': spearman_outlier_corr,
                'outlier_agreement_score75': outlier_agreement_score75,
                'multimodal_score': multimodal_score
                }

    else:
        return {'model_std': model_std, 'gt_std': gt_std,
                'error_affine': np.nan, 'R2_affine': np.nan,
                'pearson_corr': np.nan, 'spearman_corr': np.nan,
                'outlier_agreement_score75': np.nan,
                'multimodal_score': np.nan
                }


def outlier_correlation(model_embeddings: np.ndarray, gt_embeddings: np.ndarray):
    lof = LocalOutlierFactor(n_neighbors=min(gt_embeddings.shape[0], 20))
    _ = lof.fit_predict(gt_embeddings)
    scores_gt = -lof.negative_outlier_factor_

    lof = LocalOutlierFactor(n_neighbors=min(gt_embeddings.shape[0], 20))
    _ = lof.fit_predict(model_embeddings)
    scores_model = -lof.negative_outlier_factor_

    outlier_score_denominator = (scores_gt > np.quantile(scores_gt, 0.75)).sum()
    if outlier_score_denominator < 1e-8:
        outlier_agreement_score75 = np.nan
    else:
        outlier_agreement_score75 = np.sum((scores_model > np.quantile(scores_model, 0.75)) *
                                           (scores_gt > np.quantile(scores_gt, 0.75))) / outlier_score_denominator

    pearson_lof_corr = pearsonr(scores_gt, scores_model)[0]
    spearman_lof_corr = spearmanr(scores_gt, scores_model)[0]
    return pearson_lof_corr, spearman_lof_corr, outlier_agreement_score75


def unimodal_evaluation(model_embeddings: np.ndarray, gt_embeddings: np.ndarray):
    model_cov = np.cov(model_embeddings.T)
    gt_cov = np.cov(gt_embeddings.T)
    score = symmetric_kl_for_covariance(model_cov, gt_cov)
    return score


def multimodal_evaluation(model_embeddings: np.ndarray, gt_embeddings: np.ndarray, num_modes):
    model_kmean_labels = KMeans(n_clusters=num_modes).fit(model_embeddings).labels_
    gt_kmean_labels = KMeans(n_clusters=num_modes).fit(gt_embeddings).labels_
    num_nodes = gt_kmean_labels.shape[0]

    scores = []
    # Check all permutations of the labels for the best fit
    for p in permutations(range(num_modes)):
        score = np.sum(gt_kmean_labels == np.asarray(p)[model_kmean_labels]) / num_nodes
        scores.append(score)
    score = max(scores)
    return score


def symmetric_kl_for_covariance(cov1: np.ndarray, cov2: np.ndarray):
    dim = cov1.shape[0]
    if not np.linalg.det(cov1) > 0. or not np.linalg.det(cov2) > 0.:
        return np.inf
    kl_1 = np.trace(np.linalg.pinv(cov1) @ cov2)
    kl_2 = np.trace(np.linalg.pinv(cov2) @ cov1)
    sym_kl = 0.5 * (kl_1 + kl_2 - 2 * dim)
    return sym_kl


def radius_evaluation(model_embeddings, gt_embeddings):
    embeddings_radius = np.sqrt(np.sum(np.power(model_embeddings, 2), axis=1))
    gt_radius = np.sqrt(np.sum(np.power(gt_embeddings, 2), axis=1))
    pearson_corr = pearsonr(embeddings_radius, gt_radius)[0]
    spearman_corr = spearmanr(embeddings_radius, gt_radius)[0]

    outlier_score50 = np.sum((embeddings_radius > np.median(embeddings_radius)) *
                             (gt_radius > np.median(gt_radius))) / (gt_radius > np.median(gt_radius)).sum()

    outlier_score85 = np.sum((embeddings_radius > np.quantile(embeddings_radius, 0.85)) *
                             (gt_radius > np.quantile(gt_radius, 0.85))) / (
                              gt_radius > np.quantile(gt_radius, 0.85)).sum()
    return pearson_corr, spearman_corr, outlier_score50, outlier_score85
