import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import src.dataprocessing as dataproc
from src.evaluation import get_embeddings, inferred_variables_evaluation
import src.evaluation as evaluation
from src.flow_loss import ScaledFlowLoss, compute_simple_weighting
from src.utils import EMAMeter, TrainingConfig, HyperParamConfig, EvalConfig, InitConfig, LossConfig, \
    BaselineHyperParamConfig


class TrainerBase:

    def __init__(self, train_graph: dataproc.Graph, val_graph: dataproc.Graph,
                 device=torch.device('cpu'), loss_config: LossConfig = LossConfig()):
        self.device = device
        self.num_nodes = train_graph.num_vertices()

        self.train_graph = train_graph
        self.val_graph = val_graph
        # self._train_edges = train_graph.edges
        # self._val_edges = val_graph.edges

        self.train_flow = torch.from_numpy(train_graph.flow).to(device)
        self.val_flow = torch.from_numpy(val_graph.flow).to(device)

        self.train_source_nodes = torch.from_numpy(train_graph.src_nodes).to(device)
        self.train_target_nodes = torch.from_numpy(train_graph.dst_nodes).to(device)
        self.val_source_nodes = torch.from_numpy(val_graph.src_nodes).to(device)
        self.val_target_nodes = torch.from_numpy(val_graph.dst_nodes).to(device)

        nu = np.median(np.abs(train_graph.flow)) if loss_config.nu == 'auto' else loss_config.nu

        self.scaled_loss = ScaledFlowLoss(use_student_t_loss=loss_config.use_student_t_loss, nu=nu,
                                          use_squared_weighting=loss_config.use_squared_weighting)
        self.train_loss_weighting = compute_simple_weighting(self.train_flow,
                                                             min_flow_weight=loss_config.min_flow_weight,
                                                             max_flow_weight=loss_config.max_flow_weight).to(device)
        self.val_loss_weighting = compute_simple_weighting(self.val_flow, min_flow_weight=loss_config.min_flow_weight,
                                                           max_flow_weight=loss_config.max_flow_weight).to(device)
        self.return_history = False

        self.gt_norm_grad_flow, self.grad_norm, self.gt_norm_harmonic_flow, self.harmonic_norm = \
            dataproc.decompose_flow_normalized(
                source_nodes=np.concatenate((train_graph.src_nodes, val_graph.src_nodes), axis=0),
                target_nodes=np.concatenate((train_graph.dst_nodes, val_graph.dst_nodes), axis=0),
                flow=np.concatenate((train_graph.flow, val_graph.flow), axis=0),
                num_nodes=self.num_nodes
            )
        self.gt_norm_grad_flow = torch.from_numpy(self.gt_norm_grad_flow).to(device=device, dtype=torch.float)
        self.gt_norm_harmonic_flow = torch.from_numpy(self.gt_norm_harmonic_flow).to(device=device, dtype=torch.float)

    def train_val_graph(self):
        edges = np.concatenate((self.train_graph.edges, self.val_graph.edges), axis=0)
        flow = np.concatenate((self.train_graph.flow, self.val_graph.flow), axis=0)
        return dataproc.Graph(num_nodes=self.num_nodes, edges=edges, flow=flow)

    @staticmethod
    def get_embeddings(model, subtract_mean=True):
        return get_embeddings(model, subtract_mean=subtract_mean)

    @staticmethod
    def calc_convergence_crit(new_loss, current_loss):
        if np.isinf(new_loss) or np.isinf(current_loss):
            return np.inf
        return TrainerBase.stop_crit_rel_prev_value(new_loss, current_loss)

    @staticmethod
    def stop_crit_rel_prev_value(new_loss, current_loss):
        return np.abs(new_loss - current_loss) / (new_loss + 1e-4)


class Trainer:

    def __init__(self, base: TrainerBase, model: nn.Module, optimizer: optim.Optimizer,
                 train_config: TrainingConfig = TrainingConfig(), eval_config: EvalConfig = EvalConfig()):
        self.base = base
        self.optimizer = optimizer
        self.model = model
        self.use_lbfgs = train_config.use_lbfgs
        self.return_history = False
        self.use_bootstrap = train_config.use_bootstrap

        self.fast_eval_fun = self.eval_model_val
        self.history_eval_fun = self.eval_model

        self.model_chp_path = eval_config.model_chp_path
        self.train_loss_meter = EMAMeter()
        self.best_val_loss = np.inf
        self.best_val_results = {}
        self.best_val_eval_coeff = eval_config.best_val_eval_coeff
        self.iter_ = 0

    def is_finished(self, current_loss, new_loss, iteration, max_steps, tol):
        is_done = new_loss != float('inf') and self.convergence_criteria(new_loss, current_loss) < tol
        is_done = is_done or iteration >= max_steps
        if is_done:
            pass
        return is_done

    def convergence_criteria(self, new_loss, current_loss):
        return self.base.calc_convergence_crit(new_loss, current_loss)

    def _early_stopping(self, val_loss, iter_):
        if val_loss < self.best_val_eval_coeff * self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_val_results = {key + '*': val for key, val in self.eval_model().items()}
            self.best_val_results.update({"num_iter*": iter_})
            if self.model_chp_path:
                torch.save(self.model.state_dict(), self.model_chp_path)

    def train(self, tol=1e-4, max_steps=20000, verbosity=0, return_history=False):

        self.return_history = return_history
        current_loss = np.inf
        new_loss = np.inf
        fast_eval_res = {}
        history = []

        while not self.is_finished(current_loss=current_loss, new_loss=new_loss, iteration=self.iter_,
                                   max_steps=max_steps, tol=tol):
            self.iter_ += 1

            current_loss = new_loss
            new_loss, _ = self.train_step(current_loss, self.iter_)
            if self.train_loss_meter is not None:
                self.train_loss_meter.add(new_loss)
                new_loss = self.train_loss_meter.mean

            fast_eval_res = self.fast_eval_fun()
            fast_eval_res.update({'criterion': self.convergence_criteria(new_loss, current_loss)})

            self._early_stopping(fast_eval_res['val_loss'], self.iter_)

            history_entry = {'loss': new_loss, 'criterion': self.convergence_criteria(new_loss, current_loss)}
            if self.return_history:
                history_entry.update(self.history_eval_fun())
                history.append(history_entry)

            if verbosity > 1 and self.iter_ % max(max_steps // (verbosity + 1), 1) == 0:
                print(f"Iteration {self.iter_} | loss {new_loss:.5f}")
                print(" | ".join([f"{key} {item:.2f}" for key, item in fast_eval_res.items()]))

        if verbosity > 0:
            print(f"loss {new_loss:.5f} ")
            print(" | ".join([f"{key} {item:.2f}" for key, item in fast_eval_res.items()]))

        results = self.eval_model()
        results.update(self.best_val_results)
        results.update({"num_iter": self.iter_})

        return results, history

    def train_step(self, current_loss=None, iter_=None):
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.forward_pass()
        loss.backward()
        self.optimizer.step()
        return loss.detach().item(), None

    def lbfgs_train_step(self, current_loss=None, iter_=None):
        def closure():
            self.optimizer.zero_grad()
            loss_ = self.forward_pass()
            loss_.backward()
            return loss_

        loss = self.optimizer.step(closure)
        return loss, None

    def forward_pass(self):
        if self.use_bootstrap:
            bootstrap_index = torch.randint(low=0, high=len(self.base.train_flow),
                                            size=(len(self.base.train_flow),), dtype=torch.long,
                                            device=self.base.device)
            output = self.model(source_nodes=self.base.train_source_nodes[bootstrap_index],
                                target_nodes=self.base.train_target_nodes[bootstrap_index])
            loss = self.base.scaled_loss(output, self.base.train_flow[bootstrap_index],
                                         self.base.train_loss_weighting[bootstrap_index])
        else:
            output = self.model(source_nodes=self.base.train_source_nodes, target_nodes=self.base.train_target_nodes)
            loss = self.base.scaled_loss(output, self.base.train_flow, self.base.train_loss_weighting)
        return loss

    def eval_model(self):
        res = {}
        res.update(self.eval_model_val())
        res.update(self.eval_model_train())
        res.update(self.eval_flow_decomposition())
        return res

    def eval_flow_decomposition(self):
        self.model.eval()
        with torch.no_grad():
            source_nodes = torch.cat((self.base.train_source_nodes, self.base.val_source_nodes), dim=0)
            target_nodes = torch.cat((self.base.train_target_nodes, self.base.val_target_nodes), dim=0)
            output = self.model(source_nodes=source_nodes, target_nodes=target_nodes)
            output_norm = torch.norm(output)
            grad_coeff = torch.sum(self.base.gt_norm_grad_flow * output) / output_norm
            harmonic_coeff = torch.sum(self.base.gt_norm_harmonic_flow * output) / output_norm

        results = {'grad_coeff': grad_coeff.item(), 'harmonic_coeff': harmonic_coeff.item(),
                   'flow_output_norm': output_norm.item()}
        return results

    def eval_model_val(self):
        self.model.eval()
        with torch.no_grad():
            output = self.model(source_nodes=self.base.val_source_nodes, target_nodes=self.base.val_target_nodes)
            loss = self.base.scaled_loss(output, self.base.val_flow, self.base.val_loss_weighting)
            output = output.detach().cpu().numpy()
            val_flow = self.base.val_flow.detach().cpu().numpy()

        results = evaluation.calc_flow_prediction_evaluation(output, val_flow, prefix="val")
        results['val_loss'] = loss.item()

        return results

    def eval_model_train(self):
        self.model.eval()
        with torch.no_grad():
            output = self.model(source_nodes=self.base.train_source_nodes, target_nodes=self.base.train_target_nodes)
            loss = self.base.scaled_loss(output, self.base.train_flow, self.base.train_loss_weighting)
            output = output.detach().cpu().numpy()
            train_flow = self.base.train_flow.detach().cpu().numpy()

        results = evaluation.calc_flow_prediction_evaluation(output, train_flow, prefix="train")
        results['train_loss'] = loss.item()
        return results

    def calc_val_loss(self):
        self.model.eval()
        with torch.no_grad():
            output = self.model(source_nodes=self.base.val_source_nodes, target_nodes=self.base.val_target_nodes)
            loss = self.base.scaled_loss(output, self.base.val_flow, self.base.val_loss_weighting)
        return loss.detach().item()


class SheafTrainer(Trainer):

    def __init__(self, base: TrainerBase, model: nn.Module, optimizer: optim.Optimizer,
                 gt_embeddings=None, gt_gates=None,
                 train_config: TrainingConfig = TrainingConfig(), eval_config: EvalConfig = EvalConfig(),
                 hyperpara_config: HyperParamConfig = HyperParamConfig()
                 ):

        super().__init__(base=base, model=model, optimizer=optimizer, train_config=train_config,
                         eval_config=eval_config)

        self.emb_reg = hyperpara_config.embeddings_reg
        self.gates_reg = hyperpara_config.gates_reg

        self.gt_embeddings = gt_embeddings
        self.gt_gates = gt_gates
        self.gt_num_emb_modes = eval_config.gt_num_emb_modes
        self.gt_num_gate_modes = eval_config.gt_num_gate_modes

        self.emb_grad_noise = hyperpara_config.emb_grad_noise
        self.gates_grad_noise = hyperpara_config.gates_grad_noise
        self.use_proportional_noise = hyperpara_config.use_proportional_noise
        self.proportional_noise_cutoff = torch.tensor(hyperpara_config.proportional_noise_cutoff,
                                                      device=self.base.device)

        self.fast_eval_fun = self.eval_model_val
        self.history_eval_fun = self.eval_model

    def forward_pass(self):
        loss = super().forward_pass() + self.embedding_regularization_loss()
        return loss

    def train_step(self, current_loss=None, iter_=None):
        self.model.train()
        self.add_noise_to_embeddings(iter_)
        self.add_noise_to_gates(iter_)
        self.optimizer.zero_grad()
        loss = self.forward_pass()
        loss.backward()
        self.optimizer.step()
        return loss.detach().item(), None

    def add_noise_to_embeddings(self, iter_):
        if self.emb_grad_noise.add_gradient_noise and iter_ % self.emb_grad_noise.noise_interval == 0:
            with torch.no_grad():
                # noise_std = self.emb_grad_noise.std / np.power(1 + 0.002 * iter_, 0.55)
                noise_std = self.emb_grad_noise.std
                noise = noise_std * torch.randn_like(self.model.node_embeddings.weight)
                if self.use_proportional_noise:
                    noise *= torch.maximum(torch.abs(self.model.node_embeddings.weight), self.proportional_noise_cutoff)
                self.model.node_embeddings.weight += noise

    def add_noise_to_gates(self, iter_):
        if self.gates_grad_noise.add_gradient_noise and iter_ % self.gates_grad_noise.noise_interval == 0:
            with torch.no_grad():
                # noise_std = self.gates_grad_noise.std / np.power(1 + 0.002 *
                noise_std = self.gates_grad_noise.std
                noise = noise_std * torch.randn_like(self.model.gates.weight)
                if self.use_proportional_noise:
                    noise *= torch.maximum(torch.abs(self.model.gates.weight), self.proportional_noise_cutoff)
                self.model.gates.weight += noise

    def calc_full_loss(self, model_output):
        return self.base.scaled_loss(model_output, self.base.train_flow,
                                     self.base.train_loss_weighting) + self.embedding_regularization_loss()

    def eval_model(self):
        res = super(SheafTrainer, self).eval_model()
        res.update(self.eval_learned_parameters())
        return res

    def eval_learned_parameters(self):
        embedding_eval = {}
        self.model.eval()
        with torch.no_grad():
            if self.gt_embeddings is not None:
                embedding_eval = inferred_variables_evaluation(self.model.node_embeddings.weight.detach(),
                                                               self.gt_embeddings,
                                                               num_modes=self.gt_num_emb_modes)

            gate_eval = {}
            if self.gt_gates is not None:
                gate_eval = inferred_variables_evaluation(self.model.gates.weight.detach(), self.gt_gates,
                                                          num_modes=self.gt_num_gate_modes)
                gate_eval = {key + "_gates": value for key, value in gate_eval.items()}

            embedding_eval.update(gate_eval)
        return embedding_eval

    def embedding_regularization_loss(self):
        loss = 0.
        if hasattr(self.model, 'node_embeddings') and self.emb_reg.weight > 0.:
            loss += self.emb_reg.weight * self.emb_reg.loss_fun(
                self.model.node_embeddings.weight, torch.zeros_like(self.model.node_embeddings.weight.detach())
            )
        if hasattr(self.model, 'gates') and self.gates_reg.weight > 0.:
            loss += self.gates_reg.weight * self.gates_reg.loss_fun(
                self.model.gates.weight, torch.zeros_like(self.model.gates.weight.detach())
            )
        return loss


class GatesInitTrainer(SheafTrainer):

    def __init__(self, base: TrainerBase, model: nn.Module, optimizer: optim.Optimizer,
                 gt_embeddings=None, gt_gates=None,
                 train_config: TrainingConfig = TrainingConfig(), eval_config: EvalConfig = EvalConfig(),
                 hyperpara_config: HyperParamConfig = HyperParamConfig(), init_config: InitConfig = InitConfig()):
        super().__init__(base=base, model=model, optimizer=optimizer, gt_embeddings=gt_embeddings, gt_gates=gt_gates,
                         train_config=train_config, eval_config=eval_config, hyperpara_config=hyperpara_config)

        self.bce_loss = nn.BCELoss()
        truncated_flow_values = compute_simple_weighting(self.base.train_flow,
                                                         min_flow_weight=init_config.min_gates_auto_weight,
                                                         max_flow_weight=init_config.max_gates_auto_weight)
        flow_normalizer = torch.max(truncated_flow_values)
        self.flow_activation = torch.minimum(torch.abs(self.base.train_flow) / flow_normalizer,
                                             torch.tensor(1, device=self.base.train_flow.device))

    def forward_pass(self):
        output = self.model.gates_forward(source_nodes=self.base.train_source_nodes,
                                          target_nodes=self.base.train_target_nodes)
        loss = self.bce_loss(torch.mean(output, dim=1), self.flow_activation)
        return loss


class BaselineTrainer(Trainer):
    def __init__(self, base: TrainerBase, model: nn.Module, optimizer: optim.Optimizer,
                 train_config: TrainingConfig = TrainingConfig(), eval_config: EvalConfig = EvalConfig(),
                 baseline_hyperpara_config: BaselineHyperParamConfig = BaselineHyperParamConfig()
                 ):

        super().__init__(base=base, model=model, optimizer=optimizer, train_config=train_config,
                         eval_config=eval_config)

        self.reg = baseline_hyperpara_config.reg
        self.grad_noise = baseline_hyperpara_config.grad_noise

        self.use_proportional_noise = baseline_hyperpara_config.use_proportional_noise
        self.proportional_noise_cutoff = torch.tensor(baseline_hyperpara_config.proportional_noise_cutoff,
                                                      device=self.base.device)

    def forward_pass(self):
        loss = super().forward_pass() + self.parameter_regularization_loss()
        return loss

    def train_step(self, current_loss=None, iter_=None):
        self.model.train()
        self.add_noise_parameters(iter_)
        self.optimizer.zero_grad()
        loss = self.forward_pass()
        loss.backward()
        self.optimizer.step()
        return loss.detach().item(), None

    def add_noise_parameters(self, iter_):
        if self.grad_noise.add_gradient_noise and iter_ % self.grad_noise.noise_interval == 0:
            with torch.no_grad():
                # noise_std = self.emb_grad_noise.std / np.power(1 + 0.002 * iter_, 0.55)
                noise_std = self.grad_noise.std
                for parameter in self.model.parameters():
                    noise = noise_std * torch.randn_like(parameter.detach())
                    if self.use_proportional_noise:
                        noise *= torch.maximum(torch.abs(parameter.detach()), self.proportional_noise_cutoff)
                    parameter += noise

    def parameter_regularization_loss(self):

        loss = 0.
        if self.reg.weight > 0.:
            for parameter in self.model.parameters():
                loss += self.reg.weight * self.reg.loss_fun(parameter, torch.zeros_like(parameter.detach()))
        return loss
