import copy
import logging
import math
from collections import defaultdict

import numpy as np
import torch
import torch.functional as F
import torch.nn as nn
import wandb
from pytorch_metric_learning.losses import NTXentLoss
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from utils.GS_Plugin import GSPlugin
from utils.MinNormSolver import MinNormSolver


def pick_bias_infuser(agent):
    method = agent.config.model.args.get("bias_infusion", {}).get("method", False)
    if method == "OGM":
        bi = Bias_Infusion_OGM(agent)
    elif method == "MLB":
        bi = Bias_Infusion_MLB(agent)
    elif method == "MLB_3d":
        bi = Bias_Infusion_MLB_3d(agent)
    elif method == "MLB_3d_Reg":
        bi = Bias_Infusion_MLB_3d_Reg(agent)
    elif method == "MLB_Reg":
        bi = Bias_Infusion_MLB_Reg(agent)
    elif method == "AGM":
        bi = Bias_Infusion_AGM(agent)
    elif method == "MMPareto":
        bi = Bias_Infusion_MMPareto(agent)
    elif method == "MMPareto_3D":
        bi = Bias_Infusion_MMPareto_3d(agent)
    elif method == "MLA":
        bi = Bias_Infusion_MLA(agent)
    elif method == "DnR":
        bi = Bias_Infusion_DnR(agent)
    elif method == "ReconBoost":
        bi = Bias_Infusion_ReconBoost(agent)
    elif method == "MCR":
        bi = Bias_Infusion_MCR(agent)
    elif method == "MCR3D_Reg":
        bi = Bias_Infusion_MCR3D_Reg(agent)
    elif method == "MCR_Reg":
        bi = Bias_Infusion_MCR_Reg(agent)
    elif method == "MCR3D":
        bi = Bias_Infusion_MCR3D(agent)
    elif method == "AGM_3mod":
        bi = Bias_Infusion_AGM_3mod(agent)
    elif method == "AGM_3mod_reg":
        bi = Bias_Infusion_AGM_3mod_reg(agent)
    elif method == "AGM_reg":
        bi = Bias_Infusion_AGM_reg(agent)
    elif method == "MSLR":
        bi = Bias_Infusion_MSLR(agent)
    else:
        bi = General_Bias_Infusion(agent)
    return bi


class General_Bias_Infusion:
    def __init__(self, agent):
        self.agent = agent

        super(General_Bias_Infusion, self).__init__()

    def before_backward(self, total, output_losses, **kwargs):
        return total, output_losses, False

    def on_backward_end(self, **kwargs):
        return

    def on_epoch_begin(self, **kwargs):
        pass

    def plot_bias(self, **kwargs):
        pass


class Bias_Infusion_OGM(General_Bias_Infusion):
    def __init__(self, agent):
        super(Bias_Infusion_OGM, self).__init__(agent)
        logging.info("Bias Infusion OGM is being employed")
        self._initialize_logs_n_utils()

    def _initialize_logs_n_utils(self):
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.agent.logs["ratio_logs"] = {
            "ratio_mod0": [],
            "ratio_mod1": [],
            "coeff_mod0": [],
            "coeff_mod1": [],
        }

    def on_backward_end(self, label, out_mod0, out_mod1):

        if len(label.shape) > 1:
            label = label.flatten()

        score_0 = sum([self.softmax(out_mod0)[i][label[i]] for i in range(out_mod0.size(0))]).detach()
        score_1 = sum([self.softmax(out_mod1)[i][label[i]] for i in range(out_mod1.size(0))]).detach()

        ratio_0 = score_0 / score_1
        ratio_1 = 1 / ratio_0

        if ratio_0 > 1:
            coeff_0 = 1 - self.tanh(self.agent.config.model.args.bias_infusion.alpha * self.relu(ratio_0)).cpu().numpy()
            coeff_1 = 1
        else:
            coeff_1 = 1 - self.tanh(self.agent.config.model.args.bias_infusion.alpha * self.relu(ratio_1)).cpu().numpy()
            coeff_0 = 1

        self.agent.logs["ratio_logs"]["ratio_mod1"].append(ratio_1.cpu().numpy())
        self.agent.logs["ratio_logs"]["ratio_mod0"].append(ratio_0.cpu().numpy())
        self.agent.logs["ratio_logs"]["coeff_mod0"].append(coeff_0)
        self.agent.logs["ratio_logs"]["coeff_mod1"].append(coeff_1)

        wandb_output = {
            "ratio": {
                "ratio_mod1": ratio_1.cpu().numpy(),
                "ratio_mod0": ratio_0.cpu().numpy(),
                "coeff_mod0": coeff_0,
                "coeff_mod1": coeff_1,
            }
        }

        wandb.log(wandb_output)

        self._equalize_gradients(coeff_0=coeff_0, coeff_1=coeff_1)

    def _equalize_gradients(self, coeff_0=1, coeff_1=1):
        if not self.agent.config.model.args.bias_infusion.use:
            return

        if self.agent.config.model.args.bias_infusion.starting_epoch <= self.agent.logs["current_epoch"] <= self.agent.config.model.args.bias_infusion.ending_epoch:

            for name, parms in self.agent.model.named_parameters():
                if parms.grad is None:
                    continue
                if "mod0" in name or "enc_0" in name:
                    if self.agent.config.model.args.bias_infusion.method == "OGM_GE":  # bug fixed
                        parms.grad = parms.grad * coeff_0 + torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
                    elif self.agent.config.model.args.bias_infusion.method == "OGM":
                        parms.grad *= coeff_0
                    elif self.agent.config.model.args.bias_infusion.method == "Acc":
                        parms.grad *= coeff_0
                if "mod1" in name or "enc_1" in name:
                    if self.agent.config.model.args.bias_infusion.method == "OGM_GE":  # bug fixed
                        parms.grad = parms.grad * coeff_1 + torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
                    elif self.agent.config.model.args.bias_infusion.method == "OGM":
                        parms.grad *= coeff_1
                    elif self.agent.config.model.args.bias_infusion.method == "Acc":
                        parms.grad *= coeff_1


class Bias_Infusion_MMPareto(General_Bias_Infusion):
    def __init__(self, agent):
        super(Bias_Infusion_MMPareto, self).__init__(agent)
        logging.info("Bias Infusion MMPareto is being employed")
        self._initialize_logs_n_utils()

    def _initialize_logs_n_utils(self):
        pass

    def before_backward(self, total, output_losses, **kwargs):
        if not self.agent.config.model.args.bias_infusion.use:
            return

        if self.agent.config.model.args.bias_infusion.starting_epoch <= self.agent.logs["current_epoch"] <= self.agent.config.model.args.bias_infusion.ending_epoch:

            loss_mm = output_losses["ce_loss_combined"]
            loss_a = output_losses["ce_loss_c"]
            loss_v = output_losses["ce_loss_g"]

            losses = [loss_mm, loss_a, loss_v]
            all_loss = ["both", "audio", "visual"]

            grads_visual = defaultdict(dict)
            grads_audio = defaultdict(dict)

            for idx, loss_type in enumerate(all_loss):
                loss = losses[idx]
                loss.backward(retain_graph=True)
                if loss_type == "visual":
                    for name, parms in self.agent.model.named_parameters():
                        if parms.grad is None:
                            continue
                        if ("mod1" in name or "fc_1" in name or "enc_1" in name) and name in grads_visual["both"]:
                            grads_visual[loss_type][name] = parms.grad.data.clone()
                    grads_visual[loss_type]["concat"] = torch.cat(
                        [
                            grads_visual[loss_type][name].flatten()
                            for name, parms in self.agent.model.named_parameters()
                            if ("mod1" in name or "enc_1" in name) and parms.grad is not None and name in grads_visual["both"]
                        ]
                    )
                elif loss_type == "audio":
                    for name, parms in self.agent.model.named_parameters():
                        if parms.grad is None:
                            continue
                        if ("mod0" in name or "enc_0" in name) and name in grads_audio["both"]:
                            grads_audio[loss_type][name] = parms.grad.data.clone()
                    grads_audio[loss_type]["concat"] = torch.cat(
                        [
                            grads_audio[loss_type][name].flatten()
                            for name, parms in self.agent.model.named_parameters()
                            if ("mod0" in name or "enc_0" in name) and parms.grad is not None and name in grads_audio["both"]
                        ]
                    )
                else:
                    for name, parms in self.agent.model.named_parameters():
                        if parms.grad is None:
                            continue
                        if "mod0" in name or "enc_0" in name:
                            grads_audio[loss_type][name] = parms.grad.data.clone()
                        if "mod1" in name or "enc_1" in name:
                            grads_visual[loss_type][name] = parms.grad.data.clone()
                    grads_visual[loss_type]["concat"] = torch.cat(
                        [
                            grads_visual[loss_type][name].flatten()
                            for name, parms in self.agent.model.named_parameters()
                            if ("mod1" in name or "enc_1" in name) and parms.grad is not None
                        ]
                    )
                    grads_audio[loss_type]["concat"] = torch.cat(
                        [
                            grads_audio[loss_type][name].flatten()
                            for name, parms in self.agent.model.named_parameters()
                            if ("mod0" in name or "enc_0" in name) and parms.grad is not None
                        ]
                    )
                self.agent.optimizer.zero_grad()

            audio_k, visual_k = self._compute_ratio(grads_audio, grads_visual)
            total = loss_mm + loss_a + loss_v
            total.backward()
            gamma = self.agent.config.model.args.bias_infusion.alpha
            self._equalize_gradients(grads_audio, grads_visual, audio_k, visual_k, gamma)

            self.agent.optimizer.step()

            wandb_output = {"ratio": {"audio_k": audio_k, "visual_k": visual_k}}
            wandb.log(wandb_output)

            return total, output_losses, True

    def _compute_ratio(self, grads_audio, grads_visual):
        this_cos_audio = F.cosine_similarity(grads_audio["both"]["concat"], grads_audio["audio"]["concat"], dim=0)
        this_cos_visual = F.cosine_similarity(grads_visual["both"]["concat"], grads_visual["visual"]["concat"], dim=0)

        audio_task = ["both", "audio"]
        visual_task = ["both", "visual"]

        audio_k = [0, 0]
        visual_k = [0, 0]

        if this_cos_audio > 0:
            audio_k[0] = 0.5
            audio_k[1] = 0.5
        else:
            audio_k, min_norm = MinNormSolver.find_min_norm_element([list(grads_audio[t].values()) for t in audio_task])
        if this_cos_visual > 0:
            visual_k[0] = 0.5
            visual_k[1] = 0.5
        else:
            visual_k, min_norm = MinNormSolver.find_min_norm_element([list(grads_visual[t].values()) for t in visual_task])
        return audio_k, visual_k

    def _equalize_gradients(self, grads_audio, grads_visual, audio_k, visual_k, gamma):
        for name, param in self.agent.model.named_parameters():
            if param.grad is not None:
                if ("mod0" in name or "fc_0" in name or "enc_0" in name) and name in grads_audio["both"]:
                    three_norm = torch.norm(param.grad.data.clone())
                    new_grad = 2 * audio_k[0] * grads_audio["both"][name] + 2 * audio_k[1] * grads_audio["audio"][name]
                    new_norm = torch.norm(new_grad)
                    diff = three_norm / new_norm
                    if diff > 1:
                        param.grad = diff * new_grad * gamma
                    else:
                        param.grad = new_grad * gamma

                if ("mod1" in name or "fc_1" in name or "enc_1" in name) and name in grads_visual["both"]:
                    three_norm = torch.norm(param.grad.data.clone())
                    new_grad = 2 * visual_k[0] * grads_visual["both"][name] + 2 * visual_k[1] * grads_visual["visual"][name]
                    new_norm = torch.norm(new_grad)
                    diff = three_norm / new_norm
                    if diff > 1:
                        param.grad = diff * new_grad * gamma
                    else:
                        param.grad = new_grad * gamma


class Bias_Infusion_MMPareto_3d(General_Bias_Infusion):
    def __init__(self, agent):
        super(Bias_Infusion_MMPareto_3d, self).__init__(agent)
        logging.info("Bias Infusion MMPareto 3D is being employed")
        self._initialize_logs_n_utils()

    def _initialize_logs_n_utils(self):
        pass

    def before_backward(self, total, output_losses, **kwargs):
        if not self.agent.config.model.args.bias_infusion.use:
            return

        if self.agent.config.model.args.bias_infusion.starting_epoch <= self.agent.logs["current_epoch"] <= self.agent.config.model.args.bias_infusion.ending_epoch:

            loss_mm = output_losses["ce_loss_combined"]
            loss_a = output_losses["ce_loss_c"]
            loss_v = output_losses["ce_loss_g"]
            loss_t = output_losses["ce_loss_f"]

            losses = [loss_mm, loss_a, loss_v, loss_t]
            all_loss = ["both", "audio", "visual", "text"]

            grads_visual = defaultdict(dict)
            grads_audio = defaultdict(dict)
            grads_text = defaultdict(dict)

            for idx, loss_type in enumerate(all_loss):
                loss = losses[idx]
                loss.backward(retain_graph=True)
                if loss_type == "visual":
                    for name, parms in self.agent.model.named_parameters():
                        if parms.grad is None:
                            continue
                        if ("mod1" in name or "enc_1" in name or "fc_lin_1" in name) and name in grads_visual["both"]:
                            grads_visual[loss_type][name] = parms.grad.data.clone()
                    grads_visual[loss_type]["concat"] = torch.cat(
                        [
                            grads_visual[loss_type][name].flatten()
                            for name, parms in self.agent.model.named_parameters()
                            if ("mod1" in name or "enc_1" in name or "fc_lin_1" in name) and parms.grad is not None and name in grads_visual["both"]
                        ]
                    )
                elif loss_type == "audio":
                    for name, parms in self.agent.model.named_parameters():
                        if parms.grad is None:
                            continue
                        if ("mod0" in name or "enc_0" in name or "fc_lin_0" in name) and name in grads_audio["both"]:
                            grads_audio[loss_type][name] = parms.grad.data.clone()
                    grads_audio[loss_type]["concat"] = torch.cat(
                        [
                            grads_audio[loss_type][name].flatten()
                            for name, parms in self.agent.model.named_parameters()
                            if ("mod0" in name or "enc_0" in name or "fc_lin_0" in name) and parms.grad is not None and name in grads_audio["both"]
                        ]
                    )
                elif loss_type == "text":
                    for name, parms in self.agent.model.named_parameters():
                        if parms.grad is None:
                            continue
                        if ("mod2" in name or "enc_2" in name or "fc_lin_2" in name) and name in grads_text["both"]:
                            grads_text[loss_type][name] = parms.grad.data.clone()
                    grads_text[loss_type]["concat"] = torch.cat(
                        [
                            grads_text[loss_type][name].flatten()
                            for name, parms in self.agent.model.named_parameters()
                            if ("mod2" in name or "enc_2" in name or "fc_lin_2" in name) and parms.grad is not None and name in grads_text["both"]
                        ]
                    )
                else:
                    for name, parms in self.agent.model.named_parameters():
                        if parms.grad is None:
                            continue
                        if "mod0" in name or "enc_0" in name or "fc_lin_0" in name:
                            grads_audio[loss_type][name] = parms.grad.data.clone()
                        if "mod1" in name or "enc_1" in name or "fc_lin_1" in name:
                            grads_visual[loss_type][name] = parms.grad.data.clone()
                        if "mod2" in name or "enc_2" in name or "fc_lin_2" in name:
                            grads_text[loss_type][name] = parms.grad.data.clone()
                    grads_visual[loss_type]["concat"] = torch.cat(
                        [
                            grads_visual[loss_type][name].flatten()
                            for name, parms in self.agent.model.named_parameters()
                            if ("mod1" in name or "enc_1" in name or "fc_lin_1" in name) and parms.grad is not None
                        ]
                    )
                    grads_audio[loss_type]["concat"] = torch.cat(
                        [
                            grads_audio[loss_type][name].flatten()
                            for name, parms in self.agent.model.named_parameters()
                            if ("mod0" in name or "enc_0" in name or "fc_lin_0" in name) and parms.grad is not None
                        ]
                    )
                    aggr_list = []
                    for name, parms in self.agent.model.named_parameters():
                        if ("mod2" in name or "enc_2" in name or "fc_lin_2" in name) and parms.grad is not None:
                            aggr_list.append(grads_text[loss_type][name].flatten())
                    grads_text[loss_type]["concat"] = torch.cat(aggr_list)
                self.agent.optimizer.zero_grad()

            # Calculate the ratio
            audio_k, visual_k = self._compute_ratio(grads_audio, grads_visual)
            total = loss_mm + loss_a + loss_v
            total.backward()
            gamma = self.agent.config.model.args.bias_infusion.alpha
            self._equalize_gradients(grads_audio, grads_visual, audio_k, visual_k, gamma)

            self.agent.optimizer.step()

            wandb_output = {"ratio": {"audio_k": audio_k, "visual_k": visual_k}}
            wandb.log(wandb_output)

            return total, output_losses, True

    def _compute_ratio(self, grads_audio, grads_visual):
        this_cos_audio = F.cosine_similarity(grads_audio["both"]["concat"], grads_audio["audio"]["concat"], dim=0)
        this_cos_visual = F.cosine_similarity(grads_visual["both"]["concat"], grads_visual["visual"]["concat"], dim=0)

        audio_task = ["both", "audio"]
        visual_task = ["both", "visual"]

        audio_k = [0, 0]
        visual_k = [0, 0]

        if this_cos_audio > 0:
            audio_k[0] = 0.5
            audio_k[1] = 0.5
        else:
            audio_k, min_norm = MinNormSolver.find_min_norm_element([list(grads_audio[t].values()) for t in audio_task])
        if this_cos_visual > 0:
            visual_k[0] = 0.5
            visual_k[1] = 0.5
        else:
            visual_k, min_norm = MinNormSolver.find_min_norm_element([list(grads_visual[t].values()) for t in visual_task])
        return audio_k, visual_k

    def _equalize_gradients(self, grads_audio, grads_visual, audio_k, visual_k, gamma):
        for name, param in self.agent.model.named_parameters():
            if param.grad is not None:
                if ("mod0" in name or "fc_0" in name or "enc_0" in name) and name in grads_audio["both"]:
                    three_norm = torch.norm(param.grad.data.clone())
                    new_grad = 2 * audio_k[0] * grads_audio["both"][name] + 2 * audio_k[1] * grads_audio["audio"][name]
                    new_norm = torch.norm(new_grad)
                    diff = three_norm / new_norm
                    if diff > 1:
                        param.grad = diff * new_grad * gamma
                    else:
                        param.grad = new_grad * gamma

                if ("mod1" in name or "fc_1" in name or "enc_1" in name) and name in grads_visual["both"]:
                    three_norm = torch.norm(param.grad.data.clone())
                    new_grad = 2 * visual_k[0] * grads_visual["both"][name] + 2 * visual_k[1] * grads_visual["visual"][name]
                    new_norm = torch.norm(new_grad)
                    diff = three_norm / new_norm
                    if diff > 1:
                        param.grad = diff * new_grad * gamma
                    else:
                        param.grad = new_grad * gamma


class Bias_Infusion_MLA(General_Bias_Infusion):
    def __init__(self, agent):
        super(Bias_Infusion_MLA, self).__init__(agent)
        logging.info("Bias Infusion MLA is being employed")
        self._initialize_logs_n_utils()

    def _initialize_logs_n_utils(self):
        self.gs_plugin = GSPlugin()

    def before_backward(self, total, output_losses, **kwargs):
        if not self.agent.config.model.args.bias_infusion.use:
            return

        if self.agent.config.model.args.bias_infusion.starting_epoch <= self.agent.logs["current_epoch"] <= self.agent.config.model.args.bias_infusion.ending_epoch:

            loss_a = output_losses["ce_loss_c"]
            loss_a.backward()

            self.gs_plugin.before_update(
                self.agent.model.fc_out,
                kwargs["output"]["features"]["g"],
                self.agent.logs["current_step"] % len(self.agent.data_loader.train_loader),
                len(self.agent.data_loader.train_loader),
                self.gs_plugin.exp_count,
            )

            self.agent.optimizer.step()
            self.agent.optimizer.zero_grad()

            loss_v = output_losses["ce_loss_g"]
            loss_v.backward()

            self.gs_plugin.before_update(
                self.agent.model.fc_out,
                kwargs["output"]["features"]["c"],
                self.agent.logs["current_step"] % len(self.agent.data_loader.train_loader),
                len(self.agent.data_loader.train_loader),
                self.gs_plugin.exp_count,
            )
            self.agent.optimizer.step()
            self.agent.optimizer.zero_grad()

            self.gs_plugin.exp_count += 1

            return total, output_losses, True


class Bias_Infusion_ReconBoost(General_Bias_Infusion):
    def __init__(self, agent):
        super(Bias_Infusion_ReconBoost, self).__init__(agent)
        logging.info("Bias Infusion ReconBoost is being employed")
        self._initialize_logs_n_utils()

    def _initialize_logs_n_utils(self):
        pass

    def get_stage(self, epoch, epoch_stages, ensemble_stages):
        cycle_length = 2 * epoch_stages + 2 * ensemble_stages  # Total length of one full cycle
        position = epoch % cycle_length  # Position within the cycle

        if position < epoch_stages:
            return 0
        elif position < epoch_stages + ensemble_stages:
            return 1
        elif position < 2 * epoch_stages + ensemble_stages:
            return 2
        else:
            return 1  # Last phase of the cycle

    def before_backward(self, total, output_losses, **kwargs):
        if not self.agent.config.model.args.bias_infusion.use:
            return

        if self.agent.config.model.args.bias_infusion.starting_epoch <= self.agent.logs["current_epoch"] <= self.agent.config.model.args.bias_infusion.ending_epoch:

            target = (
                torch.zeros(
                    kwargs["output"]["preds"]["c"].shape[0],
                    self.agent.config.model.args.num_classes,
                )
                .cuda()
                .scatter_(1, kwargs["label"].view(-1, 1), 1)
            )

            stages = self.get_stage(
                self.agent.logs["current_epoch"],
                self.agent.config.model.args.bias_infusion.epoch_stages,
                self.agent.config.model.args.bias_infusion.ensemble_stages,
            )

            if stages == 0 or stages == 2:
                modality = stages // 2

                if modality == 0:
                    out_obj = kwargs["output"]["preds"]["c"]
                elif modality == 1:
                    out_obj = kwargs["output"]["preds"]["g"]
                out_join = kwargs["output"]["preds"]["combined"]

                boosting_loss = -self.agent.config.model.args.bias_infusion.weight1 * (target * out_obj.log_softmax(1)).mean(
                    -1
                ) + self.agent.config.model.args.bias_infusion.weight2 * (target * out_join.detach().softmax(1) * out_obj.log_softmax(1)).mean(-1)

                for name, param in self.agent.model.named_parameters():
                    if modality == 0 and ("mod1" in name or "fc_1" in name or "enc_1" in name):
                        param.requires_grad = False
                    elif modality == 1 and ("mod0" in name or "fc_0" in name or "enc_0" in name):
                        param.requires_grad = False

                self.agent.model.zero_grad()
                # disable the gradient of the other modalities

                if self.agent.config.model.args.bias_infusion.use_ga:
                    if self.agent.logs["current_epoch"] // self.agent.config.model.args.bias_infusion.epoch_stages == 0:
                        loss = boosting_loss
                    else:
                        if modality == 0:
                            pre_out_obj = kwargs["output"]["preds"]["g"]
                        elif modality == 1:
                            pre_out_obj = kwargs["output"]["preds"]["c"]
                        ga_loss = nn.MSELoss()(out_obj.detach().softmax(1), pre_out_obj.detach().softmax(1))  ## ga loss
                        loss = boosting_loss + self.agent.config.model.args.bias_infusion.alpha * ga_loss
                    loss.mean().backward()
                else:
                    boosting_loss.mean().backward()

                self.agent.optimizer.step()
                for name, param in self.agent.model.named_parameters():
                    if modality == 0 and ("mod1" in name or "fc_1" in name or "enc_1" in name):
                        param.requires_grad = True
                    elif modality == 1 and ("mod0" in name or "fc_0" in name or "enc_0" in name):
                        param.requires_grad = True

                output_losses = {"recon": boosting_loss.mean()}

            else:

                output_losses["ce_loss_combined"].backward()
                self.agent.optimizer.step()

            return total, output_losses, True

    def _compute_ratio(self, grads_audio, grads_visual):
        this_cos_audio = F.cosine_similarity(grads_audio["both"]["concat"], grads_audio["audio"]["concat"], dim=0)
        this_cos_visual = F.cosine_similarity(grads_visual["both"]["concat"], grads_visual["visual"]["concat"], dim=0)

        audio_task = ["both", "audio"]
        visual_task = ["both", "visual"]

        audio_k = [0, 0]
        visual_k = [0, 0]

        if this_cos_audio > 0:
            audio_k[0] = 0.5
            audio_k[1] = 0.5
        else:
            audio_k, min_norm = MinNormSolver.find_min_norm_element([list(grads_audio[t].values()) for t in audio_task])
        if this_cos_visual > 0:
            visual_k[0] = 0.5
            visual_k[1] = 0.5
        else:
            visual_k, min_norm = MinNormSolver.find_min_norm_element([list(grads_visual[t].values()) for t in visual_task])
        return audio_k, visual_k

    def _equalize_gradients(self, grads_audio, grads_visual, audio_k, visual_k, gamma):
        for name, param in self.agent.model.named_parameters():
            if param.grad is not None:
                if ("mod0" in name or "fc_0" in name or "enc_0" in name) and name in grads_audio["both"]:
                    three_norm = torch.norm(param.grad.data.clone())
                    new_grad = 2 * audio_k[0] * grads_audio["both"][name] + 2 * audio_k[1] * grads_audio["audio"][name]
                    new_norm = torch.norm(new_grad)
                    diff = three_norm / new_norm
                    if diff > 1:
                        param.grad = diff * new_grad * gamma
                    else:
                        param.grad = new_grad * gamma

                if ("mod1" in name or "fc_1" in name or "enc_1" in name) and name in grads_visual["both"]:
                    three_norm = torch.norm(param.grad.data.clone())
                    new_grad = 2 * visual_k[0] * grads_visual["both"][name] + 2 * visual_k[1] * grads_visual["visual"][name]
                    new_norm = torch.norm(new_grad)
                    diff = three_norm / new_norm
                    if diff > 1:
                        param.grad = diff * new_grad * gamma
                    else:
                        param.grad = new_grad * gamma


class Bias_Infusion_DnR(General_Bias_Infusion):
    def __init__(self, agent):
        super(Bias_Infusion_DnR, self).__init__(agent)
        logging.info("Bias Infusion DnR is being employed")
        self._initialize_logs_n_utils()

    def _initialize_logs_n_utils(self):
        self.checkpoint_model = None
        self.flag_reinit = 0

    def get_feature(self, args, epoch, this_dataloader, desc="Train"):
        self.agent.model.eval()
        with torch.no_grad():
            pbar = tqdm(
                enumerate(this_dataloader),
                total=len(this_dataloader),
                desc=desc,
                leave=False,
                disable=True,
                position=1,
            )
            feature_dict = defaultdict(list)
            labels = []
            for batch_idx, served_dict in pbar:

                if type(served_dict) == tuple:
                    served_dict = {
                        "data": {
                            "c": served_dict[0][0],
                            "f": served_dict[0][1],
                            "g": served_dict[0][2],
                        },
                        "label": served_dict[3].squeeze(dim=1),
                    }
                    if self.agent.config.get("task", "classification") == "classification" and len(served_dict["label"][served_dict["label"] == -1]) > 0:
                        served_dict["label"][served_dict["label"] == -1] = 0

                data = {view: served_dict["data"][view].cuda() for view in served_dict["data"] if type(served_dict["data"][view]) is torch.Tensor}
                data.update({view: data[view].float() for view in data if type(view) == int})

                label = served_dict["label"].squeeze().type(torch.LongTensor).cuda()

                output = self.agent.model(data, label=label)

                for view in output["features"]:
                    feature_dict[view].append(output["features"][view].detach().cpu().numpy())
                labels.append(label.detach().cpu().numpy())
                del output, label, data
                pbar_message = "Validation batch {0:d}/{1:d}".format(batch_idx, len(this_dataloader) - 1)
                pbar.set_description(pbar_message)
                pbar.refresh()

        labels = np.concatenate(labels)
        for view in feature_dict:
            feature_dict[view] = np.concatenate(feature_dict[view])
        return feature_dict, labels

    def purity_score(self, y_true, y_pred):

        y_voted_labels = np.zeros(y_true.shape)
        labels = np.unique(y_true)
        ordered_labels = np.arange(labels.shape[0])
        for k in range(labels.shape[0]):
            y_true[y_true == labels[k]] = ordered_labels[k]

        labels = np.unique(y_true)

        bins = np.concatenate((labels, [np.max(labels) + 1]), axis=0)

        for cluster in np.unique(y_pred):
            hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
            winner = np.argmax(hist)
            y_voted_labels[y_pred == cluster] = winner

        return accuracy_score(y_true, y_voted_labels)

    def reinit_score(self, args, train_features, train_label, val_features, val_label):

        all_feature = [
            train_features["c"],
            val_features["g"],
            train_features["g"],
            val_features["g"],
        ]
        stages = ["train_audio", "val_audio", "train_visual", "val_visual"]
        all_purity = []
        print("%%%%%%%%%%%%%%%%%%%%%%%%")
        for idx, fea in enumerate(all_feature):
            result = fea
            scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
            result = scaler.fit_transform(result)
            y_pred = KMeans(n_clusters=args.n_classes, random_state=0, n_init=10).fit_predict(result)

            if stages[idx][:5] == "train":
                purity = self.purity_score(np.array(train_label), y_pred)
            else:
                purity = self.purity_score(np.array(val_label), y_pred)
            all_purity.append(purity)

            print("%s purity= %.4f" % (stages[idx], purity))

        purity_gap_audio = np.abs(all_purity[0] - all_purity[1])
        purity_gap_visual = np.abs(all_purity[2] - all_purity[3])

        weight_audio = torch.tanh(torch.tensor(args.alpha * purity_gap_audio))
        weight_visual = torch.tanh(torch.tensor(args.alpha * purity_gap_visual))

        print("weights audio: {:.4f}, visual: {:.4f}".format(weight_audio, weight_visual))

        return weight_audio, weight_visual, all_purity

    def reinit(self, args, weight_audio, weight_visual):

        print("Start reinit ... ")

        record_names_audio = []
        record_names_visual = []
        for name, param in self.agent.model.named_parameters():
            if "mod0" in name or "fc_0" in name or "enc_0" in name:
                if "conv" in name:
                    record_names_audio.append((name, param))
            elif "mod1" in name or "fc_1" in name or "enc_1" in name:
                if "conv" in name:
                    record_names_visual.append((name, param))

        for name, param in self.agent.model.named_parameters():
            if "mod0" in name or "fc_0" in name or "enc_0" in name:
                init_weight = self.checkpoint_model[name]
                current_weight = param.data
                new_weight = weight_audio * init_weight + (1 - weight_audio).cuda() * current_weight
                param.data = new_weight
            elif "mod1" in name or "fc_1" in name or "enc_1" in name:
                init_weight = self.checkpoint_model[name]
                current_weight = param.data
                new_weight = weight_visual * init_weight + (1 - weight_visual).cuda() * current_weight
                param.data = new_weight

        return self.agent.model

    def on_epoch_begin(self, **kwargs):
        if not self.agent.config.model.args.bias_infusion.use:
            return

        epoch = self.agent.logs["current_epoch"]

        if self.checkpoint_model is None:
            self.checkpoint_model = copy.deepcopy(self.agent.model.state_dict())
        if self.agent.config.model.args.bias_infusion.starting_epoch <= epoch <= self.agent.config.model.args.bias_infusion.ending_epoch:
            args = self.agent.config.model.args.bias_infusion

            if (epoch % args.reinit_epoch == 0) & (epoch > 0):
                self.flag_reinit += 1
                if self.flag_reinit <= args.kmepoch:
                    train_features, train_label = self.get_feature(args, epoch, self.agent.data_loader.train_loader)
                    val_features, val_label = self.get_feature(args, epoch, self.agent.data_loader.valid_loader)
                    weight_audio, weight_visual, all_purity = self.reinit_score(args, train_features, train_label, val_features, val_label)
                    self.agent.model = self.reinit(args, weight_audio, weight_visual)
                    wandb_output = {
                        "ratio": {
                            "audio_k": weight_audio,
                            "visual_k": weight_visual,
                            "audio_purity_train": all_purity[0],
                            "audio_purity_train": all_purity[1],
                            "visual_purity_train": all_purity[2],
                            "visual_purity_val": all_purity[3],
                        }
                    }
                    wandb.log(wandb_output)


class Bias_Infusion_MLB(General_Bias_Infusion):
    def __init__(self, agent):
        super(Bias_Infusion_MLB, self).__init__(agent)
        logging.info("Bias Infusion MLB is being employed")
        self._initialize_logs_n_utils()

    def _initialize_logs_n_utils(self):
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.agent.logs["ratio_logs"] = {
            "ratio_audiodivvideo": [],
            "coeff_audio": [],
            "coeff_video": [],
        }
        self.balance_mode = self.agent.config.model.args.bias_infusion.get("balance_mode", False)
        # self.balance_mode = self.agent.config.model.args.bias_infusion.get("balance_mode", "whatev")
        self.alpha = self.agent.config.model.args.bias_infusion.alpha
        self.tanh_mode = self.agent.config.model.args.bias_infusion.get("tanh_mode", False)
        self.tanh_mode_beta = self.agent.config.model.args.bias_infusion.get("tanh_mode_beta", False)

        if self.tanh_mode == 2:
            print("Tanh mode 2 is being employed with beta {}".format(self.tanh_mode_beta))

    def on_backward_end(self, label, preds, **kwargs):

        def js_divergence(net_1_logits, net_2_logits):

            clip_value = 1e7

            net_1_probs = F.softmax(torch.clamp(net_1_logits, -clip_value, clip_value), dim=1)
            net_2_probs = F.softmax(torch.clamp(net_2_logits, -clip_value, clip_value), dim=1)

            total_m = 0.5 * (net_1_probs + net_2_probs)

            clip_value = 1e-20
            total_m = torch.clamp(total_m, clip_value, 1).log()

            # total_m = torch.clamp(total_m, clip_value, 1)
            net_1_probs = torch.clamp(net_1_probs, clip_value, 1)
            net_2_probs = torch.clamp(net_2_probs, clip_value, 1)

            loss = 0.0

            loss += F.kl_div(total_m, net_1_probs, reduction="batchmean")
            loss += F.kl_div(total_m, net_2_probs, reduction="batchmean")
            if torch.isnan(loss):
                # print which part is nan
                # print("total_m: ", total_m)
                print("net_1_logits: ", net_1_logits)
                print("net_2_logits: ", net_2_logits)
                print("net_1_probs: ", net_1_probs)
                print("net_2_probs: ", net_2_probs)

                raise Exception("NaN detected in loss computation")
            return 0.5 * loss

        out_color, out_gray = preds["c"], preds["g"]
        if len(label.shape) > 1:
            label = label.flatten()

        score_0 = torch.mean(torch.stack([self.softmax(out_color)[i][label[i]] for i in range(out_color.size(0))])).detach()
        score_1 = torch.mean(torch.stack([self.softmax(out_gray)[i][label[i]] for i in range(out_gray.size(0))])).detach()

        if self.tanh_mode == "2_jsd" or self.tanh_mode == "1_jsd":

            score_0 = js_divergence(out_color, preds["combined"]).detach()
            score_1 = js_divergence(out_gray, preds["combined"]).detach()

        ratio_0 = score_1 / score_0
        ratio_1 = 1 / ratio_0

        coeff_0 = 1 + self.tanh(self.alpha * (ratio_0 - 1))
        coeff_1 = 1 + self.tanh(self.alpha * (ratio_1 - 1))
        if self.tanh_mode == "2" or self.tanh_mode == "2_jsd":
            if ratio_0 > 1:
                coeff_0 = 1 + (self.tanh_mode_beta - 1) * self.tanh(self.alpha * (ratio_0 - 1))
            else:
                coeff_0 = 1 + self.tanh(self.alpha * (ratio_0 - 1))
            if ratio_1 > 1:
                coeff_1 = 1 + (self.tanh_mode_beta - 1) * self.tanh(self.alpha * (ratio_1 - 1))
            else:
                coeff_1 = 1 + self.tanh(self.alpha * (ratio_1 - 1))

        if self.tanh_mode == "linear":
            if ratio_0 > 1:
                coeff_0 = 1 + self.alpha * (ratio_0 - 1)
            else:
                coeff_0 = 1 + self.tanh(self.alpha * (ratio_0 - 1))
            if ratio_1 > 1:
                coeff_1 = 1 + self.alpha * (ratio_1 - 1)
            else:
                coeff_1 = 1 + self.tanh(self.alpha * (ratio_1 - 1))

        if self.tanh_mode == "all_linear":
            coeff_0 = self.alpha * ratio_0
            coeff_1 = self.alpha * ratio_1

        if self.tanh_mode == "softmax":
            scores = torch.tensor([score_0, score_1])
            scaled = self.alpha * scores
            softmax_weights = torch.softmax(scaled, dim=0)

            if ratio_0 > 0.5:
                coeff_0 = 1 + self.alpha * (softmax_weights[0] - 0.5)
            else:
                coeff_0 = 1 + self.tanh(self.alpha * (softmax_weights[0] - 0.5))
            if ratio_1 > 0.5:
                coeff_1 = 1 + self.alpha * (softmax_weights[1] - 0.5)
            else:
                coeff_1 = 1 + self.tanh(self.alpha * (softmax_weights[1] - 0.5))

        if self.tanh_mode == "ogm":
            if ratio_0 > 1:
                coeff_0 = 1 - self.tanh(self.alpha * self.relu(ratio_0))
                coeff_1 = torch.Tensor([1.0]).to(score_1.device)
            else:
                coeff_1 = 1 - self.tanh(self.alpha * self.relu(ratio_1))
                coeff_0 = torch.Tensor([1.0]).to(score_1.device)

        self.agent.logs["ratio_logs"]["ratio_audiodivvideo"].append(ratio_0.cpu().numpy())
        self.agent.logs["ratio_logs"]["coeff_audio"].append(coeff_0.cpu().numpy())
        self.agent.logs["ratio_logs"]["coeff_video"].append(coeff_1.cpu().numpy())

        wandb_output = {
            "ratio": {
                "ratio_0": ratio_0.cpu().numpy(),
                "ratio_1": ratio_1.cpu().numpy(),
                "coeff_0": coeff_0.cpu().numpy(),
                "coeff_1": coeff_1.cpu().numpy(),
            }
        }

        wandb.log(wandb_output)

        self._equalize_gradients(coeff_0=coeff_0, coeff_1=coeff_1)

    def _equalize_gradients(self, coeff_0=1, coeff_1=1):
        if not self.agent.config.model.args.bias_infusion.use:
            return

        if self.agent.config.model.args.bias_infusion.starting_epoch <= self.agent.logs["current_epoch"] <= self.agent.config.model.args.bias_infusion.ending_epoch:

            for name, parms in self.agent.model.named_parameters():
                if parms.grad is None:
                    continue
                if "mod0" in name or "enc_0" in name or "fc_0_lin.weight" in name:
                    parms.grad *= coeff_0
                if "mod1" in name or "enc_1" in name or "fc_1_lin.weight" in name:
                    parms.grad *= coeff_1


class Bias_Infusion_MLB_Shap(General_Bias_Infusion):
    def __init__(self, agent):
        super(Bias_Infusion_MLB_Shap, self).__init__(agent)
        logging.info("Bias Infusion MLB with Shapley Values is being employed")
        self._initialize_logs_n_utils()

    def _initialize_logs_n_utils(self):
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.agent.logs["ratio_logs"] = {
            "ratio_audiodivvideo": [],
            "coeff_audio": [],
            "coeff_video": [],
        }
        self.balance_mode = self.agent.config.model.args.bias_infusion.get("balance_mode", "shap")
        self.alpha = self.agent.config.model.args.bias_infusion.alpha
        self.tanh_mode = self.agent.config.model.args.bias_infusion.get("tanh_mode", False)
        self.tanh_mode_beta = self.agent.config.model.args.bias_infusion.get("tanh_mode_beta", False)

        if self.tanh_mode == 2:
            print("Tanh mode 2 is being employed with beta {}".format(self.tanh_mode_beta))

    # def before_backward(self, total, output_losses, **kwargs):
    #
    #     opt_done = False
    #     if self.balance_mode == "balance_only_multi":
    #         (output_losses["ce_loss_combined"]).backward(retain_graph=True)
    #         # (output_losses["ce_loss_combined"] + self.output_losses["ce_loss_c"] + self.output_losses["ce_loss_g"]).backward(retain_graph=True)
    #         self.output_losses = output_losses
    #         opt_done = True
    #
    #     return total, output_losses, opt_done

    def on_backward_end(self, label, preds):
        def js_divergence(net_1_logits, net_2_logits):

            clip_value = 1e7

            net_1_probs = F.softmax(torch.clamp(net_1_logits, -clip_value, clip_value), dim=1)
            net_2_probs = F.softmax(torch.clamp(net_2_logits, -clip_value, clip_value), dim=1)

            total_m = 0.5 * (net_1_probs + net_2_probs)

            clip_value = 1e-20
            total_m = torch.clamp(total_m, clip_value, 1).log()

            net_1_probs = torch.clamp(net_1_probs, clip_value, 1)
            net_2_probs = torch.clamp(net_2_probs, clip_value, 1)

            loss = 0.0

            loss += F.kl_div(total_m, net_1_probs, reduction="batchmean")
            loss += F.kl_div(total_m, net_2_probs, reduction="batchmean")
            if torch.isnan(loss):
                raise Exception("NaN detected in loss computation")
            return 0.5 * loss

        if len(label.shape) > 1:
            label = label.flatten()

        if self.balance_mode == "shap":
            shap_value_0 = 0.5 * (preds["combined"] - preds["sa"] + preds["sv"] - preds["sav"])
            shap_value_1 = 0.5 * (preds["combined"] + preds["sa"] - preds["sv"] - preds["sav"])

            # Compute JS divergence to normal preds vs shapley preds
            score_0 = js_divergence(shap_value_0, preds["combined"]).detach()
            score_1 = js_divergence(shap_value_1, preds["combined"]).detach()

        elif self.balance_mode == "shap_label":
            shap_value_0 = 0.5 * (preds["combined"] - preds["sa"] + preds["sv"] - preds["sav"])
            shap_value_1 = 0.5 * (preds["combined"] + preds["sa"] - preds["sv"] - preds["sav"])

            score_0 = torch.mean(torch.stack([self.softmax(shap_value_0)[i][label[i]] for i in range(shap_value_0.size(0))])).detach()
            score_1 = torch.mean(torch.stack([self.softmax(shap_value_1)[i][label[i]] for i in range(shap_value_1.size(0))])).detach()

        elif self.balance_mode == "perm":
            shap_value_0 = 0.5 * (preds["ncombined"] - preds["sa"] + preds["sv"])  # - preds["sav"])
            shap_value_1 = 0.5 * (preds["ncombined"] + preds["sa"] - preds["sv"])  # - preds["sav"])

            # Compute JS divergence to normal preds vs shapley preds
            score_0 = js_divergence(shap_value_0, preds["ncombined"]).detach()
            score_1 = js_divergence(shap_value_1, preds["ncombined"]).detach()

        elif self.balance_mode == "perm_label":
            shap_value_0 = 0.5 * (preds["ncombined"] - preds["sa"] + preds["sv"] - preds["sav"])  # )
            shap_value_1 = 0.5 * (preds["ncombined"] + preds["sa"] - preds["sv"] - preds["sav"])  # )

            # Compare as per loss using the target
            score_0 = torch.mean(torch.stack([self.softmax(shap_value_0)[i][preds["n_label"][i]] for i in range(shap_value_0.size(0))])).detach()
            score_1 = torch.mean(torch.stack([self.softmax(shap_value_1)[i][preds["n_label"][i]] for i in range(shap_value_1.size(0))])).detach()

        ratio_0 = score_1 / score_0
        ratio_1 = 1 / ratio_0

        coeff_0 = 1 + self.tanh(self.alpha * (ratio_0 - 1))
        coeff_1 = 1 + self.tanh(self.alpha * (ratio_1 - 1))
        if self.tanh_mode == 2:
            if ratio_0 > 1:
                coeff_0 = 1 + (self.tanh_mode_beta - 1) * self.tanh(self.alpha * (ratio_0 - 1))
            else:
                coeff_0 = 1 + self.tanh(self.alpha * (ratio_0 - 1))
            if ratio_1 > 1:
                coeff_1 = 1 + (self.tanh_mode_beta - 1) * self.tanh(self.alpha * (ratio_1 - 1))
            else:
                coeff_1 = 1 + self.tanh(self.alpha * (ratio_1 - 1))

        self.agent.logs["ratio_logs"]["ratio_audiodivvideo"].append(ratio_0.cpu().numpy())
        self.agent.logs["ratio_logs"]["coeff_audio"].append(coeff_0.cpu().numpy())
        self.agent.logs["ratio_logs"]["coeff_video"].append(coeff_1.cpu().numpy())

        wandb_output = {
            "ratio": {
                "ratio_0": ratio_0.cpu().numpy(),
                "ratio_1": ratio_1.cpu().numpy(),
                "score_0": score_0.cpu().numpy(),
                "score_1": score_1.cpu().numpy(),
                # "scoreperf_shap0": scoreperf_shap0.cpu().numpy(),
                # "scoreperf_shap1": scoreperf_shap1.cpu().numpy(),
                # "scoreperf_0": scoreperf_0.cpu().numpy(),
                # "scoreperf_1": scoreperf_1.cpu().numpy(),
                "coeff_0": coeff_0.cpu().numpy(),
                "coeff_1": coeff_1.cpu().numpy(),
            }
        }

        wandb.log(wandb_output)

        self._equalize_gradients(coeff_0=coeff_0, coeff_1=coeff_1)

        # if self.balance_mode == "balance_only_multi":
        #     (self.output_losses["ce_loss_c"] + self.output_losses["ce_loss_g"]).backward()
        #     del self.output_losses
        #     self.agent.optimizer.step()

    def _equalize_gradients(self, coeff_0=1, coeff_1=1):
        if not self.agent.config.model.args.bias_infusion.use:
            return

        if self.agent.config.model.args.bias_infusion.starting_epoch <= self.agent.logs["current_epoch"] <= self.agent.config.model.args.bias_infusion.ending_epoch:

            for name, parms in self.agent.model.named_parameters():
                if parms.grad is None:
                    continue
                if "mod0" in name or "enc_0" in name or "fc_0_lin.weight" in name:
                    parms.grad *= coeff_0
                if "mod1" in name or "enc_1" in name or "fc_1_lin.weight" in name:
                    parms.grad *= coeff_1

    def on_epoch_begin(self, **kwargs):

        def compute_overfit(val_loss, train_loss, prev_val_loss, prev_train_loss):
            new_O = val_loss - train_loss
            prev_O = prev_val_loss - prev_train_loss
            return new_O - prev_O

        def compute_coeff(loss_name):
            list_val = list(self.agent.logs["val_logs"].keys())
            if len(list_val) < 2:
                return 1, 1, 1
            train_loss = self.agent.logs["train_logs"][list_val[-1]]["loss"][loss_name]
            val_loss = self.agent.logs["val_logs"][list_val[-1]]["loss"][loss_name]

            prev_train_loss = self.agent.logs["train_logs"][list_val[-2]]["loss"][loss_name]
            prev_val_loss = self.agent.logs["val_logs"][list_val[-2]]["loss"][loss_name]
            print("Train Loss: {} Val Loss: {} Prev Train Loss: {} Prev Val Loss: {}".format(train_loss, val_loss, prev_train_loss, prev_val_loss))
            overfit = compute_overfit(val_loss, train_loss, prev_val_loss, prev_train_loss)
            gen = val_loss - prev_val_loss
            coef = np.abs(gen) / (overfit * overfit)
            return coef, overfit, gen

        coef_combined, over_combined, gen_combined = compute_coeff("ce_loss_combined")
        coef_c, over_c, gen_c = compute_coeff("ce_loss_c")
        coef_g, over_g, gen_g = compute_coeff("ce_loss_g")

        wandb_output = {
            "ratio": {
                "coef_combined": coef_combined,
                "coef_c": coef_c,
                "coef_g": coef_g,
                "over_combined": over_combined,
                "over_c": over_c,
                "over_g": over_g,
                "gen_combined": gen_combined,
                "gen_c": gen_c,
                "gen_g": gen_g,
            }
        }

        wandb.log(wandb_output)


class Bias_Infusion_MLB_ShapPerm(General_Bias_Infusion):
    def __init__(self, agent):
        super(Bias_Infusion_MLB_ShapPerm, self).__init__(agent)
        logging.info("Bias Infusion MLB with Shapley Values is being employed")
        self._initialize_logs_n_utils()

    def _initialize_logs_n_utils(self):
        self.criterion = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.agent.logs["ratio_logs"] = defaultdict(list)
        self.balance_mode = self.agent.config.model.args.bias_infusion.get("balance_mode", "shap")
        # self.balance_mode = self.agent.config.model.args.bias_infusion.get("balance_mode", "whatev")
        self.alpha = self.agent.config.model.args.bias_infusion.alpha
        self.tanh_mode = self.agent.config.model.args.bias_infusion.get("tanh_mode", False)
        self.tanh_mode_beta = self.agent.config.model.args.bias_infusion.get("tanh_mode_beta", False)

        if self.tanh_mode == 2:
            print("Tanh mode 2 is being employed with beta {}".format(self.tanh_mode_beta))

    def on_backward_end(self, label, preds):
        def js_divergence(net_1_logits, net_2_logits):

            clip_value = 1e7

            net_1_probs = F.softmax(torch.clamp(net_1_logits, -clip_value, clip_value), dim=1)
            net_2_probs = F.softmax(torch.clamp(net_2_logits, -clip_value, clip_value), dim=1)

            total_m = 0.5 * (net_1_probs + net_2_probs)

            clip_value = 1e-20
            total_m = torch.clamp(total_m, clip_value, 1).log()

            # total_m = torch.clamp(total_m, clip_value, 1)
            net_1_probs = torch.clamp(net_1_probs, clip_value, 1)
            net_2_probs = torch.clamp(net_2_probs, clip_value, 1)

            loss = 0.0

            loss += F.kl_div(total_m, net_1_probs, reduction="batchmean")
            loss += F.kl_div(total_m, net_2_probs, reduction="batchmean")
            if torch.isnan(loss):
                # print which part is nan
                # print("total_m: ", total_m)
                print("net_1_logits: ", net_1_logits)
                print("net_2_logits: ", net_2_logits)
                print("net_1_probs: ", net_1_probs)
                print("net_2_probs: ", net_2_probs)

                raise Exception("NaN detected in loss computation")
            return 0.5 * loss

        if len(label.shape) > 1:
            label = label.flatten()

        out_color, out_gray = preds["c"], preds["g"]
        if len(label.shape) > 1:
            label = label.flatten()

        scoreperf_0 = torch.mean(torch.stack([self.softmax(out_color)[i][label[i]] for i in range(out_color.size(0))])).detach()
        scoreperf_1 = torch.mean(torch.stack([self.softmax(out_gray)[i][label[i]] for i in range(out_gray.size(0))])).detach()
        # score_1 = sum([self.softmax(out_gray)[i][label[i]] for i in range(out_gray.size(0))]).detach()

        # scoreperf_0 = self.criterion(out_color, label.cuda()).detach()
        # scoreperf_1 = self.criterion(out_gray, label.cuda()).detach()

        ratioperf_0 = scoreperf_1 / scoreperf_0
        ratioperf_1 = 1 / ratioperf_0

        coeffperf_0 = 1 + self.tanh(self.alpha * (ratioperf_0 - 1))
        coeffperf_1 = 1 + self.tanh(self.alpha * (ratioperf_1 - 1))

        self.agent.logs["ratio_logs"]["perf_ratio_audiodivvideo"].append(ratioperf_0.cpu().numpy())
        self.agent.logs["ratio_logs"]["coeffperf_audio"].append(coeffperf_0.cpu().numpy())
        self.agent.logs["ratio_logs"]["coeffperf_video"].append(coeffperf_1.cpu().numpy())

        shap_value_0 = 0.5 * (preds["combined"] - preds["sa"] + preds["sv"] - preds["sav"])
        shap_value_1 = 0.5 * (preds["combined"] + preds["sa"] - preds["sv"] - preds["sav"])

        scoreshap_0 = self.criterion(shap_value_0, label.cuda()).detach()
        scoreshap_1 = self.criterion(shap_value_1, label.cuda()).detach()

        # Compute JS divergence to normal preds vs shapley preds
        # scoreshap_0 = js_divergence(shap_value_0, preds["combined"]).detach()
        # scoreshap_1 = js_divergence(shap_value_1, preds["combined"]).detach()
        #
        # scoreshap_0 = js_divergence(shap_value_0, F.one_hot(label, num_classes=preds["combined"].size(1)).float()).detach()
        # scoreshap_0 = js_divergence(shap_value_1, F.one_hot(label, num_classes=preds["combined"].size(1)).float()).detach()

        # # Compare as per loss using the target
        scoreperf_shap0 = torch.mean(torch.stack([self.softmax(shap_value_0)[i][label[i]] for i in range(shap_value_0.size(0))])).detach()
        scoreperf_shap1 = torch.mean(torch.stack([self.softmax(shap_value_1)[i][label[i]] for i in range(shap_value_1.size(0))])).detach()

        # ratioshapperf_0 = scoreperf_shap1 / scoreperf_shap0
        # ratioshapperf_1 = 1 / ratioshapperf_0
        #
        # coeffshapperf_0 = 1 + self.tanh(self.alpha * (ratioshapperf_0 - 1))
        # coeffshapperf_1 = 1 + self.tanh(self.alpha * (ratioshapperf_1 - 1))
        #
        # # Compare as per loss using the target
        # scoreperf_0 = torch.mean(
        #     torch.stack([self.softmax(preds["c"])[i][label[i]] for i in range(preds["c"].size(0))])).detach()
        # scoreperf_1 = torch.mean(
        #     torch.stack([self.softmax(preds["g"])[i][label[i]] for i in range(preds["g"].size(0))])).detach()

        ratioshap_0 = scoreperf_shap1 / scoreperf_shap0
        ratioshap_1 = 1 / ratioshap_0

        coeffshap_0 = 1 + self.tanh(self.alpha * (ratioshap_0 - 1))
        coeffshap_1 = 1 + self.tanh(self.alpha * (ratioshap_1 - 1))

        self.agent.logs["ratio_logs"]["shap_ratio_audiodivvideo"].append(ratioshap_0.cpu().numpy())
        self.agent.logs["ratio_logs"]["shap_coeff_audio"].append(coeffshap_0.cpu().numpy())
        self.agent.logs["ratio_logs"]["shap_coeff_video"].append(coeffshap_1.cpu().numpy())

        # self.agent.logs["ratio_logs"]["shapperf_ratio_audiodivvideo"].append(ratioshapperf_0.cpu().numpy())
        # self.agent.logs["ratio_logs"]["shapperf_coeff_audio"].append(coeffshapperf_0.cpu().numpy())
        # self.agent.logs["ratio_logs"]["shapperf_coeff_video"].append(coeffshapperf_1.cpu().numpy())

        perm_value_0 = 0.5 * (preds["perm_ncombined"] - preds["perm_sa"] + preds["perm_sv"] - preds["sav"])  # - preds["sav"])
        perm_value_1 = 0.5 * (preds["perm_ncombined"] + preds["perm_sa"] - preds["perm_sv"] - preds["sav"])  # - preds["sav"])

        # Compute JS divergence to normal preds vs shapley preds
        scoreperm_0 = js_divergence(perm_value_0, preds["perm_ncombined"]).detach()
        scoreperm_1 = js_divergence(perm_value_1, preds["perm_ncombined"]).detach()

        scoreperm_0 = self.criterion(perm_value_0, label.cuda()).detach()
        scoreperm_1 = self.criterion(perm_value_1, label.cuda()).detach()

        # Compare as per loss using the target
        scoreperf_perm0 = torch.mean(torch.stack([self.softmax(perm_value_0)[i][preds["perm_n_label"][i]] for i in range(perm_value_0.size(0))])).detach()
        scoreperf_perm1 = torch.mean(torch.stack([self.softmax(perm_value_1)[i][preds["perm_n_label"][i]] for i in range(perm_value_1.size(0))])).detach()

        # # Compare as per loss using the target
        # scoreperf_perm0 = torch.mean(
        #     torch.stack([self.softmax(preds["c"])[i][label[i]] for i in range(preds["c"].size(0))])).detach()
        # scoreperf_1 = torch.mean(
        #     torch.stack([self.softmax(preds["g"])[i][label[i]] for i in range(preds["g"].size(0))])).detach()

        ratioperm_0 = scoreperf_perm1 / scoreperf_perm0
        ratioperm_1 = 1 / ratioperm_0

        coeffperm_0 = 1 + self.tanh(self.alpha * (ratioperm_0 - 1))
        coeffperm_1 = 1 + self.tanh(self.alpha * (ratioperm_1 - 1))

        self.agent.logs["ratio_logs"]["perm_ratio_audiodivvideo"].append(ratioperm_0.cpu().numpy())
        self.agent.logs["ratio_logs"]["perm_coeff_audio"].append(coeffperm_0.cpu().numpy())
        self.agent.logs["ratio_logs"]["perm_coeff_video"].append(coeffperm_1.cpu().numpy())
        print("---------------------")
        # print("Perm-Shap", np.mean((np.array(self.agent.logs["ratio_logs"]["perm_ratio_audiodivvideo"]) - np.array(self.agent.logs["ratio_logs"]["shap_ratio_audiodivvideo"]))**2))
        # print("Perf-Shap", np.mean((np.array(self.agent.logs["ratio_logs"]["perf_ratio_audiodivvideo"]) - np.array(self.agent.logs["ratio_logs"]["shap_ratio_audiodivvideo"]))**2))
        # # print("Perf-ShapPerf", np.mean((np.array(self.agent.logs["ratio_logs"]["perf_ratio_audiodivvideo"]) - np.array(self.agent.logs["ratio_logs"]["shapperf_ratio_audiodivvideo"]))**2))
        # print("Perm-Perf", np.mean((np.array(self.agent.logs["ratio_logs"]["perm_ratio_audiodivvideo"]) - np.array(self.agent.logs["ratio_logs"]["perf_ratio_audiodivvideo"]))**2))

        print(
            "Perm-Shap",
            np.corrcoef(
                np.array(self.agent.logs["ratio_logs"]["perm_ratio_audiodivvideo"]),
                np.array(self.agent.logs["ratio_logs"]["shap_ratio_audiodivvideo"]),
            )[0, 1],
        )
        print(
            "Perf-Shap",
            np.corrcoef(
                np.array(self.agent.logs["ratio_logs"]["perf_ratio_audiodivvideo"]),
                np.array(self.agent.logs["ratio_logs"]["shap_ratio_audiodivvideo"]),
            )[0, 1],
        )
        print(
            "Perm-Perf",
            np.corrcoef(
                np.array(self.agent.logs["ratio_logs"]["perm_ratio_audiodivvideo"]),
                np.array(self.agent.logs["ratio_logs"]["perf_ratio_audiodivvideo"]),
            )[0, 1],
        )

        wandb_output = {
            "ratio": {
                "ratioshap_0": ratioshap_0.cpu().numpy(),
                "ratioperf_0": ratioperf_0.cpu().numpy(),
                "ratioperm_0": ratioperm_0.cpu().numpy(),
                # "ratioshapperf_0": ratioshapperf_0.cpu().numpy(),
                "scoreperf_0": scoreperf_0.cpu().numpy(),
                "scoreperf_1": scoreperf_1.cpu().numpy(),
                "scoreperm_0": scoreperm_0.cpu().numpy(),
                "scoreperm_1": scoreperm_1.cpu().numpy(),
                "scoreshap_0": scoreshap_0.cpu().numpy(),
                "scoreshap_1": scoreshap_1.cpu().numpy(),
                "scoreshapperf_0": scoreperf_shap0.cpu().numpy(),
                "scoreshapperf_1": scoreperf_shap1.cpu().numpy(),
                "coeffperf_0": coeffperf_0.cpu().numpy(),
                "coeffperf_1": coeffperf_1.cpu().numpy(),
                "coeffshap_0": coeffshap_0.cpu().numpy(),
                "coeffshap_1": coeffshap_1.cpu().numpy(),
                # "coeffshapperf_0": coeffshapperf_0.cpu().numpy(),
                # "coeffshapperf_1": coeffshapperf_1.cpu().numpy(),
                "coeffperm_0": coeffperm_0.cpu().numpy(),
                "coeffperm_1": coeffperm_1.cpu().numpy(),
            }
        }

        wandb.log(wandb_output)

        # self._equalize_gradients(coeff_0=coeff_0, coeff_1=coeff_1)

        # if self.balance_mode == "balance_only_multi":
        #     (self.output_losses["ce_loss_c"] + self.output_losses["ce_loss_g"]).backward()
        #     del self.output_losses
        #     self.agent.optimizer.step()

    def _equalize_gradients(self, coeff_0=1, coeff_1=1):
        if not self.agent.config.model.args.bias_infusion.use:
            return

        if self.agent.config.model.args.bias_infusion.starting_epoch <= self.agent.logs["current_epoch"] <= self.agent.config.model.args.bias_infusion.ending_epoch:

            for name, parms in self.agent.model.named_parameters():
                if parms.grad is None:
                    continue
                if "mod0" in name or "enc_0" in name or "fc_0_lin.weight" in name:
                    parms.grad *= coeff_0
                if "mod1" in name or "enc_1" in name or "fc_1_lin.weight" in name:
                    parms.grad *= coeff_1

    def on_epoch_begin(self, **kwargs):

        def compute_overfit(val_loss, train_loss, prev_val_loss, prev_train_loss):
            new_O = val_loss - train_loss
            prev_O = prev_val_loss - prev_train_loss
            return new_O - prev_O

        def compute_coeff(loss_name):
            list_val = list(self.agent.logs["val_logs"].keys())
            if len(list_val) < 2:
                return 1, 1, 1
            train_loss = self.agent.logs["train_logs"][list_val[-1]]["loss"][loss_name]
            val_loss = self.agent.logs["val_logs"][list_val[-1]]["loss"][loss_name]

            prev_train_loss = self.agent.logs["train_logs"][list_val[-2]]["loss"][loss_name]
            prev_val_loss = self.agent.logs["val_logs"][list_val[-2]]["loss"][loss_name]
            print("Train Loss: {} Val Loss: {} Prev Train Loss: {} Prev Val Loss: {}".format(train_loss, val_loss, prev_train_loss, prev_val_loss))
            overfit = compute_overfit(val_loss, train_loss, prev_val_loss, prev_train_loss)
            gen = val_loss - prev_val_loss
            coef = np.abs(gen) / (overfit * overfit)
            return coef, overfit, gen

        coef_combined, over_combined, gen_combined = compute_coeff("ce_loss_combined")
        coef_c, over_c, gen_c = compute_coeff("ce_loss_c")
        coef_g, over_g, gen_g = compute_coeff("ce_loss_g")

        wandb_output = {
            "ratio": {
                "coef_combined": coef_combined,
                "coef_c": coef_c,
                "coef_g": coef_g,
                "over_combined": over_combined,
                "over_c": over_c,
                "over_g": over_g,
                "gen_combined": gen_combined,
                "gen_c": gen_c,
                "gen_g": gen_g,
            }
        }

        wandb.log(wandb_output)


class Bias_Infusion_MLB_3d(General_Bias_Infusion):
    def __init__(self, agent):
        super(Bias_Infusion_MLB_3d, self).__init__(agent)
        logging.info("Bias Infusion MLB is being employed")
        self._initialize_logs_n_utils()

    def _initialize_logs_n_utils(self):
        self.criterion = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.agent.logs["ratio_logs"] = {
            "ratio_0": [],
            "ratio_1": [],
            "ratio_2": [],
            "coeff_v": [],
            "coeff_l": [],
            "coeff_f": [],
        }
        self.balance_mode = self.agent.config.model.args.bias_infusion.get("balance_mode", False)
        # self.balance_mode = self.agent.config.model.args.bias_infusion.get("balance_mode", "whatev")
        self.alpha = self.agent.config.model.args.bias_infusion.alpha
        self.tanh_mode = self.agent.config.model.args.bias_infusion.get("tanh_mode", False)
        self.tanh_mode_beta = self.agent.config.model.args.bias_infusion.get("tanh_mode_beta", False)

        if self.tanh_mode == 2:
            print("Tanh mode 2 is being employed with beta {}".format(self.tanh_mode_beta))

    # def before_backward(self, total, output_losses, **kwargs):
    #
    #     opt_done = False
    #     if self.balance_mode == "balance_only_multi":
    #         (output_losses["ce_loss_combined"]).backward(retain_graph=True)
    #         # (output_losses["ce_loss_combined"] + self.output_losses["ce_loss_c"] + self.output_losses["ce_loss_g"]).backward(retain_graph=True)
    #         self.output_losses = output_losses
    #         opt_done = True
    #
    #     return total, output_losses, opt_done

    def on_backward_end(self, label, out_color, out_gray, out_f):

        score_0 = torch.mean(torch.stack([self.softmax(out_color)[i][label[i]] for i in range(out_color.size(0))])).detach()
        score_1 = torch.mean(torch.stack([self.softmax(out_gray)[i][label[i]] for i in range(out_gray.size(0))])).detach()
        score_2 = torch.mean(torch.stack([self.softmax(out_f)[i][label[i]] for i in range(out_f.size(0))])).detach()

        ratio_0 = ((score_1 + score_2) / 2) / score_0
        ratio_1 = ((score_0 + score_2) / 2) / score_1
        ratio_2 = ((score_0 + score_1) / 2) / score_2

        coeff_0 = 1 + self.tanh(self.alpha * (ratio_0 - 1)).cpu().numpy()
        coeff_1 = 1 + self.tanh(self.alpha * (ratio_1 - 1)).cpu().numpy()
        coeff_2 = 1 + self.tanh(self.alpha * (ratio_2 - 1)).cpu().numpy()

        if self.tanh_mode == 2:
            if ratio_0 > 1:
                coeff_0 = 1 + (self.tanh_mode_beta - 1) * self.tanh(self.alpha * (ratio_0 - 1))
            else:
                coeff_0 = 1 + self.tanh(self.alpha * (ratio_0 - 1))
            if ratio_1 > 1:
                coeff_1 = 1 + (self.tanh_mode_beta - 1) * self.tanh(self.alpha * (ratio_1 - 1))
            else:
                coeff_1 = 1 + self.tanh(self.alpha * (ratio_1 - 1))
            if ratio_2 > 1:
                coeff_2 = 1 + (self.tanh_mode_beta - 1) * self.tanh(self.alpha * (ratio_2 - 1))
            else:
                coeff_2 = 1 + self.tanh(self.alpha * (ratio_2 - 1))

        self.agent.logs["ratio_logs"]["ratio_0"].append(ratio_0.cpu().numpy())
        self.agent.logs["ratio_logs"]["ratio_1"].append(ratio_1.cpu().numpy())
        self.agent.logs["ratio_logs"]["ratio_2"].append(ratio_2.cpu().numpy())
        self.agent.logs["ratio_logs"]["coeff_v"].append(coeff_0)
        self.agent.logs["ratio_logs"]["coeff_l"].append(coeff_1)
        self.agent.logs["ratio_logs"]["coeff_f"].append(coeff_2)

        wandb_output = {
            "ratio": {
                "ratio_0": ratio_0.cpu().numpy(),
                "ratio_1": ratio_1.cpu().numpy(),
                "ratio_2": ratio_2.cpu().numpy(),
                "coeff_v": coeff_0,
                "coeff_l": coeff_1,
                "coeff_f": coeff_2,
            }
        }

        wandb.log(wandb_output, step=self.agent.logs["current_step"])

        self._equalize_gradients(coeff_0=coeff_0, coeff_1=coeff_1, coeff_2=coeff_2)

    def _equalize_gradients(self, coeff_0=1, coeff_1=1, coeff_2=1):
        if not self.agent.config.model.args.bias_infusion.use:
            return

        if self.agent.config.model.args.bias_infusion.starting_epoch <= self.agent.logs["current_epoch"] <= self.agent.config.model.args.bias_infusion.ending_epoch:

            for name, parms in self.agent.model.named_parameters():
                if parms.grad is None:
                    continue
                if "mod0" in name or "enc_0" in name or "fc_0_lin.weight" in name:
                    parms.grad *= coeff_0
                if "mod1" in name or "enc_1" in name or "fc_1_lin.weight" in name:
                    parms.grad *= coeff_1
                if "mod2" in name or "enc_2" in name or "fc_2_lin.weight" in name:
                    parms.grad *= coeff_2


class Bias_Infusion_MLB_Reg(General_Bias_Infusion):
    def __init__(self, agent):
        super(Bias_Infusion_MLB_Reg, self).__init__(agent)
        logging.info("Bias Infusion MLB is being employed")
        self._initialize_logs_n_utils()

    def _initialize_logs_n_utils(self):
        self.criterion = nn.CrossEntropyLoss()
        self.tanh = nn.Tanh()
        self.agent.logs["ratio_logs"] = {
            "ratio_0": [],
            "ratio_1": [],
            "ratio_2": [],
            "coeff_v": [],
            "coeff_l": [],
            "coeff_f": [],
        }
        self.balance_mode = self.agent.config.model.args.bias_infusion.get("balance_mode", False)
        # self.balance_mode = self.agent.config.model.args.bias_infusion.get("balance_mode", "whatev")
        self.alpha = self.agent.config.model.args.bias_infusion.alpha
        self.tanh_mode = self.agent.config.model.args.bias_infusion.get("tanh_mode", False)
        self.tanh_mode_beta = self.agent.config.model.args.bias_infusion.get("tanh_mode_beta", False)

        if self.tanh_mode == 2:
            print("Tanh mode 2 is being employed with beta {}".format(self.tanh_mode_beta))

    def on_backward_end(self, label, out_color, out_gray):

        score_0 = F.mse_loss(out_color.squeeze(), label)
        score_1 = F.mse_loss(out_gray.squeeze(), label)

        ratio_0 = score_0 / score_1
        ratio_1 = score_1 / score_0

        coeff_0 = 1 + self.tanh(self.alpha * (ratio_0 - 1)).cpu().numpy()
        coeff_1 = 1 + self.tanh(self.alpha * (ratio_1 - 1)).cpu().numpy()

        if self.tanh_mode == 2:
            if ratio_0 > 1:
                coeff_0 = 1 + (self.tanh_mode_beta - 1) * self.tanh(self.alpha * (ratio_0 - 1))
            else:
                coeff_0 = 1 + self.tanh(self.alpha * (ratio_0 - 1))
            if ratio_1 > 1:
                coeff_1 = 1 + (self.tanh_mode_beta - 1) * self.tanh(self.alpha * (ratio_1 - 1))
            else:
                coeff_1 = 1 + self.tanh(self.alpha * (ratio_1 - 1))

        self.agent.logs["ratio_logs"]["ratio_0"].append(ratio_0.cpu().numpy())
        self.agent.logs["ratio_logs"]["ratio_1"].append(ratio_1.cpu().numpy())
        self.agent.logs["ratio_logs"]["coeff_v"].append(coeff_0)
        self.agent.logs["ratio_logs"]["coeff_l"].append(coeff_1)

        wandb_output = {
            "ratio": {
                "ratio_0": ratio_0.cpu().numpy(),
                "ratio_1": ratio_1.cpu().numpy(),
                "coeff_v": coeff_0,
                "coeff_l": coeff_1,
            }
        }

        wandb.log(wandb_output, step=self.agent.logs["current_step"])

        self._equalize_gradients(coeff_0=coeff_0, coeff_1=coeff_1)

    def _equalize_gradients(self, coeff_0=1, coeff_1=1):
        if not self.agent.config.model.args.bias_infusion.use:
            return

        if self.agent.config.model.args.bias_infusion.starting_epoch <= self.agent.logs["current_epoch"] <= self.agent.config.model.args.bias_infusion.ending_epoch:

            for name, parms in self.agent.model.named_parameters():
                if parms.grad is None:
                    continue
                if "mod0" in name or "enc_0" in name or "fc_0_lin.weight" in name:
                    parms.grad *= coeff_0
                if "mod1" in name or "enc_1" in name or "fc_1_lin.weight" in name:
                    parms.grad *= coeff_1


class Bias_Infusion_MLB_3d_Reg(General_Bias_Infusion):
    def __init__(self, agent):
        super(Bias_Infusion_MLB_3d_Reg, self).__init__(agent)
        logging.info("Bias Infusion MLB is being employed")
        self._initialize_logs_n_utils()

    def _initialize_logs_n_utils(self):
        self.criterion = nn.CrossEntropyLoss()
        self.tanh = nn.Tanh()
        self.agent.logs["ratio_logs"] = {
            "ratio_0": [],
            "ratio_1": [],
            "ratio_2": [],
            "coeff_v": [],
            "coeff_l": [],
            "coeff_f": [],
        }
        self.balance_mode = self.agent.config.model.args.bias_infusion.get("balance_mode", False)
        # self.balance_mode = self.agent.config.model.args.bias_infusion.get("balance_mode", "whatev")
        self.alpha = self.agent.config.model.args.bias_infusion.alpha
        self.tanh_mode = self.agent.config.model.args.bias_infusion.get("tanh_mode", False)
        self.tanh_mode_beta = self.agent.config.model.args.bias_infusion.get("tanh_mode_beta", False)

        if self.tanh_mode == 2:
            print("Tanh mode 2 is being employed with beta {}".format(self.tanh_mode_beta))

    # def before_backward(self, total, output_losses, **kwargs):
    #
    #     opt_done = False
    #     if self.balance_mode == "balance_only_multi":
    #         (output_losses["ce_loss_combined"]).backward(retain_graph=True)
    #         # (output_losses["ce_loss_combined"] + self.output_losses["ce_loss_c"] + self.output_losses["ce_loss_g"]).backward(retain_graph=True)
    #         self.output_losses = output_losses
    #         opt_done = True
    #
    #     return total, output_losses, opt_done

    def on_backward_end(self, label, out_color, out_gray, out_f):

        score_0 = F.mse_loss(out_color.squeeze(), label)
        score_1 = F.mse_loss(out_gray.squeeze(), label)
        score_2 = F.mse_loss(out_f.squeeze(), label)

        ratio_0 = score_0 / ((score_1 + score_2) / 2)
        ratio_1 = score_1 / ((score_0 + score_2) / 2)
        ratio_2 = score_2 / ((score_0 + score_1) / 2)

        coeff_0 = 1 + self.tanh(self.alpha * (ratio_0 - 1)).cpu().numpy()
        coeff_1 = 1 + self.tanh(self.alpha * (ratio_1 - 1)).cpu().numpy()
        coeff_2 = 1 + self.tanh(self.alpha * (ratio_2 - 1)).cpu().numpy()

        if self.tanh_mode == 2:
            if ratio_0 > 1:
                coeff_0 = 1 + (self.tanh_mode_beta - 1) * self.tanh(self.alpha * (ratio_0 - 1))
            else:
                coeff_0 = 1 + self.tanh(self.alpha * (ratio_0 - 1))
            if ratio_1 > 1:
                coeff_1 = 1 + (self.tanh_mode_beta - 1) * self.tanh(self.alpha * (ratio_1 - 1))
            else:
                coeff_1 = 1 + self.tanh(self.alpha * (ratio_1 - 1))
            if ratio_2 > 1:
                coeff_2 = 1 + (self.tanh_mode_beta - 1) * self.tanh(self.alpha * (ratio_2 - 1))
            else:
                coeff_2 = 1 + self.tanh(self.alpha * (ratio_2 - 1))

        self.agent.logs["ratio_logs"]["ratio_0"].append(ratio_0.cpu().numpy())
        self.agent.logs["ratio_logs"]["ratio_1"].append(ratio_1.cpu().numpy())
        self.agent.logs["ratio_logs"]["ratio_2"].append(ratio_2.cpu().numpy())
        self.agent.logs["ratio_logs"]["coeff_v"].append(coeff_0)
        self.agent.logs["ratio_logs"]["coeff_l"].append(coeff_1)
        self.agent.logs["ratio_logs"]["coeff_f"].append(coeff_2)

        wandb_output = {
            "ratio": {
                "ratio_0": ratio_0.cpu().numpy(),
                "ratio_1": ratio_1.cpu().numpy(),
                "ratio_2": ratio_2.cpu().numpy(),
                "coeff_v": coeff_0,
                "coeff_l": coeff_1,
                "coeff_f": coeff_2,
            }
        }

        wandb.log(wandb_output, step=self.agent.logs["current_step"])

        self._equalize_gradients(coeff_0=coeff_0, coeff_1=coeff_1, coeff_2=coeff_2)

    def _equalize_gradients(self, coeff_0=1, coeff_1=1, coeff_2=1):
        if not self.agent.config.model.args.bias_infusion.use:
            return

        if self.agent.config.model.args.bias_infusion.starting_epoch <= self.agent.logs["current_epoch"] <= self.agent.config.model.args.bias_infusion.ending_epoch:

            for name, parms in self.agent.model.named_parameters():
                if parms.grad is None:
                    continue
                if "mod0" in name or "enc_0" in name or "fc_0_lin.weight" in name:
                    parms.grad *= coeff_0
                if "mod1" in name or "enc_1" in name or "fc_1_lin.weight" in name:
                    parms.grad *= coeff_1
                if "mod2" in name or "enc_2" in name or "fc_2_lin.weight" in name:
                    parms.grad *= coeff_2


class Bias_Infusion_MSLR(General_Bias_Infusion):
    def __init__(self, agent):
        super(Bias_Infusion_MSLR, self).__init__(agent)
        self.agent.logger.info("Bias Infusion MSLR is used")

        self._initialize_logs_n_utils()

    def before_backward(self, total, output_losses, **kwargs):
        return output_losses["ce_loss_combined"], output_losses, False

    def _initialize_logs_n_utils(self):

        self.init_learning_rate = self.agent.config.model.args.bias_infusion.get("init_learning_rate", {"c": 1, "g": 1})
        self.coeff_memory = defaultdict(list)
        self.keep_memory_epoch = self.agent.config.model.args.bias_infusion.get("keep_memory_epoch", 5)
        self.starting_epoch = self.agent.config.model.args.bias_infusion.get("starting_epoch", 0)
        self.ending_epoch = self.agent.config.model.args.bias_infusion.get("ending_epoch", 1500)
        self.ratio = defaultdict(lambda: 1)
        self.agent.logs["ratio_logs"] = {
            "ratio_color": [],
            "ratio_gray": [],
            "coeff_color": [],
            "coeff_gray": [],
        }

    def on_epoch_begin(self, **kwargs):

        if self.agent.logs["current_epoch"] < 1:
            return

        acc_0 = self.agent.logs["val_logs"][list(self.agent.logs["val_logs"].keys())[-1]]["acc"]["c"]
        acc_1 = self.agent.logs["val_logs"][list(self.agent.logs["val_logs"].keys())[-1]]["acc"]["g"]

        if self.agent.logs["current_epoch"] > self.keep_memory_epoch:
            ratio_0 = acc_0 / np.array(self.coeff_memory[0][-self.keep_memory_epoch :]).mean()
            ratio_1 = acc_1 / np.array(self.coeff_memory[1][-self.keep_memory_epoch :]).mean()
        else:
            ratio_0, ratio_1 = 1, 1

        self.ratio[0] = ratio_0
        self.ratio[1] = ratio_1

        self.coeff_memory[0].append(acc_0)
        self.coeff_memory[1].append(acc_1)

    def on_backward_end(self, label, preds):
        coeff_0 = self.ratio[0] * self.init_learning_rate["c"]
        coeff_1 = self.ratio[1] * self.init_learning_rate["g"]

        self._equalize_gradients(coeff_0=coeff_0, coeff_1=coeff_1)

        self.agent.logs["ratio_logs"]["ratio_gray"].append(self.ratio[0])
        self.agent.logs["ratio_logs"]["ratio_color"].append(self.ratio[1])
        self.agent.logs["ratio_logs"]["coeff_color"].append(coeff_0)
        self.agent.logs["ratio_logs"]["coeff_gray"].append(coeff_1)
        wandb.log({"ratio": self.agent.logs["ratio_logs"]})

    def _equalize_gradients(self, coeff_0=1, coeff_1=1):
        if not self.agent.config.model.args.bias_infusion.use:
            return
        if self.starting_epoch <= self.agent.logs["current_epoch"] <= self.ending_epoch:
            for name, params in self.agent.model.named_parameters():
                if params.grad is None:
                    continue
                if "mod0" in name or "enc_0" in name:
                    params.grad *= coeff_0
                if "mod1" in name or "enc_1" in name:
                    params.grad *= coeff_1


class Bias_Infusion_AGM(General_Bias_Infusion):
    def __init__(self, agent):
        super(Bias_Infusion_AGM, self).__init__(agent)

        self.agent.logger.info("Bias_Infusion_AGM is used")
        self._initialize_logs_n_utils()

    def _initialize_logs_n_utils(self):
        self.criterion = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        self.bias_infuser = self.agent.config.model.args.get("bias_infusion", {})
        self.train_score_a, self.train_score_v = 0, 0
        self.ra_score_a, self.ra_score_v = 0, 0
        self.agent.logs["ratio_logs"] = {
            "ratio_color": [],
            "ratio_color_ra": [],
            "ratio_gray": [],
            "ratio_gray_ra": [],
            "coeff_color": [],
            "coeff_gray": [],
        }

    def before_backward(self, total, output_losses, label, output, data, **kwargs):

        # if self.bias_infuser.get("starting_epoch",0) <= self.agent.logs["current_epoch"] <= self.bias_infuser.get("ending_epoch",1500):
        #     return total, output_losses, False

        # self.agent.model.eval()
        # if 0 in data and 1 in data:
        #     this_data = copy.deepcopy(data)
        #     this_data[0] = torch.zeros_like(this_data[0])
        #     pad_audio = self.agent.model(this_data, return_features=True)["preds"]["combined"]
        #     this_data = copy.deepcopy(data)
        #     this_data[1] = torch.zeros_like(this_data[1])
        #     pad_video = self.agent.model(this_data, return_features=True)["preds"]["combined"]
        # elif 2 in data and 3 in data:
        #     this_data = copy.deepcopy(data)
        #     this_data[2] = torch.zeros_like(this_data[2])
        #     pad_audio = self.agent.model(this_data, return_features=True)["preds"]["combined"]
        #     this_data = copy.deepcopy(data)
        #     this_data[3] = torch.zeros_like(this_data[3])
        #
        #     pad_video = self.agent.model(this_data, return_features=True)["preds"]["combined"]
        #
        # self.agent.model.train()

        # output["preds"]["c"] = 0.5 * (output["preds"]["combined"] - pad_audio + pad_video).detach()
        # output["preds"]["g"] = 0.5 * (output["preds"]["combined"] - pad_video + pad_audio).detach()
        # del pad_audio, pad_video, this_data

        out_color = output["preds"]["c"]
        out_gray = output["preds"]["g"]

        if len(label.shape) > 1:
            label = label.flatten()
        score_audio = 0.0
        score_visual = 0.0
        for k in range(out_color.size(0)):
            if torch.isinf(self.softmax(out_color)[k][label[k]]) or self.softmax(out_color)[k][label[k]] < 1e-8:
                score_audio += -torch.log(torch.tensor(1e-8, dtype=out_color.dtype, device=out_color.device))
            else:
                score_audio += -torch.log(self.softmax(out_color)[k][label[k]])
            if torch.isinf(self.softmax(out_gray)[k][label[k]]) or self.softmax(out_gray)[k][label[k]] < 1e-8:
                score_visual += -torch.log(torch.tensor(1e-8, dtype=out_gray.dtype, device=out_gray.device))
            else:
                score_visual += -torch.log(self.softmax(out_gray)[k][label[k]])

        score_audio = score_audio / out_color.size(0)
        score_visual = score_visual / out_gray.size(0)

        r_a = math.exp((score_visual.item() - score_audio.item()))
        r_v = math.exp((score_audio.item() - score_visual.item()))

        optimal_ratio_a = math.exp((self.train_score_v - self.train_score_a))
        optimal_ratio_v = math.exp((self.train_score_a - self.train_score_v))

        coeff_a = math.exp(self.bias_infuser.alpha * (min(optimal_ratio_a - r_a, 10)))
        coeff_v = math.exp(self.bias_infuser.alpha * (min(optimal_ratio_v - r_v, 10)))

        # Shouldnt this go above optimal_ratio?
        iteration = self.agent.logs["current_step"]
        self.train_score_a = self.train_score_a * iteration / (iteration + 1) + score_audio.item() / (iteration + 1)
        self.train_score_v = self.train_score_v * iteration / (iteration + 1) + score_visual.item() / (iteration + 1)

        self.ra_score_a = self.ra_score_a * iteration / (iteration + 1) + score_audio.item() / (iteration + 1)
        self.ra_score_v = self.ra_score_v * iteration / (iteration + 1) + score_visual.item() / (iteration + 1)

        self.agent.logs["ratio_logs"]["ratio_color"].append(self.train_score_a)
        self.agent.logs["ratio_logs"]["ratio_color_ra"].append(self.ra_score_a)
        self.agent.logs["ratio_logs"]["ratio_gray"].append(self.train_score_v)
        self.agent.logs["ratio_logs"]["ratio_gray_ra"].append(self.ra_score_v)
        self.agent.logs["ratio_logs"]["coeff_color"].append(coeff_a)
        self.agent.logs["ratio_logs"]["coeff_gray"].append(coeff_v)

        wandb_output = {
            "ratio": {
                "ratio_gray": self.train_score_a,
                "ratio_color": self.train_score_v,
                "ratio_gray_ra": self.ra_score_a,
                "ratio_color_ra": self.ra_score_v,
                "coeff_color": coeff_a,
                "coeff_gray": coeff_v,
            }
        }

        wandb.log(wandb_output, step=self.agent.logs["current_step"])

        self.agent.model.update_scale(coeff_a, coeff_v)

        return total, output_losses, False


class Bias_Infusion_AGM_reg(General_Bias_Infusion):
    def __init__(self, agent):
        super(Bias_Infusion_AGM_reg, self).__init__(agent)

        self.agent.logger.info("Bias_Infusion_AGM is used")
        self._initialize_logs_n_utils()

    def _initialize_logs_n_utils(self):
        self.criterion = nn.MSELoss()
        self.bias_infuser = self.agent.config.model.args.get("bias_infusion", {})
        self.train_score_a, self.train_score_v = 0, 0
        self.ra_score_a, self.ra_score_v = 0, 0
        self.agent.logs["ratio_logs"] = {
            "ratio_color": [],
            "ratio_color_ra": [],
            "ratio_gray": [],
            "ratio_gray_ra": [],
            "coeff_color": [],
            "coeff_gray": [],
        }

    def before_backward(self, total, output_losses, label, output, data, **kwargs):
        out_color = output["preds"]["c"]
        out_gray = output["preds"]["g"]

        score_visual = self.criterion(out_gray.squeeze(), label)
        score_audio = self.criterion(out_color.squeeze(), label)

        score_audio = score_audio / out_color.size(0)
        score_visual = score_visual / out_gray.size(0)

        score_visual = 1 / score_visual
        score_audio = 1 / score_audio

        mean_score = (score_visual.item() + score_audio.item()) / 2

        r_v = math.exp((mean_score - score_visual.item()))
        r_a = math.exp((mean_score - score_audio.item()))

        optimal_ratio_a = math.exp((self.train_score_v - self.train_score_a))
        optimal_ratio_v = math.exp((self.train_score_a - self.train_score_v))

        coeff_a = math.exp(self.bias_infuser.alpha * (min(optimal_ratio_a - r_a, 10)))
        coeff_v = math.exp(self.bias_infuser.alpha * (min(optimal_ratio_v - r_v, 10)))

        # Shouldnt this go above optimal_ratio?
        iteration = self.agent.logs["current_step"]
        self.train_score_a = self.train_score_a * iteration / (iteration + 1) + score_audio.item() / (iteration + 1)
        self.train_score_v = self.train_score_v * iteration / (iteration + 1) + score_visual.item() / (iteration + 1)

        self.ra_score_a = self.ra_score_a * iteration / (iteration + 1) + score_audio.item() / (iteration + 1)
        self.ra_score_v = self.ra_score_v * iteration / (iteration + 1) + score_visual.item() / (iteration + 1)

        self.agent.logs["ratio_logs"]["ratio_color"].append(self.train_score_a)
        self.agent.logs["ratio_logs"]["ratio_color_ra"].append(self.ra_score_a)
        self.agent.logs["ratio_logs"]["ratio_gray"].append(self.train_score_v)
        self.agent.logs["ratio_logs"]["ratio_gray_ra"].append(self.ra_score_v)
        self.agent.logs["ratio_logs"]["coeff_color"].append(coeff_a)
        self.agent.logs["ratio_logs"]["coeff_gray"].append(coeff_v)

        wandb_output = {
            "ratio": {
                "ratio_gray": self.train_score_a,
                "ratio_color": self.train_score_v,
                "ratio_gray_ra": self.ra_score_a,
                "ratio_color_ra": self.ra_score_v,
                "coeff_color": coeff_a,
                "coeff_gray": coeff_v,
            }
        }

        wandb.log(wandb_output, step=self.agent.logs["current_step"])

        self.agent.model.update_scale(coeff_a, coeff_v)

        return total, output_losses, False


class Bias_Infusion_AGM_3mod(General_Bias_Infusion):
    def __init__(self, agent):
        super(Bias_Infusion_AGM_3mod, self).__init__(agent)

        self.agent.logger.info("Bias_Infusion_AGM is used")
        self._initialize_logs_n_utils()

    def _initialize_logs_n_utils(self):
        self.criterion = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        self.bias_infuser = self.agent.config.model.args.get("bias_infusion", {})
        self.train_score_a, self.train_score_v, self.train_score_f = 0, 0, 0
        self.ra_score_a, self.ra_score_v = 0, 0
        self.agent.logs["ratio_logs"] = {
            "ratio_color": [],
            "ratio_gray": [],
            "ratio_flow": [],
            "coeff_color": [],
            "coeff_gray": [],
            "coeff_flow": [],
        }

    def before_backward(self, total, output_losses, label, output, data, **kwargs):

        out_color = output["preds"]["c"]
        out_gray = output["preds"]["g"]
        out_flow = output["preds"]["f"]

        score_visual = 0.0
        score_audio = 0.0
        score_flow = 0.0

        for k in range(out_color.size(0)):
            if torch.isinf(self.softmax(out_color)[k][label[k]]) or self.softmax(out_color)[k][label[k]] < 1e-8:
                score_audio += -torch.log(torch.tensor(1e-8, dtype=out_color.dtype, device=out_color.device))
            else:
                score_audio += -torch.log(self.softmax(out_color)[k][label[k]])
            if torch.isinf(self.softmax(out_gray)[k][label[k]]) or self.softmax(out_gray)[k][label[k]] < 1e-8:
                score_visual += -torch.log(torch.tensor(1e-8, dtype=out_gray.dtype, device=out_gray.device))
            else:
                score_visual += -torch.log(self.softmax(out_gray)[k][label[k]])

            if torch.isinf(self.softmax(out_flow)[k][label[k]]) or self.softmax(out_flow)[k][label[k]] < 1e-8:
                score_flow += -torch.log(torch.tensor(1e-8, dtype=out_flow.dtype, device=out_gray.device))
            else:
                score_flow += -torch.log(self.softmax(out_flow)[k][label[k]])

        score_audio = score_audio / out_color.size(0)
        score_visual = score_visual / out_gray.size(0)
        score_flow = score_flow / out_flow.size(0)

        mean_score = (score_visual.item() + score_audio.item() + score_flow.item()) / 3

        r_v = math.exp((mean_score - score_visual.item()) * 3 / 2)
        r_a = math.exp((mean_score - score_audio.item()) * 3 / 2)
        r_f = math.exp((mean_score - score_flow.item()) * 3 / 2)

        iteration = self.agent.logs["current_step"]

        self.train_score_a = self.train_score_a * iteration / (iteration + 1) + score_audio.item() / (iteration + 1)
        self.train_score_v = self.train_score_v * iteration / (iteration + 1) + score_visual.item() / (iteration + 1)
        self.train_score_f = self.train_score_f * iteration / (iteration + 1) + score_flow.item() / (iteration + 1)

        optimal_mean_score = (self.train_score_v + self.train_score_a + self.train_score_f) / 3
        optimal_ratio_a = math.exp((optimal_mean_score - self.train_score_a) * 3 / 2)
        optimal_ratio_v = math.exp((optimal_mean_score - self.train_score_v) * 3 / 2)
        optimal_ratio_f = math.exp((optimal_mean_score - self.train_score_f) * 3 / 2)

        coeff_a = math.exp(self.bias_infuser.alpha * (min(optimal_ratio_a - r_a, 7)))
        coeff_v = math.exp(self.bias_infuser.alpha * (min(optimal_ratio_v - r_v, 7)))
        coeff_f = math.exp(self.bias_infuser.alpha * (min(optimal_ratio_f - r_f, 7)))

        self.agent.logs["ratio_logs"]["ratio_color"].append(self.train_score_a)
        self.agent.logs["ratio_logs"]["ratio_gray"].append(self.train_score_v)
        self.agent.logs["ratio_logs"]["ratio_gray"].append(self.train_score_f)

        self.agent.logs["ratio_logs"]["coeff_color"].append(coeff_a)
        self.agent.logs["ratio_logs"]["coeff_gray"].append(coeff_v)
        self.agent.logs["ratio_logs"]["coeff_flow"].append(coeff_f)

        wandb.log(
            {
                "coeff_v": coeff_a,
                "coeff_l": coeff_v,
                "coeff_f": coeff_f,
                "running_ratio_v": self.train_score_a,
                "running_ratio_l": self.train_score_v,
                "running_ratio_f": self.train_score_f,
            },
            step=self.agent.logs["current_step"],
        )

        self.agent.model.update_scale(coeff_c=coeff_a, coeff_g=coeff_v, coeff_f=coeff_f)

        return total, output_losses, False


class Bias_Infusion_AGM_3mod_reg(General_Bias_Infusion):
    def __init__(self, agent):
        super(Bias_Infusion_AGM_3mod_reg, self).__init__(agent)

        self.agent.logger.info("Bias_Infusion_AGM is used")
        self._initialize_logs_n_utils()

    def _initialize_logs_n_utils(self):
        self.criterion = nn.MSELoss()
        self.bias_infuser = self.agent.config.model.args.get("bias_infusion", {})
        self.train_score_a, self.train_score_v, self.train_score_f = 0, 0, 0
        self.ra_score_a, self.ra_score_v = 0, 0
        self.agent.logs["ratio_logs"] = {
            "ratio_color": [],
            "ratio_gray": [],
            "ratio_flow": [],
            "coeff_color": [],
            "coeff_gray": [],
            "coeff_flow": [],
        }

    def before_backward(self, total, output_losses, label, output, data, **kwargs):
        out_color = output["preds"]["c"]
        out_gray = output["preds"]["g"]
        out_flow = output["preds"]["f"]

        score_visual = self.criterion(out_gray.squeeze(), label)
        score_audio = self.criterion(out_color.squeeze(), label)
        score_flow = self.criterion(out_flow.squeeze(), label)

        score_audio = score_audio / out_color.size(0)
        score_visual = score_visual / out_gray.size(0)
        score_flow = score_flow / out_flow.size(0)

        score_visual = 1 / score_visual
        score_audio = 1 / score_audio
        score_flow = 1 / score_flow

        mean_score = (score_visual.item() + score_audio.item() + score_flow.item()) / 3

        r_v = math.exp((mean_score - score_visual.item()) * 3 / 2)
        r_a = math.exp((mean_score - score_audio.item()) * 3 / 2)
        r_f = math.exp((mean_score - score_flow.item()) * 3 / 2)

        iteration = self.agent.logs["current_step"]

        self.train_score_a = self.train_score_a * iteration / (iteration + 1) + score_audio.item() / (iteration + 1)
        self.train_score_v = self.train_score_v * iteration / (iteration + 1) + score_visual.item() / (iteration + 1)
        self.train_score_f = self.train_score_f * iteration / (iteration + 1) + score_flow.item() / (iteration + 1)

        optimal_mean_score = (self.train_score_v + self.train_score_a + self.train_score_f) / 3
        optimal_ratio_a = math.exp((optimal_mean_score - self.train_score_a) * 3 / 2)
        optimal_ratio_v = math.exp((optimal_mean_score - self.train_score_v) * 3 / 2)
        optimal_ratio_f = math.exp((optimal_mean_score - self.train_score_f) * 3 / 2)

        coeff_a = math.exp(self.bias_infuser.alpha * (min(optimal_ratio_a - r_a, 7)))
        coeff_v = math.exp(self.bias_infuser.alpha * (min(optimal_ratio_v - r_v, 7)))
        coeff_f = math.exp(self.bias_infuser.alpha * (min(optimal_ratio_f - r_f, 7)))

        self.agent.logs["ratio_logs"]["ratio_color"].append(self.train_score_a)
        self.agent.logs["ratio_logs"]["ratio_gray"].append(self.train_score_v)
        self.agent.logs["ratio_logs"]["ratio_gray"].append(self.train_score_f)

        self.agent.logs["ratio_logs"]["coeff_color"].append(coeff_a)
        self.agent.logs["ratio_logs"]["coeff_gray"].append(coeff_v)
        self.agent.logs["ratio_logs"]["coeff_flow"].append(coeff_f)

        wandb.log(
            {
                "coeff_v": coeff_a,
                "coeff_l": coeff_v,
                "coeff_f": coeff_f,
                "running_ratio_v": self.train_score_a,
                "running_ratio_l": self.train_score_v,
                "running_ratio_f": self.train_score_f,
            },
            step=self.agent.logs["current_step"],
        )

        self.agent.model.update_scale(coeff_c=coeff_a, coeff_g=coeff_v, coeff_f=coeff_f)

        return total, output_losses, False


def nt_xent_loss(z_x, z_y, z_k=None, label=None, temperature=0.5):
    z_x = F.normalize(z_x, p=2, dim=1)
    z_y = F.normalize(z_y, p=2, dim=1)
    z = torch.cat([z_x, z_y], dim=0)
    if label is not None:
        labels = torch.cat([label, label], dim=0)
    else:
        labels = torch.cat([torch.arange(z_x.shape[0]), torch.arange(z_y.shape[0])], dim=0)
    if z_k is not None:
        z_k = F.normalize(z_k, p=2, dim=1)
        z = torch.cat([z, z_k], dim=0)
        if label is not None:
            labels = torch.cat([label, label, label], dim=0)
        else:
            labels = torch.cat(
                [
                    torch.arange(z_x.shape[0]),
                    torch.arange(z_y.shape[0]),
                    torch.arange(z_k.shape[0]),
                ],
                dim=0,
            )

    loss = NTXentLoss(temperature=temperature)(z, labels)

    return loss


class Bias_Infusion_MCR(General_Bias_Infusion):
    """
    NGMine = Norm-Gradient my version
    """

    def __init__(self, agent):
        super(Bias_Infusion_MCR, self).__init__(agent)

        self._initialize_logs_n_utils()

    def _initialize_logs_n_utils(self):
        self.losses = []
        self.agent.logs["reg_logs"] = defaultdict(list)
        self.agent.logs["ratio_logs"] = defaultdict(list)
        self.agent.logs["ratio_logs"] = defaultdict(list)
        self.bias_infuser = self.agent.config.model.args.get("bias_infusion", {})
        self.reg_by = self.bias_infuser.get("reg_by", "dist_pred")
        self.l = self.bias_infuser.get("l", 0)
        self.contr_coeff = self.bias_infuser.get("contr_coeff", False)
        self.temperature = self.bias_infuser.get("temperature", 0.5)

    def get_perturbed_gradients(self, output):

        def log_likelihood_ratio(logits_Z, logits_Z_hat):
            """
            Compute the negative average log likelihood ratio between two sets of logits.

            Args:
            - logits_Z (torch.Tensor): Logits from predictions using (X, Z), shape (batch_size, num_classes)
            - logits_Z_hat (torch.Tensor): Logits from predictions using (X, Z_hat), shape (batch_size, num_classes)
            - targets (torch.Tensor): True class labels, shape (batch_size,)

            Returns:
            - torch.Tensor: Difference of negative average log likelihood ratios
            """
            # Compute probabilities from logits

            probs_Z = F.softmax(logits_Z, dim=1)
            probs_Z_hat = F.softmax(logits_Z_hat, dim=1)

            # label_norm = targets.repeat(len(pert_data))[ids_of_each_sample]

            # torch.concatenate([targets[pert_data[i]["data"]] for i in range(len(pert_data))], dim=0)

            # Select the probabilities corresponding to the true labels
            # target_probs_Z_hat = probs_Z_hat[range(len(label_norm)), label_norm]
            # target_probs_Z = probs_Z[range(len(targets)), targets]

            # Compute log probabilities
            # log_probs_Z = torch.log(target_probs_Z + 1e-9)  # Add epsilon to avoid log(0)
            # log_probs_Z_hat = torch.log(target_probs_Z_hat + 1e-9)  # Add epsilon to avoid log(0)

            # Compute log probabilities
            # log_probs_Z = torch.log(probs_Z + 1e-7)  # Add epsilon to avoid log(0)
            # log_probs_Z_hat = torch.log(probs_Z_hat + 1e-7)  # Add epsilon to avoid log(0)
            log_probs_Z = F.log_softmax(logits_Z, dim=1)
            log_probs_Z_hat = F.log_softmax(logits_Z_hat, dim=1)

            per_sample_avg_log_prob_Z_hat = torch.concatenate([(log_probs_Z_hat * probs_Z_hat).mean().unsqueeze(0)])
            per_sample_avg_log_prob_Z_hat_onlylog = torch.concatenate([(log_probs_Z_hat).mean().unsqueeze(0)])

            #     print(log_probs_Z[ids_of_each_sample==i].mean())
            # # Compute the separate empirical averages
            # avg_log_prob_Z = torch.concat([log_probs_Z[ids_of_each_sample==i].mean() for i in len(pert_data[0]["data"])])
            avg_log_prob_Z = -torch.mean(log_probs_Z * probs_Z)
            avg_log_prob_Z_hat = -torch.mean(per_sample_avg_log_prob_Z_hat)

            avg_log_prob_Z_hat_onlylog = -torch.mean(per_sample_avg_log_prob_Z_hat_onlylog)
            avg_log_prob_Z_onlylog = -torch.mean(log_probs_Z)

            # avg_log_prob_Z_hat = (per_sample_avg_log_prob_Z * per_sample_avg_prob_Z).mean()

            # Compute the difference of the negative average log likelihood ratios
            difference = -avg_log_prob_Z + avg_log_prob_Z_hat
            difference_onlylog = -avg_log_prob_Z_onlylog + avg_log_prob_Z_hat_onlylog

            return (
                difference_onlylog,
                avg_log_prob_Z_hat_onlylog,
                avg_log_prob_Z_onlylog,
            )

        def js_divergence(net_1_logits, net_2_logits):

            clip_value = 1e7

            net_1_probs = F.softmax(torch.clamp(net_1_logits, -clip_value, clip_value), dim=1)
            net_2_probs = F.softmax(torch.clamp(net_2_logits, -clip_value, clip_value), dim=1)

            total_m = 0.5 * (net_1_probs + net_2_probs)

            clip_value = 1e-20
            total_m = torch.clamp(total_m, clip_value, 1).log()

            # total_m = torch.clamp(total_m, clip_value, 1)
            net_1_probs = torch.clamp(net_1_probs, clip_value, 1)
            net_2_probs = torch.clamp(net_2_probs, clip_value, 1)

            loss = 0.0

            loss += F.kl_div(total_m, net_1_probs, reduction="batchmean")
            loss += F.kl_div(total_m, net_2_probs, reduction="batchmean")
            if torch.isnan(loss):
                raise Exception("NaN detected in loss computation")
            return 0.5 * loss

        def get_losses(
            predicted_logits_00,
            soft_labels_norm,
            cross_permod,
            label,
            label_shuffled,
            title,
        ):

            jsd_label = js_divergence(predicted_logits_00, label)
            jsd_pred = js_divergence(predicted_logits_00, soft_labels_norm)
            loglike_pred, entropy_y, norm_ent = log_likelihood_ratio(soft_labels_norm, predicted_logits_00)

            cross_permod["{}_yent".format(title)].append(entropy_y)
            cross_permod["{}_nll".format(title)].append(loglike_pred)
            cross_permod["{}_label".format(title)].append(jsd_label)
            cross_permod["{}_pred".format(title)].append(jsd_pred)

            return cross_permod

        cross_permod = defaultdict(list)
        modimport_permod = {"enc_0": [], "enc_1": []}

        if "sa_detv" in output["preds"] and "sa_deta" in output["preds"] and "sa" in output["preds"]:
            label = F.one_hot(
                output["preds"]["n_label"],
                num_classes=self.agent.config.model.args.num_classes,
            )
            label_shuffled = F.one_hot(
                output["preds"]["n_label_shuffled"],
                num_classes=self.agent.config.model.args.num_classes,
            )

            cross_permod = get_losses(
                output["preds"]["sa_detv"],
                output["preds"]["ncombined"],
                cross_permod,
                label=label,
                label_shuffled=label_shuffled,
                title="enc0_p0",
            )
            cross_permod = get_losses(
                output["preds"]["sa_deta"],
                output["preds"]["ncombined"],
                cross_permod,
                label=label,
                label_shuffled=label_shuffled,
                title="enc1_p0",
            )
            cross_permod = get_losses(
                output["preds"]["sa"],
                output["preds"]["ncombined"],
                cross_permod,
                label=label,
                label_shuffled=label_shuffled,
                title="all_p0",
            )
            norm_acc = (torch.argmax(output["preds"]["ncombined"], dim=1) == output["preds"]["n_label"]).float().mean()
            acc_pert0 = (torch.argmax(output["preds"]["sa_detv"], dim=1) == output["preds"]["n_label"]).float().mean()
        else:
            cross_permod["enc0_p0_pred"].append(torch.tensor(0.0).to(self.agent.device))
            cross_permod["enc1_p0_pred"].append(torch.tensor(0.0).to(self.agent.device))
            cross_permod["all_p0_pred"].append(torch.tensor(0.0).to(self.agent.device))
            cross_permod["enc0_p0_nll"].append(torch.tensor(0.0).to(self.agent.device))
            cross_permod["enc1_p0_nll"].append(torch.tensor(0.0).to(self.agent.device))

            norm_acc = torch.tensor(0.0).to(self.agent.device)
            acc_pert0 = torch.tensor(0.0).to(self.agent.device)

        if "sv_detv" in output["preds"] and "sv_deta" in output["preds"] and "sv" in output["preds"]:
            label = F.one_hot(
                output["preds"]["n_label"],
                num_classes=self.agent.config.model.args.num_classes,
            )
            label_shuffled = F.one_hot(
                output["preds"]["n_label_shuffled"],
                num_classes=self.agent.config.model.args.num_classes,
            )

            cross_permod = get_losses(
                output["preds"]["sv_deta"],
                output["preds"]["ncombined"],
                cross_permod,
                label=label,
                label_shuffled=label_shuffled,
                title="enc1_p1",
            )
            cross_permod = get_losses(
                output["preds"]["sv_detv"],
                output["preds"]["ncombined"],
                cross_permod,
                label=label,
                label_shuffled=label_shuffled,
                title="enc0_p1",
            )
            cross_permod = get_losses(
                output["preds"]["sv"],
                output["preds"]["ncombined"],
                cross_permod,
                label=label,
                label_shuffled=label_shuffled,
                title="all_p1",
            )
            acc_pert1 = (torch.argmax(output["preds"]["sv_deta"], dim=1) == output["preds"]["n_label"]).float().mean()
        else:
            cross_permod["enc0_p1_pred"].append(torch.tensor(0.0).to(self.agent.device))
            cross_permod["enc1_p1_pred"].append(torch.tensor(0.0).to(self.agent.device))
            cross_permod["all_p1_pred"].append(torch.tensor(0.0).to(self.agent.device))
            cross_permod["enc0_p1_nll"].append(torch.tensor(0.0).to(self.agent.device))
            cross_permod["enc1_p1_nll"].append(torch.tensor(0.0).to(self.agent.device))
            acc_pert1 = torch.tensor(0.0).to(self.agent.device)

        modimport_permod["enc_0"].append(acc_pert1 / norm_acc)
        modimport_permod["enc_1"].append(acc_pert0 / norm_acc)

        out = {name: torch.stack(cross_permod[name]).mean() for name in cross_permod}

        modimport_permod = {name: torch.stack(modimport_permod[name]).mean() for name in modimport_permod}

        return out, modimport_permod

    def before_backward(self, total, output_losses, w_loss, loss_fun, data, label, output, **kwargs):

        if not (self.bias_infuser.get("starting_epoch", 0) <= self.agent.logs["current_epoch"] <= self.bias_infuser.get("ending_epoch", 1000)):
            return total, output_losses, False

        num_samples = self.agent.config.model.args.bias_infusion.num_samples

        pert_loss, import_permod = self.get_perturbed_gradients(output)

        l = self.l

        if self.reg_by == "dist_pred_cjsd":
            reg_term = self.l * (-pert_loss["enc0_p0_pred"] + pert_loss["enc1_p0_pred"] - pert_loss["enc1_p1_pred"] + pert_loss["enc0_p1_pred"])
            reg_term_0 = pert_loss["enc0_p0_pred"]
            reg_term_1 = pert_loss["enc1_p1_pred"]
        elif self.reg_by == "dist_pred_max_cjsd":
            reg_term = self.l * (-pert_loss["enc0_p0_pred"] - pert_loss["enc1_p1_pred"])
            reg_term_0 = pert_loss["enc0_p0_pred"]
            reg_term_1 = pert_loss["enc1_p1_pred"]
        elif self.reg_by == "dist_pred_all_max_cjsd":
            reg_term_0 = pert_loss["all_p0_pred"]
            reg_term_1 = pert_loss["all_p1_pred"]
            reg_term = self.l * (-reg_term_0 - reg_term_1)

        if self.contr_coeff:
            if self.bias_infuser.get("contr_type", "label") == "label":
                if len(label.shape) > 1:
                    label_partitions = 30
                    label = label.flatten()
                    cca_loss = []
                    for each_part in range(label_partitions):
                        from_id = int(len(label) / label_partitions) * each_part
                        to_id = int(len(label) / label_partitions) * (each_part + 1)
                        if each_part == label_partitions - 1:
                            to_id = len(label)
                        cca_loss.append(
                            nt_xent_loss(
                                output["features"]["c"][from_id:to_id],
                                output["features"]["g"][from_id:to_id],
                                label=label[from_id:to_id],
                                temperature=self.temperature,
                            ).unsqueeze(0)
                        )
                    cca_loss = torch.cat(cca_loss).mean()
                else:
                    # label = label.flatten()
                    if "f" in output["features"]:
                        cca_loss = nt_xent_loss(
                            output["features"]["c"],
                            output["features"]["g"],
                            output["features"]["f"],
                            label=label,
                            temperature=self.temperature,
                        )
                    else:
                        cca_loss = nt_xent_loss(
                            output["features"]["c"],
                            output["features"]["g"],
                            label=label,
                            temperature=self.temperature,
                        )

                # else:
                #
                #     cca_loss = self.cca_loss_fun.loss(output["features"]["c"]["features"]["combined"],output["features"]["g"]["features"]["combined"])
            elif self.bias_infuser.get("contr_type", "label") == "pure":
                if "f" in output["features"]:
                    cca_loss = nt_xent_loss(
                        output["features"]["c"],
                        output["features"]["g"],
                        output["features"]["f"],
                        temperature=self.temperature,
                    )
                else:
                    cca_loss = nt_xent_loss(
                        output["features"]["c"],
                        output["features"]["g"],
                        temperature=self.temperature,
                    )

            cca_loss_reg = self.contr_coeff * cca_loss

        reg_terms = {}

        if "reg_term_0" in locals() and "reg_term_1" in locals():
            reg_terms["reg_term_0"] = reg_term_0
            reg_terms["reg_term_1"] = reg_term_1
            output_losses["reg_term_0"] = reg_term_0
            output_losses["reg_term_1"] = reg_term_1

        wandb.log(
            {"perturb": pert_loss, "reg_terms": reg_terms},
            step=self.agent.logs["current_step"] + 1,
        )

        return total, output_losses, False


class Bias_Infusion_MCR3D(General_Bias_Infusion):
    """
    NGMine = Norm-Gradient my version
    """

    def __init__(self, agent):
        super(Bias_Infusion_MCR3D, self).__init__(agent)

        self._initialize_logs_n_utils()

    def _initialize_logs_n_utils(self):
        self.losses = []
        self.agent.logs["reg_logs"] = defaultdict(list)
        self.agent.logs["ratio_logs"] = defaultdict(list)
        self.agent.logs["ratio_logs"] = defaultdict(list)
        self.bias_infuser = self.agent.config.model.args.get("bias_infusion", {})
        self.reg_by = self.bias_infuser.get("reg_by", "dist_pred")
        self.l = self.bias_infuser.get("l", 0)
        self.task = self.bias_infuser.get("task", "reg")
        self.contr_coeff = self.bias_infuser.get("contr_coeff", False)
        self.temperature = self.bias_infuser.get("temperature", 0.5)

    def _perturb_tensor(
        self,
        tens: dict,
        n_samples: int,
        label: torch.Tensor,
        shuffle: bool = True,
        noise: bool = True,
    ) -> torch.Tensor:

        batch_size = tens[0].shape[0] if 0 in tens else tens[2].shape[0]

        final_data = []
        while len(final_data) < n_samples:

            if shuffle:
                shuffle_idx_0 = torch.randperm(batch_size)
                shuffle_idx_1 = torch.randperm(batch_size)
                shuffle_idx_2 = torch.randperm(batch_size)
                nonequal_label_0 = torch.ones(batch_size, dtype=torch.bool)
                nonequal_label_1 = torch.ones(batch_size, dtype=torch.bool)
                nonequal_label_2 = torch.ones(batch_size, dtype=torch.bool)

                if nonequal_label_0.sum() <= 1 or nonequal_label_1.sum() <= 1 or nonequal_label_2.sum() <= 1:
                    continue
                shuffle_idx_0 = shuffle_idx_0[nonequal_label_0.cpu()]
                shuffle_idx_1 = shuffle_idx_1[nonequal_label_1.cpu()]
                shuffle_idx_2 = shuffle_idx_2[nonequal_label_2.cpu()]

            if nonequal_label_0.sum() <= 1 or nonequal_label_1.sum() <= 1 or nonequal_label_2.sum() <= 1:
                continue
            final_data.append(
                {
                    "shuffle_idx_0": shuffle_idx_0,
                    "shuffle_idx_1": shuffle_idx_1,
                    "shuffle_idx_2": shuffle_idx_2,
                    "data_0": nonequal_label_0,
                    "data_1": nonequal_label_1,
                    "data_2": nonequal_label_2,
                }
            )

        return final_data

    def get_perturbed_gradients(self, output):
        def log_likelihood_ratio(logits_Z, logits_Z_hat):
            """
            Compute the negative average log likelihood ratio between two sets of logits.

            Args:
            - logits_Z (torch.Tensor): Logits from predictions using (X, Z), shape (batch_size, num_classes)
            - logits_Z_hat (torch.Tensor): Logits from predictions using (X, Z_hat), shape (batch_size, num_classes)
            - targets (torch.Tensor): True class labels, shape (batch_size,)

            Returns:
            - torch.Tensor: Difference of negative average log likelihood ratios
            """
            # Compute probabilities from logits

            probs_Z = F.softmax(logits_Z, dim=1)
            probs_Z_hat = F.softmax(logits_Z_hat, dim=1)

            # label_norm = targets.repeat(len(pert_data))[ids_of_each_sample]

            # torch.concatenate([targets[pert_data[i]["data"]] for i in range(len(pert_data))], dim=0)

            # Select the probabilities corresponding to the true labels
            # target_probs_Z_hat = probs_Z_hat[range(len(label_norm)), label_norm]
            # target_probs_Z = probs_Z[range(len(targets)), targets]

            # Compute log probabilities
            # log_probs_Z = torch.log(target_probs_Z + 1e-9)  # Add epsilon to avoid log(0)
            # log_probs_Z_hat = torch.log(target_probs_Z_hat + 1e-9)  # Add epsilon to avoid log(0)

            # Compute log probabilities
            # log_probs_Z = torch.log(probs_Z + 1e-7)  # Add epsilon to avoid log(0)
            # log_probs_Z_hat = torch.log(probs_Z_hat + 1e-7)  # Add epsilon to avoid log(0)
            log_probs_Z = F.log_softmax(logits_Z, dim=1)
            log_probs_Z_hat = F.log_softmax(logits_Z_hat, dim=1)

            per_sample_avg_log_prob_Z_hat = torch.concatenate([(log_probs_Z_hat * probs_Z_hat).mean().unsqueeze(0)])
            per_sample_avg_log_prob_Z_hat_onlylog = torch.concatenate([(log_probs_Z_hat).mean().unsqueeze(0)])

            #     print(log_probs_Z[ids_of_each_sample==i].mean())
            # # Compute the separate empirical averages
            # avg_log_prob_Z = torch.concat([log_probs_Z[ids_of_each_sample==i].mean() for i in len(pert_data[0]["data"])])
            avg_log_prob_Z = -torch.mean(log_probs_Z * probs_Z)
            avg_log_prob_Z_hat = -torch.mean(per_sample_avg_log_prob_Z_hat)

            avg_log_prob_Z_hat_onlylog = -torch.mean(per_sample_avg_log_prob_Z_hat_onlylog)
            avg_log_prob_Z_onlylog = -torch.mean(log_probs_Z)

            # avg_log_prob_Z_hat = (per_sample_avg_log_prob_Z * per_sample_avg_prob_Z).mean()

            # Compute the difference of the negative average log likelihood ratios
            difference = -avg_log_prob_Z + avg_log_prob_Z_hat
            difference_onlylog = -avg_log_prob_Z_onlylog + avg_log_prob_Z_hat_onlylog

            return (
                difference_onlylog,
                avg_log_prob_Z_hat_onlylog,
                avg_log_prob_Z_onlylog,
            )

        def js_divergence(net_1_logits, net_2_logits):
            clip_value = 1e7
            net_1_probs = F.softmax(torch.clamp(net_1_logits, -clip_value, clip_value), dim=1)
            net_2_probs = F.softmax(torch.clamp(net_2_logits, -clip_value, clip_value), dim=1)

            total_m = 0.5 * (net_1_probs + net_2_probs)

            clip_value = 1e-20
            total_m = torch.clamp(total_m, clip_value, 1).log()

            # total_m = torch.clamp(total_m, clip_value, 1)
            net_1_probs = torch.clamp(net_1_probs, clip_value, 1)
            net_2_probs = torch.clamp(net_2_probs, clip_value, 1)

            loss = 0.0

            loss += F.kl_div(total_m, net_1_probs, reduction="batchmean")
            loss += F.kl_div(total_m, net_2_probs, reduction="batchmean")
            if torch.isnan(loss):
                raise Exception("NaN detected in loss computation")
            return 0.5 * loss

        def get_losses(pred_perm, pred_normal, cross_permod, title):
            if self.task == "reg":
                mse_pred = F.mse_loss(pred_perm, pred_normal)
                cross_permod["{}_pred".format(title)].append(mse_pred)
            else:
                jsd_pred = js_divergence(pred_perm, pred_normal)
                loglike_pred, entropy_y, norm_ent = log_likelihood_ratio(pred_normal, pred_perm)

                cross_permod["{}_yent".format(title)].append(entropy_y)
                cross_permod["{}_nll".format(title)].append(loglike_pred)
                cross_permod["{}_pred".format(title)].append(jsd_pred)
            return cross_permod

        cross_permod = defaultdict(list)

        soft_labels_norm = output["preds"]["ncombined"]

        cross_permod = get_losses(output["preds"]["sc_detc"], soft_labels_norm, cross_permod, title="enc12_p0")
        # cross_permod = get_losses(predicted_logits_0["preds"]["sg_detc"], soft_labels_norm, cross_permod, title="enc12_p1", sign="minus")
        # cross_permod = get_losses(predicted_logits_0["preds"]["sf_detc"], soft_labels_norm, cross_permod, title="enc12_p2", sign="minus")
        # cross_permod = get_losses(predicted_logits_0["preds"]["scf_detc"], soft_labels_norm, cross_permod, title="enc12_p02", sign="minus")
        cross_permod = get_losses(
            output["preds"]["sgf_detc"],
            soft_labels_norm,
            cross_permod,
            title="enc12_p12",
        )
        # cross_permod = get_losses(predicted_logits_0["preds"]["scg_detc"], soft_labels_norm, cross_permod, title="enc12_p01")

        # cross_permod = get_losses(predicted_logits_0["preds"]["sc_detg"], soft_labels_norm, cross_permod, title="enc02_p0")
        cross_permod = get_losses(output["preds"]["sg_detg"], soft_labels_norm, cross_permod, title="enc02_p1")
        # cross_permod = get_losses(predicted_logits_0["preds"]["sf_detg"], soft_labels_norm, cross_permod, title="enc02_p2")
        cross_permod = get_losses(
            output["preds"]["scf_detg"],
            soft_labels_norm,
            cross_permod,
            title="enc02_p02",
        )
        # cross_permod = get_losses(predicted_logits_0["preds"]["sgf_detg"], soft_labels_norm, cross_permod, title="enc02_p12")
        # cross_permod = get_losses(predicted_logits_0["preds"]["scg_detg"], soft_labels_norm, cross_permod, title="enc02_p01")

        # cross_permod = get_losses(predicted_logits_0["preds"]["sc_detf"], soft_labels_norm, cross_permod, title="enc01_p0")
        # cross_permod = get_losses(predicted_logits_0["preds"]["sg_detf"], soft_labels_norm, cross_permod, title="enc01_p1")
        cross_permod = get_losses(output["preds"]["sf_detf"], soft_labels_norm, cross_permod, title="enc01_p2")
        # cross_permod = get_losses(predicted_logits_0["preds"]["scf_detf"], soft_labels_norm, cross_permod, title="enc01_p02")
        # cross_permod = get_losses(predicted_logits_0["preds"]["sgf_detf"], soft_labels_norm, cross_permod, title="enc01_p12")
        cross_permod = get_losses(
            output["preds"]["scg_detf"],
            soft_labels_norm,
            cross_permod,
            title="enc01_p01",
        )

        cross_permod = get_losses(output["preds"]["sc_detgf"], soft_labels_norm, cross_permod, title="enc0_p0")
        cross_permod = get_losses(output["preds"]["sg_detgf"], soft_labels_norm, cross_permod, title="enc0_p1")
        cross_permod = get_losses(output["preds"]["sf_detgf"], soft_labels_norm, cross_permod, title="enc0_p2")
        cross_permod = get_losses(
            output["preds"]["scf_detgf"],
            soft_labels_norm,
            cross_permod,
            title="enc0_p02",
        )
        cross_permod = get_losses(
            output["preds"]["scf_detgf_agree"],
            soft_labels_norm,
            cross_permod,
            title="enc0_p02_agree",
        )
        cross_permod = get_losses(
            output["preds"]["sgf_detgf"],
            soft_labels_norm,
            cross_permod,
            title="enc0_p12",
        )
        cross_permod = get_losses(
            output["preds"]["sgf_detgf_agree"],
            soft_labels_norm,
            cross_permod,
            title="enc0_p12_agree",
        )
        cross_permod = get_losses(
            output["preds"]["scg_detgf"],
            soft_labels_norm,
            cross_permod,
            title="enc0_p01",
        )
        cross_permod = get_losses(
            output["preds"]["scg_detgf_agree"],
            soft_labels_norm,
            cross_permod,
            title="enc0_p01_agree",
        )

        cross_permod = get_losses(output["preds"]["sc_detcf"], soft_labels_norm, cross_permod, title="enc1_p0")
        cross_permod = get_losses(output["preds"]["sg_detcf"], soft_labels_norm, cross_permod, title="enc1_p1")
        cross_permod = get_losses(output["preds"]["sf_detcf"], soft_labels_norm, cross_permod, title="enc1_p2")
        cross_permod = get_losses(
            output["preds"]["scf_detcf"],
            soft_labels_norm,
            cross_permod,
            title="enc1_p02",
        )
        cross_permod = get_losses(
            output["preds"]["scf_detcf_agree"],
            soft_labels_norm,
            cross_permod,
            title="enc1_p02_agree",
        )
        cross_permod = get_losses(
            output["preds"]["sgf_detcf"],
            soft_labels_norm,
            cross_permod,
            title="enc1_p12",
        )
        cross_permod = get_losses(
            output["preds"]["sgf_detcf_agree"],
            soft_labels_norm,
            cross_permod,
            title="enc1_p12_agree",
        )
        cross_permod = get_losses(
            output["preds"]["scg_detcf"],
            soft_labels_norm,
            cross_permod,
            title="enc1_p01",
        )
        cross_permod = get_losses(
            output["preds"]["scg_detcf_agree"],
            soft_labels_norm,
            cross_permod,
            title="enc1_p01_agree",
        )

        cross_permod = get_losses(output["preds"]["sc_detcg"], soft_labels_norm, cross_permod, title="enc2_p0")
        cross_permod = get_losses(output["preds"]["sg_detcg"], soft_labels_norm, cross_permod, title="enc2_p1")
        cross_permod = get_losses(output["preds"]["sf_detcg"], soft_labels_norm, cross_permod, title="enc2_p2")
        cross_permod = get_losses(
            output["preds"]["scf_detcg"],
            soft_labels_norm,
            cross_permod,
            title="enc2_p02",
        )
        cross_permod = get_losses(
            output["preds"]["scf_detcg_agree"],
            soft_labels_norm,
            cross_permod,
            title="enc2_p02_agree",
        )
        cross_permod = get_losses(
            output["preds"]["sgf_detcg"],
            soft_labels_norm,
            cross_permod,
            title="enc2_p12",
        )
        cross_permod = get_losses(
            output["preds"]["sgf_detcg_agree"],
            soft_labels_norm,
            cross_permod,
            title="enc2_p12_agree",
        )
        cross_permod = get_losses(
            output["preds"]["scg_detcg"],
            soft_labels_norm,
            cross_permod,
            title="enc2_p01",
        )
        cross_permod = get_losses(
            output["preds"]["scg_detcg_agree"],
            soft_labels_norm,
            cross_permod,
            title="enc2_p01_agree",
        )

        cross_permod = get_losses(output["preds"]["sc"], soft_labels_norm, cross_permod, title="all_p0")
        cross_permod = get_losses(output["preds"]["sg"], soft_labels_norm, cross_permod, title="all_p1")
        cross_permod = get_losses(output["preds"]["sf"], soft_labels_norm, cross_permod, title="all_p2")
        cross_permod = get_losses(output["preds"]["scf"], soft_labels_norm, cross_permod, title="all_p02")
        cross_permod = get_losses(output["preds"]["sgf"], soft_labels_norm, cross_permod, title="all_p12")
        cross_permod = get_losses(output["preds"]["scg"], soft_labels_norm, cross_permod, title="all_p01")

        out = {name: torch.stack(cross_permod[name]).mean() for name in cross_permod}

        return out

    def before_backward(self, total, output_losses, w_loss, loss_fun, data, label, output, **kwargs):

        if not (self.bias_infuser.get("starting_epoch", 0) <= self.agent.logs["current_epoch"] <= self.bias_infuser.get("ending_epoch", 1000)):
            return total, output_losses, False

        pert_loss = self.get_perturbed_gradients(output)

        if self.reg_by == "dist_pred_cjsd":
            reg_term_0 = pert_loss["enc0_p0_pred"] + pert_loss["enc0_p01_pred"] + pert_loss["enc0_p02_pred"]
            reg_term_1 = pert_loss["enc1_p1_pred"] + pert_loss["enc1_p01_pred"] + pert_loss["enc1_p12_pred"]
            reg_term_2 = pert_loss["enc2_p2_pred"] + pert_loss["enc2_p12_pred"] + pert_loss["enc2_p02_pred"]
            reg_term_0_min = pert_loss["enc0_p1_pred"] + pert_loss["enc0_p2_pred"] + pert_loss["enc0_p12_pred"]
            reg_term_1_min = pert_loss["enc1_p0_pred"] + pert_loss["enc1_p2_pred"] + pert_loss["enc1_p02_pred"]
            reg_term_2_min = pert_loss["enc2_p1_pred"] + pert_loss["enc2_p0_pred"] + pert_loss["enc2_p01_pred"]
            reg_term = self.l * (-reg_term_0 - reg_term_1 - reg_term_2 + reg_term_0_min + reg_term_1_min + reg_term_2_min)

        elif self.reg_by == "dist_pred_max_cjsd":
            reg_term_0 = pert_loss["enc0_p0_pred"] + pert_loss["enc0_p01_pred"] + pert_loss["enc0_p02_pred"]
            reg_term_1 = pert_loss["enc1_p1_pred"] + pert_loss["enc1_p01_pred"] + pert_loss["enc1_p12_pred"]
            reg_term_2 = pert_loss["enc2_p2_pred"] + pert_loss["enc2_p12_pred"] + pert_loss["enc2_p02_pred"]
            reg_term = self.l * (-reg_term_0 - reg_term_1 - reg_term_2)

        elif self.reg_by == "dist_pred_all_max_cjsd":
            reg_term_0 = pert_loss["all_p0_pred"]
            reg_term_1 = pert_loss["all_p0_pred"]
            reg_term_2 = pert_loss["all_p0_pred"]
            reg_term_01 = pert_loss["all_p01_pred"]
            reg_term_12 = pert_loss["all_p12_pred"]
            reg_term_02 = pert_loss["all_p02_pred"]
            reg_term = self.l * (-reg_term_0 - reg_term_1 - reg_term_2 - reg_term_01 - reg_term_12 - reg_term_02)

        cca_loss_reg = 0
        if self.contr_coeff:
            cca_loss = nt_xent_loss(
                output["features"]["c"],
                output["features"]["g"],
                output["features"]["f"],
                label=label,
                temperature=self.temperature,
            )

            cca_loss_reg = self.contr_coeff * cca_loss

        reg_terms = {
            "reg_term": reg_term,
            "reg_term_0": reg_term_0,
            "reg_term_1": reg_term_1,
            "reg_term_2": reg_term_2,
            "contrastive": cca_loss_reg,
        }

        wandb.log(
            {"perturb": pert_loss, "reg_terms": reg_terms},
            step=self.agent.logs["current_step"] + 1,
        )

        if self.contr_coeff:
            output_losses["cca"] = cca_loss

        output_losses["reg_term"] = reg_term
        output_losses["reg_term_0"] = reg_term_0
        output_losses["reg_term_1"] = reg_term_1
        output_losses["reg_term_2"] = reg_term_2

        total += reg_term + cca_loss_reg

        return total, output_losses, False


class Bias_Infusion_MCR_Reg(General_Bias_Infusion):
    """
    NGMine = Norm-Gradient my version
    """

    def __init__(self, agent):
        super(Bias_Infusion_MCR_Reg, self).__init__(agent)

        self._initialize_logs_n_utils()

    def _initialize_logs_n_utils(self):
        self.losses = []
        self.agent.logs["reg_logs"] = defaultdict(list)
        self.agent.logs["ratio_logs"] = defaultdict(list)
        self.agent.logs["ratio_logs"] = defaultdict(list)
        self.bias_infuser = self.agent.config.model.args.get("bias_infusion", {})
        self.reg_by = self.bias_infuser.get("reg_by", "dist_pred")
        self.l = self.bias_infuser.get("l", 0)
        self.task = self.bias_infuser.get("task", "reg")
        self.contr_coeff = self.bias_infuser.get("contr_coeff", False)
        self.temperature = self.bias_infuser.get("temperature", 0.5)

    def get_perturbed_gradients(self, output):
        def log_likelihood_ratio(logits_Z, logits_Z_hat):

            # Compute probabilities from logits

            probs_Z = F.softmax(logits_Z, dim=1)
            probs_Z_hat = F.softmax(logits_Z_hat, dim=1)

            log_probs_Z = F.log_softmax(logits_Z, dim=1)
            log_probs_Z_hat = F.log_softmax(logits_Z_hat, dim=1)

            per_sample_avg_log_prob_Z_hat = torch.concatenate([(log_probs_Z_hat * probs_Z_hat).mean().unsqueeze(0)])
            per_sample_avg_log_prob_Z_hat_onlylog = torch.concatenate([(log_probs_Z_hat).mean().unsqueeze(0)])

            avg_log_prob_Z = -torch.mean(log_probs_Z * probs_Z)
            avg_log_prob_Z_hat = -torch.mean(per_sample_avg_log_prob_Z_hat)

            avg_log_prob_Z_hat_onlylog = -torch.mean(per_sample_avg_log_prob_Z_hat_onlylog)
            avg_log_prob_Z_onlylog = -torch.mean(log_probs_Z)

            difference = -avg_log_prob_Z + avg_log_prob_Z_hat
            difference_onlylog = -avg_log_prob_Z_onlylog + avg_log_prob_Z_hat_onlylog

            return (
                difference_onlylog,
                avg_log_prob_Z_hat_onlylog,
                avg_log_prob_Z_onlylog,
            )

        def js_divergence(net_1_logits, net_2_logits):
            clip_value = 1e7
            net_1_probs = F.softmax(torch.clamp(net_1_logits, -clip_value, clip_value), dim=1)
            net_2_probs = F.softmax(torch.clamp(net_2_logits, -clip_value, clip_value), dim=1)

            total_m = 0.5 * (net_1_probs + net_2_probs)

            clip_value = 1e-20
            total_m = torch.clamp(total_m, clip_value, 1).log()

            # total_m = torch.clamp(total_m, clip_value, 1)
            net_1_probs = torch.clamp(net_1_probs, clip_value, 1)
            net_2_probs = torch.clamp(net_2_probs, clip_value, 1)

            loss = 0.0

            loss += F.kl_div(total_m, net_1_probs, reduction="batchmean")
            loss += F.kl_div(total_m, net_2_probs, reduction="batchmean")
            if torch.isnan(loss):
                raise Exception("NaN detected in loss computation")
            return 0.5 * loss

        def get_losses(pred_perm, pred_normal, cross_permod, title):
            if self.task == "reg":
                mse_pred = F.mse_loss(pred_perm, pred_normal)
                cross_permod["{}_pred".format(title)].append(mse_pred)
            else:
                jsd_pred = js_divergence(pred_perm, pred_normal)
                loglike_pred, entropy_y, norm_ent = log_likelihood_ratio(pred_normal, pred_perm)

                cross_permod["{}_yent".format(title)].append(entropy_y)
                cross_permod["{}_nll".format(title)].append(loglike_pred)
                cross_permod["{}_pred".format(title)].append(jsd_pred)
            return cross_permod

        cross_permod = defaultdict(list)

        if "ncombined" not in output["preds"]:
            return None

        soft_labels_norm = output["preds"]["ncombined"]

        cross_permod = get_losses(output["preds"]["sa_detv"], soft_labels_norm, cross_permod, title="enc0_p0")
        cross_permod = get_losses(output["preds"]["sa_deta"], soft_labels_norm, cross_permod, title="enc1_p0")
        cross_permod = get_losses(output["preds"]["sv_detv"], soft_labels_norm, cross_permod, title="enc0_p1")
        cross_permod = get_losses(output["preds"]["sv_deta"], soft_labels_norm, cross_permod, title="enc1_p1")
        cross_permod = get_losses(output["preds"]["sv"], soft_labels_norm, cross_permod, title="all_p1")
        cross_permod = get_losses(output["preds"]["sa"], soft_labels_norm, cross_permod, title="all_p0")

        out = {name: torch.stack(cross_permod[name]).mean() for name in cross_permod}

        return out

    def before_backward(self, total, output_losses, w_loss, loss_fun, data, label, output, **kwargs):

        if not (self.bias_infuser.get("starting_epoch", 0) <= self.agent.logs["current_epoch"] <= self.bias_infuser.get("ending_epoch", 1000)):
            return total, output_losses, False

        pert_loss = self.get_perturbed_gradients(output)

        if self.reg_by == "dist_pred_cjsd":
            reg_term = self.l * (-pert_loss["enc0_p0_pred"] + pert_loss["enc1_p0_pred"] - pert_loss["enc1_p1_pred"] + pert_loss["enc0_p1_pred"])
            reg_term_0 = pert_loss["enc0_p0_pred"]
            reg_term_1 = pert_loss["enc1_p1_pred"]
        elif self.reg_by == "dist_pred_max_cjsd":
            reg_term = self.l * (-pert_loss["enc0_p0_pred"] - pert_loss["enc1_p1_pred"])
            reg_term_0 = pert_loss["enc0_p0_pred"]
            reg_term_1 = pert_loss["enc1_p1_pred"]
        elif self.reg_by == "dist_pred_all_max_cjsd":
            reg_term_0 = pert_loss["all_p0_pred"]
            reg_term_1 = pert_loss["all_p1_pred"]
            reg_term = self.l * (-reg_term_0 - reg_term_1)

        cca_loss_reg = 0
        if self.contr_coeff:
            if self.bias_infuser.get("contr_type", "label") == "label":
                cca_loss = nt_xent_loss(
                    output["features"]["c"],
                    output["features"]["g"],
                    label=label,
                    temperature=self.temperature,
                )
            elif self.bias_infuser.get("contr_type", "label") == "pure":
                cca_loss = nt_xent_loss(
                    output["features"]["c"],
                    output["features"]["g"],
                    temperature=self.temperature,
                )

            cca_loss_reg = self.contr_coeff * cca_loss

        reg_terms = {
            "reg_term": reg_term,
            "reg_term_0": reg_term_0,
            "reg_term_1": reg_term_1,
            "contrastive": cca_loss_reg,
        }

        wandb.log(
            {"perturb": pert_loss, "reg_terms": reg_terms},
            step=self.agent.logs["current_step"] + 1,
        )

        if self.contr_coeff:
            output_losses["cca"] = cca_loss

        output_losses["reg_term"] = reg_term
        output_losses["reg_term_0"] = reg_term_0
        output_losses["reg_term_1"] = reg_term_1

        total += reg_term + cca_loss_reg

        return total, output_losses, False


class Bias_Infusion_MCR3D_Reg(General_Bias_Infusion):
    """
    NGMine = Norm-Gradient my version
    """

    def __init__(self, agent):
        super(Bias_Infusion_MCR3D_Reg, self).__init__(agent)

        self._initialize_logs_n_utils()

    def _initialize_logs_n_utils(self):
        self.losses = []
        self.agent.logs["reg_logs"] = defaultdict(list)
        self.agent.logs["ratio_logs"] = defaultdict(list)
        self.agent.logs["ratio_logs"] = defaultdict(list)
        self.bias_infuser = self.agent.config.model.args.get("bias_infusion", {})
        self.reg_by = self.bias_infuser.get("reg_by", "dist_pred")
        self.l = self.bias_infuser.get("l", 0)

        self.task = self.bias_infuser.get("task", "reg")
        self.contr_coeff = self.bias_infuser.get("contr_coeff", False)

        self.temperature = self.bias_infuser.get("temperature", 0.5)

    # def _perturb_tensor(self, tens: dict, n_samples: int, label: torch.Tensor, shuffle: bool = True,
    #                     noise: bool = True) -> torch.Tensor:
    #
    #     batch_size = tens[0].shape[0] if 0 in tens else tens[2].shape[0]
    #
    #     final_data = []
    #     while len(final_data) < n_samples:
    #
    #         if shuffle:
    #             shuffle_idx_0 = torch.randperm(batch_size)
    #             shuffle_idx_1 = torch.randperm(batch_size)
    #             shuffle_idx_2 = torch.randperm(batch_size)
    #             shuffle_idx_3 = torch.randperm(batch_size)
    #
    #             if self.reg_by == "dist_pred_3d_agree":
    #                 shuffle_idx_1 = shuffle_idx_0
    #                 shuffle_idx_2 = shuffle_idx_0
    #             nonequal_label_0 = torch.ones(batch_size, dtype=torch.bool)
    #             nonequal_label_1 = nonequal_label_0
    #             nonequal_label_2 = nonequal_label_0
    #             nonequal_label_3 = nonequal_label_0
    #
    #             # nonequal_label_0 = ~(label[shuffle_idx_0] == label)
    #             # nonequal_label_1 = ~(label[shuffle_idx_1] == label)
    #             # nonequal_label_2 = ~(label[shuffle_idx_2] == label)
    #             if nonequal_label_0.sum() <= 1 or nonequal_label_1.sum() <= 1 or nonequal_label_2.sum() <= 1 or nonequal_label_2.sum() <= 1:
    #                 continue
    #             shuffle_idx_0 = shuffle_idx_0[nonequal_label_0.cpu()]
    #             shuffle_idx_1 = shuffle_idx_1[nonequal_label_1.cpu()]
    #             shuffle_idx_2 = shuffle_idx_2[nonequal_label_2.cpu()]
    #             shuffle_idx_3 = shuffle_idx_3[nonequal_label_3.cpu()]
    #
    #         if nonequal_label_0.sum() <= 1 or nonequal_label_1.sum() <= 1 or nonequal_label_2.sum() <= 1 or nonequal_label_2.sum() <= 1:
    #             continue
    #         final_data.append({"shuffle_idx_0": shuffle_idx_0, "shuffle_idx_1": shuffle_idx_1,"shuffle_idx_2": shuffle_idx_2,
    #                            "data_0": nonequal_label_0, "data_1": nonequal_label_1, "data_2": nonequal_label_2, "data_3": nonequal_label_3})
    #
    #     return final_data

    def get_perturbed_gradients(self, output):
        def log_likelihood_ratio(logits_Z, logits_Z_hat):
            """
            Compute the negative average log likelihood ratio between two sets of logits.

            Args:
            - logits_Z (torch.Tensor): Logits from predictions using (X, Z), shape (batch_size, num_classes)
            - logits_Z_hat (torch.Tensor): Logits from predictions using (X, Z_hat), shape (batch_size, num_classes)
            - targets (torch.Tensor): True class labels, shape (batch_size,)

            Returns:
            - torch.Tensor: Difference of negative average log likelihood ratios
            """
            # Compute probabilities from logits

            probs_Z = F.softmax(logits_Z, dim=1)
            probs_Z_hat = F.softmax(logits_Z_hat, dim=1)

            # label_norm = targets.repeat(len(pert_data))[ids_of_each_sample]

            # torch.concatenate([targets[pert_data[i]["data"]] for i in range(len(pert_data))], dim=0)

            # Select the probabilities corresponding to the true labels
            # target_probs_Z_hat = probs_Z_hat[range(len(label_norm)), label_norm]
            # target_probs_Z = probs_Z[range(len(targets)), targets]

            # Compute log probabilities
            # log_probs_Z = torch.log(target_probs_Z + 1e-9)  # Add epsilon to avoid log(0)
            # log_probs_Z_hat = torch.log(target_probs_Z_hat + 1e-9)  # Add epsilon to avoid log(0)

            # Compute log probabilities
            # log_probs_Z = torch.log(probs_Z + 1e-7)  # Add epsilon to avoid log(0)
            # log_probs_Z_hat = torch.log(probs_Z_hat + 1e-7)  # Add epsilon to avoid log(0)
            log_probs_Z = F.log_softmax(logits_Z, dim=1)
            log_probs_Z_hat = F.log_softmax(logits_Z_hat, dim=1)

            per_sample_avg_log_prob_Z_hat = torch.concatenate([(log_probs_Z_hat * probs_Z_hat).mean().unsqueeze(0)])
            per_sample_avg_log_prob_Z_hat_onlylog = torch.concatenate([(log_probs_Z_hat).mean().unsqueeze(0)])

            #     print(log_probs_Z[ids_of_each_sample==i].mean())
            # # Compute the separate empirical averages
            # avg_log_prob_Z = torch.concat([log_probs_Z[ids_of_each_sample==i].mean() for i in len(pert_data[0]["data"])])
            avg_log_prob_Z = -torch.mean(log_probs_Z * probs_Z)
            avg_log_prob_Z_hat = -torch.mean(per_sample_avg_log_prob_Z_hat)

            avg_log_prob_Z_hat_onlylog = -torch.mean(per_sample_avg_log_prob_Z_hat_onlylog)
            avg_log_prob_Z_onlylog = -torch.mean(log_probs_Z)

            # avg_log_prob_Z_hat = (per_sample_avg_log_prob_Z * per_sample_avg_prob_Z).mean()

            # Compute the difference of the negative average log likelihood ratios
            difference = -avg_log_prob_Z + avg_log_prob_Z_hat
            difference_onlylog = -avg_log_prob_Z_onlylog + avg_log_prob_Z_hat_onlylog

            return (
                difference_onlylog,
                avg_log_prob_Z_hat_onlylog,
                avg_log_prob_Z_onlylog,
            )

        def js_divergence(net_1_logits, net_2_logits):
            clip_value = 1e7
            net_1_probs = F.softmax(torch.clamp(net_1_logits, -clip_value, clip_value), dim=1)
            net_2_probs = F.softmax(torch.clamp(net_2_logits, -clip_value, clip_value), dim=1)

            total_m = 0.5 * (net_1_probs + net_2_probs)

            clip_value = 1e-20
            total_m = torch.clamp(total_m, clip_value, 1).log()

            # total_m = torch.clamp(total_m, clip_value, 1)
            net_1_probs = torch.clamp(net_1_probs, clip_value, 1)
            net_2_probs = torch.clamp(net_2_probs, clip_value, 1)

            loss = 0.0

            loss += F.kl_div(total_m, net_1_probs, reduction="batchmean")
            loss += F.kl_div(total_m, net_2_probs, reduction="batchmean")
            if torch.isnan(loss):
                raise Exception("NaN detected in loss computation")
            return 0.5 * loss

        def get_losses(pred_perm, pred_normal, cross_permod, title):
            if self.task == "reg":
                mse_pred = F.mse_loss(pred_perm, pred_normal)
                cross_permod["{}_pred".format(title)].append(mse_pred)
            else:
                jsd_pred = js_divergence(pred_perm, pred_normal)
                loglike_pred, entropy_y, norm_ent = log_likelihood_ratio(pred_normal, pred_perm)

                cross_permod["{}_yent".format(title)].append(entropy_y)
                cross_permod["{}_nll".format(title)].append(loglike_pred)
                cross_permod["{}_pred".format(title)].append(jsd_pred)
            return cross_permod

        cross_permod = defaultdict(list)

        soft_labels_norm = output["preds"]["ncombined"]

        cross_permod = get_losses(output["preds"]["sc_detc"], soft_labels_norm, cross_permod, title="enc12_p0")
        # cross_permod = get_losses(predicted_logits_0["preds"]["sg_detc"], soft_labels_norm, cross_permod, title="enc12_p1", sign="minus")
        # cross_permod = get_losses(predicted_logits_0["preds"]["sf_detc"], soft_labels_norm, cross_permod, title="enc12_p2", sign="minus")
        # cross_permod = get_losses(predicted_logits_0["preds"]["scf_detc"], soft_labels_norm, cross_permod, title="enc12_p02", sign="minus")
        cross_permod = get_losses(
            output["preds"]["sgf_detc"],
            soft_labels_norm,
            cross_permod,
            title="enc12_p12",
        )
        # cross_permod = get_losses(predicted_logits_0["preds"]["scg_detc"], soft_labels_norm, cross_permod, title="enc12_p01")

        # cross_permod = get_losses(predicted_logits_0["preds"]["sc_detg"], soft_labels_norm, cross_permod, title="enc02_p0")
        cross_permod = get_losses(output["preds"]["sg_detg"], soft_labels_norm, cross_permod, title="enc02_p1")
        # cross_permod = get_losses(predicted_logits_0["preds"]["sf_detg"], soft_labels_norm, cross_permod, title="enc02_p2")
        cross_permod = get_losses(
            output["preds"]["scf_detg"],
            soft_labels_norm,
            cross_permod,
            title="enc02_p02",
        )
        # cross_permod = get_losses(predicted_logits_0["preds"]["sgf_detg"], soft_labels_norm, cross_permod, title="enc02_p12")
        # cross_permod = get_losses(predicted_logits_0["preds"]["scg_detg"], soft_labels_norm, cross_permod, title="enc02_p01")

        # cross_permod = get_losses(predicted_logits_0["preds"]["sc_detf"], soft_labels_norm, cross_permod, title="enc01_p0")
        # cross_permod = get_losses(predicted_logits_0["preds"]["sg_detf"], soft_labels_norm, cross_permod, title="enc01_p1")
        cross_permod = get_losses(output["preds"]["sf_detf"], soft_labels_norm, cross_permod, title="enc01_p2")
        # cross_permod = get_losses(predicted_logits_0["preds"]["scf_detf"], soft_labels_norm, cross_permod, title="enc01_p02")
        # cross_permod = get_losses(predicted_logits_0["preds"]["sgf_detf"], soft_labels_norm, cross_permod, title="enc01_p12")
        cross_permod = get_losses(
            output["preds"]["scg_detf"],
            soft_labels_norm,
            cross_permod,
            title="enc01_p01",
        )

        cross_permod = get_losses(output["preds"]["sc_detgf"], soft_labels_norm, cross_permod, title="enc0_p0")
        cross_permod = get_losses(output["preds"]["sg_detgf"], soft_labels_norm, cross_permod, title="enc0_p1")
        cross_permod = get_losses(output["preds"]["sf_detgf"], soft_labels_norm, cross_permod, title="enc0_p2")
        cross_permod = get_losses(
            output["preds"]["scf_detgf"],
            soft_labels_norm,
            cross_permod,
            title="enc0_p02",
        )
        cross_permod = get_losses(
            output["preds"]["sgf_detgf"],
            soft_labels_norm,
            cross_permod,
            title="enc0_p12",
        )
        if "scg_detgf_agree" in output["preds"]:
            cross_permod = get_losses(
                output["preds"]["sgf_detgf_agree"],
                soft_labels_norm,
                cross_permod,
                title="enc0_p12_agree",
            )
        cross_permod = get_losses(
            output["preds"]["scg_detgf"],
            soft_labels_norm,
            cross_permod,
            title="enc0_p01",
        )

        cross_permod = get_losses(output["preds"]["sc_detcf"], soft_labels_norm, cross_permod, title="enc1_p0")
        cross_permod = get_losses(output["preds"]["sg_detcf"], soft_labels_norm, cross_permod, title="enc1_p1")
        cross_permod = get_losses(output["preds"]["sf_detcf"], soft_labels_norm, cross_permod, title="enc1_p2")
        cross_permod = get_losses(
            output["preds"]["scf_detcf"],
            soft_labels_norm,
            cross_permod,
            title="enc1_p02",
        )
        if "scf_detcf_agree" in output["preds"]:
            cross_permod = get_losses(
                output["preds"]["scf_detcf_agree"],
                soft_labels_norm,
                cross_permod,
                title="enc1_p02_agree",
            )
        cross_permod = get_losses(
            output["preds"]["sgf_detcf"],
            soft_labels_norm,
            cross_permod,
            title="enc1_p12",
        )
        cross_permod = get_losses(
            output["preds"]["scg_detcf"],
            soft_labels_norm,
            cross_permod,
            title="enc1_p01",
        )

        cross_permod = get_losses(output["preds"]["sc_detcg"], soft_labels_norm, cross_permod, title="enc2_p0")
        cross_permod = get_losses(output["preds"]["sg_detcg"], soft_labels_norm, cross_permod, title="enc2_p1")
        cross_permod = get_losses(output["preds"]["sf_detcg"], soft_labels_norm, cross_permod, title="enc2_p2")
        cross_permod = get_losses(
            output["preds"]["scf_detcg"],
            soft_labels_norm,
            cross_permod,
            title="enc2_p02",
        )
        cross_permod = get_losses(
            output["preds"]["sgf_detcg"],
            soft_labels_norm,
            cross_permod,
            title="enc2_p12",
        )
        cross_permod = get_losses(
            output["preds"]["scg_detcg"],
            soft_labels_norm,
            cross_permod,
            title="enc2_p01",
        )
        if "scg_detcg_agree" in output["preds"]:
            cross_permod = get_losses(
                output["preds"]["scg_detcg_agree"],
                soft_labels_norm,
                cross_permod,
                title="enc2_p01_agree",
            )

        cross_permod = get_losses(output["preds"]["sc"], soft_labels_norm, cross_permod, title="all_p0")
        cross_permod = get_losses(output["preds"]["sg"], soft_labels_norm, cross_permod, title="all_p1")
        cross_permod = get_losses(output["preds"]["sf"], soft_labels_norm, cross_permod, title="all_p2")
        cross_permod = get_losses(output["preds"]["scf"], soft_labels_norm, cross_permod, title="all_p02")
        cross_permod = get_losses(output["preds"]["sgf"], soft_labels_norm, cross_permod, title="all_p12")
        cross_permod = get_losses(output["preds"]["scg"], soft_labels_norm, cross_permod, title="all_p01")

        out = {name: torch.stack(cross_permod[name]).mean() for name in cross_permod}

        return out

    def before_backward(self, total, output_losses, w_loss, loss_fun, data, label, output, **kwargs):

        if not (self.bias_infuser.get("starting_epoch", 0) <= self.agent.logs["current_epoch"] <= self.bias_infuser.get("ending_epoch", 1000)):
            return total, output_losses, False

        pert_loss = self.get_perturbed_gradients(output)

        if self.reg_by == "dist_pred_cjsd":
            reg_term_0 = pert_loss["enc0_p0_pred"] + pert_loss["enc0_p01_pred"] + pert_loss["enc0_p02_pred"]
            reg_term_1 = pert_loss["enc1_p1_pred"] + pert_loss["enc1_p01_pred"] + pert_loss["enc1_p12_pred"]
            reg_term_2 = pert_loss["enc2_p2_pred"] + pert_loss["enc2_p12_pred"] + pert_loss["enc2_p02_pred"]
            reg_term_0_min = pert_loss["enc0_p1_pred"] + pert_loss["enc0_p2_pred"] + pert_loss["enc0_p12_pred"]
            reg_term_1_min = pert_loss["enc1_p0_pred"] + pert_loss["enc1_p2_pred"] + pert_loss["enc1_p02_pred"]
            reg_term_2_min = pert_loss["enc2_p1_pred"] + pert_loss["enc2_p0_pred"] + pert_loss["enc2_p01_pred"]
            reg_term = self.l * (-reg_term_0 - reg_term_1 - reg_term_2 + reg_term_0_min + reg_term_1_min + reg_term_2_min)
        elif self.reg_by == "dist_pred_max_cjsd":
            reg_term_0 = pert_loss["enc0_p0_pred"] + pert_loss["enc0_p01_pred"] + pert_loss["enc0_p02_pred"]
            reg_term_1 = pert_loss["enc1_p1_pred"] + pert_loss["enc1_p01_pred"] + pert_loss["enc1_p12_pred"]
            reg_term_2 = pert_loss["enc2_p2_pred"] + pert_loss["enc2_p12_pred"] + pert_loss["enc2_p02_pred"]
            reg_term = self.l * (-reg_term_0 - reg_term_1 - reg_term_2)
        elif self.reg_by == "dist_pred_all_max_cjsd":
            reg_term_0 = pert_loss["all_p0_pred"]
            reg_term_1 = pert_loss["all_p0_pred"]
            reg_term_2 = pert_loss["all_p0_pred"]
            reg_term_01 = pert_loss["all_p01_pred"]
            reg_term_12 = pert_loss["all_p12_pred"]
            reg_term_02 = pert_loss["all_p02_pred"]
            reg_term = self.l * (-reg_term_0 - reg_term_1 - reg_term_2 - reg_term_01 - reg_term_12 - reg_term_02)
        else:
            raise NotImplementedError

        cca_loss_reg = 0
        if self.contr_coeff:
            if self.bias_infuser.get("contr_type", "label") == "label":
                # cca_loss = self.cca_loss_fun.loss(output["features"]["c"]["features"]["combined"],output["features"]["g"]["features"]["combined"])
                cca_loss = nt_xent_loss(
                    output["features"]["c"],
                    output["features"]["g"],
                    output["features"]["f"],
                    label=label,
                    temperature=self.temperature,
                )
            elif self.bias_infuser.get("contr_type", "label") == "pure":
                cca_loss = nt_xent_loss(
                    output["features"]["c"],
                    output["features"]["g"],
                    output["features"]["f"],
                    temperature=self.temperature,
                )

            cca_loss_reg = self.contr_coeff * cca_loss

        reg_terms = {
            "reg_term": reg_term,
            "reg_term_0": reg_term_0,
            "reg_term_1": reg_term_1,
            "reg_term_2": reg_term_2,
            "contrastive": cca_loss_reg,
        }

        if "reg_term_0_min" in locals() and "reg_term_1_min" in locals() and "reg_term_2_min" in locals():
            reg_terms["reg_term_0_min"] = reg_term_0_min
            reg_terms["reg_term_1_min"] = reg_term_1_min
            reg_terms["reg_term_2_min"] = reg_term_2_min
        wandb.log(
            {"perturb": pert_loss, "reg_terms": reg_terms},
            step=self.agent.logs["current_step"] + 1,
        )

        if self.contr_coeff:
            output_losses["cca"] = cca_loss

        output_losses["reg_term"] = reg_term
        output_losses["reg_term_0"] = reg_term_0
        output_losses["reg_term_1"] = reg_term_1
        output_losses["reg_term_2"] = reg_term_2
        if "reg_term_0_min" in locals() and "reg_term_1_min" in locals() and "reg_term_2_min" in locals():
            output_losses["reg_term_0_min"] = reg_term_0_min
            output_losses["reg_term_1_min"] = reg_term_1_min
            output_losses["reg_term_2_min"] = reg_term_2_min

        total += reg_term + cca_loss_reg

        return total, output_losses, False
