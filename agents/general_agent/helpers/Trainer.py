import copy
import time
from collections import defaultdict

import torch
from colorama import Fore
from tqdm import tqdm


class Trainer:

    def __init__(self, agent):
        self.agent = agent

        if self.agent.config.get("task", "classification") == "classification" or self.agent.config.get("task", "classification") == "bias_measure":
            self.train_step_func = "train_one_step"  # self._find_train_step_func()
        elif self.agent.config.get("task", "classification") == "regression":
            self.train_step_func = "train_one_step_regression"
        elif self.agent.config.get("task", "classification") == "shapley_classification":
            self.train_step_func = "train_one_step_shapley"

        self.this_train_step_func = getattr(self, self.train_step_func)
        self._get_loss_weights()
        self.end_of_epoch_check = self.agent.config.early_stopping.get("end_of_epoch_check", False)
        if self.end_of_epoch_check:
            self.agent.config.early_stopping.validate_every = len(self.agent.data_loader.train_loader)

    def train_steps(self):

        # self.agent.validator_tester.validate()
        # self.agent.monitor_n_saver.monitoring()

        self.agent.model.train()
        self.agent.mem_loader._my_numel(self.agent.model, only_trainable=True)
        self.agent.start = time.time()

        self.running_values = {
            "targets": [],
            "preds": [],
            "batch_loss": [],
            "cond_speed": [],
            "early_stop": False,
            "saved_at_valstep": 0,
            "prev_epoch_time": 0,
            "val_loss": {"combined": 0},
        }

        for current_epoch in range(self.agent.logs["current_epoch"], self.agent.config.early_stopping.max_epoch):
            self.agent.logs["current_epoch"] = copy.deepcopy(current_epoch)
            self.agent.bias_infuser.on_epoch_begin(current_epoch=self.agent.logs["current_epoch"])
            self.agent.evaluators.train_evaluator.reset()

            pbar = tqdm(
                enumerate(self.agent.data_loader.train_loader),
                total=len(self.agent.data_loader.train_loader),
                desc="Training",
                leave=None,
                disable=self.agent.config.training_params.tdqm_disable or not self.agent.accelerator.is_main_process,
                position=0,
            )
            for batch_idx, served_dict in pbar:

                if type(served_dict) == tuple:  # FACTORCL DATASETS
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

                if self.agent.config.model.load_ongoing:
                    if self.agent.logs["current_step"] > self.agent.logs["current_epoch"] * len(self.agent.data_loader.train_loader) + batch_idx:
                        self.agent.logger.info(f"Skipping batch {batch_idx} due to load_ongoing experiment")
                        continue

                self.agent.optimizer.zero_grad()
                step_outcome, optstep_done = self.this_train_step_func(served_dict)
                self._clip_grads()

                if not optstep_done:
                    self.agent.optimizer.step()
                self.agent.scheduler.step(
                    step=self.agent.logs["current_step"] + 1,
                    loss=step_outcome["loss"]["total"].item(),
                )

                all_outputs = self.agent.accelerator.gather(step_outcome)

                self.agent.evaluators.train_evaluator.process(all_outputs)

                del served_dict, step_outcome, all_outputs
                # torch.cuda.empty_cache()
                pbar_message = self.local_logging(batch_idx, False)
                pbar.set_description(pbar_message)
                pbar.refresh()

                if self.agent.evaluators.train_evaluator.get_early_stop():
                    return
                self.agent.logs["current_step"] += 1
                if self.agent.logs["current_step"] - self.agent.logs["saved_step"] > self.agent.config.early_stopping.get("save_every_step", float("inf")):
                    self.agent.accelerator.wait_for_everyone()
                    if self.agent.accelerator.is_main_process:
                        self.agent.monitor_n_saver.save(verbose=True)

            self.agent.logs["current_epoch"] += 1
            self.local_logging(batch_idx, True)

    def train_one_step(self, served_dict, **kwargs):

        data = {view: served_dict["data"][view].to(self.agent.accelerator.device) for view in served_dict["data"] if type(served_dict["data"][view]) is torch.Tensor}

        data.update({view: data[view].float() for view in data if type(view) == int})

        label = served_dict["label"].type(torch.LongTensor).to(self.agent.accelerator.device)

        bias_method = self.agent.config.model.args.get("bias_infusion", {}).get("method", False)

        self.agent.optimizer.zero_grad()

        output = self.agent.model(data, label=label, return_features=True)

        def calculate_loss(output, label):
            total_loss = torch.zeros(1).squeeze().to(self.agent.accelerator.device)
            output_losses, ce_loss = {}, {}

            if hasattr(self.agent.config.model.args, "multi_loss"):
                for k, v in output["preds"].items():

                    if k in self.agent.config.model.args.multi_loss.multi_supervised_w and self.agent.config.model.args.multi_loss.multi_supervised_w[k] != 0:
                        if len(label) > 0:  # TODO: Check if this one needs to be one or zero
                            if len(label.shape) > 1:
                                label = label.flatten()

                            ce_loss[k] = self.agent.loss(v, label.to(self.agent.accelerator.device))
                            total_loss += self.w_loss[k] * ce_loss[k]
                            output_losses.update({"ce_loss_{}".format(k): self.w_loss[k] * ce_loss[k]})

            return total_loss, output_losses

        total_loss, output_losses = calculate_loss(output, label)
        if "losses" in output:
            if self.agent.logs["current_epoch"] < self.agent.config.model.args.get("bias_infusion", {}).get("starting_epoch", 0):
                total_loss = torch.cat(
                    [output["losses"][i].unsqueeze(dim=0) for i in output["losses"]],
                    dim=0,
                ).sum()
            else:
                total_loss += torch.cat(
                    [output["losses"][i].unsqueeze(dim=0) for i in output["losses"]],
                    dim=0,
                ).sum()
            output_losses.update({i: output["losses"][i] for i in output["losses"]})

        total_loss, output_losses, optstep_done = self.agent.bias_infuser.before_backward(
            total=total_loss,
            output_losses=output_losses,
            w_loss=self.w_loss,
            loss_fun=calculate_loss,
            data=data,
            label=label,
            output=output,
        )

        if total_loss.requires_grad and not optstep_done:
            self.agent.accelerator.backward(total_loss)
        else:
            optstep_done = True

        if "c" in output["preds"] and "g" in output["preds"]:
            if bias_method == "OGM-Mine_3d":
                self.agent.bias_infuser.on_backward_end(
                    label=label.detach(),
                    out_color=output["preds"]["c"].detach(),
                    out_gray=output["preds"]["g"].detach(),
                    out_f=output["preds"]["f"].detach(),
                )
            else:
                self.agent.bias_infuser.on_backward_end(label=label.detach().cpu(), preds=output["preds"])
        this_output = {}

        for i in output_losses:
            output_losses[i] = output_losses[i].detach()
        total_loss = total_loss.detach()
        output_losses.update({"total": total_loss})
        this_output.update(
            {
                "loss": output_losses,
                "pred": {pred: output["preds"][pred].detach() for pred in output["preds"]},
                "label": label.detach().to(self.agent.accelerator.device),
            }
        )

        return this_output, optstep_done

    def train_one_step_regression(self, served_dict, **kwargs):

        data = {view: served_dict["data"][view].to(self.agent.accelerator.device) for view in served_dict["data"] if type(served_dict["data"][view]) is torch.Tensor}
        data.update({view: data[view].float() for view in data if type(view) == int})
        bias_method = self.agent.config.model.args.get("bias_infusion", {}).get("method", False)

        label = served_dict["label"].to(self.agent.accelerator.device)

        self.agent.optimizer.zero_grad()

        output = self.agent.model(data, return_features=True, label=label)

        def calculate_loss(output, label):
            total_loss = torch.zeros(1).squeeze().to(self.agent.accelerator.device)
            output_losses, ce_loss = {}, {}

            if hasattr(self.agent.config.model.args, "multi_loss"):
                for k, v in output["preds"].items():
                    if k in self.agent.config.model.args.multi_loss.multi_supervised_w and self.agent.config.model.args.multi_loss.multi_supervised_w[k] != 0:
                        ce_loss[k] = torch.nn.L1Loss()(v.squeeze(), label.to(self.agent.accelerator.device))
                        total_loss += self.w_loss[k] * ce_loss[k]
                        output_losses.update({"ce_loss_{}".format(k): self.w_loss[k] * ce_loss[k]})

            return total_loss, output_losses

        total_loss, output_losses = calculate_loss(output, label)

        total_loss, output_losses, optstep_done = self.agent.bias_infuser.before_backward(
            total=total_loss,
            output_losses=output_losses,
            w_loss=self.w_loss,
            loss_fun=calculate_loss,
            data=data,
            label=label,
            output=output,
        )

        if total_loss.requires_grad and not optstep_done:
            self.agent.accelerator.backward(total_loss)
        else:
            optstep_done = True

        for i in output_losses:
            output_losses[i] = output_losses[i].detach()

        if "c" in output["preds"] and "g" in output["preds"]:
            if bias_method == "MLB_3d" or bias_method == "MLB_3d_Reg":
                self.agent.bias_infuser.on_backward_end(
                    label=label.detach(),
                    out_mod0=output["preds"]["c"].detach(),
                    out_mod1=output["preds"]["g"].detach(),
                    out_mod2=output["preds"]["f"].detach(),
                )
            else:
                self.agent.bias_infuser.on_backward_end(
                    label=label.detach().cpu().cpu(),
                    out_mod0=output["preds"]["c"].detach().cpu(),
                    out_mod1=output["preds"]["g"].detach().cpu(),
                )

        this_output = {}

        total_loss = total_loss.detach()
        output_losses.update({"total": total_loss})
        this_output.update(
            {
                "loss": output_losses,
                "pred": {pred: output["preds"][pred].detach() for pred in output["preds"]},
                "label": label.detach(),
            }
        )

        return this_output, optstep_done

    def local_logging(self, batch_idx, end_of_epoch=None):

        mean_batch_loss, mean_batch_loss_message = self.agent.evaluators.train_evaluator.mean_batch_loss()

        if (
            self.end_of_epoch_check
            and end_of_epoch
            or not self.end_of_epoch_check
            and self.agent.logs["current_step"] % self.agent.config.early_stopping.validate_every == 0
            and self.agent.logs["current_step"] // self.agent.config.early_stopping.validate_every >= self.agent.config.early_stopping.validate_after
            and batch_idx != 0
        ):

            self.agent.model.load_state_dict(self.agent.accelerator.get_state_dict(self.agent.model))

            self.agent.bias_infuser.plot_bias()
            self.agent.validator_tester.validate()
            if self.agent.config.training_params.rec_test:
                self.agent.validator_tester.validate(test_set=True)
            self.agent.monitor_n_saver.monitoring()
            if self.agent.evaluators.train_evaluator.get_early_stop():
                return
            self.agent.model.train()

        pbar_message = Fore.WHITE + "Training batch {0:d}/{1:d} steps no improve {2:d} with {3:}".format(
            batch_idx,
            len(self.agent.data_loader.train_loader) - 1,
            self.agent.logs["steps_no_improve"],
            mean_batch_loss_message,
        )
        return pbar_message

    def _clip_grads(self):

        clip_method = self.agent.config.model.args.get("clip_grad", False)
        bias_method = self.agent.config.model.args.get("bias_infusion", {}).get("method", False)

        if (bias_method == "AGM" or bias_method == "AGM_3mod") and clip_method == "AGM":
            named_modules = [i[0] for i in self.agent.model.named_children()]

            grad_max = torch.Tensor([-100]).to(self.agent.accelerator.device)
            grad_min = torch.Tensor([100]).to(self.agent.accelerator.device)

            if "fc_0_lin" in named_modules:
                if self.agent.model.fc_0_lin.weight.grad is not None:
                    grad_max = max(grad_max, torch.max(self.agent.model.fc_0_lin.weight.grad))
                    grad_min = min(grad_min, torch.min(self.agent.model.fc_0_lin.weight.grad))
                if self.agent.model.fc_1_lin.weight.grad is not None:
                    grad_max = max(grad_max, torch.max(self.agent.model.fc_1_lin.weight.grad))
                    grad_min = min(grad_min, torch.min(self.agent.model.fc_1_lin.weight.grad))
            if "common_fc" in named_modules:
                for i in self.agent.model.common_fc.parameters():
                    if i.grad is not None:
                        grad_max = max(grad_max, torch.max(i.grad))
                        grad_min = min(grad_min, torch.min(i.grad))
            if "classifier" in named_modules:
                for i in self.agent.model.classifier.parameters():
                    if i.grad is not None:
                        grad_max = max(grad_max, torch.max(i.grad))
                        grad_min = min(grad_min, torch.min(i.grad))
            if "fusion" in named_modules:
                for i in self.agent.model.fusion.parameters():
                    if i.grad is not None:
                        grad_max = max(grad_max, torch.max(i.grad))
                        grad_min = min(grad_min, torch.min(i.grad))
            if grad_max == -100 and self.agent.model.cls_type != "dec":
                raise NotImplementedError("We have not implemented clip grad for this model")
            if bias_method == "AGM":
                if grad_max > 1 or grad_min < -1:
                    self.agent.accelerator.clip_grad_norm_(
                        self.agent.model.parameters(),
                        max_norm=self.agent.config.model.args.get("clip_value", 1.0),
                    )
            elif bias_method == "AGM_3mod":
                if grad_max > 5 or grad_min < -5:
                    self.agent.accelerator.clip_grad_norm_(
                        self.agent.model.parameters(),
                        max_norm=self.agent.config.model.args.get("clip_value", 5.0),
                    )
        elif clip_method == True:
            self.agent.accelerator.clip_grad_norm_(
                self.agent.model.parameters(),
                max_norm=self.agent.config.model.args.get("clip_value", 1.0),
            )

    def _get_loss_weights(self):

        w_loss = defaultdict(int)
        w_loss["total"] = 1
        if "multi_loss" in self.agent.config.model.args:
            if "multi_supervised_w" in self.agent.config.model.args.multi_loss:
                for (
                    k,
                    v,
                ) in self.agent.config.model.args.multi_loss.multi_supervised_w.items():
                    w_loss[k] = v
        else:
            w_loss["total"] = 1
        if hasattr(self.agent.logs, "w_loss") and self.agent.config.model.get("load_ongoing", False):
            self.w_loss = self.agent.logs.w_loss
        else:
            self.w_loss = w_loss
            self.agent.logs["w_loss"] = w_loss

        self.agent.logger.info("Loss Weights are {}".format(dict(self.w_loss)))
