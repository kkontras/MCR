import os

import torch
import wandb
from colorama import Fore


class Monitor_n_Save:

    def __init__(self, agent):
        self.agent = agent

    def save(
        self,
        verbose=False,
        is_best_model=False,
        model_save=True,
        post_test_results=False,
    ):

        file_name = self.agent.config.model.save_dir
        if "data_split" in self.agent.config.dataset and self.agent.config.dataset.data_split.get("split_method", False) == "patients_folds":
            file_name = file_name.format(self.agent.config.dataset.data_split.fold)

        if "save_base_dir" in self.agent.config.model:
            file_name = os.path.join(self.agent.config.model.save_base_dir, file_name)

        self.agent.logs["saved_step"] = self.agent.logs["current_step"]

        unwrapped_model = self.agent.accelerator.unwrap_model(self.agent.model)

        save_dict = {}
        savior = {}
        savior["optimizer_state_dict"] = self.agent.optimizer.state_dict()
        savior["scheduler_state_dict"] = self.agent.scheduler.state_dict()
        savior["logs"] = self.agent.logs
        savior["configs"] = self.agent.config
        if hasattr(self.agent.data_loader.train_loader, "generator"):
            savior["training_dataloder_generator_state"] = self.agent.data_loader.train_loader.generator.get_state()

        if not model_save:
            if os.path.exists(file_name):
                prev_checkpoint = torch.load(file_name, map_location="cpu", weights_only=False)
                if "best_model_state_dict" in prev_checkpoint:
                    savior["best_model_state_dict"] = prev_checkpoint["best_model_state_dict"]
                if "model_state_dict" in prev_checkpoint:
                    savior["model_state_dict"] = prev_checkpoint["model_state_dict"]
        else:
            savior["model_state_dict"] = unwrapped_model.state_dict()

        if is_best_model:
            savior["best_model_state_dict"] = unwrapped_model.state_dict()
        else:
            if os.path.exists(file_name):
                prev_checkpoint = torch.load(file_name, map_location="cpu", weights_only=False)
                if "best_model_state_dict" in prev_checkpoint:
                    savior["best_model_state_dict"] = prev_checkpoint["best_model_state_dict"]
            else:
                savior["best_model_state_dict"] = unwrapped_model.state_dict()
        if post_test_results:
            savior["post_test_results"] = post_test_results
        if hasattr(self.agent.data_loader, "metrics"):
            savior["metrics"] = self.agent.data_loader.metrics

        save_dict.update(savior)

        try:
            self.agent.accelerator.save(save_dict, file_name)
            if verbose:
                self.agent.logger.info(Fore.WHITE + "Models has saved successfully in {}".format(file_name))
        except:
            raise Exception("Problem in model saving")

    def monitoring(self):
        self.agent.accelerator.wait_for_everyone()
        if self.agent.accelerator.is_main_process:
            train_metrics = self.agent.evaluators.train_evaluator.evaluate()
            val_metrics = self.agent.evaluators.val_evaluator.evaluate()

            self._find_learning_rate()

            self._update_train_val_logs(train_metrics=train_metrics, val_metrics=val_metrics)
            wandb.log(
                {"train": train_metrics, "val": val_metrics},
                step=self.agent.logs["current_step"] + 1,
            )

            is_best = self.agent.evaluators.val_evaluator.is_best(metrics=val_metrics, best_logs=self.agent.logs["best_logs"])
            self.print_valid_results(val_metrics, self.agent.logs["current_step"])

            not_saved = True
            if is_best:
                wandb_out = {"best_val": val_metrics}
                self._update_best_logs(
                    current_step=self.agent.logs["current_step"],
                    val_metrics=val_metrics,
                )

                if self.agent.config.training_params.rec_test:
                    test_metrics = self._test_n_update()
                    wandb_out["test"] = test_metrics
                wandb.log(wandb_out, step=self.agent.logs["current_step"] + 1)

                self.agent.logs["steps_no_improve"] = 0
                self.save(verbose=True, is_best_model=True)
                not_saved = False
            else:
                self.agent.logs["steps_no_improve"] += 1
                if self.agent.config.training_params.rec_test and self.agent.config.training_params.test_on_bottoms:
                    self._test_n_update()

            self._early_stop_check_n_save(not_saved)

    def _find_learning_rate(self):
        for param_group in self.agent.optimizer.param_groups:
            self.lr = param_group["lr"]

    def _update_train_val_logs(self, train_metrics, val_metrics):

        train_metrics.update(
            {
                "validate_every": self.agent.config.early_stopping.validate_every,
                "batch_size": self.agent.config.training_params.batch_size,
                "learning_rate": self.agent.scheduler.lr_history[
                    max(
                        self.agent.logs["current_step"] - self.agent.config.early_stopping.validate_every,
                        0,
                    ) : self.agent.logs["current_step"]
                ],
            }
        )

        self.agent.logs["val_logs"][self.agent.logs["current_step"]] = val_metrics
        self.agent.logs["train_logs"][self.agent.logs["current_step"]] = train_metrics

    def _update_best_logs(self, current_step, val_metrics):

        val_metrics.update({"step": current_step})
        self.agent.logs["best_logs"] = val_metrics

    def print_valid_results(self, val_metrics, current_step=None, test=False):

        if self.agent.config.training_params.verbose:
            if current_step is not None:
                step = int(current_step / self.agent.config.early_stopping.validate_every)
                message = Fore.WHITE + "Epoch {0:d} step {1:d} with ".format(self.agent.logs["current_epoch"], step)

            if not self.agent.config.training_params.tdqm_disable and not self.agent.trainer.end_of_epoch_check:
                print()

            if test:
                message = Fore.WHITE + "Test "
            else:
                message = Fore.WHITE + "Val "

            if current_step is not None:
                step = int(current_step / self.agent.config.early_stopping.validate_every)
                message += "Epoch {0:d} step {1:d} with ".format(self.agent.logs["current_epoch"], step)
            else:
                message += "Epoch {0:d} with ".format(self.agent.logs["current_epoch"])

            if "loss" in val_metrics:
                for i, v in val_metrics["loss"].items():
                    message += Fore.RED + "{} : {:.6f} ".format(i, v)
            if "acc" in val_metrics:
                for i, v in val_metrics["acc"].items():
                    message += Fore.LIGHTBLUE_EX + "Acc_{}: {:.2f} ".format(i, v * 100)
            if "top5_acc" in val_metrics:
                for i, v in val_metrics["top5_acc"].items():
                    message += Fore.LIGHTBLUE_EX + "Top5_Acc_{}: {:.2f} ".format(i, v * 100)
            if "acc_exzero" in val_metrics:
                for i, v in val_metrics["acc_exzero"].items():
                    message += Fore.LIGHTBLUE_EX + "Acc_ExZ_{}: {:.2f} ".format(i, v * 100)
            if "f1" in val_metrics:
                for i, v in val_metrics["f1"].items():
                    message += Fore.LIGHTGREEN_EX + "F1_{}: {:.2f} ".format(i, v * 100)
            if "k" in val_metrics:
                for i, v in val_metrics["k"].items():
                    message += Fore.LIGHTGREEN_EX + "K_{}: {:.4f} ".format(i, v)
            if "acc_7" in val_metrics:
                for i, v in val_metrics["acc_7"].items():
                    message += Fore.MAGENTA + "Acc7_{}: {:.4f} ".format(i, v * 100)
            if "acc_5" in val_metrics:
                for i, v in val_metrics["acc_5"].items():
                    message += Fore.LIGHTMAGENTA_EX + "Acc5_{}: {:.4f} ".format(i, v * 100)
            if "mae" in val_metrics:
                for i, v in val_metrics["mae"].items():
                    message += Fore.LIGHTBLUE_EX + "MAE_{}: {:.4f} ".format(i, v)
            if "corr" in val_metrics:
                for i, v in val_metrics["corr"].items():
                    message += Fore.LIGHTWHITE_EX + "Corr_{}: {:.4f} ".format(i, v)
            # if "val_perclassf1" in val_metrics:
            #     for i, v in val_metrics["val_perclassf1"].items(): message += Fore.BLUE + "F1_perclass_{}: {} ".format(i,"{}".format(str(list((v*100).round(2)))))

            if self.agent.accelerator.is_main_process:
                self.agent.logger.info(message)

    def _print_epoch_metrics(self):
        if self.agent.config.training_params.verbose:
            if self.agent.accelerator.is_main_process:
                self.agent.logger.info(
                    "Epoch {0:d}, N: {1:d}, lr: {2:.8f} Validation loss: {3:.6f}, accuracy: {4:.2f}% f1 :{5:.4f},  :{6:.4f}  Training loss: {7:.6f}, accuracy: {8:.2f}% f1 :{9:.4f}, k :{10:.4f},".format(
                        self.agent.logs["current_epoch"],
                        self.agent.logs["current_step"] * self.agent.config.training_params.batch_size * self.agent.config.dataset.seq_legth[0],
                        self.lr,
                        self.agent.logs["val_logs"][self.agent.logs["current_step"]]["loss"],
                        self.agent.logs["val_logs"][self.agent.logs["current_step"]]["acc"] * 100,
                        self.agent.logs["val_logs"][self.agent.logs["current_step"]]["f1"],
                        self.agent.logs["val_logs"][self.agent.logs["current_step"]]["k"],
                        self.agent.logs["train_logs"][self.agent.logs["current_step"]]["loss"],
                        self.agent.logs["train_logs"][self.agent.logs["current_step"]]["acc"] * 100,
                        self.agent.logs["train_logs"][self.agent.logs["current_step"]]["f1"],
                        self.agent.logs["train_logs"][self.agent.logs["current_step"]]["k"],
                    )
                )

    def _test_n_update(self):
        test_metrics = self.agent.evaluators.test_evaluator.evaluate()

        self.agent.logs["test_logs"][self.agent.logs["current_step"]] = test_metrics
        self.print_valid_results(test_metrics, self.agent.logs["current_step"], test=True)
        return test_metrics

    def print_test_results(self, val_metrics):

        if self.agent.config.training_params.verbose:
            message = Fore.WHITE + "Test "
            if "loss" in val_metrics:
                for i, v in val_metrics["loss"].items():
                    if "combined" in i:
                        message += Fore.RED + "{} : {:.6f} ".format(i, v)
            if "acc" in val_metrics:
                for i, v in val_metrics["acc"].items():
                    message += Fore.LIGHTBLUE_EX + "Acc_{}: {:.2f} ".format(i, v * 100)

            if "acc_exzero" in val_metrics:
                for i, v in val_metrics["acc_exzero"].items():
                    message += Fore.LIGHTBLUE_EX + "Acc_ExZ_{}: {:.2f} ".format(i, v * 100)

            # if "test_f1" in val_metrics:
            #     for i, v in val_metrics["test_f1"].items(): message += Fore.LIGHTGREEN_EX + "F1_{}: {:.2f} ".format(i,
            #                                                                                                        v * 100)
            # if "test_k" in val_metrics:
            #     for i, v in val_metrics["test_k"].items(): message += Fore.LIGHTGREEN_EX + "K_{}: {:.4f} ".format(i, v)
            # if "test_perclassf1" in val_metrics:
            #     for i, v in val_metrics["test_perclassf1"].items(): message += Fore.BLUE + "F1_perclass_{}: {} ".format(
            #         i,
            #         "{}".format(
            #             str(list(
            #                 (
            #                         v * 100).round(
            #                     2)))))
            if self.agent.accelerator.is_main_process:
                self.agent.logger.info(message)

    def _early_stop_check_n_save(self, not_saved):

        training_cycle = self.agent.logs["current_step"] // self.agent.config.early_stopping.validate_every
        if not_saved and training_cycle % self.agent.config.early_stopping.save_every_valstep == 0:
            # Some epochs without improvement have passed, we save to avoid losing progress even if its not giving new best
            self.save()

        if training_cycle == self.agent.config.early_stopping.n_steps_stop_after:
            # After 'n_steps_stop_after' we need to start counting till we reach the earlystop_threshold
            self.steps_at_earlystop_threshold = self.agent.logs[
                "steps_no_improve"
            ]  # we dont need to initialize that since training_cycle > self.agent.config.n_steps_stop_after will not be true before ==

        if (
            training_cycle > self.agent.config.early_stopping.n_steps_stop_after
            and self.agent.logs["steps_no_improve"] >= self.agent.config.early_stopping.n_steps_stop - 1
        ):
            self.agent.evaluators.train_evaluator.set_early_stop()

        print(self.agent.evaluators.train_evaluator.early_stop)
