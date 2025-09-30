from colorama import Fore, Style

from datasets.AVE.AVE_Dataset import *
from datasets.CREMAD.CREMAD_Dataset import *
from datasets.SthSth.dataset_factory import *
from datasets.UCF.UCF_Dataset import *
from models.MCR_Models import *
from models.SthSth_models import *
from utils.config import process_config, process_config_default

# from datasets.Synthetic_FactorCL.Synthetic_FCL_Dataset import *
# from datasets.Factor_CL_Datasets.FactorCL_Datasets import *


class Importer:
    def __init__(
        self,
        config_name: str,
        device: str = "cuda:0",
        default_files: list = None,
        fold: int = None,
    ):

        if default_files is not None:
            self.config = process_config_default(json_file=config_name, default_files=default_files, printing=False)
        else:
            self.config = process_config(json_file=config_name, printing=False)
        self.device = device
        self.fold = fold

    def load_checkpoint(self):

        if "save_base_dir" in self.config.model:
            self.config.model.save_dir = os.path.join(self.config.model.save_base_dir, self.config.model.save_dir)

        if "model" not in self.config:
            self.checkpoint = torch.load(self.config.save_dir, map_location="cpu", weights_only=False)
        else:
            self.checkpoint = torch.load(self.config.model.save_dir, map_location="cpu", weights_only=False)

    def change_config(self, attr, value, c=None):
        if c == None:
            c = self.config
        attr_splits = attr.split(".")
        for attr_split in attr_splits[:-1]:
            c = getattr(c, attr_split)
        setattr(c, attr_splits[-1], value)

    def get_dataloaders(self):

        dataloader = globals()[self.config.dataset.dataloader_class]
        data_loader = dataloader(config=self.config)
        if hasattr(self, "checkpoint") and hasattr(self.checkpoint, "metrics"):
            data_loader.load_metrics_ongoing(self.checkpoint["metrics"])
        if hasattr(self, "checkpoint") and hasattr(self.checkpoint, "logs") and hasattr(self.checkpoint["logs"], "weights"):
            data_loader.weights = self.checkpoint["logs"]["weights"]

        return data_loader

    def get_model(self, model=None, return_model: str = "best_model"):

        if not model:
            model_class = globals()[self.config.model.model_class]

            if "save_base_dir" in self.config.model and "swin_backbone" in self.config.model.args:
                self.config.model.args.swin_backbone = os.path.join(
                    self.config.model.save_base_dir,
                    self.config.model.args.swin_backbone,
                )
            if "save_base_dir" in self.config.model and "pretraining_paths" in self.config.model.args:
                self.config.model.args.pretraining_paths = {
                    i: os.path.join(
                        self.config.model.save_base_dir,
                        self.config.model.args.pretraining_paths[i],
                    )
                    for i in self.config.model.args.pretraining_paths
                }

            enc = self._load_encoder(encoders=self.config.model.get("encoders", []))
            model = model_class(encs=enc, args=self.config.model.args)

            model = model.to(self.device)

        if return_model == "untrained_model":
            return model
        elif return_model == "best_model":

            self.checkpoint["best_model_state_dict"] = {key.replace("module.", ""): value for key, value in self.checkpoint["best_model_state_dict"].items()}

            self.checkpoint["best_model_state_dict"] = {
                key.replace("parametrizations.weight.original0", "weight_g"): value for key, value in self.checkpoint["best_model_state_dict"].items()
            }
            self.checkpoint["best_model_state_dict"] = {
                key.replace("parametrizations.weight.original1", "weight_v"): value for key, value in self.checkpoint["best_model_state_dict"].items()
            }

            print("Loading best model from {}".format(self.config.model.save_dir))
            model.load_state_dict(self.checkpoint["best_model_state_dict"])
            return model
        elif return_model == "running_model":
            model.load_state_dict(self.checkpoint["model_state_dict"])
            return model
        else:
            raise ValueError('Return such model does not exits as option, choose from "best_model","running_model", "untrained_model" ')

    def print_progress(
        self,
        multi_fold_results,
        verbose=True,
        print_entropy=False,
        latex_version=False,
        print_post_test=True,
    ):

        val_metrics = self.checkpoint["logs"]["best_logs"]

        multi_fold_results[self.fold] = val_metrics
        if verbose:
            print("-- Best Validation --")
        latex_message = {}
        if "acc" not in val_metrics:
            current_epoch = self.checkpoint["logs"]["current_epoch"] if "current_epoch" not in val_metrics else val_metrics["current_epoch"]
            message = Style.BRIGHT + Fore.WHITE + "Epoch: {}, No_improve: {} ".format(current_epoch, self.checkpoint["logs"]["steps_no_improve"])
            if "loss" in val_metrics:
                for i, v in val_metrics["loss"].items():
                    message += Fore.RED + "{} : {:.6f} ".format(i, val_metrics["loss"][i])
            if verbose:
                print(message + Style.RESET_ALL)

        else:
            current_epoch = self.checkpoint["logs"]["current_epoch"] if "current_epoch" not in val_metrics else val_metrics["current_epoch"]

            for pred in val_metrics["acc"]:
                message = Style.BRIGHT + Fore.WHITE + "Step: {}, No_improve: {} ".format(current_epoch, self.checkpoint["logs"]["steps_no_improve"])
                if "loss" in val_metrics:
                    for i, v in val_metrics["loss"].items():
                        if pred in i or i == "total":
                            message += Fore.RED + "{} : {:.6f} ".format(i, val_metrics["loss"][i])
                if "acc" in val_metrics:
                    if pred in val_metrics["acc"]:
                        message += Fore.LIGHTBLUE_EX + "Acc_{}: {:.2f} ".format(pred, val_metrics["acc"][pred] * 100)
                if "k" in val_metrics:
                    if pred in val_metrics["k"]:
                        message += Fore.LIGHTGREEN_EX + "K_{}: {:.3f} ".format(pred, val_metrics["k"][pred])
                if "f1" in val_metrics:
                    if pred in val_metrics["f1"]:
                        message += Fore.LIGHTGREEN_EX + "F1_{}: {:.2f} ".format(pred, val_metrics["f1"][pred] * 100)
                if "perclassf1" in val_metrics:
                    if pred in val_metrics["perclassf1"]:
                        message += Fore.BLUE + "F1_perclass_{}: {} ".format(
                            pred,
                            "{}".format(str(list((val_metrics["perclassf1"][pred] * 100).round(2)))),
                        )

                if verbose:
                    print(message + Style.RESET_ALL)

        if self.config.training_params.rec_test and "test_logs" in self.checkpoint["logs"] and len(self.checkpoint["logs"]["test_logs"]) > 0 and "step" in val_metrics:

            test_best_logs = self.checkpoint["logs"]["test_logs"][val_metrics["step"]]

            if "acc" not in test_best_logs:
                if "test_acc" in test_best_logs:
                    test_best_logs = {k.replace("test_", ""): v for k, v in test_best_logs.items()}
            if "acc" in test_best_logs:
                if verbose:
                    print("-- Best Test --")

                for pred in test_best_logs["acc"]:

                    message = Style.BRIGHT + Fore.WHITE + "Best Test "
                    if "loss" in test_best_logs:
                        for i, v in test_best_logs["loss"].items():
                            if pred in i or i == "total":
                                message += Fore.RED + "{} : {:.6f} ".format(i, test_best_logs["loss"][i])
                    if "acc" in test_best_logs:
                        if pred in test_best_logs["acc"]:
                            message += Fore.LIGHTBLUE_EX + "Acc_{}: {:.2f} ".format(pred, test_best_logs["acc"][pred] * 100)
                    if "k" in test_best_logs:
                        if pred in test_best_logs["k"]:
                            message += Fore.LIGHTGREEN_EX + "K_{}: {:.3f} ".format(pred, test_best_logs["k"][pred])
                    if "f1" in test_best_logs:
                        if pred in test_best_logs["f1"]:
                            message += Fore.LIGHTGREEN_EX + "F1_{}: {:.2f} ".format(pred, test_best_logs["f1"][pred] * 100)
                    if "perclassf1" in test_best_logs:
                        if pred in test_best_logs["perclassf1"]:
                            message += Fore.BLUE + "F1_perclass_{}: {} ".format(
                                pred,
                                "{}".format(str(list((test_best_logs["perclassf1"][pred] * 100).round(2)))),
                            )

                if verbose:
                    print(message)

        def _print_test_results(metrics, verbose, description, multi_fold_results, print_entropy=False):
            # description = "--- Post Test ---"
            latex_message = {}
            # message = Style.BRIGHT + Fore.WHITE + "{} ".format(description)
            if verbose:
                print(Style.BRIGHT + Fore.WHITE + "{} ".format(description))
            for pred in metrics["acc"]:
                message = "{} ".format(pred)
                latex_message[pred] = "{} & ".format(pred)

                if "acc" in metrics:
                    if pred in metrics["acc"]:
                        message += Fore.LIGHTBLUE_EX + "Acc: {:.1f} ".format(metrics["acc"][pred] * 100)
                        latex_message[pred] += " {:.1f} &".format(metrics["acc"][pred] * 100)

                if "k" in metrics:
                    if pred in metrics["k"]:
                        message += Fore.LIGHTGREEN_EX + "K: {:.3f} ".format(metrics["k"][pred])
                        latex_message[pred] += " {:.3f} &".format(metrics["k"][pred])

                if "f1" in metrics:
                    if pred in metrics["f1"]:
                        message += Fore.LIGHTGREEN_EX + "F1: {:.1f} ".format(metrics["f1"][pred] * 100)
                        latex_message[pred] += " {:.1f} &".format(metrics["f1"][pred] * 100)

                if "ece" in metrics:
                    if pred in metrics["ece"]:
                        message += Fore.LIGHTRED_EX + "ECE: {:.3f} ".format(metrics["ece"][pred])
                        latex_message[pred] += " {:.3f} &".format(metrics["ece"][pred])

                # if "f1_perclass" in metrics:
                #     if pred in metrics["f1_perclass"]:
                #         message += Fore.BLUE + "F1_perclass: {} ".format("{}".format(
                #             str(list((metrics["f1_perclass"][pred] * 100).round(1)))))
                #         for i in list((metrics["f1_perclass"][pred] * 100).round(2)):
                #             latex_message[pred] += " {:.1f} &".format(i)

                if verbose:
                    print(message + Style.RESET_ALL)
                if verbose:
                    print(latex_message[pred] + Style.RESET_ALL)

                # TODO: Make sure that this works to accumulate both the skipped and the normal cases, combined tags could get confused together
                multi_fold_results.update({self.fold: metrics})
                if "step" not in val_metrics:
                    val_metrics["step"] = -1
                multi_fold_results[self.fold]["best_step"] = int(val_metrics["step"] / self.config.early_stopping.validate_every)
                multi_fold_results[self.fold]["steps_no_improve"] = self.checkpoint["logs"]["steps_no_improve"]

                if print_entropy:
                    for pred in metrics["entropy"]:
                        message = ""
                        if "entropy" in metrics:
                            if pred in metrics["entropy"]:
                                message += Fore.LIGHTRED_EX + "E_{}: {:.4f} ".format(pred, metrics["entropy"][pred])
                        if "entropy_var" in metrics:
                            if pred in metrics["entropy_var"]:
                                message += Fore.LIGHTRED_EX + "E_var_{}: {:.4f} ".format(pred, metrics["entropy_var"][pred])
                        if "entropy_correct" in metrics:
                            if pred in metrics["entropy_correct"]:
                                message += Fore.LIGHTMAGENTA_EX + "EC_{}: {:.4f} ".format(pred, metrics["entropy_correct"][pred])
                        if "entropy_correct_var" in metrics:
                            if pred in metrics["entropy_correct_var"]:
                                message += Fore.LIGHTMAGENTA_EX + "EC_{}: {:.4f} ".format(pred, metrics["entropy_correct_var"][pred])
                        if "entropy_wrong" in metrics:
                            if pred in metrics["entropy_wrong"]:
                                message += Fore.LIGHTYELLOW_EX + "EW_{}: {:.4f} ".format(pred, metrics["entropy_wrong"][pred])
                        if "entropy_wrong_var" in metrics:
                            if pred in metrics["entropy_wrong_var"]:
                                message += Fore.LIGHTYELLOW_EX + "EW_var_{}: {:.4f} ".format(pred, metrics["entropy_wrong_var"][pred])

                        if verbose:
                            print(message + Style.RESET_ALL)
            return metrics

        test_results = False
        if "post_test_results" in self.checkpoint and print_post_test:
            # test_flag = True
            metrics = self.checkpoint["post_test_results"]
            test_results = _print_test_results(
                metrics=metrics,
                verbose=verbose,
                description="--- Post Test ---",
                print_entropy=print_entropy,
                multi_fold_results=multi_fold_results,
            )
        else:
            if "test_best_logs" in locals():
                test_results = test_best_logs
                # multi_fold_results.update({self.fold: test_best_logs})

        step = self.checkpoint["logs"]["current_step"] if "step" not in val_metrics else val_metrics["step"]
        val_metrics["best_step"] = int(step / self.config.early_stopping.validate_every)
        val_metrics["steps_no_improve"] = self.checkpoint["logs"]["steps_no_improve"]
        val_metrics["current_epoch"] = self.checkpoint["logs"]["current_epoch"]

        return val_metrics, test_results

    def _my_numel(self, m: torch.nn.Module, only_trainable: bool = False, verbose=True):

        parameters = list(m.parameters())
        if only_trainable:
            parameters = [p for p in parameters if p.requires_grad]
        unique = {p.data_ptr(): p for p in parameters}.values()
        model_total_params = sum(p.numel() for p in unique)
        if verbose:
            print("Total number of trainable parameters are: {}".format(model_total_params))

        return model_total_params

    def _load_encoder(self, encoders):
        encs = []
        for num_enc in range(len(encoders)):
            enc_class = globals()[encoders[num_enc]["model"]]
            args = encoders[num_enc]["args"]
            if "encoders" in encoders[num_enc]:
                enc_enc = self._load_encoder(encoders=encoders[num_enc]["encoders"])
                enc = enc_class(encs=enc_enc, args=args)
            else:
                enc = enc_class(args=args, encs=[])

            pretrained_enc_args = encoders[num_enc].get("pretrainedEncoder", {})
            if pretrained_enc_args.get("use", False):
                pre_dir = pretrained_enc_args.get("dir", "")
                if self.fold is not None:
                    pre_dir = pre_dir.format(self.fold)
                else:
                    pre_dir = pre_dir.format("")
                if "save_base_dir" in self.config.model:
                    pre_dir = os.path.join(self.config.model.save_base_dir, pre_dir)
                print("Loading encoder from {}".format(pre_dir))
                checkpoint = torch.load(pre_dir)
                if "encoder_state_dict" in checkpoint:
                    checkpoint["encoder_state_dict"] = {key.replace("module.", ""): value for key, value in checkpoint["encoder_state_dict"].items()}
                    enc.load_state_dict(checkpoint["encoder_state_dict"])
                elif "model_state_dict" in checkpoint:
                    if "VaVL" not in encoders[num_enc]["model"]:
                        print("Replacing module")
                        checkpoint["best_model_state_dict"] = {key.replace("module.", ""): value for key, value in checkpoint["best_model_state_dict"].items()}
                    enc.load_state_dict(checkpoint["best_model_state_dict"])

            encs.append(enc)
        return encs
