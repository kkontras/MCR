from collections import defaultdict

import numpy as np
import torch
import torchmetrics
from colorama import Fore, Style
from scipy.special import softmax
from scipy.stats import entropy
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score, roc_auc_score
from torchmetrics import Accuracy
from tqdm.auto import tqdm


def multiclass_acc(preds, truths):
    """
    Compute the multiclass accuracy w.r.t. groundtruth

    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))


class Validator:
    def __init__(self, config, device, model=None, data_loader=None):
        self.config = config
        self.device = device
        self.model = model
        self.data_loader = data_loader

    def get_set(self, set):
        if set == "Validation":
            this_dataloader = self.data_loader.valid_loader
        elif set == "Test":
            this_dataloader = self.data_loader.test_loader
        elif set == "Train":
            this_dataloader = self.data_loader.train_loader
        elif set == "Total":
            this_dataloader = self.data_loader.total_loader
        else:
            raise ValueError('This set {} does not exist, options are "Validation", "Test", Train" "Total"'.format(set))
        return this_dataloader

    def get_results(
        self,
        set: str = "Validation",
        print_results: bool = False,
        show_border_info: bool = False,
    ):

        this_dataloader = self.get_set(set)
        metrics = self.validate(
            data_loader=this_dataloader,
            description=set,
            show_border_info=show_border_info,
        )
        self._print_results(metrics=metrics, description=set)
        # if print_results:
        #     self.norm_n_plot_confusion_matrix(metrics["conf"]["combined"], description=set)
        return metrics

    def get_features(self, set: str = "Validation"):
        this_dataloader = self.get_set(set)
        self.model.eval()
        self.model.cuda()
        with torch.no_grad():
            tts, features, inits = [], defaultdict(list), []
            pbar = tqdm(
                enumerate(this_dataloader),
                total=len(this_dataloader),
                desc="Calculating Features",
                leave=False,
            )
            # for batch_idx, (data, target, init, _) in pbar:
            for batch_idx, served_dict in pbar:

                data = {view: served_dict["data"][view].cuda() for view in served_dict["data"] if type(served_dict["data"][view]) is torch.Tensor}

                data.update({view: data[view].float().cuda() for view in data if type(view) == int})

                target = served_dict["label"].type(torch.LongTensor).cuda()

                output = self.model(data, return_features=True)

                for i in output["features"]:
                    features[i].append(output["features"][i].cpu().detach())

                tts.append(target.cpu())

        for key in features:
            features[key] = torch.concat(features[key])
        tts = torch.concat(tts)

        # save in a file that will open up the fastest both features and tts
        torch.save(
            {"features": features, "labels": tts},
            "pretrained_feature_dataset_{}.pt".format(set),
        )

        return features, tts

    def _perf_measure(self, y_actual, y_hat):
        TP = 0
        FP = 0
        TN = 0
        FN = 0

        for i in range(len(y_hat)):
            if y_actual[i] == y_hat[i] == 1:
                TP += 1
            if y_hat[i] == 1 and y_actual[i] != y_hat[i]:
                FP += 1
            if y_actual[i] == y_hat[i] == 0:
                TN += 1
            if y_hat[i] == 0 and y_actual[i] != y_hat[i]:
                FN += 1

        return (TP, FP, TN, FN)

    def _get_loss_weights(self):

        if "multi_loss" in self.config.model.args and "renew_each_step" in self.config.model.args.multi_loss and self.config.model.args.multi_loss.renew_each_step:
            w_loss = defaultdict(int)
            if "multi_loss" in self.config.model.args and "multi_loss_weights" in self.config.model.args.multi_loss:

                if "multi_supervised_loss" in self.config.model.args.multi_loss.multi_loss_weights:
                    for (
                        k,
                        v,
                    ) in self.config.model.args.multi_loss.multi_loss_weights.multi_supervised_loss.items():
                        w_loss[k] = v
                w_loss["alignments"] = (
                    self.config.model.args.multi_loss.multi_loss_weights["alignment_loss"]
                    if "alignment_loss" in self.config.model.args.multi_loss.multi_loss_weights
                    else 0
                )
                w_loss["order"] = (
                    self.config.model.args.multi_loss.multi_loss_weights["order_loss"] if "order_loss" in self.config.model.args.multi_loss.multi_loss_weights else 0
                )
                w_loss["consistency"] = (
                    self.config.model.args.multi_loss.multi_loss_weights["consistency_loss"]
                    if "consistency_loss" in self.config.model.args.multi_loss.multi_loss_weights
                    else 0
                )
                w_loss["reconstruction"] = (
                    self.config.model.args.multi_loss.multi_loss_weights["reconstruction"]
                    if "reconstruction" in self.config.model.args.multi_loss.multi_loss_weights
                    else 0
                )
            else:
                w_loss["total"] = 1
                # raise Warning("We dont have multi supervised loss weights")
            self.w_loss = w_loss
        print("Loss Weights are", dict(self.w_loss))

    def _calculate_classification_metrics(self, tts, preds):
        multiclass = True

        ece = torchmetrics.CalibrationError(num_classes=self.config.model.args.num_classes, task="MULTICLASS")
        total_preds, total_preds_nonargmaxed, metrics = {}, {}, defaultdict(dict)
        for pred_key in preds[0].keys():
            total_preds_nonargmaxed[pred_key] = np.concatenate([pred[pred_key].cpu().numpy() for pred in preds], axis=0)
            metrics["loss"][pred_key] = torch.nn.CrossEntropyLoss()(torch.tensor(total_preds_nonargmaxed[pred_key]), torch.tensor(tts)).item()
            metrics["entropy"][pred_key] = entropy(softmax(total_preds_nonargmaxed[pred_key], axis=1), axis=1)
            total_preds[pred_key] = total_preds_nonargmaxed[pred_key].argmax(axis=-1)
            metrics["entropy_correct"][pred_key] = metrics["entropy"][pred_key][total_preds[pred_key] == tts].mean()
            metrics["entropy_correct_var"][pred_key] = metrics["entropy"][pred_key][total_preds[pred_key] == tts].std()
            metrics["entropy_wrong"][pred_key] = metrics["entropy"][pred_key][total_preds[pred_key] != tts].mean()
            metrics["entropy_wrong_var"][pred_key] = metrics["entropy"][pred_key][total_preds[pred_key] != tts].std()
            metrics["entropy_var"][pred_key] = metrics["entropy"][pred_key].std()
            metrics["entropy"][pred_key] = metrics["entropy"][pred_key].mean()
            metrics["ece_correct"][pred_key] = ece(
                torch.from_numpy(total_preds_nonargmaxed[pred_key][total_preds[pred_key] == tts]),
                torch.from_numpy(tts[total_preds[pred_key] == tts]),
            )
            metrics["ece_false"][pred_key] = ece(
                torch.from_numpy(total_preds_nonargmaxed[pred_key][total_preds[pred_key] != tts]),
                torch.from_numpy(tts[total_preds[pred_key] != tts]),
            )
            metrics["ece"][pred_key] = ece(
                torch.from_numpy(total_preds_nonargmaxed[pred_key]),
                torch.from_numpy(tts),
            )
            metrics["acc"][pred_key] = np.equal(tts, total_preds[pred_key]).sum() / len(tts)
            # if self.config.model.args.num_classes >5:
            #     metrics["top5_acc"][pred_key] = top_k_accuracy_score(tts, total_preds_nonargmaxed[pred_key], k=5)
            metrics["f1"][pred_key] = f1_score(total_preds[pred_key], tts, average="macro")
            metrics["k"][pred_key] = cohen_kappa_score(total_preds[pred_key], tts)
            metrics["f1_perclass"][pred_key] = f1_score(total_preds[pred_key], tts, average=None)
            metrics["auc"][pred_key] = roc_auc_score(tts, total_preds[pred_key]) if not multiclass else 0
            metrics["conf"][pred_key] = confusion_matrix(tts, total_preds[pred_key])

            # if pred_key == "combined":
            #     ceu = self.ceu(total_preds[pred_key], tts)
            #     if ceu is not None:
            #         metrics["ceu"][pred_key] = ceu
            #         print(metrics["ceu"][pred_key])

            tp, fp, tn, fn = self._perf_measure(tts, preds)
            metrics["spec"][pred_key] = tn / (tn + fp) if (tn + fp) != 0 else 0
            metrics["sens"][pred_key] = tp / (tp + fn) if (tp + fn) != 0 else 0
        metrics["total_preds"] = total_preds_nonargmaxed
        metrics["total_preds_target"] = tts
        metrics = dict(metrics)  # Avoid passing empty dicts to logs, better return an error!

        return metrics

    def _calculate_regression_metrics(self, targets_tens, preds):
        metrics = defaultdict(dict)
        for pred_key in preds[0].keys():
            total_preds = np.concatenate([pred[pred_key].cpu() for pred in preds], axis=0).squeeze()

            # total_preds = torch.concatenate(total_preds_nonargmaxed[pred_key]).cpu().squeeze()#[:self.processed_instances]

            binary_truth_nozeros = targets_tens[targets_tens != 0] > 0
            binary_preds_nozeros = total_preds[targets_tens != 0] > 0

            binary_truth = targets_tens > 0
            binary_preds = total_preds > 0

            # turn them to torch
            binary_truth_nozeros = torch.tensor(binary_truth_nozeros)
            binary_preds_nozeros = torch.tensor(binary_preds_nozeros)

            binary_truth = torch.tensor(binary_truth)
            binary_preds = torch.tensor(binary_preds)

            metrics["acc"][pred_key] = Accuracy(task="binary")(binary_preds_nozeros, binary_truth_nozeros).item()
            metrics["acc_has0"][pred_key] = Accuracy(task="binary")(binary_preds, binary_truth).item()

            metrics["f1"][pred_key] = f1_score(
                binary_preds_nozeros.cpu().numpy(),
                binary_truth_nozeros.cpu().numpy(),
                average="weighted",
            )
            metrics["f1_has0"][pred_key] = f1_score(
                binary_preds.cpu().numpy(),
                binary_truth.cpu().numpy(),
                average="weighted",
            )

            test_preds_a7 = np.clip(total_preds, a_min=-3.0, a_max=3.0)
            test_truth_a7 = np.clip(targets_tens, a_min=-3.0, a_max=3.0)
            test_preds_a5 = np.clip(total_preds, a_min=-2.0, a_max=2.0)
            test_truth_a5 = np.clip(targets_tens, a_min=-2.0, a_max=2.0)

            metrics["mae"][pred_key] = np.mean(np.absolute(total_preds - targets_tens))  # Average L1 distance between preds and truths
            metrics["corr"][pred_key] = np.corrcoef(total_preds, targets_tens)[0][1]
            metrics["acc_7"][pred_key] = multiclass_acc(test_preds_a7, test_truth_a7)
            metrics["acc_5"][pred_key] = multiclass_acc(test_preds_a5, test_truth_a5)
            if pred_key == "combined":
                metrics["total_preds"] = total_preds
                metrics["total_preds_target"] = targets_tens

        metrics = dict(metrics)  # Avoid passing empty dicts to logs, better return an error!

        return metrics

    def validate(self, data_loader, description):
        self.model.eval()
        self.model.train(False)
        with torch.no_grad():
            tts, preds, inits = [], [], []
            pbar = tqdm(
                enumerate(data_loader),
                total=len(data_loader),
                desc=description,
                leave=False,
            )
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
                    if self.config.get("task", "classification") == "classification" and len(served_dict["label"][served_dict["label"] == -1]) > 0:
                        served_dict["label"][served_dict["label"] == -1] = 0

                served_dict["data"] = {view: served_dict["data"][view].to(self.device) for view in served_dict["data"] if type(served_dict["data"][view]) is torch.Tensor}
                served_dict["data"].update({view: served_dict["data"][view].float() for view in served_dict["data"] if type(view) == int})

                target = served_dict["label"].to(self.device)

                output = self.model(served_dict["data"])

                if len(target.shape) > 1:
                    target = target.flatten()

                tts.append(target)
                preds.append(output["preds"])
                pbar.set_description("{} batch {}/{}".format(description, int(batch_idx), int(len(data_loader))))
                pbar.refresh()

            if "softlabels" in self.config.dataset and self.config.dataset.softlabels:
                tts = torch.cat(tts).argmax(dim=1).cpu().numpy()
            else:
                tts = torch.cat(tts).cpu().numpy().squeeze()

        if self.config.get("task", "classification") == "classification":
            metrics = self._calculate_classification_metrics(tts, preds)
        elif self.config.get("task", "classification") == "regression":
            metrics = self._calculate_regression_metrics(tts, preds)

        return metrics

    def _print_results(self, metrics, description):

        for i, v in metrics["acc"].items():
            message = Style.BRIGHT + Fore.WHITE + "{} ".format(description)
            message += Fore.LIGHTBLUE_EX + "Acc_{}: {:.2f} ".format(i, metrics["acc"][i] * 100)
            # message += Fore.LIGHTGREEN_EX + "K_{}: {:.4f} ".format(i, metrics["k"][i])
            # message += Fore.LIGHTGREEN_EX + "F1_{}: {:.2f} ".format(i, metrics["f1"][i] * 100)
            # message += Fore.LIGHTRED_EX + "ECE_{}: {:.3f} ".format(i, metrics["ece"][i])
            # message += Fore.BLUE + "F1_perclass_{}: {} ".format(i,"{}".format(str(list((metrics["f1_perclass"][i] * 100).round(2)))))
            print(message + Style.RESET_ALL)

    def save_test_results(self, checkpoint, save_dir, test_results):

        test_results_dict = {"post_test_results": test_results}
        checkpoint.update(test_results_dict)
        try:
            torch.save(checkpoint, save_dir)
            print("Models has saved successfully in {}".format(save_dir))
        except:
            raise Exception("Problem in model saving")
