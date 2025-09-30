# -*- coding: UTF-8 -*-
# Local modules
import argparse
import os
import sys

import pandas as pd

# PyTorch Modules
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

# 3rd-Party Modules
from tqdm import tqdm

# Self-Written Modules
sys.path.append(os.getcwd())
import net

import utils


def main(args):
    utils.print_config_description(args.conf_path)
    config_dict = utils.load_env(args.conf_path)
    assert config_dict.get("config_root", None) != None, "No config_root in config/conf.json"
    config_path = os.path.join(config_dict["config_root"], config_dict[args.corpus_type])
    utils.print_config_description(config_path)

    # Make model directory
    model_path = args.model_path
    os.makedirs(model_path, exist_ok=True)

    # Initialize dataset
    DataManager = utils.DataManager(config_path)
    lab_type = args.label_type

    audio_path, video_path, label_path = utils.load_audio_and_label_file_paths(args)

    fnames_aud, fnames_vid = [], []
    v_fnames = os.listdir(video_path)
    for fname_aud in os.listdir(audio_path):
        if fname_aud.replace(".wav", ".npy") in v_fnames:
            fnames_aud.append(fname_aud)
            fnames_vid.append(fname_aud.replace(".wav", ""))
    fnames_aud.sort()
    fnames_vid.sort()

    snum = 100000000000000000
    train_wav_path = DataManager.get_wav_path(split_type="train", wav_loc=audio_path, fnames=fnames_aud, lbl_loc=label_path)[:snum]
    train_vid_path = DataManager.get_vid_path(split_type="train", vid_loc=video_path, fnames=fnames_vid, lbl_loc=label_path)[:snum]

    train_utts = [fname.split("/")[-1] for fname in train_wav_path]

    train_labs = DataManager.get_msp_labels(train_utts, lab_type=lab_type, lbl_loc=label_path)

    train_wavs = utils.WavExtractor(train_wav_path).extract()
    train_vids = utils.VidExtractor(train_vid_path).extract()

    dev_wav_path = DataManager.get_wav_path(split_type="dev", wav_loc=audio_path, fnames=fnames_aud, lbl_loc=label_path)[:snum]
    dev_vid_path = DataManager.get_vid_path(split_type="dev", vid_loc=video_path, fnames=fnames_vid, lbl_loc=label_path)[:snum]

    dev_utts = [fname.split("/")[-1] for fname in dev_wav_path]
    dev_labs = DataManager.get_msp_labels(dev_utts, lab_type=lab_type, lbl_loc=label_path)
    dev_wavs = utils.WavExtractor(dev_wav_path).extract()
    dev_vids = utils.VidExtractor(dev_vid_path).extract()
    ###################################################################################################

    test_wav_path = DataManager.get_wav_path(split_type="test", wav_loc=audio_path, fnames=fnames_aud, lbl_loc=label_path)
    test_vid_path = DataManager.get_vid_path(split_type="test", vid_loc=video_path, fnames=fnames_vid, lbl_loc=label_path)

    test_wav_path.sort()
    test_utts = [fname.split("/")[-1] for fname in test_wav_path]
    test_utts.sort()
    test_labs = DataManager.get_msp_labels(test_utts, lab_type=lab_type, lbl_loc=label_path)
    test_wavs = utils.WavExtractor(test_wav_path).extract()
    test_vids = utils.VidExtractor(test_vid_path).extract()

    train_set = utils.AudVidSet(
        train_wavs,
        train_vids,
        train_labs,
        train_utts,
        print_dur=True,
        lab_type=lab_type,
        print_utt=True,
        label_config=DataManager.get_label_config(lab_type),
    )

    dev_set = utils.AudVidSet(
        dev_wavs,
        dev_vids,
        dev_labs,
        dev_utts,
        print_dur=True,
        lab_type=lab_type,
        print_utt=True,
        wav_mean=train_set.wav_mean,
        wav_std=train_set.wav_std,
        vid_mean=train_set.vid_mean,
        vid_std=train_set.vid_std,
        label_config=DataManager.get_label_config(lab_type),
    )

    test_set = utils.AudVidSet(
        test_wavs,
        test_vids,
        test_labs,
        test_utts,
        print_dur=True,
        lab_type=lab_type,
        print_utt=True,
        wav_mean=train_set.wav_mean,
        wav_std=train_set.wav_std,
        vid_mean=train_set.vid_mean,
        vid_std=train_set.vid_std,
        label_config=DataManager.get_label_config(lab_type),
    )

    def zero_out_percentage(x_in, percentage):
        """
        Zero out a percentage of features along the length of the tensor.

        Args:
        x_in (torch.Tensor): The input tensor with shape (batch size, length, feature dim).
        percentage (float): The percentage of the length to zero out.

        Returns:
        torch.Tensor: The resulting tensor with the specified percentage zeroed out.
        """
        # Make sure the percentage is between 0 and 100
        if not (0 <= percentage <= 100):
            raise ValueError("Percentage should be between 0 and 100.")

        # Calculate the number of features to zero out
        num_features_to_zero = int(x_in.shape[1] * (percentage / 100.0))

        # Create the mask
        batch_size, length, feature_dim = x_in.shape
        mask = torch.ones_like(x_in)

        # Randomly choose indices along the length to zero out
        # Ensure indices_to_zero is on the same device as x_in
        indices_to_zero = torch.rand(batch_size, length, device=x_in.device).argsort(dim=1)[:, :num_features_to_zero]

        # Expand the indices to the feature dimension
        expanded_indices_to_zero = indices_to_zero.unsqueeze(-1).expand(-1, -1, feature_dim)

        # Use the indices to zero out the corresponding features
        mask.scatter_(1, expanded_indices_to_zero, 0)

        # Apply the mask to the input tensor
        x_out = x_in * mask

        return x_out

    # Example usage
    # x_in = torch.randn(2, 20, 10)  # Sample tensor
    # percentage_to_zero_out = 30  # For example, 10%
    # x_out = zero_out_percentage(x_in, percentage_to_zero_out)

    # Verify the result
    # print("Percentage of zeros:", (x_out == 0).float().mean().item())

    # print(x_in,'in')
    # print(x_out,'out')

    # print(train_set.wav_mean, train_set.wav_std, train_set.vid_mean, train_set.vid_std)

    train_set.save_norm_stat(model_path + "/train_norm_stat.pkl")
    print(args.batch_size, "batch_size")
    total_dataloader = {
        "train": DataLoader(
            train_set,
            batch_size=args.batch_size,
            collate_fn=utils.collate_fn_padd,
            shuffle=False,
        ),
        "dev": DataLoader(
            dev_set,
            batch_size=args.batch_size,
            collate_fn=utils.collate_fn_padd,
            shuffle=False,
        ),
        "test": DataLoader(
            test_set,
            batch_size=args.batch_size,
            collate_fn=utils.collate_fn_padd,
            shuffle=False,
        ),
    }

    # Initialize model
    modelWrapper = net.ModelWrapper(args)  # Change this to use custom model
    w2v_mod = modelWrapper.init_model()
    modelWrapper.init_optimizer()
    modelWrapper.load_model("/media/lucas/08AE364B3D909CD8/ICMI_2023/wav2vec2", "train")

    # Initialize loss function
    lm = utils.LogManager()
    if args.label_type == "dimensional":
        lm.alloc_stat_type_list(
            [
                "train_aro",
                "train_dom",
                "train_val",
                "dev_aro",
                "dev_dom",
                "dev_val",
                "test_aro",
                "test_dom",
                "test_val",
                "test_aro_a",
                "test_dom_a",
                "test_val_a",
                "test_aro_v",
                "test_dom_v",
                "test_val_v",
            ]
        )
    elif args.label_type == "categorical":
        lm.alloc_stat_type_list(["train_loss", "train_acc", "dev_loss", "dev_acc", "test_loss", "test_acc"])

    epochs = args.epochs
    scaler = GradScaler()
    min_epoch = 0
    min_loss = 99999999999
    temp_dev = 99999999999
    losses_train, losses_dev, losses_test = [], [], []
    for epoch in range(epochs):
        print("Epoch:", epoch)
        lm.init_stat()
        for xy_pair in tqdm(total_dataloader["train"]):
            modelWrapper.set_train()
            xa = xy_pair[0]
            xv = xy_pair[1]
            y = xy_pair[2]
            mask = xy_pair[3]

            # print(len(xv.size()), len(xa.size()), 'xvsize')

            xa = xa.cuda(non_blocking=True).float()
            xv = xv.cuda(non_blocking=True).float()
            y = y.cuda(non_blocking=True).float()
            mask = mask.cuda(non_blocking=True).float()

            with autocast():

                preds_v, x_in, rec_pred = modelWrapper.feed_forward(None, xv, mode="visual", attention_mask=mask)
                loss_rec_v = utils.MSE_loss(rec_pred, x_in)
                total_loss_v = 0.0
                if args.label_type == "dimensional":
                    ccc = utils.CCC_loss(preds_v, y)
                    loss = 1.0 - ccc
                    total_loss_v += loss[0] + loss[1] + loss[2] + loss_rec_v

                elif args.label_type == "categorical":
                    total_loss_v += utils.CE_category(preds_v, y) + 2 * loss_rec_v

            modelWrapper.backprop(total_loss_v, "visual")

            with autocast():

                preds_a, x_in, rec_pred = modelWrapper.feed_forward(xa, None, mode="acoustic", attention_mask=mask)
                loss_rec_a = utils.MSE_loss(rec_pred, x_in)

                total_loss_a = 0.0
                if args.label_type == "dimensional":
                    ccc = utils.CCC_loss(preds_a, y)
                    loss = 1.0 - ccc
                    total_loss_a += loss[0] + loss[1] + loss[2] + loss_rec_a

                elif args.label_type == "categorical":
                    total_loss_a += utils.CE_category(preds_a, y) + 2 * loss_rec_a

            ## Backpropagation
            modelWrapper.backprop(total_loss_a, "acoustic")

            with autocast():

                preds_a, preds_v, preds = modelWrapper.feed_forward(xa, xv, mode="weights", attention_mask=mask)
                total_loss = 0.0
                if args.label_type == "dimensional":
                    ccc = utils.CCC_loss(preds, y)
                    loss = 1.0 - ccc
                    total_loss += loss[0] + loss[1] + loss[2]

                elif args.label_type == "categorical":
                    total_loss += utils.CE_category(preds, y)

                    acc = utils.calc_acc(preds, y)

            ## Backpropagation
            modelWrapper.backprop(total_loss, "weights")

            # Logging
            if args.label_type == "dimensional":
                lm.add_torch_stat("train_aro", ccc[0])
                lm.add_torch_stat("train_dom", ccc[1])
                lm.add_torch_stat("train_val", ccc[2])
            elif args.label_type == "categorical":
                lm.add_torch_stat("train_loss", total_loss)
                lm.add_torch_stat("train_acc", acc)

        modelWrapper.set_eval()

        with torch.no_grad():
            total_pred = []
            total_y = []
            for xy_pair in tqdm(total_dataloader["dev"]):
                xa = xy_pair[0]
                xv = xy_pair[1]
                y = xy_pair[2]
                mask = xy_pair[3]

                xa = xa.cuda(non_blocking=True).float()
                xv = xv.cuda(non_blocking=True).float()
                y = y.cuda(non_blocking=True).float()
                mask = mask.cuda(non_blocking=True).float()

                xa = w2v_mod(xa, attention_mask=mask).last_hidden_state

                preds_a, preds_v, preds_av = modelWrapper.feed_forward(xa, xv, mode="weights", attention_mask=mask)

                total_pred.append(preds_av)
                total_y.append(y)

            total_pred = torch.cat(total_pred, 0)
            total_y = torch.cat(total_y, 0)

            # Define the list of zeroing percentages
            zeroing_percentages = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
            pmodes = ["audio", "video"]

            trigger = False

            # Iterate over the zeroing percentages
            for pmode in pmodes:
                for percentage_to_zero_out in zeroing_percentages:

                    # Initialize lists to store predictions and actual values for each percentage
                    total_pred_t, total_pred_a, total_pred_v = [], [], []
                    total_y_t = []
                    total_utts = []

                    # Test loop
                    for xy_pair in tqdm(total_dataloader["test"]):
                        xa = xy_pair[0].cuda(non_blocking=True).float()
                        xv = xy_pair[1].cuda(non_blocking=True).float()
                        y = xy_pair[2].cuda(non_blocking=True).float()
                        mask = xy_pair[3].cuda(non_blocking=True).float()
                        utt_ids = xy_pair[4]

                        # Embed the audio features
                        xa = w2v_mod(xa, attention_mask=mask).last_hidden_state

                        print(xv.size(), xa.size(), "shapes")

                        print(
                            "Percentage of zeros:",
                            (xv == 0).float().mean().item(),
                            "xv in",
                        )

                        # Zero out the specified percentage of the data
                        if pmode == "audio":
                            xa = zero_out_percentage(xa, percentage_to_zero_out)
                            print(
                                "Percentage of zeros:",
                                (xa == 0).float().mean().item(),
                                "xa",
                            )

                        elif pmode == "video":
                            xv = zero_out_percentage(xv, percentage_to_zero_out)
                            print(
                                "Percentage of zeros:",
                                (xv == 0).float().mean().item(),
                                "xv",
                            )

                        # Get model predictions
                        preds_a, preds_v, preds_av = modelWrapper.feed_forward(xa, xv, mode="weights", attention_mask=mask)

                        # Store predictions and true values
                        total_pred_t.append(preds_av)
                        total_pred_a.append(preds_a)
                        total_pred_v.append(preds_v)
                        total_y_t.append(y)
                        total_utts.extend(utt_ids)

                    # Concatenate the list of tensors into a single tensor
                    total_pred_t = torch.cat(total_pred_t, 0)
                    total_pred_a = torch.cat(total_pred_a, 0)
                    total_pred_v = torch.cat(total_pred_v, 0)
                    total_y_t = torch.cat(total_y_t, 0)

                    # Perform evaluations
                    if args.label_type == "categorical":
                        if args.label_learning == "hard-label":
                            loss = utils.CE_category(total_pred, total_y)
                            loss_t = utils.CE_category(total_pred_t, total_y_t)

                        acc = utils.calc_acc(total_pred, total_y)
                        lm.add_torch_stat("dev_loss", loss)
                        lm.add_torch_stat("dev_acc", acc)

                        acc_t = utils.calc_acc(total_pred_t, total_y_t)
                        lm.add_torch_stat("test_loss", loss_t)
                        lm.add_torch_stat("test_acc", acc_t)

                    if args.label_type == "dimensional":
                        ccc_dev = utils.CCC_loss(total_pred, total_y)
                        ccc_test = utils.CCC_loss(total_pred_t, total_y_t)
                        ccc_test_a = utils.CCC_loss(total_pred_a, total_y_t)
                        ccc_test_v = utils.CCC_loss(total_pred_v, total_y_t)

                        lm.add_torch_stat("dev_aro", ccc_dev[0])
                        lm.add_torch_stat("dev_dom", ccc_dev[1])
                        lm.add_torch_stat("dev_val", ccc_dev[2])
                        lm.add_torch_stat("test_aro", ccc_test[0])
                        lm.add_torch_stat("test_dom", ccc_test[1])
                        lm.add_torch_stat("test_val", ccc_test[2])
                        lm.add_torch_stat("test_aro_a", ccc_test_a[0])
                        lm.add_torch_stat("test_dom_a", ccc_test_a[1])
                        lm.add_torch_stat("test_val_a", ccc_test_a[2])
                        lm.add_torch_stat("test_aro_v", ccc_test_v[0])
                        lm.add_torch_stat("test_dom_v", ccc_test_v[1])
                        lm.add_torch_stat("test_val_v", ccc_test_v[2])

                    lm.print_stat()
                    if args.label_type == "dimensional":
                        dev_loss = 3.0 - lm.get_stat("dev_aro") - lm.get_stat("dev_dom") - lm.get_stat("dev_val")
                        test_loss = 3.0 - lm.get_stat("test_aro") - lm.get_stat("test_dom") - lm.get_stat("test_val")
                    elif args.label_type == "categorical":
                        dev_loss = lm.get_stat("dev_loss")
                        tr_loss = lm.get_stat("train_loss")
                        test_loss = lm.get_stat("test_loss")
                        losses_dev.append(dev_loss)
                        losses_train.append(tr_loss)
                        losses_test.append(test_loss)
                    if min_loss > dev_loss:
                        min_epoch = epoch
                        min_loss = dev_loss

                    if trigger == True or float(dev_loss) < float(temp_dev):

                        if trigger == False:
                            temp_dev = float(dev_loss)
                            print("better dev loss found:" + str(float(dev_loss)) + " saving model")
                            modelWrapper.save_model(epoch)

                        trigger = True

                        # total_pred_np = total_pred_t.detach().cpu().numpy()
                        # total_pred_np_a = total_pred_a.detach().cpu().numpy()
                        # total_pred_np_v = total_pred_v.detach().cpu().numpy()
                        # total_y_np = total_y_t.detach().cpu().numpy()

                        # total_pred_pd = pd.DataFrame(total_pred_np, index=total_utts)
                        # total_pred_pd_a = pd.DataFrame(total_pred_np_a, index=total_utts)
                        # total_pred_pd_v = pd.DataFrame(total_pred_np_v, index=total_utts)
                        # total_y_pd = pd.DataFrame(total_y_np, index=total_utts)

                        # Use the percentage in the save path to differentiate the results
                        for mode in ["acoustic", "visual", "audiovisual"]:
                            save_path = f"{model_path}/predictions/test/{mode}/{pmode}/{percentage_to_zero_out}pct/"

                            print(f"this is percentage {percentage_to_zero_out}% of less {pmode} frames.")

                            if not os.path.exists(save_path):
                                os.makedirs(save_path)

                            # Save the predictions and true values as CSV files
                            if mode == "acoustic":
                                pd.DataFrame(
                                    total_pred_a.detach().cpu().numpy(),
                                    index=total_utts,
                                ).to_csv(f"{save_path}y_pred.csv")
                                pd.DataFrame(total_y_t.detach().cpu().numpy(), index=total_utts).to_csv(f"{save_path}y_true.csv")

                            elif mode == "visual":
                                pd.DataFrame(
                                    total_pred_v.detach().cpu().numpy(),
                                    index=total_utts,
                                ).to_csv(f"{save_path}y_pred.csv")
                                pd.DataFrame(total_y_t.detach().cpu().numpy(), index=total_utts).to_csv(f"{save_path}y_true.csv")

                            elif mode == "audiovisual":
                                pd.DataFrame(
                                    total_pred_t.detach().cpu().numpy(),
                                    index=total_utts,
                                ).to_csv(f"{save_path}y_pred.csv")
                                pd.DataFrame(total_y_t.detach().cpu().numpy(), index=total_utts).to_csv(f"{save_path}y_true.csv")

                            # Additional evaluations and printing can go here
                            if args.label_type == "categorical":
                                print("This is mode:", mode)
                                utils.scores(save_path)

        #     total_pred_t, total_pred_a, total_pred_v  = [] , [], []
        #     total_y_t = []
        #     total_utts = []
        #     for xy_pair in tqdm(total_dataloader["test"]):
        #         xa = xy_pair[0]
        #         xv = xy_pair[1]
        #         y = xy_pair[2]
        #         mask = xy_pair[3]
        #         utt_ids = xy_pair[4]

        #         xa=xa.cuda(non_blocking=True).float()
        #         xv=xv.cuda(non_blocking=True).float()
        #         y=y.cuda(non_blocking=True).float()
        #         mask=mask.cuda(non_blocking=True).float()

        #         xa = w2v_mod(xa, attention_mask=mask).last_hidden_state

        #         percentage_to_zero_out = 10  # For example, 10%
        #         xa = zero_out_percentage(xa, percentage_to_zero_out)

        #         preds_a, preds_v, preds_av = modelWrapper.feed_forward(xa, xv, mode = 'weights', attention_mask=mask)

        #         total_pred_t.append(preds_av)
        #         total_pred_a.append(preds_a)
        #         total_pred_v.append(preds_v)
        #         total_y_t.append(y)
        #         total_utts.extend(utt_ids)

        #     total_pred_t = torch.cat(total_pred_t, 0)
        #     total_pred_a = torch.cat(total_pred_a, 0)
        #     total_pred_v = torch.cat(total_pred_v, 0)
        #     total_y_t = torch.cat(total_y_t, 0)

        # if args.label_type == "categorical":
        #     if args.label_learning == "hard-label":
        #         loss = utils.CE_category(total_pred, total_y)
        #         loss_t = utils.CE_category(total_pred_t, total_y_t)

        #     acc = utils.calc_acc(total_pred, total_y)
        #     lm.add_torch_stat("dev_loss", loss)
        #     lm.add_torch_stat("dev_acc", acc)

        #     acc_t = utils.calc_acc(total_pred_t, total_y_t)
        #     lm.add_torch_stat("test_loss", loss_t)
        #     lm.add_torch_stat("test_acc", acc_t)

        # if args.label_type == "dimensional":
        #     ccc_dev = utils.CCC_loss(total_pred, total_y)
        #     ccc_test = utils.CCC_loss(total_pred_t, total_y_t)
        #     ccc_test_a = utils.CCC_loss(total_pred_a, total_y_t)
        #     ccc_test_v = utils.CCC_loss(total_pred_v, total_y_t)

        #     lm.add_torch_stat("dev_aro", ccc_dev[0])
        #     lm.add_torch_stat("dev_dom", ccc_dev[1])
        #     lm.add_torch_stat("dev_val", ccc_dev[2])
        #     lm.add_torch_stat("test_aro", ccc_test[0])
        #     lm.add_torch_stat("test_dom", ccc_test[1])
        #     lm.add_torch_stat("test_val", ccc_test[2])
        #     lm.add_torch_stat("test_aro_a", ccc_test_a[0])
        #     lm.add_torch_stat("test_dom_a", ccc_test_a[1])
        #     lm.add_torch_stat("test_val_a", ccc_test_a[2])
        #     lm.add_torch_stat("test_aro_v", ccc_test_v[0])
        #     lm.add_torch_stat("test_dom_v", ccc_test_v[1])
        #     lm.add_torch_stat("test_val_v", ccc_test_v[2])

        # lm.print_stat()
        # if args.label_type == "dimensional":
        #     dev_loss = 3.0 - lm.get_stat("dev_aro") - lm.get_stat("dev_dom") - lm.get_stat("dev_val")
        #     test_loss = 3.0 - lm.get_stat("test_aro") - lm.get_stat("test_dom") - lm.get_stat("test_val")
        # elif args.label_type == "categorical":
        #     dev_loss = lm.get_stat("dev_loss")
        #     tr_loss = lm.get_stat("train_loss")
        #     test_loss = lm.get_stat("test_loss")
        #     losses_dev.append(dev_loss)
        #     losses_train.append(tr_loss)
        #     losses_test.append(test_loss)
        # if min_loss > dev_loss:
        #     min_epoch = epoch
        #     min_loss = dev_loss

        # if float(dev_loss) < float(temp_dev):
        #     temp_dev = float(dev_loss)
        #     print('better dev loss found:' + str(float(dev_loss)) + ' saving model')
        #     modelWrapper.save_model(epoch)

        #     total_pred_np = total_pred_t.detach().cpu().numpy()
        #     total_pred_np_a = total_pred_a.detach().cpu().numpy()
        #     total_pred_np_v = total_pred_v.detach().cpu().numpy()
        #     total_y_np = total_y_t.detach().cpu().numpy()

        #     total_pred_pd = pd.DataFrame(total_pred_np, index=total_utts)
        #     total_pred_pd_a = pd.DataFrame(total_pred_np_a, index=total_utts)
        #     total_pred_pd_v = pd.DataFrame(total_pred_np_v, index=total_utts)
        #     total_y_pd = pd.DataFrame(total_y_np, index=total_utts)

        #     for mode in ['acoustic', 'visual', 'audiovisual']:
        #         save_path = model_path + '/predictions/test/' + mode + '/'

        #         if not os.path.exists(save_path):
        #             os.makedirs(save_path)

        #         if mode == 'acoustic':
        #             total_pred_pd_a.to_csv(save_path+'y_pred.csv')
        #             total_y_pd.to_csv(save_path+'y_true.csv')

        #         elif mode == 'visual':
        #             total_pred_pd_v.to_csv(save_path+'y_pred.csv')
        #             total_y_pd.to_csv(save_path+'y_true.csv')

        #         elif mode == 'audiovisual':
        #             total_pred_pd.to_csv(save_path+'y_pred.csv')
        #             total_y_pd.to_csv(save_path+'y_true.csv')

        #         if args.label_type == "categorical":
        #             print("This is mode:", mode)
        #             utils.scores(save_path)
    print("Save", end=" ")
    print(min_epoch, end=" ")
    print("")

    with open(model_path + "/train_loss.txt", "w") as f:
        for item in losses_train:
            f.write("%s\n" % item)

    with open(model_path + "/dev_loss.txt", "w") as f:
        for item in losses_dev:
            f.write("%s\n" % item)

    with open(model_path + "/test_loss.txt", "w") as f:
        for item in losses_test:
            f.write("%s\n" % item)

    print("Loss", end=" ")
    if args.label_type == "dimensional":
        print(3.0 - min_loss, end=" ")
    elif args.label_type == "categorical":
        print(min_loss, end=" ")
    print("")


if __name__ == "__main__":
    # Inputs for the main function
    parser = argparse.ArgumentParser()

    # Experiment Arguments
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--conf_path", default="config/conf.json", type=str)

    # Data Arguments
    parser.add_argument("--corpus_type", default="podcast_v1.7", type=str)
    parser.add_argument("--model_type", default="wav2vec2", type=str)
    parser.add_argument(
        "--label_type",
        choices=["dimensional", "categorical"],
        default="categorical",
        type=str,
    )

    # Model Arguments
    parser.add_argument("--model_path", default=None, type=str)
    parser.add_argument("--output_num", default=4, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--hidden_dim", default=256, type=int)
    parser.add_argument("--num_layers", default=3, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--lr", default=1e-5, type=float)

    # Label Learning Arguments
    parser.add_argument("--label_learning", default="multi-label", type=str)
    parser.add_argument("--corpus", default="USC-IEMOCAP", type=str)
    parser.add_argument("--num_classes", default="four", type=str)
    parser.add_argument("--label_rule", default="M", type=str)
    parser.add_argument("--partition_number", default="1", type=str)
    parser.add_argument("--data_mode", default="primary", type=str)
    parser.add_argument("--output_dim", default=6, type=int)
    parser.add_argument("--relu_dropout", type=float, default=0.1, help="relu dropout")
    parser.add_argument(
        "--out_dropout",
        type=float,
        default=0.2,
        help="output layer dropout (default: 0.2",
    )
    parser.add_argument("--optim", type=str, default="Adam", help="optimizer to use (default: Adam)")

    args = parser.parse_args()

    # Call main function
    main(args)
