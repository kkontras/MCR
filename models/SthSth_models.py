import copy
import logging
from collections import OrderedDict
from typing import Dict

from yacs.config import CfgNode

from datasets.SthSth.utils_data import get_normalizer
from models.model_utils.fusion_gates import *
from models.model_utils.sthsth_handmodels import Stlt
from models.model_utils.sthsth_swin import SwinTransformer3D


class SwinModel(nn.Module):
    def __init__(self, encs, args, checkpoint: str = None, frozen_encoders: bool = False):
        super(SwinModel, self).__init__()
        self.args = args
        self.frozen_encoders = frozen_encoders
        # Create backbone
        self.backbone = SwinTransformer3D(
            patch_size=(2, 4, 4),
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=(8, 7, 7),
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.2,
            patch_norm=True,
        )

        if self.args.get("swin_backbone", False):
            logging.info(f"Loading Swin Transformer backbone from {self.args.swin_backbone}")
            this_checkpoint = torch.load(self.args.swin_backbone, map_location="cpu")
            new_state_dict = OrderedDict()
            for k, v in this_checkpoint["state_dict"].items():
                if "backbone" in k:
                    name = k[9:]
                    new_state_dict[name] = v
            self.backbone.load_state_dict(new_state_dict)
        # Build classifier
        # actions = {"action": self.cfg.num_classes}
        # self.classifiers = nn.ModuleDict(
        #     {
        #         actions_name: nn.Linear(768, actions_num)
        #         for actions_name, actions_num in actions.items()
        #         if actions_num is not None
        #     }
        # )

        self.classifiers = nn.ModuleDict({"ACTION": nn.Linear(768, self.args.num_classes)})
        self.modality = self.get_modality()

        # Load existing checkpoint, if any
        if checkpoint:
            logging.info(f"Loading model checkpoint from {checkpoint}")
            ch = torch.load(checkpoint, map_location="cpu")
            if "best_model_state_dict" in ch:
                self.load_state_dict(ch["model_state_dict"])
            else:
                self.load_state_dict(ch)

        if self.frozen_encoders:
            self.backbone.requires_grad_(False)

    def get_modality(self):
        mapping = {
            "video": 1,
            "video_flow": "video",
            "flow": 2,
            "depth": "depth",
            "video_layout": "video",
            "video_depth": "video",
            "layout": 0,
            "video_layout_flow": "video",
            "omnivore": "omnivore",
            "audio": "audio",
            "video_audio": "video",
            "video_flow_audio": "video",
            "segmentation": "segmentation",
            "video_segmentation": "video",
        }

        return mapping[self.args.modality]

    def forward(self, batch: Dict[str, torch.Tensor], return_features=False, detach=False):
        if self.frozen_encoders:
            with torch.no_grad():
                self.backbone.eval()
                video_frames = batch[self.modality].permute(0, 2, 1, 3, 4)
                normalizer = get_normalizer(input_type=self.args.modality, model_name="swin")
                video_frames = normalizer(video_frames).permute(0, 2, 1, 3, 4)
                features = self.backbone(video_frames)
        else:
            video_frames = batch[self.modality].permute(0, 2, 1, 3, 4)
            normalizer = get_normalizer(input_type=self.args.modality, model_name="swin")
            video_frames = normalizer(video_frames).permute(0, 2, 1, 3, 4)
            features = self.backbone(video_frames)

        features = features.mean(dim=[2, 3, 4])

        output = {"preds": {}}
        if detach:
            output["preds"]["combined"] = self.classifiers["ACTION"](features.detach())
        else:
            output["preds"]["combined"] = self.classifiers["ACTION"](features)
        if return_features:
            output["features"] = {"combined": features}

            # return output, features
        return output


class All3Model(nn.Module):
    def __init__(self, encs, args: CfgNode):
        super(All3Model, self).__init__()
        self.args = copy.deepcopy(args)
        pretraining_paths = self.args.get("pretraining_paths", {})
        frozen_encoders = self.args.get("frozen_encoders", False)
        self.fusion_gates = self.args.get("fusion_gates", False)

        if self.args.activate_model["video"]:
            video_args = copy.deepcopy(args)
            video_args.modality = "video"
            self.mod0_video_model = SwinModel(
                encs=encs,
                args=video_args,
                checkpoint=pretraining_paths.get("video", None),
            )
            if frozen_encoders:
                self.mod0_video_model.requires_grad_(False)
        if self.args.activate_model["flow"]:
            flow_args = copy.deepcopy(args)
            flow_args.modality = "flow"
            self.mod2_flow_model = SwinModel(
                encs=encs,
                args=flow_args,
                checkpoint=pretraining_paths.get("flow", None),
            )
            if frozen_encoders:
                self.mod2_flow_model.requires_grad_(False)
        if self.args.activate_model["layout"]:
            layout_args = copy.deepcopy(args)
            layout_args.modality = "layout"
            self.mod1_layout_model = Stlt(
                encs=encs,
                args=layout_args,
                checkpoint=pretraining_paths.get("layout", None),
            )
            if frozen_encoders:
                self.mod1_layout_model.requires_grad_(False)

        if self.args.activate_model["video"] and self.args.activate_model["flow"] and self.args.activate_model["layout"]:
            if self.args.get("bnorm", False):
                self.classifier = nn.Sequential(nn.BatchNorm1d(768 * 3), nn.Linear(768 * 3, 174))
            else:
                self.classifier = nn.Linear(768 * 3, 174)

        elif (
            self.args.activate_model["video"]
            and self.args.activate_model["flow"]
            or self.args.activate_model["video"]
            and self.args.activate_model["layout"]
            or self.args.activate_model["flow"]
            and self.args.activate_model["layout"]
        ):
            # self.classifier = nn.Linear(768*2,174)
            if self.fusion_gates == "film":
                self.classifier = FiLM(768, 768, 174)
            else:
                if self.args.get("bnorm", False):
                    self.classifier = nn.Sequential(nn.BatchNorm1d(768 * 2), nn.Linear(768 * 2, 174))
                else:
                    self.classifier = nn.Linear(768 * 2, 174)

        # if cfg.CHECKPOINT_PATH:
        #     print("We are loading from {}".format(cfg.CHECKPOINT_PATH))
        #     self.load_state_dict(torch.load(cfg.CHECKPOINT_PATH, map_location="cpu"))

    def forward(self, batch: Dict[str, torch.Tensor], return_features: bool = False, **kwargs):

        frozen_encoders = self.args.get("frozen_encoders", False)
        pred = {}
        features = {}
        if frozen_encoders:
            with torch.no_grad():
                if self.args.activate_model["video"]:
                    video_output = self.mod0_video_model(batch, return_features=True)
                    pred.update({"c": video_output["preds"]["combined"]})
                    features.update({"c": video_output["features"]["combined"]})

                if self.args.activate_model["layout"]:
                    layout_output = self.mod1_layout_model(batch, return_features=True)
                    pred.update({"g": layout_output["preds"]["combined"]})
                    features.update({"g": layout_output["features"]["combined"]})

                if self.args.activate_model["flow"]:
                    flow_output = self.mod2_flow_model(batch, return_features=True)
                    pred.update({"flow": flow_output["preds"]["combined"]})
                    features.update({"flow": flow_output["features"]["combined"]})
        else:
            if self.args.activate_model["video"]:
                video_output = self.mod0_video_model(batch, return_features=True)
                pred.update({"c": video_output["preds"]["combined"]})
                features.update({"c": video_output["features"]["combined"]})

            if self.args.activate_model["layout"]:
                layout_output = self.mod1_layout_model(batch, return_features=True)
                pred.update({"g": layout_output["preds"]["combined"]})
                features.update({"g": layout_output["features"]["combined"]})

            if self.args.activate_model["flow"]:
                flow_output = self.mod2_flow_model(batch, return_features=True)
                pred.update({"flow": flow_output["preds"]["combined"]})
                features.update({"flow": flow_output["features"]["combined"]})

        if self.args.activate_model["video"] and self.args.activate_model["flow"] and self.args.activate_model["layout"]:
            feature_list = [features["c"], features["g"], features["flow"]]
        elif self.args.activate_model["video"] and self.args.activate_model["flow"]:
            feature_list = [features["c"], features["flow"]]
        elif self.args.activate_model["video"] and self.args.activate_model["layout"]:
            feature_list = [features["c"], features["g"]]
        elif self.args.activate_model["flow"] and self.args.activate_model["layout"]:
            feature_list = [features["g"], features["flow"]]
        elif self.args.activate_model["video"]:
            feature_list = [features["c"]]
        elif self.args.activate_model["flow"]:
            feature_list = [features["flow"]]
        elif self.args.activate_model["layout"]:
            feature_list = [features["g"]]

        if self.fusion_gates:
            preds = self.classifier(feature_list)
        else:
            preds = self.classifier(torch.cat(feature_list, dim=1))

        pred.update({"combined": preds})

        return {"preds": pred, "features": features}


class All3Model_ShuffleGrad(nn.Module):
    def __init__(self, encs, args: CfgNode):
        super(All3Model_ShuffleGrad, self).__init__()
        self.args = copy.deepcopy(args)
        pretraining_paths = self.args.get("pretraining_paths", {})
        frozen_encoders = self.args.get("frozen_encoders", False)
        self.fusion_gates = self.args.get("fusion_gates", False)

        if self.args.activate_model["video"]:
            video_args = copy.deepcopy(args)
            video_args.modality = "video"
            self.mod0_video_model = SwinModel(
                encs=encs,
                args=video_args,
                checkpoint=pretraining_paths.get("video", None),
            )
            if frozen_encoders:
                self.mod0_video_model.requires_grad_(False)
        if self.args.activate_model["flow"]:
            flow_args = copy.deepcopy(args)
            flow_args.modality = "flow"
            self.mod2_flow_model = SwinModel(
                encs=encs,
                args=flow_args,
                checkpoint=pretraining_paths.get("flow", None),
            )
            if frozen_encoders:
                self.mod2_flow_model.requires_grad_(False)
        if self.args.activate_model["layout"]:
            layout_args = copy.deepcopy(args)
            layout_args.modality = "layout"
            self.mod1_layout_model = Stlt(
                encs=encs,
                args=layout_args,
                checkpoint=pretraining_paths.get("layout", None),
            )
            if frozen_encoders:
                self.mod1_layout_model.requires_grad_(False)

        self.fc_mod0 = nn.Linear(768, 174, bias=False)
        self.fc_mod1 = nn.Linear(768, 174, bias=False)
        self.fc_mod2 = nn.Linear(768, 174, bias=False)
        self.cls_bias = nn.Parameter(torch.zeros(174), requires_grad=True)

        if self.args.bias_infusion.get("lib", 0) != 0:

            self.fc_yz = nn.Sequential(
                nn.Linear(174, 768, bias=False),
                nn.ReLU(),
                nn.Linear(768, 768 * 3, bias=False),
            )

    def get_features(self, batch):
        frozen_encoders = self.args.get("frozen_encoders", False)
        pred = {}
        features = {}
        if frozen_encoders:
            with torch.no_grad():
                if self.args.activate_model["video"]:
                    video_output = self.mod0_video_model(batch, return_features=True, detach=True)
                    pred.update({"c": video_output["preds"]["combined"]})
                    features.update({"c": video_output["features"]["combined"]})

                if self.args.activate_model["layout"]:
                    layout_output = self.mod1_layout_model(batch, return_features=True, detach=True)
                    pred.update({"g": layout_output["preds"]["combined"]})
                    features.update({"g": layout_output["features"]["combined"]})

                if self.args.activate_model["flow"]:
                    flow_output = self.mod2_flow_model(batch, return_features=True, detach=True)
                    pred.update({"flow": flow_output["preds"]["combined"]})
                    features.update({"flow": flow_output["features"]["combined"]})
        else:
            if self.args.activate_model["video"]:
                video_output = self.mod0_video_model(batch, return_features=True, detach=True)
                pred.update({"c": video_output["preds"]["combined"]})
                features.update({"c": video_output["features"]["combined"]})

            if self.args.activate_model["layout"]:
                layout_output = self.mod1_layout_model(batch, return_features=True, detach=True)
                pred.update({"g": layout_output["preds"]["combined"]})
                features.update({"g": layout_output["features"]["combined"]})

            if self.args.activate_model["flow"]:
                flow_output = self.mod2_flow_model(batch, return_features=True, detach=True)
                pred.update({"f": flow_output["preds"]["combined"]})
                features.update({"f": flow_output["features"]["combined"]})

        return pred, features

    def multimodal_pred(self, c, g, flow, **kwargs):
        pred_c = torch.matmul(c, self.fc_mod0.weight.T)  # + self.fc_0_lin.bias / 2
        pred_g = torch.matmul(g, self.fc_mod1.weight.T)  # + self.fc_0_lin.bias / 2
        pred_f = torch.matmul(flow, self.fc_mod2.weight.T)  # + self.fc_0_lin.bias / 2
        if "detach_c" in kwargs and kwargs["detach_a"]:
            pred_c = pred_c.detach()
        if "detach_g" in kwargs and kwargs["detach_a"]:
            pred_g = pred_g.detach()
        if "detach_f" in kwargs and kwargs["detach_v"]:
            pred_f = pred_f.detach()

        total_pred = pred_c + pred_g + pred_f

        return total_pred, pred_c, pred_g, pred_f

    def shuffle_ids(self, label):
        batch_size = label.size(0)
        shuffle_data = []
        random_shuffling = True
        if "rand" in self.args.bias_infusion.shuffle_type:
            while len(shuffle_data) < self.args.bias_infusion.num_samples:
                shuffle_idx_0 = torch.randperm(batch_size)
                shuffle_idx_1 = torch.randperm(batch_size)
                shuffle_idx_2 = torch.randperm(batch_size)
                shuffle_idx_3 = torch.randperm(batch_size)
                nonequal_label = torch.ones(batch_size, dtype=torch.bool)

                if nonequal_label.sum() <= 1:
                    continue
                shuffle_data.append(
                    {
                        "shuffle_idx_0": shuffle_idx_0,
                        "shuffle_idx_1": shuffle_idx_1,
                        "shuffle_idx_2": shuffle_idx_2,
                        "shuffle_idx_3": shuffle_idx_3,
                        "data": nonequal_label,
                    }
                )
        return shuffle_data

    def shuffle_data_func(self, label, pred_c, pred_g, pred_flow, pred_mm):

        shuffle_data = self.shuffle_ids(label)

        # We should have splitted the n_pred_c into 1,2 and 12 to use it along. It get tricky so we allow same labels for now.
        s_pred_c = torch.concatenate([pred_c[sh_data_i["shuffle_idx_0"]] for sh_data_i in shuffle_data], dim=0)
        s_pred_g = torch.concatenate([pred_g[sh_data_i["shuffle_idx_1"]] for sh_data_i in shuffle_data], dim=0)
        s_pred_flow = torch.concatenate([pred_flow[sh_data_i["shuffle_idx_2"]] for sh_data_i in shuffle_data], dim=0)

        s_pred_c_3 = torch.concatenate([pred_c[sh_data_i["shuffle_idx_3"]] for sh_data_i in shuffle_data], dim=0)
        s_pred_g_3 = torch.concatenate([pred_g[sh_data_i["shuffle_idx_3"]] for sh_data_i in shuffle_data], dim=0)
        s_pred_flow_3 = torch.concatenate([pred_flow[sh_data_i["shuffle_idx_3"]] for sh_data_i in shuffle_data], dim=0)

        n_pred_c = torch.concatenate([pred_c[sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0)
        n_pred_g = torch.concatenate([pred_g[sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0)
        n_pred_flow = torch.concatenate([pred_flow[sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0)

        n_pred = torch.concatenate([pred_mm[sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0)

        return (
            s_pred_c,
            s_pred_g,
            s_pred_flow,
            n_pred_c,
            n_pred_g,
            n_pred_flow,
            n_pred,
            s_pred_c_3,
            s_pred_g_3,
            s_pred_flow_3,
        )

    def forward(self, batch: Dict[str, torch.Tensor], return_features: bool = False, **kwargs):

        pred, features = self.get_features(batch)

        norm_preds, pred_c, pred_g, pred_f = self.multimodal_pred(features["c"], features["g"], features["f"])

        output = {
            "preds": {
                "combined": norm_preds + self.cls_bias,
                "c": pred_c + self.cls_bias,
                "g": pred_g + self.cls_bias,
                "f": pred_f + self.cls_bias,
            }
        }

        # if "detach" in kwargs and kwargs["detach"]:
        if self.training:
            if self.args.bias_infusion.get("lib", 0) > 0:
                pred_feat = self.fc_yz(pred.detach())
                combined_features = torch.cat([features["c"], features["g"], features["f"]], dim=1)
                CMI_yz_Loss = torch.nn.MSELoss()(combined_features, pred_feat) * self.args.bias_infusion.get("lib", 0)
                output["losses"] = {"CMI_yz_Loss": CMI_yz_Loss}

            (
                s_pred_c,
                s_pred_g,
                s_pred_flow,
                n_pred_c,
                n_pred_g,
                n_pred_flow,
                n_pred,
                s_pred_c_3,
                s_pred_g_3,
                s_pred_flow_3,
            ) = self.shuffle_data_func(kwargs["label"], pred_c, pred_g, pred_f, norm_preds)

            output["preds"]["sc_detc"] = s_pred_c.detach() + n_pred_g + n_pred_flow + self.cls_bias
            output["preds"]["sg_detc"] = s_pred_g + n_pred_c.detach() + n_pred_flow + self.cls_bias
            output["preds"]["sf_detc"] = s_pred_flow + n_pred_c.detach() + n_pred_g + self.cls_bias
            output["preds"]["scf_detc"] = s_pred_c.detach() + +s_pred_flow + n_pred_g + self.cls_bias
            output["preds"]["sgf_detc"] = s_pred_g + s_pred_flow + n_pred_c.detach() + self.cls_bias
            output["preds"]["scg_detc"] = s_pred_c.detach() + s_pred_g + n_pred_flow + self.cls_bias

            output["preds"]["sc_detg"] = s_pred_c + n_pred_g.detach() + n_pred_flow + self.cls_bias
            output["preds"]["sg_detg"] = s_pred_g.detach() + n_pred_c + n_pred_flow + self.cls_bias
            output["preds"]["sf_detg"] = s_pred_flow + n_pred_c + n_pred_g.detach() + self.cls_bias
            output["preds"]["scf_detg"] = s_pred_c + +s_pred_flow + n_pred_g.detach() + self.cls_bias
            output["preds"]["sgf_detg"] = s_pred_g.detach() + s_pred_flow + n_pred_c + self.cls_bias
            output["preds"]["scg_detg"] = s_pred_c + s_pred_g.detach() + n_pred_flow + self.cls_bias

            output["preds"]["sc_detf"] = s_pred_c + n_pred_g + n_pred_flow.detach() + self.cls_bias
            output["preds"]["sg_detf"] = s_pred_g + n_pred_c + n_pred_flow.detach() + self.cls_bias
            output["preds"]["sf_detf"] = s_pred_flow.detach() + n_pred_c + n_pred_g + self.cls_bias
            output["preds"]["scf_detf"] = s_pred_c + +s_pred_flow.detach() + n_pred_g + self.cls_bias
            output["preds"]["sgf_detf"] = s_pred_g + s_pred_flow.detach() + n_pred_c + self.cls_bias
            output["preds"]["scg_detf"] = s_pred_c + s_pred_g + n_pred_flow.detach() + self.cls_bias

            output["preds"]["sc_detgf"] = s_pred_c + n_pred_g.detach() + n_pred_flow.detach() + self.cls_bias
            output["preds"]["sg_detgf"] = s_pred_g.detach() + n_pred_c + n_pred_flow.detach() + self.cls_bias
            output["preds"]["sf_detgf"] = s_pred_flow.detach() + n_pred_c + n_pred_g.detach() + self.cls_bias
            output["preds"]["scf_detgf"] = s_pred_c + +s_pred_flow.detach() + n_pred_g.detach() + self.cls_bias
            output["preds"]["scf_detgf_agree"] = s_pred_c_3 + +s_pred_flow_3.detach() + n_pred_g.detach() + self.cls_bias
            output["preds"]["sgf_detgf"] = +s_pred_g.detach() + s_pred_flow.detach() + n_pred_c + self.cls_bias
            output["preds"]["sgf_detgf_agree"] = +s_pred_g_3.detach() + s_pred_flow_3.detach() + n_pred_c + self.cls_bias
            output["preds"]["scg_detgf"] = s_pred_c + s_pred_g.detach() + n_pred_flow.detach() + self.cls_bias
            output["preds"]["scg_detgf_agree"] = s_pred_c_3 + s_pred_g_3.detach() + n_pred_flow.detach() + self.cls_bias

            output["preds"]["sc_detcf"] = s_pred_c.detach() + n_pred_g.detach() + n_pred_flow.detach() + self.cls_bias
            output["preds"]["sg_detcf"] = s_pred_g.detach() + n_pred_c.detach() + n_pred_flow.detach() + self.cls_bias
            output["preds"]["sf_detcf"] = s_pred_flow.detach() + n_pred_c.detach() + n_pred_g.detach() + self.cls_bias
            output["preds"]["scf_detcf"] = s_pred_c.detach() + +s_pred_flow.detach() + n_pred_g.detach() + self.cls_bias
            output["preds"]["scf_detcf_agree"] = s_pred_c_3.detach() + s_pred_flow_3.detach() + n_pred_g.detach() + self.cls_bias
            output["preds"]["sgf_detcf"] = +s_pred_g.detach() + s_pred_flow.detach() + n_pred_c.detach() + self.cls_bias
            output["preds"]["sgf_detcf_agree"] = +s_pred_g_3.detach() + s_pred_flow_3.detach() + n_pred_c.detach() + self.cls_bias
            output["preds"]["scg_detcf"] = s_pred_c.detach() + s_pred_g.detach() + n_pred_flow.detach() + self.cls_bias
            output["preds"]["scg_detcf_agree"] = s_pred_c_3.detach() + s_pred_g_3.detach() + n_pred_flow.detach() + self.cls_bias

            output["preds"]["sc_detcg"] = s_pred_c.detach() + n_pred_g.detach() + n_pred_flow + self.cls_bias
            output["preds"]["sg_detcg"] = s_pred_g.detach() + n_pred_c.detach() + n_pred_flow + self.cls_bias
            output["preds"]["sf_detcg"] = s_pred_flow + n_pred_c.detach() + n_pred_g.detach() + self.cls_bias
            output["preds"]["scf_detcg"] = s_pred_c.detach() + s_pred_flow + n_pred_g.detach() + self.cls_bias
            output["preds"]["scf_detcg_agree"] = s_pred_c_3.detach() + s_pred_flow_3 + n_pred_g.detach() + self.cls_bias
            output["preds"]["sgf_detcg"] = s_pred_g.detach() + s_pred_flow + n_pred_c.detach() + self.cls_bias
            output["preds"]["sgf_detcg_agree"] = s_pred_g_3.detach() + s_pred_flow_3 + n_pred_c.detach() + self.cls_bias
            output["preds"]["scg_detcg"] = s_pred_c.detach() + s_pred_g.detach() + n_pred_flow + self.cls_bias
            output["preds"]["scg_detcg_agree"] = s_pred_c_3.detach() + s_pred_g_3.detach() + n_pred_flow + self.cls_bias

            output["preds"]["sc_detcgf"] = s_pred_c.detach() + n_pred_g.detach() + n_pred_flow.detach() + self.cls_bias.detach()
            output["preds"]["sg_detcgf"] = s_pred_g.detach() + n_pred_c.detach() + n_pred_flow.detach() + self.cls_bias.detach()
            output["preds"]["sf_detcgf"] = s_pred_flow.detach() + n_pred_c.detach() + n_pred_g.detach() + self.cls_bias.detach()
            output["preds"]["scf_detcgf"] = s_pred_c.detach() + s_pred_flow.detach() + n_pred_g.detach() + self.cls_bias.detach()
            output["preds"]["sgf_detcgf"] = s_pred_g.detach() + s_pred_flow.detach() + n_pred_c.detach() + self.cls_bias.detach()
            output["preds"]["scg_detcgf"] = s_pred_c.detach() + s_pred_g.detach() + n_pred_flow.detach() + self.cls_bias.detach()

            output["preds"]["sc"] = s_pred_c + n_pred_g + n_pred_flow + self.cls_bias
            output["preds"]["sg"] = s_pred_g + n_pred_c + n_pred_flow + self.cls_bias
            output["preds"]["sf"] = s_pred_flow + n_pred_c + n_pred_g + self.cls_bias
            output["preds"]["scf"] = s_pred_c + +s_pred_flow + n_pred_g + self.cls_bias
            output["preds"]["sgf"] = +s_pred_g + s_pred_flow + n_pred_c + self.cls_bias
            output["preds"]["scg"] = s_pred_c + s_pred_g + n_pred_flow + self.cls_bias

            output["preds"]["ncombined"] = n_pred

        output["features"] = features
        return output


class All2Model_ShuffleGrad(nn.Module):
    def __init__(self, encs, args: CfgNode):
        super(All2Model_ShuffleGrad, self).__init__()
        self.args = copy.deepcopy(args)
        pretraining_paths = self.args.get("pretraining_paths", {})
        frozen_encoders = self.args.get("frozen_encoders", False)
        self.fusion_gates = self.args.get("fusion_gates", False)

        if self.args.activate_model["video"]:
            video_args = copy.deepcopy(args)
            video_args.modality = "video"
            self.mod0_video_model = SwinModel(
                encs=encs,
                args=video_args,
                checkpoint=pretraining_paths.get("video", None),
            )
            if frozen_encoders:
                self.mod0_video_model.requires_grad_(False)
        if self.args.activate_model["flow"]:
            flow_args = copy.deepcopy(args)
            flow_args.modality = "flow"
            self.mod1_flow_model = SwinModel(
                encs=encs,
                args=flow_args,
                checkpoint=pretraining_paths.get("flow", None),
            )
            if frozen_encoders:
                self.mod1_flow_model.requires_grad_(False)

        self.fc_mod0 = nn.Linear(768, 174, bias=False)
        self.fc_mod1 = nn.Linear(768, 174, bias=False)
        self.cls_bias = nn.Parameter(torch.zeros(174), requires_grad=True)

        if self.args.bias_infusion.get("lib", 0) != 0:
            num_classes = 174
            d_model = 768
            self.fc_yz = nn.Sequential(
                nn.Linear(num_classes, d_model, bias=False),
                nn.ReLU(),
                nn.Linear(d_model, d_model * 2, bias=False),
            )

    def get_features(self, batch):
        frozen_encoders = self.args.get("frozen_encoders", False)
        pred = {}
        features = {}
        if frozen_encoders:
            with torch.no_grad():
                if self.args.activate_model["video"]:
                    video_output = self.mod0_video_model(batch, return_features=True, detach=True)
                    pred.update({"c": video_output["preds"]["combined"]})
                    features.update({"c": video_output["features"]["combined"]})

                if self.args.activate_model["flow"]:
                    flow_output = self.mod1_flow_model(batch, return_features=True, detach=True)
                    pred.update({"g": flow_output["preds"]["combined"]})
                    features.update({"g": flow_output["features"]["combined"]})
        else:
            if self.args.activate_model["video"]:
                video_output = self.mod0_video_model(batch, return_features=True, detach=True)
                pred.update({"c": video_output["preds"]["combined"]})
                features.update({"c": video_output["features"]["combined"]})

            if self.args.activate_model["flow"]:
                flow_output = self.mod1_flow_model(batch, return_features=True, detach=True)
                pred.update({"g": flow_output["preds"]["combined"]})
                features.update({"g": flow_output["features"]["combined"]})

        return pred, features

    def multimodal_pred(self, c, g, **kwargs):
        pred_c = torch.matmul(c, self.fc_mod0.weight.T)  # + self.fc_0_lin.bias / 2
        pred_g = torch.matmul(g, self.fc_mod1.weight.T)  # + self.fc_0_lin.bias / 2
        if "detach_c" in kwargs and kwargs["detach_a"]:
            pred_c = pred_c.detach()
        if "detach_g" in kwargs and kwargs["detach_a"]:
            pred_g = pred_g.detach()

        total_pred = pred_c + pred_g + self.cls_bias

        return total_pred, pred_c, pred_g

    def shuffle_ids(self, label):
        batch_size = label.size(0)
        shuffle_data = []
        random_shuffling = True
        if "rand" in self.args.bias_infusion.shuffle_type:
            while len(shuffle_data) < self.args.bias_infusion.num_samples:
                shuffle_idx_0 = torch.randperm(batch_size)
                nonequal_label = torch.ones(batch_size, dtype=torch.bool)

                if nonequal_label.sum() <= 1:
                    continue
                shuffle_data.append({"shuffle_idx": shuffle_idx_0, "data": nonequal_label})
        return shuffle_data

    def shuffle_data_func(self, label, pred_c, pred_g, pred_mm):

        shuffle_data = self.shuffle_ids(label)

        # We should have splitted the n_pred_c into 1,2 and 12 to use it along. It get tricky so we allow same labels for now.
        s_pred_c = torch.concatenate([pred_c[sh_data_i["shuffle_idx"]] for sh_data_i in shuffle_data], dim=0)
        s_pred_g = torch.concatenate([pred_g[sh_data_i["shuffle_idx"]] for sh_data_i in shuffle_data], dim=0)

        n_pred_c = torch.concatenate([pred_c[sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0)
        n_pred_g = torch.concatenate([pred_g[sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0)

        n_pred = torch.concatenate([pred_mm[sh_data_i["data"]].detach() for sh_data_i in shuffle_data], dim=0)

        n_label_shuffled = torch.concatenate([label[sh_data_i["shuffle_idx"]] for sh_data_i in shuffle_data], dim=0)
        n_label = torch.concatenate([label[sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0)

        return s_pred_c, s_pred_g, n_pred_c, n_pred_g, n_pred, n_label, n_label_shuffled

    def forward(self, batch: Dict[str, torch.Tensor], return_features: bool = False, **kwargs):

        pred, features = self.get_features(batch)

        norm_preds, pred_c, pred_g = self.multimodal_pred(features["c"], features["g"])

        output = {
            "preds": {
                "combined": norm_preds + self.cls_bias,
                "c": pred_c + self.cls_bias,
                "g": pred_g + self.cls_bias,
            }
        }

        # if "detach" in kwargs and kwargs["detach"]:
        if self.training:
            if self.args.bias_infusion.get("lib", 0) > 0:
                this_pred = norm_preds + self.cls_bias
                pred_feat = self.fc_yz(this_pred.detach())
                combined_features = torch.cat([features["c"], features["g"]], dim=1)
                CMI_yz_Loss = torch.nn.MSELoss()(combined_features, pred_feat) * self.args.bias_infusion.get("lib", 0)
                output["losses"] = {"CMI_yz_Loss": CMI_yz_Loss}

            if self.args.bias_infusion.get("l", 0) > 0:
                if not self.args.bias_infusion.get("training_mode", False):
                    self.eval()

                (
                    s_pred_c,
                    s_pred_g,
                    n_pred_c,
                    n_pred_g,
                    n_pred,
                    n_label,
                    n_label_shuffled,
                ) = self.shuffle_data_func(kwargs["label"], pred_c, pred_g, norm_preds)

                output["preds"]["sa_deta"] = s_pred_c.detach() + n_pred_g + self.cls_bias
                output["preds"]["sv_deta"] = s_pred_g + n_pred_c.detach() + self.cls_bias
                output["preds"]["sa_detv"] = s_pred_c + n_pred_g.detach() + self.cls_bias
                output["preds"]["sv_detv"] = s_pred_g.detach() + n_pred_c + self.cls_bias

                output["preds"]["sa"] = s_pred_c + n_pred_g + self.cls_bias
                output["preds"]["sv"] = s_pred_g + n_pred_c + self.cls_bias

                output["preds"]["ncombined"] = n_pred

                if not self.args.bias_infusion.get("training_mode", False):
                    self.train()

                output["preds"]["n_label"] = n_label
                output["preds"]["n_label_shuffled"] = n_label_shuffled

        output["features"] = features
        return output


class All3Model_OGM(nn.Module):
    def __init__(self, encs, args: CfgNode):
        super(All3Model_OGM, self).__init__()
        self.args = copy.deepcopy(args)
        pretraining_paths = self.args.get("pretraining_paths", {})
        frozen_encoders = self.args.get("frozen_encoders", False)
        self.fusion_gates = self.args.get("fusion_gates", False)

        if self.args.activate_model["video"]:
            video_args = copy.deepcopy(args)
            video_args.modality = "video"
            self.mod0_video_model = SwinModel(
                encs=encs,
                args=video_args,
                checkpoint=pretraining_paths.get("video", None),
            )
            if frozen_encoders:
                self.mod0_video_model.requires_grad_(False)
        if self.args.activate_model["flow"]:
            flow_args = copy.deepcopy(args)
            flow_args.modality = "flow"
            self.mod2_flow_model = SwinModel(
                encs=encs,
                args=flow_args,
                checkpoint=pretraining_paths.get("flow", None),
            )
            if frozen_encoders:
                self.mod2_flow_model.requires_grad_(False)
        if self.args.activate_model["layout"]:
            layout_args = copy.deepcopy(args)
            layout_args.modality = "layout"
            self.mod1_layout_model = Stlt(
                encs=encs,
                args=layout_args,
                checkpoint=pretraining_paths.get("layout", None),
            )
            if frozen_encoders:
                self.mod1_layout_model.requires_grad_(False)

        if self.args.activate_model["video"] and self.args.activate_model["flow"] and self.args.activate_model["layout"]:
            self.fc_0_lin = nn.Linear(768, 174)
            self.fc_1_lin = nn.Linear(768, 174, bias=False)
            self.fc_2_lin = nn.Linear(768, 174, bias=False)

            # if self.args.get("bnorm", False):
            #     self.classifier = nn.Sequential(nn.BatchNorm1d(768 * 3), nn.Linear(768 * 3, 174))
            # else:
            #     self.classifier = nn.Linear(768 * 3, 174)

        elif (
            self.args.activate_model["video"]
            and self.args.activate_model["flow"]
            or self.args.activate_model["video"]
            and self.args.activate_model["layout"]
            or self.args.activate_model["flow"]
            and self.args.activate_model["layout"]
        ):
            self.fc_0_lin = nn.Linear(768, 174)
            self.fc_1_lin = nn.Linear(768, 174, bias=False)

        # if cfg.CHECKPOINT_PATH:
        #     print("We are loading from {}".format(cfg.CHECKPOINT_PATH))
        #     self.load_state_dict(torch.load(cfg.CHECKPOINT_PATH, map_location="cpu"))

    def forward(self, batch: Dict[str, torch.Tensor], return_features: bool = False, **kwargs):

        frozen_encoders = self.args.get("frozen_encoders", False)
        pred = {}
        features = {}
        if frozen_encoders:
            with torch.no_grad():
                if self.args.activate_model["video"]:
                    video_output = self.mod0_video_model(batch, return_features=True)
                    pred.update({"c": video_output["preds"]["combined"]})
                    features.update({"c": video_output["features"]["combined"]})

                if self.args.activate_model["layout"]:
                    layout_output = self.mod1_layout_model(batch, return_features=True)
                    pred.update({"g": layout_output["preds"]["combined"]})
                    features.update({"g": layout_output["features"]["combined"]})

                if self.args.activate_model["flow"]:
                    flow_output = self.mod2_flow_model(batch, return_features=True)
                    pred.update({"flow": flow_output["preds"]["combined"]})
                    features.update({"flow": flow_output["features"]["combined"]})
        else:
            if self.args.activate_model["video"]:
                video_output = self.mod0_video_model(batch, return_features=True)
                pred.update({"c": video_output["preds"]["combined"]})
                features.update({"c": video_output["features"]["combined"]})

            if self.args.activate_model["layout"]:
                layout_output = self.mod1_layout_model(batch, return_features=True)
                pred.update({"g": layout_output["preds"]["combined"]})
                features.update({"g": layout_output["features"]["combined"]})

            if self.args.activate_model["flow"]:
                flow_output = self.mod2_flow_model(batch, return_features=True)
                pred.update({"flow": flow_output["preds"]["combined"]})
                features.update({"flow": flow_output["features"]["combined"]})

        if self.args.activate_model["video"] and self.args.activate_model["flow"] and self.args.activate_model["layout"]:
            feature_list = [features["c"], features["g"], features["flow"]]
        elif self.args.activate_model["video"] and self.args.activate_model["flow"]:
            feature_list = [features["c"], features["flow"]]
        elif self.args.activate_model["video"] and self.args.activate_model["layout"]:
            feature_list = [features["c"], features["g"]]
        elif self.args.activate_model["flow"] and self.args.activate_model["layout"]:
            feature_list = [features["g"], features["flow"]]
        elif self.args.activate_model["video"]:
            feature_list = [features["c"]]
        elif self.args.activate_model["flow"]:
            feature_list = [features["flow"]]
        elif self.args.activate_model["layout"]:
            feature_list = [features["g"]]

        pred_v = torch.matmul(feature_list[0], self.fc_0_lin.weight.T) + self.fc_0_lin.bias / 3
        pred_l = torch.matmul(feature_list[1], self.fc_1_lin.weight.T) + self.fc_0_lin.bias / 3
        pred_f = torch.matmul(feature_list[2], self.fc_2_lin.weight.T) + self.fc_0_lin.bias / 3

        preds = pred_v + pred_l + pred_f

        pred.update({"combined": preds})
        pred.update({"c": pred_v})
        pred.update({"g": pred_l})
        pred.update({"f": pred_f})

        return {"preds": pred, "features": features}


class All2Model_OGM(nn.Module):
    def __init__(self, encs, args: CfgNode):
        super(All2Model_OGM, self).__init__()
        self.args = copy.deepcopy(args)
        pretraining_paths = self.args.get("pretraining_paths", {})
        frozen_encoders = self.args.get("frozen_encoders", False)
        self.fusion_gates = self.args.get("fusion_gates", False)

        if self.args.activate_model["video"]:
            video_args = copy.deepcopy(args)
            video_args.modality = "video"
            self.mod0_video_model = SwinModel(
                encs=encs,
                args=video_args,
                checkpoint=pretraining_paths.get("video", None),
            )
            if frozen_encoders:
                self.mod0_video_model.requires_grad_(False)
        if self.args.activate_model["flow"]:
            flow_args = copy.deepcopy(args)
            flow_args.modality = "flow"
            self.mod1_flow_model = SwinModel(
                encs=encs,
                args=flow_args,
                checkpoint=pretraining_paths.get("flow", None),
            )
            if frozen_encoders:
                self.mod1_flow_model.requires_grad_(False)

        self.fc_0_lin = nn.Linear(768, 174)
        self.fc_1_lin = nn.Linear(768, 174, bias=False)

    def forward(self, batch: Dict[str, torch.Tensor], return_features: bool = False, **kwargs):

        frozen_encoders = self.args.get("frozen_encoders", False)
        pred = {}
        features = {}
        if frozen_encoders:
            with torch.no_grad():
                if self.args.activate_model["video"]:
                    video_output = self.mod0_video_model(batch, return_features=True)
                    pred.update({"c": video_output["preds"]["combined"]})
                    features.update({"c": video_output["features"]["combined"]})

                if self.args.activate_model["flow"]:
                    flow_output = self.mod1_flow_model(batch, return_features=True)
                    pred.update({"g": flow_output["preds"]["combined"]})
                    features.update({"g": flow_output["features"]["combined"]})
        else:
            if self.args.activate_model["video"]:
                video_output = self.mod0_video_model(batch, return_features=True)
                pred.update({"c": video_output["preds"]["combined"]})
                features.update({"c": video_output["features"]["combined"]})

            if self.args.activate_model["flow"]:
                flow_output = self.mod1_flow_model(batch, return_features=True)
                pred.update({"g": flow_output["preds"]["combined"]})
                features.update({"g": flow_output["features"]["combined"]})

        feature_list = [features["c"], features["g"]]

        pred_v = torch.matmul(feature_list[0], self.fc_0_lin.weight.T) + self.fc_0_lin.bias / 2
        pred_l = torch.matmul(feature_list[1], self.fc_1_lin.weight.T) + self.fc_0_lin.bias / 2

        preds = pred_v + pred_l

        pred.update({"combined": preds})
        pred.update({"c": pred_v})
        pred.update({"g": pred_l})

        return {"preds": pred, "features": features}


class Modality_out(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Modality_Text(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        total_out,
        pad_visual_out,
        pad_audio_out,
        pad_text_out,
        pad_visual_audio_out,
        pad_visual_text_out,
        pad_audio_text_out,
        zero_padding_out,
    ):
        return (total_out - pad_text_out + pad_visual_audio_out) / 3 + (pad_visual_out - pad_audio_text_out + pad_audio_out - pad_visual_text_out) / 6


class Modality_Audio(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        total_out,
        pad_visual_out,
        pad_audio_out,
        pad_text_out,
        pad_visual_audio_out,
        pad_visual_text_out,
        pad_audio_text_out,
        zero_padding_out,
    ):
        return (total_out - pad_audio_out + pad_visual_text_out) / 3 + (pad_visual_out - pad_audio_text_out + pad_text_out - pad_visual_audio_out) / 6


class Modality_Visual(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        total_out,
        pad_visual_out,
        pad_audio_out,
        pad_text_out,
        pad_visual_audio_out,
        pad_visual_text_out,
        pad_audio_text_out,
        zero_padding_out,
    ):
        return (total_out - pad_visual_out + pad_audio_text_out) / 3 + (pad_audio_out - pad_visual_text_out + pad_text_out - pad_visual_audio_out) / 6


class All3Model_AGM(nn.Module):
    def __init__(self, encs, args: CfgNode):
        super(All3Model_AGM, self).__init__()
        self.args = copy.deepcopy(args)
        pretraining_paths = self.args.get("pretraining_paths", {})
        frozen_encoders = self.args.get("frozen_encoders", False)
        self.fusion_gates = self.args.get("fusion_gates", False)

        if self.args.activate_model["video"]:
            video_args = copy.deepcopy(args)
            video_args.modality = "video"
            self.mod0_video_model = SwinModel(
                encs=encs,
                args=video_args,
                checkpoint=pretraining_paths.get("video", None),
            )
            if frozen_encoders:
                self.mod0_video_model.requires_grad_(False)
        if self.args.activate_model["flow"]:
            flow_args = copy.deepcopy(args)
            flow_args.modality = "flow"
            self.mod2_flow_model = SwinModel(
                encs=encs,
                args=flow_args,
                checkpoint=pretraining_paths.get("flow", None),
            )
            if frozen_encoders:
                self.mod2_flow_model.requires_grad_(False)
        if self.args.activate_model["layout"]:
            layout_args = copy.deepcopy(args)
            layout_args.modality = "layout"
            self.mod1_layout_model = Stlt(
                encs=encs,
                args=layout_args,
                checkpoint=pretraining_paths.get("layout", None),
            )
            if frozen_encoders:
                self.mod1_layout_model.requires_grad_(False)

        if self.args.activate_model["video"] and self.args.activate_model["flow"] and self.args.activate_model["layout"]:
            if self.args.get("bnorm", False):
                self.classifier = nn.Sequential(nn.BatchNorm1d(768 * 3), nn.Linear(768 * 3, 174))
            else:
                self.classifier = nn.Linear(768 * 3, 174)

        elif (
            self.args.activate_model["video"]
            and self.args.activate_model["flow"]
            or self.args.activate_model["video"]
            and self.args.activate_model["layout"]
            or self.args.activate_model["flow"]
            and self.args.activate_model["layout"]
        ):
            # self.classifier = nn.Linear(768*2,174)
            if self.fusion_gates == "film":
                self.classifier = FiLM(768, 768, 174)
            else:
                if self.args.get("bnorm", False):
                    self.classifier = nn.Sequential(nn.BatchNorm1d(768 * 2), nn.Linear(768 * 2, 174))
                else:
                    self.classifier = nn.Linear(768 * 2, 174)

        self.m_v_o = Modality_out()
        self.m_f_o = Modality_out()
        self.m_l_o = Modality_out()

        self.m_f = Modality_Text()
        self.m_l = Modality_Audio()
        self.m_v = Modality_Visual()
        self.m_v_o = Modality_out()
        self.m_f_o = Modality_out()
        self.m_l_o = Modality_out()

        self.scale_f = 1.0
        self.scale_v = 1.0
        self.scale_l = 1.0

        self.m_f_o.register_full_backward_hook(self.hookf)
        self.m_v_o.register_full_backward_hook(self.hookv)
        self.m_l_o.register_full_backward_hook(self.hookl)

        # if cfg.CHECKPOINT_PATH:
        #     print("We are loading from {}".format(cfg.CHECKPOINT_PATH))
        #     self.load_state_dict(torch.load(cfg.CHECKPOINT_PATH, map_location="cpu"))

    def hookl(self, m, ginp, gout):
        gnew = ginp[0].clone()
        return (gnew * self.scale_l,)

    def hookv(self, m, ginp, gout):
        gnew = ginp[0].clone()
        return (gnew * self.scale_v,)

    def hookf(self, m, ginp, gout):
        gnew = ginp[0].clone()
        return (gnew * self.scale_f,)

    def update_scale(self, coeff_c, coeff_g, coeff_f):
        self.scale_v = coeff_c
        self.scale_l = coeff_g
        self.scale_f = coeff_f

    def make_zero_batch(self, batch: Dict[str, torch.Tensor]):
        zero_input = {}
        for key in batch:
            zero_input[key] = torch.zeros_like(batch[key])
        return zero_input

    def forward(self, batch: Dict[str, torch.Tensor], return_features: bool = False, **kwargs):

        pred = {}
        features = {}
        video_output = self.mod0_video_model(batch, return_features=True)
        pred.update({"c": video_output["preds"]["combined"]})
        features.update({"c": video_output["features"]["combined"]})

        layout_output = self.mod1_layout_model(batch, return_features=True)
        pred.update({"g": layout_output["preds"]["combined"]})
        features.update({"g": layout_output["features"]["combined"]})

        flow_output = self.mod2_flow_model(batch, return_features=True)
        pred.update({"flow": flow_output["preds"]["combined"]})
        features.update({"flow": flow_output["features"]["combined"]})

        feature_list = [features["c"], features["g"], features["flow"]]
        preds = self.classifier(torch.cat(feature_list, dim=1))

        train_flag = self.training == "train"

        self.eval()
        with torch.no_grad():
            zero_input = self.make_zero_batch(batch)

            video_zero_features = self.mod0_video_model(zero_input, return_features=True)["features"]["combined"]
            layout_zero_features = self.mod1_layout_model(zero_input, return_features=True)["features"]["combined"]
            flow_zero_features = self.mod2_flow_model(zero_input, return_features=True)["features"]["combined"]

            feature_list = [video_zero_features, features["g"], features["flow"]]
            preds_zv = self.classifier(torch.cat(feature_list, dim=1))

            feature_list = [features["c"], layout_zero_features, features["flow"]]
            preds_zl = self.classifier(torch.cat(feature_list, dim=1))

            feature_list = [features["c"], features["g"], flow_zero_features]
            preds_zf = self.classifier(torch.cat(feature_list, dim=1))

            feature_list = [video_zero_features, layout_zero_features, features["flow"]]
            preds_zvl = self.classifier(torch.cat(feature_list, dim=1))

            feature_list = [video_zero_features, features["g"], flow_zero_features]
            preds_zvf = self.classifier(torch.cat(feature_list, dim=1))

            feature_list = [features["c"], layout_zero_features, flow_zero_features]
            preds_zlf = self.classifier(torch.cat(feature_list, dim=1))

            feature_list = [
                video_zero_features,
                layout_zero_features,
                flow_zero_features,
            ]
            preds_zvlf = self.classifier(torch.cat(feature_list, dim=1))

        if train_flag:
            self.train()
        m_v_out = self.m_v_o(
            self.m_v(
                preds,
                preds_zv,
                preds_zl,
                preds_zf,
                preds_zvl,
                preds_zvf,
                preds_zlf,
                preds_zvlf,
            )
        )
        m_l_out = self.m_l_o(
            self.m_l(
                preds,
                preds_zv,
                preds_zl,
                preds_zf,
                preds_zvl,
                preds_zvf,
                preds_zlf,
                preds_zvlf,
            )
        )
        m_f_out = self.m_f_o(
            self.m_f(
                preds,
                preds_zv,
                preds_zl,
                preds_zf,
                preds_zvl,
                preds_zvf,
                preds_zlf,
                preds_zvlf,
            )
        )

        # individual marginal contribution (contain zero padding)
        m_l_mc = m_l_out - preds_zvlf / 3
        m_v_mc = m_v_out - preds_zvlf / 3
        m_f_mc = m_f_out - preds_zvlf / 3

        pred.update({"both": preds})
        pred.update({"combined": m_v_out + m_l_out + m_f_out})
        pred.update({"c_mc": m_v_mc})
        pred.update({"g_mc": m_l_mc})
        pred.update({"f_mc": m_f_mc})
        pred.update({"c": m_v_out})
        pred.update({"g": m_l_out})
        pred.update({"f": m_f_out})

        return {"preds": pred, "features": features}


class All2Model_AGM(nn.Module):
    def __init__(self, encs, args: CfgNode):
        super(All2Model_AGM, self).__init__()
        self.args = copy.deepcopy(args)
        pretraining_paths = self.args.get("pretraining_paths", {})
        frozen_encoders = self.args.get("frozen_encoders", False)
        self.fusion_gates = self.args.get("fusion_gates", False)

        if self.args.activate_model["video"]:
            video_args = copy.deepcopy(args)
            video_args.modality = "video"
            self.mod0_video_model = SwinModel(
                encs=encs,
                args=video_args,
                checkpoint=pretraining_paths.get("video", None),
            )
            if frozen_encoders:
                self.mod0_video_model.requires_grad_(False)
        if self.args.activate_model["flow"]:
            flow_args = copy.deepcopy(args)
            flow_args.modality = "flow"
            self.mod1_flow_model = SwinModel(
                encs=encs,
                args=flow_args,
                checkpoint=pretraining_paths.get("flow", None),
            )
            if frozen_encoders:
                self.mod1_flow_model.requires_grad_(False)

        self.classifier = nn.Linear(768 * 2, 174)

        self.m_v_o = Modality_out()
        self.m_f_o = Modality_out()

        self.scale_f = 1.0
        self.scale_v = 1.0

        self.m_f_o.register_full_backward_hook(self.hookf)
        self.m_v_o.register_full_backward_hook(self.hookv)

    def hookv(self, m, ginp, gout):
        gnew = ginp[0].clone()
        return (gnew * self.scale_v,)

    def hookf(self, m, ginp, gout):
        gnew = ginp[0].clone()
        return (gnew * self.scale_f,)

    def update_scale(self, coeff_c, coeff_g):
        self.scale_v = coeff_c
        self.scale_l = coeff_g

    def make_zero_batch(self, batch: Dict[str, torch.Tensor]):
        zero_input = {}
        for key in batch:
            zero_input[key] = torch.zeros_like(batch[key])
        return zero_input

    def forward(self, batch: Dict[str, torch.Tensor], return_features: bool = False, **kwargs):

        pred = {}
        features = {}
        video_output = self.mod0_video_model(batch, return_features=True)
        pred.update({"c": video_output["preds"]["combined"]})
        features.update({"c": video_output["features"]["combined"]})

        flow_output = self.mod1_flow_model(batch, return_features=True)
        pred.update({"g": flow_output["preds"]["combined"]})
        features.update({"g": flow_output["features"]["combined"]})

        feature_list = [features["c"], features["g"]]
        preds = self.classifier(torch.cat(feature_list, dim=1))

        train_flag = self.training == "train"

        self.eval()
        with torch.no_grad():
            zero_input = self.make_zero_batch(batch)

            video_zero_features = self.mod0_video_model(zero_input, return_features=True)["features"]["combined"]
            flow_zero_features = self.mod1_flow_model(zero_input, return_features=True)["features"]["combined"]

            feature_list = [video_zero_features, features["g"]]
            preds_zv = self.classifier(torch.cat(feature_list, dim=1))

            feature_list = [features["c"], flow_zero_features]
            preds_zf = self.classifier(torch.cat(feature_list, dim=1))

        if train_flag:
            self.train()

        m_v_out = self.m_v_o(0.5 * (preds - preds_zv + preds_zf))
        m_f_out = self.m_f_o(0.5 * (preds - preds_zf + preds_zv))

        pred.update({"both": preds})
        pred.update({"combined": m_v_out + m_f_out})
        pred.update({"c": m_v_out})
        pred.update({"g": m_f_out})

        return {"preds": pred, "features": features}


class PF_All3Model(nn.Module):
    def __init__(self, encs, args: CfgNode):
        super(PF_All3Model, self).__init__()
        self.args = copy.deepcopy(args)

        d_model = args.d_model
        fc_inner = args.get("fc_inner", 512)
        dropout = args.get("dropout", 0.1)
        self.cls_type = args.get("cls_type", "linear")

        if self.args.activate_model["video"] and self.args.activate_model["flow"] and self.args.activate_model["layout"]:
            # self.classifier = [nn.Linear(768,174), nn.Linear(768,174), nn.Linear(768,174)]
            if self.cls_type == "linear":
                if self.args.get("wxnorm", False):
                    self.classifier_0 = nn.Sequential(nn.Linear(768, 174), nn.BatchNorm1d(174))
                    self.classifier_1 = nn.Sequential(nn.Linear(768, 174), nn.BatchNorm1d(174))
                    self.classifier_2 = nn.Sequential(nn.Linear(768, 174), nn.BatchNorm1d(174))
                elif self.args.get("bnorm", False):
                    self.classifier_0 = nn.Sequential(nn.BatchNorm1d(768), nn.Linear(768, 174))
                    self.classifier_1 = nn.Sequential(nn.BatchNorm1d(768), nn.Linear(768, 174))
                    self.classifier_2 = nn.Sequential(nn.BatchNorm1d(768), nn.Linear(768, 174))
                else:
                    self.classifier_0 = nn.Linear(768, 174)
                    self.classifier_1 = nn.Linear(768, 174)
                    self.classifier_2 = nn.Linear(768, 174)

                if self.args.get("cbnorm", False):
                    self.classifier = nn.Sequential(nn.BatchNorm1d(768 * 3), nn.Linear(768 * 3, 174))

            elif self.cls_type == "nonlinear":
                if self.args.wxnorm:
                    self.classifier_0 = nn.Sequential(nn.Linear(768, fc_inner), nn.BatchNorm1d(fc_inner))
                    self.classifier_1 = nn.Sequential(nn.Linear(768, fc_inner), nn.BatchNorm1d(fc_inner))
                    self.classifier_2 = nn.Sequential(nn.Linear(768, fc_inner), nn.BatchNorm1d(fc_inner))
                elif self.args.get("bnorm", False):
                    self.classifier_0 = nn.Sequential(nn.BatchNorm1d(768), nn.Linear(768, fc_inner))
                    self.classifier_1 = nn.Sequential(nn.BatchNorm1d(768), nn.Linear(768, fc_inner))
                    self.classifier_2 = nn.Sequential(nn.BatchNorm1d(768), nn.Linear(768, fc_inner))
                else:
                    self.classifier_0 = nn.Linear(768, fc_inner)
                    self.classifier_1 = nn.Linear(768, fc_inner)
                    self.classifier_2 = nn.Linear(768, fc_inner)
                self.classifier = nn.Sequential(
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(fc_inner, fc_inner),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(fc_inner, 174),
                )
                if self.args.get("cbnorm", False):
                    self.classifier = nn.Sequential(
                        nn.BatchNorm1d(768 * 3),
                        nn.Linear(768 * 3, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(fc_inner, 174),
                    )
            elif self.cls_type == "highlynonlinear":
                if self.args.wxnorm:
                    self.classifier_0 = nn.Sequential(nn.Linear(768, fc_inner), nn.BatchNorm1d(fc_inner))
                    self.classifier_1 = nn.Sequential(nn.Linear(768, fc_inner), nn.BatchNorm1d(fc_inner))
                    self.classifier_2 = nn.Sequential(nn.Linear(768, fc_inner), nn.BatchNorm1d(fc_inner))
                elif self.args.get("bnorm", False):
                    self.classifier_0 = nn.Sequential(nn.BatchNorm1d(768), nn.Linear(768, fc_inner))
                    self.classifier_1 = nn.Sequential(nn.BatchNorm1d(768), nn.Linear(768, fc_inner))
                    self.classifier_2 = nn.Sequential(nn.BatchNorm1d(768), nn.Linear(768, fc_inner))
                else:
                    self.classifier_0 = nn.Linear(768, fc_inner)
                    self.classifier_1 = nn.Linear(768, fc_inner)
                    self.classifier_2 = nn.Linear(768, fc_inner)
                self.classifier = nn.Sequential(
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.BatchNorm1d(fc_inner),
                    nn.Linear(fc_inner, fc_inner),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.BatchNorm1d(fc_inner),
                    nn.Linear(fc_inner, 174),
                )
                if self.args.get("cbnorm", False):
                    self.classifier = nn.Sequential(
                        nn.BatchNorm1d(768 * 3),
                        nn.Linear(768 * 3, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.BatchNorm1d(fc_inner),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.BatchNorm1d(fc_inner),
                        nn.Linear(fc_inner, 174),
                    )

        elif (
            self.args.activate_model["video"]
            and self.args.activate_model["flow"]
            or self.args.activate_model["video"]
            and self.args.activate_model["layout"]
            or self.args.activate_model["flow"]
            and self.args.activate_model["layout"]
        ):
            if self.args.get("wxnorm", False):
                self.classifier_0 = nn.Sequential(nn.Linear(768, fc_inner), nn.BatchNorm1d(fc_inner))
                self.classifier_1 = nn.Sequential(nn.Linear(768, fc_inner), nn.BatchNorm1d(fc_inner))
            elif self.args.get("bnorm", False):
                self.classifier_0 = nn.Sequential(nn.BatchNorm1d(768), nn.Linear(768, fc_inner))
                self.classifier_1 = nn.Sequential(nn.BatchNorm1d(768), nn.Linear(768, fc_inner))
            else:
                self.classifier_0 = nn.Linear(768, 174)
                self.classifier_1 = nn.Linear(768, 174)

            if self.args.get("cbnorm", False):
                self.classifier = nn.Sequential(nn.BatchNorm1d(768 * 2), nn.Linear(768 * 2, 174))

            # self.classifier = FiLM(768, 768, 174)
            # self.classifier = GatedFusion(768, 768, 174)

        else:
            if self.args.get("bnorm", False):
                self.classifier = nn.Sequential(nn.BatchNorm1d(768), nn.Linear(768, 174))
            else:
                self.classifier = nn.Linear(768, 174)

    def forward(self, batch: Dict[str, torch.Tensor], return_features: bool = False):

        features = batch
        pred = {}

        if self.args.activate_model["video"] and self.args.activate_model["flow"] and self.args.activate_model["layout"]:
            feature_list = [features[0], features[1], features[2]]

            if self.args.get("cbnorm", False):
                pred.update({"combined": self.classifier(torch.cat(feature_list, dim=1))})
            else:
                pred.update({"video": self.classifier_0(features[0])})
                pred.update({"flow": self.classifier_1(features[1])})
                pred.update({"layout": self.classifier_2(features[2])})
                if self.cls_type == "linear":
                    pred.update({"combined": pred["video"] + pred["flow"] + pred["layout"]})
                else:
                    pred.update({"combined": self.classifier(pred["video"] + pred["flow"] + pred["layout"])})

        elif self.args.activate_model["video"] and self.args.activate_model["flow"]:

            pred.update({"video": self.classifier_0[0](features[0])})
            pred.update({"flow": self.classifier_1[1](features[1])})
            pred.update({"combined": pred["video"] + pred["flow"]})

        elif self.args.activate_model["video"] and self.args.activate_model["layout"]:
            feature_list = [features[0], features[1]]

            if self.args.get("cbnorm", False):
                pred.update({"combined": self.classifier(torch.cat(feature_list, dim=1))})
            else:
                pred.update({"video": self.classifier_0(features[0])})
                pred.update({"layout": self.classifier_1(features[2])})
                if self.args.get("wxnorm", False):
                    pred["video"] = (pred["video"] - pred["video"].mean()) / pred["video"].std()
                    pred["layout"] = (pred["layout"] - pred["layout"].mean()) / pred["layout"].std()
                # pred.update({"combined": self.classifier(feature_list)})
                pred.update({"combined": pred["video"] + pred["layout"]})

        elif self.args.activate_model["flow"] and self.args.activate_model["layout"]:
            pred.update({"flow": self.classifier_0(features[1])})
            pred.update({"layout": self.classifier_1(features[2])})
            pred.update({"combined": pred["flow"] + pred["layout"]})

        elif self.args.activate_model["video"]:
            pred.update({"combined": self.classifier(features[0])})
        elif self.args.activate_model["flow"]:
            pred.update({"combined": self.classifier(features[1])})
        elif self.args.activate_model["layout"]:
            pred.update({"combined": self.classifier(features[2])})

        if return_features:
            return {"preds": pred, "features": features}
        return {"preds": pred}


class All3Model_Ens(nn.Module):
    def __init__(self, encs, args: CfgNode):
        super(All3Model_Ens, self).__init__()
        self.args = copy.deepcopy(args)
        pretraining_paths = self.args.get("pretraining_paths", {})
        frozen_encoders = self.args.get("frozen_encoders", False)
        self.norm_preds = self.args.get("norm_preds", False)

        if self.args.activate_model["video"]:
            video_args = copy.deepcopy(args)
            video_args.modality = "video"
            self.mod0_video_model = SwinModel(
                encs=encs,
                args=video_args,
                checkpoint=pretraining_paths.get("video", None),
                frozen_encoders=frozen_encoders,
            )

        if self.args.activate_model["flow"]:
            flow_args = copy.deepcopy(args)
            flow_args.modality = "flow"
            self.mod2_flow_model = SwinModel(
                encs=encs,
                args=flow_args,
                checkpoint=pretraining_paths.get("flow", None),
                frozen_encoders=frozen_encoders,
            )

        if self.args.activate_model["layout"]:
            layout_args = copy.deepcopy(args)
            layout_args.modality = "layout"
            self.mod1_layout_model = Stlt(
                encs=encs,
                args=layout_args,
                checkpoint=pretraining_paths.get("layout", None),
                frozen_encoders=frozen_encoders,
            )

    def forward(self, batch: Dict[str, torch.Tensor], return_features: bool = False, **kwargs):

        pred = {}
        features = {}

        if self.args.activate_model["video"]:
            video_output = self.mod0_video_model(batch, return_features=True)
            pred.update({"c": video_output["preds"]["combined"]})
            features.update({"c": video_output["features"]["combined"]})

        if self.args.activate_model["layout"]:
            layout_output = self.mod1_layout_model(batch, return_features=True)
            pred.update({"g": layout_output["preds"]["combined"]})
            features.update({"g": layout_output["features"]["combined"]})

        if self.args.activate_model["flow"]:
            flow_output = self.mod2_flow_model(batch, return_features=True)
            pred.update({"flow": flow_output["preds"]["combined"]})
            features.update({"flow": flow_output["features"]["combined"]})

        if self.norm_preds:
            for k in pred.keys():
                pred[k] = nn.functional.softmax(pred[k], dim=1)

        if self.args.activate_model["video"] and self.args.activate_model["flow"] and self.args.activate_model["layout"]:
            comb_pred = (pred["c"] + pred["flow"] + pred["g"]) / 3
        elif self.args.activate_model["video"] and self.args.activate_model["flow"]:
            comb_pred = (pred["c"] + pred["flow"]) / 2
        elif self.args.activate_model["video"] and self.args.activate_model["layout"]:
            comb_pred = (pred["c"] + pred["g"]) / 2
        elif self.args.activate_model["flow"] and self.args.activate_model["layout"]:
            comb_pred = (pred["flow"] + pred["g"]) / 2
        elif self.args.activate_model["video"]:
            comb_pred = pred["c"]
        elif self.args.activate_model["flow"]:
            comb_pred = pred["flow"]
        elif self.args.activate_model["layout"]:
            comb_pred = pred["g"]
        pred.update({"combined": comb_pred})

        return {"preds": pred, "features": features}


class All2Model_Ens(nn.Module):
    def __init__(self, encs, args: CfgNode):
        super(All2Model_Ens, self).__init__()
        self.args = copy.deepcopy(args)
        pretraining_paths = self.args.get("pretraining_paths", {})
        frozen_encoders = self.args.get("frozen_encoders", False)
        self.norm_preds = self.args.get("norm_preds", False)

        if self.args.activate_model["video"]:
            video_args = copy.deepcopy(args)
            video_args.modality = "video"
            self.mod0_video_model = SwinModel(
                encs=encs,
                args=video_args,
                checkpoint=pretraining_paths.get("video", None),
                frozen_encoders=frozen_encoders,
            )

        if self.args.activate_model["flow"]:
            flow_args = copy.deepcopy(args)
            flow_args.modality = "flow"
            self.mod1_flow_model = SwinModel(
                encs=encs,
                args=flow_args,
                checkpoint=pretraining_paths.get("flow", None),
                frozen_encoders=frozen_encoders,
            )

    def forward(self, batch: Dict[str, torch.Tensor], return_features: bool = False, **kwargs):

        pred = {}
        features = {}

        if self.args.activate_model["video"]:
            video_output = self.mod0_video_model(batch, return_features=True)
            pred.update({"c": video_output["preds"]["combined"]})
            features.update({"c": video_output["features"]["combined"]})

        if self.args.activate_model["flow"]:
            flow_output = self.mod1_flow_model(batch, return_features=True)
            pred.update({"g": flow_output["preds"]["combined"]})
            features.update({"g": flow_output["features"]["combined"]})

        if self.norm_preds:
            for k in pred.keys():
                pred[k] = nn.functional.softmax(pred[k], dim=1)

        comb_pred = (pred["c"] + pred["g"]) / 2

        pred.update({"combined": comb_pred})

        return {"preds": pred, "features": features}


class Video_Layout_Model(nn.Module):
    def __init__(self, encs, args):
        super(Video_Layout_Model, self).__init__()

        self.args = copy.deepcopy(args)
        pretraining_paths = self.args.get("pretraining_paths", {})

        self.mod0_video_model = SwinModel(encs=encs, args=self.args, checkpoint=pretraining_paths.get("video", None))
        self.mod1_layout_model = Stlt(encs=encs, args=self.args, checkpoint=pretraining_paths.get("layout", None))

        self.classifier = nn.Linear(768 * 2, 174)

        if self.args.get("freeze_encoders", False):
            self.mod0_video_model.requires_grad_(False)
            self.mod1_layout_model.requires_grad_(False)

        # if cfg.CHECKPOINT_PATH:
        #     print("We are loading from {}".format(cfg.CHECKPOINT_PATH))
        #     self.load_state_dict(torch.load(cfg.CHECKPOINT_PATH, map_location="cpu"))

    def forward(self, batch: Dict[str, torch.Tensor], return_features: bool = False):

        if self.args.get("freeze_encoders", False):
            self.mod0_video_model.eval()
            self.mod1_layout_model.eval()

        video_output, video_features = self.mod0_video_model(batch, return_features=True)

        layout_output, layout_features = self.mod1_layout_model(batch, return_features=True)

        pred = {"combined": self.classifier(torch.cat([video_features, layout_features], dim=1))}
        pred.update({"c": video_output["preds"]["combined"]})
        pred.update({"g": layout_output["preds"]["combined"]})

        # print(flow_output.keys())
        # print(layout_output.keys())

        # output = {"ACTION": (video_output["ACTION"] + flow_output["ACTION"] + layout_output["ACTION"])/3}
        if return_features:
            return {
                "preds": pred,
                "features": {"video": video_features, "layout": layout_features},
            }
        return {"preds": pred}


class Flow_Model(nn.Module):
    def __init__(self, cfg: CfgNode):
        super(Flow_Model, self).__init__()
        self.cfg = copy.deepcopy(cfg)

        cfg_flow = copy.deepcopy(cfg)
        cfg_flow.defrost()
        # cfg_flow.BACKBONE_MODEL_PATH = None
        cfg_flow.CHECKPOINT_PATH = cfg.FLOW_PRETRAINED_PATH if "FLOW_PRETRAINED_PATH" in cfg else None
        cfg_flow.freeze()
        self.flow_model = SwinModel(cfg=cfg_flow)

    def forward(self, batch: Dict[str, torch.Tensor], return_features: bool = False):

        flow_output, flow_features = self.flow_model(batch, return_features=True)

        pred = {"combined": flow_output["ACTION"]["combined"]}

        if return_features:
            return {"ACTION": pred, "features": {"flow": flow_features}}
        return {"ACTION": pred}


class Video_Layout_Model_Ens(nn.Module):
    def __init__(self, cfg: CfgNode):
        super(Video_Layout_Model_Ens, self).__init__()
        self.cfg = copy.deepcopy(cfg)

        # self.cfg.TRAIN_DATASET_PATH = "project_default_directory/Sth-Sth/something_something_detections/kkontras_flow_train.json"
        # self.cfg.VAL_DATASET_PATH = "project_default_directory/Sth-Sth/something_something_detections/kkontras_flow_val.json"
        cfg_video = copy.deepcopy(cfg)
        cfg_video.defrost()
        cfg_video.CHECKPOINT_PATH = cfg.VIDEO_PRETRAINED_PATH if "VIDEO_PRETRAINED_PATH" in cfg else None
        cfg_video.freeze()
        self.video_model = SwinModel(cfg=cfg_video)

        cfg_layout = copy.deepcopy(cfg)
        cfg_layout.defrost()
        cfg_layout.BACKBONE_MODEL_PATH = None
        cfg_layout.CHECKPOINT_PATH = cfg.LAYOUT_PRETRAINED_PATH if "LAYOUT_PRETRAINED_PATH" in cfg else None
        cfg_layout.freeze()
        self.layout_model = Stlt(cfg=cfg_layout)

        # if cfg.CHECKPOINT_PATH:
        #     print("We are loading from {}".format(cfg.CHECKPOINT_PATH))
        #     self.load_state_dict(torch.load(cfg.CHECKPOINT_PATH, map_location="cpu"))

    def forward(self, batch: Dict[str, torch.Tensor], return_features: bool = False):

        video_output, video_features = self.video_model(batch, return_features=True)

        layout_output, layout_features = self.layout_model(batch, return_features=True)

        pred = {"combined": (video_output["ACTION"]["combined"] + layout_output["ACTION"]["combined"]) / 2}
        pred.update({"video": video_output["ACTION"]["combined"]})
        pred.update({"layout": layout_output["ACTION"]["combined"]})

        if return_features:
            return {
                "ACTION": pred,
                "features": {"video": video_features, "layout": layout_features},
            }
        return {"ACTION": pred}


model_factory = {
    "stlt": Stlt,
    "swin": SwinModel,
}
