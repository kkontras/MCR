import torch.nn.functional as F
import torch.nn as nn
import torch
import wandb
from models.model_utils.backbone import resnet18
from pytorch_metric_learning.losses import NTXentLoss



def nt_xent_loss(z_x, z_y, z_k=None, label=None, temperature=0.5):
    # z_x = F.normalize(z_x, p=2, dim=1)
    # z_y = F.normalize(z_y, p=2, dim=1)
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
            labels = torch.cat([torch.arange(z_x.shape[0]), torch.arange(z_y.shape[0]), torch.arange(z_k.shape[0])],
                               dim=0)

    loss = NTXentLoss(temperature=temperature)(z, labels)

    return loss

class Audio_ResNet(nn.Module):
    """
    Audio encoder using a ResNet backend.
    Produces:
        - features["combined"] : pooled feature vector (B × d_model)
        - preds["combined"]    : logits (B × num_classes)
        - nonaggr_features     : spatial feature map (B × C × HW)
    """
    def __init__(self, args, encs=None):
        super().__init__()

        self.num_classes = args.num_classes
        self.d_model = args.d_model
        self.audio_net = resnet18(modality="audio")

        # Final classifier on pooled vector
        self.aclassifier = nn.Linear(512, self.num_classes)

    def forward(self, x, **kwargs):
        """
        x : list or tuple
            x[0] must be audio input of shape (B, H, W)
        """
        audio_in = x[0].unsqueeze(1)            # (B, 1, H, W)
        feat_map = self.audio_net(audio_in)     # (B, C, H', W')

        pooled = F.adaptive_avg_pool2d(feat_map, 1).flatten(1)   # (B, C)

        # detaching if required
        if kwargs.get("detach_enc0", False):
            pooled = pooled.detach()
            feat_map = feat_map.detach()

        pred = self.aclassifier(pooled.detach() if kwargs.get("detach_pred", False) else pooled)

        return {
            "preds": {"combined": pred},
            "features": {"combined": pooled},
            "nonaggr_features": {"combined": feat_map.flatten(start_dim=2)},  # (B, C, H'*W')
        }

class Video_ResNet(nn.Module):
    """
    Video encoder using a ResNet backend.
    Produces:
        - features["combined"] : pooled video feature (B × d_model)
        - preds["combined"]    : logits (B × num_classes)
        - nonaggr_features     : spatial+temporal map (B × C × T*H*W)
    """
    def __init__(self, args, encs=None):
        super().__init__()

        self.num_classes = args.num_classes
        self.d_model = args.d_model
        self.visual_net = resnet18(modality="visual")

        self.vclassifier = nn.Linear(self.d_model, self.num_classes)

    def forward(self, x, **kwargs):
        """
        x : list or tuple
            x[1] must be video input of shape (B, T, C, H, W)
        """
        video_in = x[1]                           # (B, T, C, H, W)

        B, T, C, H, W = video_in.shape
        v = self.visual_net(video_in)

        # reshape back into temporal sequence
        Fdim, Hp, Wp = v.shape[1:]
        v = v.view(B, T, Fdim, Hp, Wp).permute(0, 2, 1, 3, 4)  # (B, Fdim, T, Hp, Wp)

        pooled = F.adaptive_avg_pool3d(v, 1).flatten(1)        # (B, Fdim)

        if kwargs.get("detach_enc1", False):
            pooled = pooled.detach()
            v = v.detach()

        pred = self.vclassifier(pooled.detach() if kwargs.get("detach_pred", False) else pooled)

        return {
            "preds": {"combined": pred},
            "features": {"combined": pooled},
            "nonaggr_features": {"combined": v.flatten(start_dim=2)},  # (B, C, T*H*W)
        }

class MCR_Linear(nn.Module):

    def __init__(self, args: dict, encs: list[nn.Module]):
        super().__init__()

        self.args: dict = args
        num_classes: int = args["num_classes"]
        d_model: int = args["d_model"]

        self.enc_0: nn.Module = encs[0]
        self.enc_1: nn.Module = encs[1]

        self.use_bias_infusion: bool = args.get("bias_infusion", {}).get("l", 0) != 0

        self.fc_0: nn.Linear = nn.Linear(d_model, num_classes, bias=False)
        self.fc_1: nn.Linear = nn.Linear(d_model, num_classes, bias=False)
        self.bias_lin: nn.Parameter = nn.Parameter(torch.zeros(num_classes))

        self.bias_infusion = args.get("bias_infusion", {})
        self.lib = self.bias_infusion.get("lib", 0) > 0
        self.l = self.bias_infusion.get("l", 0) > 0
        self.temperature = self.bias_infusion.get("temperature", 0.1)
        self.contr_coeff = self.bias_infusion.get("contr_coeff", 0.0)
        self.contr_type = self.bias_infusion.get("contr_type", "label")
        self.regby = self.bias_infusion.get("regby", "greedy")

        if self.lib > 0:
            self.fc_yz: nn.Sequential = nn.Sequential(
                nn.Linear(num_classes, d_model, bias=False),
                nn.ReLU(),
                nn.Linear(d_model, d_model * 2, bias=False),
            )

    def _get_features(self, x: torch.Tensor, **enc_kwargs):
        """
        Run both encoders and extract their combined feature tensors.

        Returns
        -------
        a_feat : torch.Tensor
            Features from enc_0, shape (B, d_model)
        v_feat : torch.Tensor
            Features from enc_1, shape (B, d_model)
        a_pred : torch.Tensor
            Unimodal predictions from enc_0, shape (B, num_class)
        v_pred : torch.Tensor
            Unimodal predictions from enc_1, shape (B, num_class)
        """
        out_a = self.enc_0(x, **enc_kwargs)
        out_v = self.enc_1(x, **enc_kwargs)

        a_feat = out_a["features"]["combined"]
        v_feat = out_v["features"]["combined"]
        a_pred = out_a["preds"]["combined"]
        v_pred = out_v["preds"]["combined"]

        return a_feat, v_feat, a_pred, v_pred

    def _linear_forward(
        self,
        a_feat: torch.Tensor,
        v_feat: torch.Tensor,
        detach_a: bool = False,
        detach_v: bool = False,
        skip_bias: bool = False,
    ) -> torch.Tensor:
        """
        Compute the linear fusion prediction:
            pred = a_feat * fc_0 + v_feat * fc_1 + bias_lin

        Parameters
        ----------
        a_feat, v_feat : torch.Tensor
            Feature matrices for each modality.
        detach_a, detach_v : bool
            If True, detach the corresponding features.
        skip_bias : bool
            If True, do not add the bias term.

        Returns
        -------
        torch.Tensor
            Logits of shape (B, num_classes).
        """
        if detach_a:
            a_feat = a_feat.detach()
        if detach_v:
            v_feat = v_feat.detach()

        pred_a = self.fc_0(a_feat)
        pred_v = self.fc_1(v_feat)

        pred = pred_a + pred_v

        if not skip_bias:
            pred = pred + self.bias_lin

        return pred

    def _random_shuffles(self, batch_size: int, num_samples: int):
        """
        Generate `num_samples` random permutations over the batch.
        Returns a list of shuffle_idx tensors, each of length batch_size.
        """
        return [torch.randperm(batch_size) for _ in range(num_samples)]

    def _build_shuffled_batches(
            self,
            a_feat: torch.Tensor,
            v_feat: torch.Tensor,
            pred: torch.Tensor,
            labels: torch.Tensor,
    ):
        """
        Build expanded shuffled and repeated batches.
        Shapes: (B * num_samples, d_model) or (B * num_samples,)
        """
        B = labels.size(0)
        num_samples = self.args["bias_infusion"].get("num_samples", 1)

        shuffle_indices = self._random_shuffles(B, num_samples)

        # Expanded shuffled features
        shuffled_a = torch.cat([a_feat[idx] for idx in shuffle_indices], dim=0)
        shuffled_v = torch.cat([v_feat[idx] for idx in shuffle_indices], dim=0)
        label_shuffled = torch.cat([labels[idx] for idx in shuffle_indices], dim=0)

        # Repeated originals
        repeated = torch.cat([torch.arange(B) for _ in range(num_samples)], dim=0)

        a_repeated = a_feat[repeated]
        v_repeated = v_feat[repeated]
        pred_repeated = pred[repeated]
        label_repeated = labels[repeated]

        return {
            "shuffled_a": shuffled_a,
            "shuffled_v": shuffled_v,
            "a_repeated": a_repeated,
            "v_repeated": v_repeated,
            "pred_repeated": pred_repeated,
            "label_repeated": label_repeated,
            "label_shuffled": label_shuffled,
        }

    def forward(self, x: torch.Tensor, labels: torch.Tensor = None, **enc_kwargs):
        """
        Forward pass combining encoders + linear fusion.
        Produces additional outputs when bias infusion is enabled.

        Parameters
        ----------
        x : torch.Tensor
            Input batch.
        labels : torch.Tensor, optional
            Ground-truth labels for bias infusion.
        enc_kwargs : dict
            Additional arguments forwarded to enc_0 and enc_1.

        Returns
        -------
        dict
            Contains:
                "pred" : logits
                "features" : {"a": a_feat, "v": v_feat}
                "bias_infusion": {...}   # optional
        """
        a_feat, v_feat, a_pred, v_pred = self._get_features(x, **enc_kwargs)
        pred = self._linear_forward(a_feat, v_feat)

        out = {
            "preds": {"combined": pred,
                      "c": a_pred,
                      "g": v_pred
                      },
            "features": {
                "a": a_feat,
                "v": v_feat,
            },
        }

        if (
            self.training
            and self.l>0
            and labels is not None
        ):
            s = self._build_shuffled_batches(a_feat, v_feat, pred, labels)

            out["preds"].update({

                # --- shuffled A with repeated V ---
                "sa_detv": self._linear_forward(s["shuffled_a"], s["v_repeated"], detach_v=True, skip_bias=True),
                "sa_deta": self._linear_forward(s["shuffled_a"], s["v_repeated"], detach_a=True, skip_bias=True),
                "sa": self._linear_forward(s["shuffled_a"], s["v_repeated"], skip_bias=True),

                # --- repeated A with shuffled V ---
                "sv_detv": self._linear_forward(s["a_repeated"], s["shuffled_v"], detach_v=True, skip_bias=True),
                "sv_deta": self._linear_forward(s["a_repeated"], s["shuffled_v"], detach_a=True, skip_bias=True),
                "sv": self._linear_forward(s["a_repeated"], s["shuffled_v"], skip_bias=True),

                # --- repeated predictions and labels ---
                "pred_repeated": s["pred_repeated"],
                "label_repeated": s["label_repeated"],
                "label_shuffled": s["label_shuffled"],
            })

            out["losses"], out["permutation_importance"] = self._compute_loss(out, labels)

        return out

    def _apply_game(self, lmipds):
        """
        Compute LMIPD-based regularization according to the selected game setting.

        lmipds keys expected:
            enc0_sa_jsd, enc0_sv_jsd,
            enc1_sa_jsd, enc1_sv_jsd,
            all_sa_jsd,  all_sv_jsd
        """

        # JSD scores for each encoder under each perturbation
        enc0_sa = lmipds["enc0_sa_jsd"]  # encoder 0 under A-shuffle
        enc0_sv = lmipds["enc0_sv_jsd"]  # encoder 0 under V-shuffle
        enc1_sa = lmipds["enc1_sa_jsd"]  # encoder 1 under A-shuffle
        enc1_sv = lmipds["enc1_sv_jsd"]  # encoder 1 under V-shuffle

        if self.regby == "greedy":
            # Use encoder-specific JSDs
            reg0 = enc0_sa
            reg1 = enc1_sv
            total = self.l * (-enc0_sa + enc1_sa - enc1_sv + enc0_sv)

        elif self.regby == "ind":
            # Penalize each encoder individually
            reg0 = enc0_sa
            reg1 = enc1_sv
            total = self.l * (-(enc0_sa + enc1_sv))

        else:  # "colab"
            # Cooperative minimization using modality-aggregated JSDs
            reg0 = lmipds["all_sa_jsd"]
            reg1 = lmipds["all_sv_jsd"]
            total = self.l * (-(reg0 + reg1))

        return {
            "total_reg": total,
            "reg0": reg0,
            "reg1": reg1,
        }

    def _compute_contrastive_term(self, output, label):
        if not self.contr_coeff:
            return 0.0

        feat_a = output["features"]["a"]
        feat_v = output["features"]["v"]
        t = self.temperature

        if self.contr_type == "label":
            loss = nt_xent_loss(feat_a, feat_v, label=label, temperature=t)
        else:
            loss = nt_xent_loss(feat_a, feat_v, temperature=t)

        return self.contr_coeff * loss

    def _get_lmipds(self, output):
        P = output["preds"]
        device = self.agent.device

        def jsd(p, q):
            p = F.softmax(p, dim=1)
            q = F.softmax(q, dim=1)
            m = 0.5 * (p + q)
            return 0.5 * (
                    F.kl_div(m.log(), p, reduction="batchmean") +
                    F.kl_div(m.log(), q, reduction="batchmean")
            )

        def eval_side(side):
            if side == "sa":
                keys = ("sa_detv", "sa_deta", "sa")
            else:
                keys = ("sv_detv", "sv_deta", "sv")

            if all(k in P for k in keys):
                detv, deta, both = P[keys[0]], P[keys[1]], P[keys[2]]
                base = P["pred_repeated"]
                return {
                    "enc0": jsd(detv, base),
                    "enc1": jsd(deta, base),
                    "all": jsd(both, base),
                }
            else:
                zero = torch.tensor(0.0, device=device)
                return {"enc0": zero, "enc1": zero, "all": zero}

        sa = eval_side("sa")
        sv = eval_side("sv")

        base = P["pred_repeated"]
        base_acc = (base.argmax(1) == P["label_repeated"]).float().mean() + 1e-8
        acc_sa = (P["sa_detv"].argmax(1) == P["label_repeated"]).float().mean() if "sa_detv" in P else torch.tensor(0.0, device=device)
        acc_sv = (P["sv_deta"].argmax(1) == P["label_repeated"]).float().mean() if "sv_deta" in P else torch.tensor(0.0, device=device)

        permutation_importance = {
            "enc_0": acc_sv / base_acc,
            "enc_1": acc_sa / base_acc,
        }

        out = {
            "enc0_sa_jsd": sa["enc0"],
            "enc1_sa_jsd": sa["enc1"],
            "all_sa_jsd": sa["all"],

            "enc0_sv_jsd": sv["enc0"],
            "enc1_sv_jsd": sv["enc1"],
            "all_sv_jsd": sv["all"],
        }

        return out, permutation_importance

    def _reconstruction_loss(self, output):
        """
        Reconstruct features from predicted labels using fc_yz.
        Returns MSE loss or 0 if reconstruction is disabled.
        """
        if self.lib <=0.0:
            return torch.tensor(0.0, device=self.bias_lin.device)

        preds = output["preds"]["combined"].detach()              # B × num_classes
        a_feat = output["features"]["a"]                          # B × d_model
        v_feat = output["features"]["v"]                          # B × d_model
        true_feat = torch.cat([a_feat, v_feat], dim=1)            # B × (2*d_model)

        recon = self.fc_yz(preds)                                 # B × (2*d_model)

        return F.mse_loss(recon, true_feat)

    def _log_reg_terms(self, lmipds, lmipd_reg, contrastive_reg, reconstr_reg):
        try:
            if hasattr(wandb, "run") and wandb.run is not None:
                wandb.log(
                    {
                        "lmipds": lmipds,
                        "lmipd_reg": lmipd_reg,
                        "contrastive_reg": contrastive_reg,
                        "reconstruction_reg": reconstr_reg,
                    },
                    step=self.agent.logs["current_step"] + 1
                )
        except Exception:
            pass  # completely silent fail

    def _compute_loss(self, output, labels):
        """
        Compute:
            - LMIPD regularizer
            - Contrastive regularizer
            - Reconstruction regularizer
        plus logs each component.
        """

        lmipds, permutation_importance = self._get_lmipds(output)
        lmipd_reg = self._apply_game(lmipds)

        contrastive_reg = self._compute_contrastive_term(output, labels)

        reconstr_reg = self._reconstruction_loss(output)

        output_losses = {
            "lmipd": lmipd_reg["total_reg"],
            "contrastive": contrastive_reg,
            "reconstruction": reconstr_reg,
        }

        self._log_reg_terms(
            lmipds=lmipds,
            lmipd_reg=lmipd_reg,
            contrastive_reg=contrastive_reg,
            reconstr_reg=reconstr_reg
        )

        return output_losses, permutation_importance



if __name__ == "__main__":

    class Args:
        def __init__(self):
            self.num_classes = 5
            self.d_model = 512
            self.fc_inner = 256
            self.dropout = 0.1
            self.bias_infusion = {
                "num_samples": 3,
                "l": 1,
                "lib": 1,
                "contr_type": "label",
                "temperature": 0.1,
                "contr_coeff": 1.0,
                "regby": "greedy"
            }


    torch.manual_seed(0)
    args = Args()

    # unimodal encoders
    enc0 = Audio_ResNet(args)
    enc1 = Video_ResNet(args)

    # multimodal
    model = MCR_Linear(vars(args), encs=[enc0, enc1])
    model.training = True

    # fake agent
    model.agent = type("A", (), {})()
    model.agent.device = "cpu"
    model.agent.logs = {"current_step": 0}

    # dummy data
    B = 4
    audio = torch.randn(B, 257, 188)
    video = torch.randn(B, 3, 3, 224, 224)
    x = {0: audio, 1: video}
    labels = torch.randint(0, args.num_classes, (B,))


    out = model(x, labels=labels)

    print("\n=== OUTPUT KEYS ===")
    print(list(out.keys()))

    losses = out["losses"]
    print("\n=== LOSSES ===")
    for k, v in losses.items():
        print(f"{k:12s}: {float(v):.4f}")

    print("\n=== PERMUTATION IMPORTANCE ===")
    importance = out["permutation_importance"]
    for k, v in importance.items():
        print(f"{k:20s}: {float(v):.4f}")


