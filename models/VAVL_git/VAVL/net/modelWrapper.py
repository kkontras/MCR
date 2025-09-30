import os
import sys

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler
from transformers import Wav2Vec2Model

from . import avmodel

sys.path.append(os.getcwd())


class ModelWrapper:
    def __init__(self, args, **kwargs):
        self.args = args
        self.device = args.device
        self.model_type = args.model_type
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.output_num = args.output_num
        self.lab_type = args.label_type
        self.lbl_learning = args.label_learning
        self.lr = args.lr
        self.model_path = args.model_path

        return

    def init_model(self):
        """
        Define model and load pretrained weights
        """
        assert self.model_type in ["wav2vec2-large", "wav2vec2-large-robust"], print("Wrong model type")

        default_models = {
            "wav2vec2": "wav2vec2-large-robust",
        }
        real_model_name = default_models.get(self.model_type, self.model_type)
        assert real_model_name not in ["wav2vec2"], print(
            "Model name is not properly converted.\n \
                Current model_name:",
            real_model_name,
        )

        root_model_type = real_model_name.split("-")[0]
        assert root_model_type in ["wav2vec2"], print(
            "Can't specify the root model type\n \
                Current root_model_type:",
            root_model_type,
        )

        #### Wav2vec2
        if root_model_type == "wav2vec2":
            """
            Additional settings
            - Freeze feature encoder (for all wav2vec2 models)
            - Prune top 12 transformer layers (for wav2vec2-large-robust)
            """
            self.wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/" + real_model_name)
            self.wav2vec_model.freeze_feature_encoder()
            if real_model_name == "wav2vec2-large-robust":
                del self.wav2vec_model.encoder.layers[12:]

        self.acoustic_model = avmodel.Acoustic(self.args)
        self.visual_model = avmodel.Visual(self.args)
        self.shared_model = avmodel.Shared(self.args)

        self.MLP_a = avmodel.MLP(self.args)
        self.MLP_av = avmodel.MLP(self.args)
        self.MLP_v = avmodel.MLP(self.args)

        self.weights = avmodel.MultitaskFusion(self.args)

        self.MLP_rec_a = avmodel.MLP_reconst_a(self.args)
        self.MLP_rec_v = avmodel.MLP_reconst_v(self.args)

        self.wav2vec_model.to("cuda:0")

        self.acoustic_model.to(self.device)
        self.visual_model.to(self.device)
        self.weights.to(self.device)
        self.shared_model.to(self.device)

        self.MLP_a.to(self.device)
        self.MLP_av.to(self.device)
        self.MLP_v.to(self.device)

        self.MLP_rec_a.to(self.device)
        self.MLP_rec_v.to(self.device)

        self.model_type_list = ["head", "wav2vec"]

        return self.wav2vec_model

    def init_optimizer(self):
        """
        Define optimizer for pre-trained model
        """

        assert self.wav2vec_model is not None and self.shared_model is not None, print("Model is not initialized")

        self.acoustic_model_opt = optim.Adam(
            self.acoustic_model.parameters(),
            lr=self.lr,
            weight_decay=5e-7,
            betas=(0.95, 0.999),
        )
        self.visual_model_opt = optim.Adam(
            self.visual_model.parameters(),
            lr=self.lr,
            weight_decay=5e-7,
            betas=(0.95, 0.999),
        )
        self.weights_opt = optim.Adam(
            self.weights.parameters(),
            lr=self.lr,
            weight_decay=5e-7,
            betas=(0.95, 0.999),
        )
        self.shared_model_opt = optim.Adam(
            self.shared_model.parameters(),
            lr=self.lr,
            weight_decay=5e-7,
            betas=(0.95, 0.999),
        )

        self.MLP_a_opt = optim.Adam(self.MLP_a.parameters(), lr=self.lr, weight_decay=5e-7, betas=(0.95, 0.999))
        self.MLP_v_opt = optim.Adam(self.MLP_v.parameters(), lr=self.lr, weight_decay=5e-7, betas=(0.95, 0.999))
        self.MLP_av_opt = optim.Adam(self.MLP_av.parameters(), lr=self.lr, weight_decay=5e-7, betas=(0.95, 0.999))

        self.MLP_rec_a_opt = optim.Adam(
            self.MLP_rec_a.parameters(),
            lr=self.lr,
            weight_decay=5e-7,
            betas=(0.95, 0.999),
        )
        self.MLP_rec_v_opt = optim.Adam(
            self.MLP_rec_v.parameters(),
            lr=self.lr,
            weight_decay=5e-7,
            betas=(0.95, 0.999),
        )

        self.scaler = GradScaler()

    def feed_forward(self, xa, xv, eval=False, **kwargs):
        """
        Feed forward the model
        """

        def __inference__(self, x_aud, x_vid, mode, **kwargs):

            mask = kwargs.get("attention_mask", None)

            if mode == "acoustic":
                # print(0)
                x_in = self.wav2vec_model(x_aud, attention_mask=mask).last_hidden_state
                representation_aud = self.acoustic_model(x_in)
                rep = self.shared_model(representation_aud)

                pred = self.MLP_a(rep)
                rec_pred = self.MLP_rec_a(rep)

                return pred, torch.mean(x_in, dim=1), rec_pred

            elif mode == "visual":
                # print(1)
                representation_vid = self.visual_model(x_vid)
                rep = self.shared_model(representation_vid)

                pred = self.MLP_v(rep)
                rec_pred = self.MLP_rec_v(rep)

                return pred, torch.mean(x_vid, dim=1), rec_pred

            elif mode == "weights":
                self.acoustic_model.eval()
                self.visual_model.eval()
                self.MLP_a.eval()
                self.MLP_v.eval()
                self.shared_model.eval()

                if len(x_aud.size()) == 2:
                    x_aud = self.wav2vec_model(x_aud, attention_mask=mask).last_hidden_state
                # print(x_in.size(), 'x_in')
                representation_aud = self.acoustic_model(x_aud)
                rep_a = self.shared_model(representation_aud)

                pred_a = self.MLP_a(rep_a)

                # print(x_vid.size(), 'x_vid')

                representation_vid = self.visual_model(x_vid)
                rep_v = self.shared_model(representation_vid)

                pred_v = self.MLP_v(rep_v)

                pred = self.weights(rep_a, rep_v)  # Shape: [batch_size, num_tasks]

                return pred_a, pred_v, pred

        if eval:
            with torch.no_grad():
                return __inference__(self, xa, xv, **kwargs)
        else:
            return __inference__(self, xa, xv, **kwargs)

    def backprop(self, total_loss, mode):
        """
        Update the model given loss
        """

        if mode == "acoustic":
            self.acoustic_model_opt.zero_grad(set_to_none=True)
            self.shared_model_opt.zero_grad(set_to_none=True)
            self.MLP_a_opt.zero_grad(set_to_none=True)
            self.MLP_rec_a_opt.zero_grad(set_to_none=True)
            total_loss.backward()
            self.acoustic_model_opt.step()
            self.shared_model_opt.step()
            self.MLP_a_opt.step()
            self.MLP_rec_a_opt.step()

        elif mode == "visual":
            self.visual_model_opt.zero_grad(set_to_none=True)
            self.shared_model_opt.zero_grad(set_to_none=True)
            self.MLP_v_opt.zero_grad(set_to_none=True)
            self.MLP_rec_v_opt.zero_grad(set_to_none=True)
            total_loss.backward()
            self.visual_model_opt.step()
            self.shared_model_opt.step()
            self.MLP_v_opt.step()
            self.MLP_rec_v_opt.step()

        elif mode == "weights":
            self.weights_opt.zero_grad(set_to_none=True)
            total_loss.backward()
            self.weights_opt.step()

    def save_model(self, epoch):
        """
        Save the model for each epoch
        """
        torch.save(
            self.acoustic_model.state_dict(),
            os.path.join(self.model_path, "final_acoustic_head.pt"),
        )
        torch.save(
            self.visual_model.state_dict(),
            os.path.join(self.model_path, "final_visual_head.pt"),
        )
        torch.save(
            self.weights.state_dict(),
            os.path.join(self.model_path, "final_weights_head.pt"),
        )
        torch.save(
            self.shared_model.state_dict(),
            os.path.join(self.model_path, "final_shared_head.pt"),
        )

        torch.save(self.MLP_a.state_dict(), os.path.join(self.model_path, "MLP_a_head.pt"))
        torch.save(self.MLP_v.state_dict(), os.path.join(self.model_path, "MLP_v_head.pt"))
        torch.save(self.MLP_av.state_dict(), os.path.join(self.model_path, "MLP_av_head.pt"))

        torch.save(
            self.MLP_rec_a.state_dict(),
            os.path.join(self.model_path, "MLP_rec_a_head.pt"),
        )
        torch.save(
            self.MLP_rec_v.state_dict(),
            os.path.join(self.model_path, "MLP_rec_v_head.pt"),
        )

    def set_eval(self):
        """
        Set the model to eval mode
        """
        self.wav2vec_model.eval()
        self.acoustic_model.eval()
        self.visual_model.eval()
        self.weights.eval()
        self.shared_model.eval()
        self.MLP_a.eval()
        self.MLP_av.eval()
        self.MLP_v.eval()
        self.MLP_rec_a.eval()
        self.MLP_rec_v.eval()

    def set_train(self):
        """
        Set the model to train mode
        """
        self.wav2vec_model.eval()
        self.acoustic_model.train()
        self.visual_model.train()
        self.weights.train()
        self.shared_model.train()
        self.MLP_a.train()
        self.MLP_av.train()
        self.MLP_v.train()
        self.MLP_rec_a.train()
        self.MLP_rec_v.train()

    def load_model(self, model_path, run_type):
        if run_type == "train":
            self.wav2vec_model.load_state_dict(torch.load(model_path + "/final_wav2vec.pt"))
        else:
            self.acoustic_model.load_state_dict(torch.load(model_path + "/final_acoustic_head.pt"))
            self.visual_model.load_state_dict(torch.load(model_path + "/final_visual_head.pt"))
            self.weights.load_state_dict(torch.load(model_path + "/final_weights_head.pt"))
            self.shared_model.load_state_dict(torch.load(model_path + "/final_shared_head.pt"))

            self.MLP_a.load_state_dict(torch.load(model_path + "/MLP_a_head.pt"))
            self.MLP_av.load_state_dict(torch.load(model_path + "/MLP_av_head.pt"))
            self.MLP_v.load_state_dict(torch.load(model_path + "/MLP_v_head.pt"))

            self.MLP_rec_a.load_state_dict(torch.load(model_path + "MLP_rec_a_head.pt"))
            self.MLP_rec_v.load_state_dict(torch.load(model_path + "MLP_rec_v_head.pt"))
