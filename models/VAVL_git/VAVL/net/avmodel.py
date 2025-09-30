import torch
import torch.nn.functional as F
from torch import nn

from models.VAVL_git.VAVL.conformer.model import Conformer


class MultitaskFusion(nn.Module):
    def __init__(self, model_args):
        super(MultitaskFusion, self).__init__()
        output_dim = model_args.output_num
        self.out_dropout = model_args.out_dropout
        self.hidden_1 = 256
        self.hidden_2 = 512

        self.projav1 = nn.Linear(2 * self.hidden_2, self.hidden_2)
        self.projav2 = nn.Linear(self.hidden_2, self.hidden_1)
        self.out_layer = nn.Linear(self.hidden_1, output_dim)

    def forward(self, feats_a, feats_v):

        # Main network output
        feats = torch.cat((feats_a, feats_v), dim=1)
        linear_hs_proj_av1 = self.projav2(F.dropout(F.relu(self.projav1(feats)), p=self.out_dropout, training=self.training))
        output = self.out_layer(linear_hs_proj_av1)

        return output


class Acoustic(nn.Module):
    def __init__(self, model_args):
        super(Acoustic, self).__init__()

        # Model Hyperparameters
        self.a_dim, self.v_dim = 1024, 1408
        self.d_v = 50
        self.hidden_2 = 512

        # 1D convolutional projection layers
        self.conv_1d_a = nn.Conv1d(self.a_dim, self.d_v, kernel_size=1, padding=0, bias=False)

        self.x_acoustic = Conformer(input_dim=self.d_v, encoder_dim=self.hidden_2, num_encoder_layers=3)

    def forward(self, x_aud):
        x_aud = x_aud.transpose(1, 2)

        # 1-D Convolution visual/audio features
        audio = x_aud if self.a_dim == self.d_v else self.conv_1d_a(x_aud)
        proj_x_a = audio.permute(2, 0, 1)
        acoustic_feats = self.x_acoustic(proj_x_a)

        return acoustic_feats


class Visual(nn.Module):
    def __init__(self, model_args):
        super(Visual, self).__init__()

        # Model Hyperparameters
        self.a_dim, self.v_dim = 1024, 1408
        self.d_v = 50
        self.hidden_2 = 512

        # 1D convolutional projection layers
        self.conv_1d_v = nn.Conv1d(self.v_dim, self.d_v, kernel_size=1, padding=0, bias=False)

        self.x_visual = Conformer(input_dim=self.d_v, encoder_dim=self.hidden_2, num_encoder_layers=3)

    def forward(self, x_vid):
        x_vid = x_vid.transpose(1, 2)

        # 1-D Convolution visual/audio features
        visual = x_vid if self.v_dim == self.d_v else self.conv_1d_v(x_vid)

        proj_x_v = visual.permute(2, 0, 1)
        visual_feats = self.x_visual(proj_x_v)

        return visual_feats


class Shared(nn.Module):
    def __init__(self, model_args):
        super(Shared, self).__init__()

        # Model Hyperparameters
        self.hidden_2 = 512

        self.x_shared = Conformer(input_dim=self.hidden_2, encoder_dim=self.hidden_2, num_encoder_layers=2)

        self.layer_norm = nn.LayerNorm(self.hidden_2)

    def forward(self, input):

        feats = self.x_shared(input)
        feats += input
        normalized_tensor = self.layer_norm(feats)
        normalized_tensor = normalized_tensor.permute(1, 0, 2)

        representation_feats = nn.AdaptiveAvgPool1d(1)(normalized_tensor.permute(0, 2, 1)).squeeze(2)

        return representation_feats


class MLP(nn.Module):
    def __init__(self, model_args):
        super(MLP, self).__init__()

        # Model Hyperparameters
        output_dim = model_args.output_num
        self.out_dropout = model_args.out_dropout
        self.hidden_1 = 256
        self.hidden_2 = 512

        # print('Out dim:', output_dim)
        self.projav1 = nn.Linear(self.hidden_2, self.hidden_2)
        self.projav2 = nn.Linear(self.hidden_2, self.hidden_1)
        self.out_layer = nn.Linear(self.hidden_1, output_dim)

    def forward(self, feats):

        # Main network output
        linear_hs_proj_av1 = self.projav2(F.dropout(F.relu(self.projav1(feats)), p=self.out_dropout, training=self.training))
        output = self.out_layer(linear_hs_proj_av1)

        return output


class MLP_reconst_v(nn.Module):
    def __init__(self, model_args):
        super(MLP_reconst_v, self).__init__()

        # Model Hyperparameters
        self.a_dim, self.v_dim = 1024, 1408
        self.out_dropout = model_args.out_dropout
        self.hidden_1 = 256
        self.hidden_2 = 512

        self.projav1 = nn.Linear(self.hidden_2, self.hidden_2)
        self.projav2 = nn.Linear(self.hidden_2, self.hidden_1)
        self.out_layer = nn.Linear(self.hidden_1, self.v_dim)

    def forward(self, feats):

        # Main network output
        linear_hs_proj_av1 = self.projav2(F.dropout(F.relu(self.projav1(feats)), p=self.out_dropout, training=self.training))
        output = self.out_layer(linear_hs_proj_av1)

        return output


class MLP_reconst_a(nn.Module):
    def __init__(self, model_args):
        super(MLP_reconst_a, self).__init__()

        # Model Hyperparameters
        self.a_dim, self.v_dim = 1024, 1408
        self.out_dropout = model_args.out_dropout
        self.hidden_1 = 256
        self.hidden_2 = 512

        self.projav1 = nn.Linear(self.hidden_2, self.hidden_2)
        self.projav2 = nn.Linear(self.hidden_2, self.hidden_1)
        self.out_layer = nn.Linear(self.hidden_1, self.a_dim)

    def forward(self, feats):

        # Main network output
        linear_hs_proj_av1 = self.projav2(F.dropout(F.relu(self.projav1(feats)), p=self.out_dropout, training=self.training))
        output = self.out_layer(linear_hs_proj_av1)

        return output
