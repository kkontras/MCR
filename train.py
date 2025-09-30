import argparse
import logging
import shutil

from agents.general_agent import *
from utils.config import process_config_default, setup_logger

# xrandr --output DP-4 --scale 0.8x0.8


shutil._USE_CP_SENDFILE = False


def main(config_path, default_config_path, args):
    setup_logger()

    config = process_config_default(config_path, default_config_path)

    m = ""
    enc_m = ""

    if "fold" in args and args.fold is not None:
        if "data_split" in config.dataset:
            config.dataset.data_split.fold = int(args.fold)
        config.dataset.fold = int(args.fold)
        m += "fold{}".format(args.fold)
        enc_m += "fold{}".format(args.fold)
        seeds = [0, 109, 19, 337] if "UCF" in config_path else [109, 19, 337]
        config.training_params.seed = int(seeds[int(args.fold)])
        if "norm_wav_path" in config.dataset:
            config.dataset.norm_wav_path = config.dataset.norm_wav_path.format(args.fold)
        if "norm_face_path" in config.dataset:
            config.dataset.norm_face_path = config.dataset.norm_face_path.format(args.fold)
        if hasattr(config.model, "encoders"):
            for i in range(len(config.model.encoders)):
                config.model.encoders[i].pretrainedEncoder.dir = config.model.encoders[i].pretrainedEncoder.dir.format(args.fold)
        if "pretraining_paths" in config.model.args:
            for i in config.model.args.pretraining_paths:
                config.model.args.pretraining_paths[i] = config.model.args.pretraining_paths[i].format(args.fold)
    if "alpha" in args and args.alpha is not None:
        config.model.args.bias_infusion.alpha = float(args.alpha)
        m += "_alpha{}".format(args.alpha)
    if "recon_weight1" in args and args.recon_weight1 is not None:
        config.model.args.bias_infusion.weight1 = float(args.recon_weight1)
        m += "_w1{}".format(args.recon_weight1)
    if "recon_weight2" in args and args.recon_weight2 is not None:
        config.model.args.bias_infusion.weight2 = float(args.recon_weight2)
        m += "_w2{}".format(args.recon_weight2)
    if "recon_epochstages" in args and args.recon_epochstages is not None:
        config.model.args.bias_infusion.epoch_stages = int(args.recon_epochstages)
        m += "_epochstage{}".format(args.recon_epochstages)
    if "recon_ensemblestages" in args and args.recon_ensemblestages is not None:
        config.model.args.bias_infusion.ensemble_stages = int(args.recon_ensemblestages)
        m += "_ensstage{}".format(args.recon_ensemblestages)
    if "num_classes" in args and args.num_classes is not None:
        config.model.args.num_classes = int(args.num_classes)
        if hasattr(config.model, "encoders"):
            for i in range(len(config.model.encoders)):
                config.model.encoders[i].args.num_classes = int(args.num_classes)
        enc_m += "_numclasses{}".format(args.num_classes)
        m += "_numclasses{}".format(args.num_classes)
    if "tanh_mode_beta" in args and args.tanh_mode_beta is not None:
        config.model.args.bias_infusion.tanh_mode = "2"
        config.model.args.bias_infusion.tanh_mode_beta = float(args.tanh_mode_beta)
        m += "_beta{}".format(args.tanh_mode_beta)
    if "reg_by" in args and args.reg_by is not None:
        config.model.args.bias_infusion.reg_by = args.reg_by
        m += "_regby{}".format(args.reg_by)
        m += "_clip{}".format(args.clip)
    if "l" in args and args.l is not None:
        config.model.args.bias_infusion.l = float(args.l)
        m += "_l{}".format(args.l)
    if "multil" in args and args.multil is not None:
        for i in config.model.args.multi_loss.multi_supervised_w:
            if i != "combined" and config.model.args.multi_loss.multi_supervised_w[i] != 0:
                config.model.args.multi_loss.multi_supervised_w[i] = float(args.multil)
        m += "_multil{}".format(args.multil)
    if "lib" in args and args.lib is not None:
        config.model.args.bias_infusion.lib = float(args.lib)
        m += "_lib{}".format(args.lib)
    if "kmepoch" in args and args.kmepoch is not None:
        config.model.args.bias_infusion.keep_memory_epoch = int(args.kmepoch)
        m += "_kmepoch{}".format(args.kmepoch)
    if "mmcosine_scaling" in args and args.mmcosine_scaling is not None:
        config.model.args.bias_infusion.mmcosine_scaling = float(args.mmcosine_scaling)
        m += "_mmcosinescaling{}".format(args.mmcosine_scaling)
    if "ilr_c" in args and "ilr_g" in args and args.ilr_c is not None and args.ilr_g is not None:
        config.model.args.bias_infusion.init_learning_rate = {
            "c": float(args.ilr_c),
            "g": float(args.ilr_g),
        }
        m += "_ilrcg{}_{}".format(args.ilr_c, args.ilr_g)
    if "num_samples" in args and args.num_samples is not None:
        config.model.args.bias_infusion.num_samples = int(args.num_samples)
        m += "_numsamples{}".format(args.num_samples)

    if "contr_coeff" in args and args.contr_coeff is not None:
        config.model.args.bias_infusion.contr_coeff = float(args.contr_coeff)
        m += "_contrcoeff{}".format(args.contr_coeff)
    if "validate_with" in args and args.validate_with is not None:
        config.early_stopping.validate_with = args.validate_with
        enc_m += "_vld{}".format(args.validate_with)
        m += "_vld{}".format(args.validate_with)
    if "lr" in args and args.lr is not None:
        config.optimizer.learning_rate = float(args.lr)
        m += "_lr{}".format(args.lr)
        enc_m += "_lr{}".format(args.lr)
    if "wd" in args and args.wd is not None:
        config.optimizer.weight_decay = float(args.wd)
        m += "_wd{}".format(args.wd)
        enc_m += "_wd{}".format(args.wd)
    if "cls" in args and args.cls is not None:
        config.model.args.cls_type = args.cls
        m += "_{}".format(args.cls)
    if "batch_size" in args and args.batch_size is not None:
        config.training_params.batch_size = int(args.batch_size)
        m += "_bs{}".format(args.batch_size)
        enc_m += "_bs{}".format(args.batch_size)
    if "pre" in args and args.pre:
        m += "_pre"
        if hasattr(config.model, "encoders"):
            for i in range(len(config.model.encoders)):
                config.model.encoders[i].pretrainedEncoder.use = True

    config.model.save_dir = config.model.save_dir.format(m)

    # if enc_m != "":
    if hasattr(config.model, "encoders"):
        for i in range(len(config.model.encoders)):
            config.model.encoders[i].pretrainedEncoder.dir = config.model.encoders[i].pretrainedEncoder.dir.format(enc_m)

    logging.info("save_dir: {}".format(config.model.save_dir))
    agent_class = globals()[config.agent]
    agent = agent_class(config)
    agent.run()
    agent.finalize()


parser = argparse.ArgumentParser(description="My Command Line Program")
parser.add_argument("--config", help="Number of config file")
parser.add_argument("--default_config", help="Number of config file")
parser.add_argument("--fold", help="Fold")
parser.add_argument("--alpha", help="Alpha")
parser.add_argument("--tanh_mode_beta", help="tanh_mode_beta")
parser.add_argument("--reg_by", help="reg_by")
parser.add_argument("--batch_size", help="batch_size")
parser.add_argument("--l", help="L for Gat")
parser.add_argument("--multil", help="Coeff of Multi-Loss")
parser.add_argument("--lib", help="lib for Gat")
parser.add_argument("--kmepoch", help="keep memory epoch")
parser.add_argument("--num_samples", help="Number of samples for Gat")
parser.add_argument("--contr_coeff", help="ShuffleGrad Contrastive Coefficient")
parser.add_argument("--validate_with", help="validate_with")
parser.add_argument("--ilr_c", help="Initial Learning Rate Audio")
parser.add_argument("--ilr_g", help="Initial Learning Rate Video")
parser.add_argument("--mmcosine_scaling", help="mmcosine_scaling")
parser.add_argument("--recon_weight1", help="ReconBoost Parameters")
parser.add_argument("--recon_weight2", help="ReconBoost Parameters")
parser.add_argument("--recon_epochstages", help="ReconBoost Parameters")
parser.add_argument("--recon_ensemblestages", help="ReconBoost Parameters")
parser.add_argument("--lr", required=False, help="Learning Rate", default=None)
parser.add_argument("--wd", required=False, help="Weight Decay", default=None)
parser.add_argument("--cls", required=False, help="CLS linear, nonlinear, highlynonlinear", default=None)
parser.add_argument("--pre", action="store_true")
parser.set_defaults(pre=False)

args = parser.parse_args()

for var_name in vars(args):
    var_value = getattr(args, var_name)
    if var_value == "None":
        setattr(args, var_name, None)

main(config_path=args.config, default_config_path=args.default_config, args=args)
