import copy
import json
import logging
import os
import sys
from logging import Formatter
from logging.handlers import RotatingFileHandler
from pprint import pprint

from easydict import EasyDict


def _resolve_path_placeholders(value, data_dir, checkpoints_dir):
    if isinstance(value, EasyDict):
        return EasyDict({k: _resolve_path_placeholders(v, data_dir, checkpoints_dir) for k, v in value.items()})
    if isinstance(value, dict):
        return {k: _resolve_path_placeholders(v, data_dir, checkpoints_dir) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_path_placeholders(v, data_dir, checkpoints_dir) for v in value]
    if isinstance(value, str):
        # Backward compatibility for existing configs.
        value = value.replace("project_default_directory/data/Balance_Final", checkpoints_dir)
        value = value.replace("project_default_directory", data_dir)
        # Explicit placeholders preferred going forward.
        value = value.replace("project_checkpoints_directory", checkpoints_dir)
        value = value.replace("project_data_directory", data_dir)
        # Support shell-style variables directly in config files.
        value = os.path.expandvars(value)
    return value


def _apply_project_paths(config):
    project_default_dir = os.path.abspath(os.getenv("PROJECT_DEFAULT_DIR", os.getcwd()))
    data_dir = os.path.abspath(os.getenv("PROJECT_DATA_DIR", os.path.join(project_default_dir, "data")))
    checkpoints_env = os.getenv("PROJECT_CHECKPOINTS_DIR", os.getenv("PROJECT_CHECKPOINT_DIR"))
    checkpoints_dir = os.path.abspath(checkpoints_env or os.path.join(project_default_dir, "checkpoints"))
    os.environ.setdefault("PROJECT_DEFAULT_DIR", project_default_dir)
    os.environ.setdefault("PROJECT_DATA_DIR", data_dir)
    os.environ.setdefault("PROJECT_CHECKPOINT_DIR", checkpoints_dir)
    os.environ.setdefault("PROJECT_CHECKPOINTS_DIR", checkpoints_dir)
    return _resolve_path_placeholders(config, data_dir, checkpoints_dir)


def setup_logging(log_dir):
    log_file_format = "[%(levelname)s] - %(asctime)s - %(name)s - : %(message)s in %(pathname)s:%(lineno)d"
    log_console_format = "[%(levelname)s]: %(message)s"

    logging.shutdown()

    # Main logger
    main_logger = logging.getLogger()
    main_logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(Formatter(log_console_format))

    exp_file_handler = RotatingFileHandler("{}exp_debug.log".format(log_dir), maxBytes=10**6, backupCount=5)
    exp_file_handler.setLevel(logging.DEBUG)
    exp_file_handler.setFormatter(Formatter(log_file_format))

    exp_errors_file_handler = RotatingFileHandler("{}exp_error.log".format(log_dir), maxBytes=10**6, backupCount=5)
    exp_errors_file_handler.setLevel(logging.WARNING)
    exp_errors_file_handler.setFormatter(Formatter(log_file_format))

    main_logger.addHandler(console_handler)
    main_logger.addHandler(exp_file_handler)
    main_logger.addHandler(exp_errors_file_handler)


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file: the path of the config file
    :return: config(namespace), config(dictionary)
    """

    # parse the configurations from the config json file provided
    with open(json_file, "r") as config_file:
        try:
            config_dict = json.load(config_file)
            # EasyDict allows to access dict values as attributes (works recursively).
            config = EasyDict(config_dict)
            return config, config_dict
        except ValueError:
            print("INVALID JSON file format.. Please provide a good json file")
            exit(-1)


def setup_logger():

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=[handler])

    # main_logger = logging.getLogger()
    # main_logger.setLevel(logging.WARNING)

    logger = logging.getLogger("wandb")
    logger.setLevel(logging.WARNING)


def merge_dicts(default_dict: EasyDict, dict2: EasyDict) -> EasyDict:
    """
    Recursively merges two dictionaries, combining their values.
    If a key exists in both dictionaries, the value from default_dict takes precedence.
    """
    merged = copy.deepcopy(default_dict)  # Start with a copy of default_dict

    for key, value in dict2.items():
        if key in merged and isinstance(merged[key], EasyDict) and isinstance(value, EasyDict):
            # Recursively merge nested dictionaries
            merged[key] = merge_dicts(merged[key], value)
        else:
            # Otherwise, simply update the value
            merged[key] = value

    return merged


def process_config_default(json_file, default_files=False, printing=True):
    """
    Get the json file
    Processing it with EasyDict to be accessible as attributes
    then editing the path of the experiments folder
    creating some important directories in the experiment folder
    Then setup the logging in the whole program
    Then return the config
    :param json_file: the path of the config file
    :return: config object(namespace)
    """
    config_logger = logging.getLogger("Config")
    config_logger.setLevel(logging.INFO)

    if printing:
        config_logger.info(json_file)
    config, _ = get_config_from_json(json_file)

    if default_files:
        default_config, _ = get_config_from_json(default_files)
        config = merge_dicts(default_config, config)
    config = _apply_project_paths(config)

    if printing:
        config_logger.info(" THE Configuration of your experiment ..")
        pprint(config)
        # making sure that you have provided the exp_name.
        try:
            config_logger.info(" *************************************** ")
            config_logger.info("The experiment name is {}".format(config.exp_name))
            config_logger.info(" *************************************** ")
        except AttributeError:
            config_logger.info("ERROR!!..Please provide the exp_name in json file..")
            exit(-1)

    # create some important directories to be used for that experiment.
    # config.summary_dir = os.path.join("experiments", config.exp_name, "summaries/")
    # config.checkpoint_dir = os.path.join("experiments", config.exp_name, "checkpoints/")
    # config.out_dir = os.path.join("experiments", config.exp_name, "out/")
    # config.log_dir = os.path.join("experiments", config.exp_name, "logs/")
    # create_dirs([config.summary_dir, config.checkpoint_dir, config.out_dir, config.log_dir])
    # logging.shutdown()
    # importlib.reload(logging)

    return config


def process_config(json_file, printing=True):
    """
    Get the json file
    Processing it with EasyDict to be accessible as attributes
    then editing the path of the experiments folder
    creating some important directories in the experiment folder
    Then setup the logging in the whole program
    Then return the config
    :param json_file: the path of the config file
    :return: config object(namespace)
    """
    config_logger = logging.getLogger("Config")
    config_logger.setLevel(logging.INFO)

    if printing:
        config_logger.info(json_file)
    config, _ = get_config_from_json(json_file)
    config = _apply_project_paths(config)
    if printing:
        config_logger.info(" THE Configuration of your experiment ..")
        pprint(config)
        # making sure that you have provided the exp_name.
        try:
            config_logger.info(" *************************************** ")
            config_logger.info("The experiment name is {}".format(config.exp_name))
            config_logger.info(" *************************************** ")
        except AttributeError:
            config_logger.info("ERROR!!..Please provide the exp_name in json file..")
            exit(-1)

    # create some important directories to be used for that experiment.
    # config.summary_dir = os.path.join("experiments", config.exp_name, "summaries/")
    # config.checkpoint_dir = os.path.join("experiments", config.exp_name, "checkpoints/")
    # config.out_dir = os.path.join("experiments", config.exp_name, "out/")
    # config.log_dir = os.path.join("experiments", config.exp_name, "logs/")
    # create_dirs([config.summary_dir, config.checkpoint_dir, config.out_dir, config.log_dir])
    # logging.shutdown()
    # importlib.reload(logging)

    return config
