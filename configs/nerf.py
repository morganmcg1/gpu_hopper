import random
import ml_collections


def get_hopper_config() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.hopper_services = ["kaggle"]
    
    return configs


def get_wandb_config() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.entity = "morgan"
    configs.project = "nerf-gpu-hopper"
    
    return configs


def get_model_config() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.POS_ENCODE_DIMS = 16
    
    return configs
           

def get_train_config() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.BATCH_SIZE = 5
    configs.NUM_SAMPLES = 32
    configs.EPOCHS = 30
    configs.SAVE_FREQ = 5
    configs.MODEL_NAME = "my_nerf"

    return configs


def get_config() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.wandb = get_wandb_config()
    configs.model = get_model_config()
    configs.train = get_train_config()
    configs.hopper = get_hopper_config()
    
    return configs