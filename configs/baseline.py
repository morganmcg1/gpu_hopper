import random
import ml_collections


def get_wandb_config() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.entity = "morgan"
    configs.project = "gpu-hopper"
    
    return configs


def get_model_config() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.layer_1 = 512
    configs.activation_1 = "relu"
    configs.dropout = random.uniform(0.01, 0.80)
    configs.layer_2 = 10
    configs.activation_2 = "softmax"
    
    return configs
           

def get_train_config() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.optimizer = "sgd"
    configs.loss_func = "sparse_categorical_crossentropy"
    configs.metric = "accuracy"
    configs.epoch = 6
    configs.batch_size = 256

    return configs


def get_config() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.wandb = get_wandb_config()
    configs.model = get_model_config()
    configs.train = get_train_config()
    
    return configs