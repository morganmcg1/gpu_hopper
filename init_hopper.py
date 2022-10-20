import os
import wandb
from configs.utils import import_config
from fastcore.script import *

os.environ["WANDB_SILENT"] = "True"

@call_parse
def init(
        config_path:str="configs/baseline.py"
        ):

    "The main training function"

    config = import_config(config_path=config_path)
    wandb_run_id = wandb.util.generate_id()

    # initialize wandb
    wandb.init(
        id=wandb_run_id,
        resume="allow",
        entity=config.wandb.entity, 
        project=config.wandb.project, 
        config={"config_path": config_path}
        )

    # Log all configs except wandb configs
    for config_key in config.keys():
        if config_key != "wandb":
            wandb.config.update(config[config_key].to_dict())

    print(f"\n*** Initialing Hopper, your wandb run id is: {wandb_run_id} ***")
    print(f"\n*** You can view your wandb run at: {wandb.run.get_url()} ***\n")

    wandb.config['hopper_initialised'] = True
    wandb.finish()

    return 