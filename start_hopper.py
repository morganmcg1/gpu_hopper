import time
from datetime import datetime
import wandb
from fastcore.script import *
from configs.utils import import_config

@call_parse
def main(
        wandb_run_id:Param("wandb run id from", str)=None,
        config:Param("config file to use", str)="configs/baseline.py"
        ):

    configs = import_config(config_path=config)
    entity = configs.wandb.entity
    project = configs.wandb.project
    services = configs.hopper.hopper_services

    api = wandb.Api()
    run = api.from_path(f"{entity}/{project}/runs/{wandb_run_id}")
    
    print("\nStarting Hopper watch...\n")

    while True:
        s_count = 0
        api = wandb.Api()
        run = api.from_path(f"{entity}/{project}/runs/{wandb_run_id}")
        if run.state == "running":
            dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print(f"{dt_string} : Run {wandb_run_id} is still running")
            time.sleep(60)
        elif run.state in ["failed", "killed", "crashed"] and s_count < len(services):
            dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print(f"\n{dt_string} : Detected run {wandb_run_id} has failed, crashed or been killed")
            print(f'{dt_string} : Launching {services[s_count]}...\n')
            s_count += 1
            break
        else:
            dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print(f"\n{dt_string} : No more services to launch, exiting Hopper...\n")
            break

    
