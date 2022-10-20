
from importlib import import_module

# dynamically import the config file
def import_config(config_path):
    """
    Import the config file dynamically
    """
    module_name = config_path.replace("/", ".").replace(".py", "")
    config_module = import_module(module_name)
    get_config = getattr(config_module, "get_config")
    return get_config()