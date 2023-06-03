from task.mano_block_assembly import ManoBlockAssembly
from task.franka_block_assembly import FrankaBlockAssembly
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *


custom_parameters = [
    {"name": "--num_envs", "type": int, "default": 256, "help": "Number of environments to create"},
    {"name": "--headless", "action": "store_true", "help": "Run headless"},
    {"name": "--goal", "type": str, "default":'1',"help": ""},
    {"name": "--save", "action": "store_true"},
]
args = gymutil.parse_arguments(
    description="Franka block assembly demonstration",
    custom_parameters=custom_parameters,
)

issac = FrankaBlockAssembly(args)
info=issac.simulate()

mano = ManoBlockAssembly(info,args)
mano.simulate()
