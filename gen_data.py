from tasks.franka_block_assembly import FrankaBlockAssembly
from tasks.mano_block_assembly import ManoBlockAssembly
from tasks.franka_peg_insertion import FrankaPegInsertion

from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *


custom_parameters = [
    {"name": "--num_envs", "type": int, "default": 256, "help": "Number of environments to create"},
    {"name": "--task", "type": str, "default": "peg_insertion", "help": "task"},
    {"name": "--headless", "action": "store_true", "help": "Run headless"},
    {"name": "--goal", "type": str, "default":'1',"help": ""},
    {"name": "--save", "action": "store_true"},
]
args = gymutil.parse_arguments(
    description="Franka block assembly demonstration",
    custom_parameters=custom_parameters,
)

if args.task == 'block_assembly':
    franka = FrankaBlockAssembly(args)
    info = franka.simulate()
    mano = ManoBlockAssembly(info,args)

if args.task == 'peg_insertion':
    franka = FrankaPegInsertion(args)
    info = franka.simulate()