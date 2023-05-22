from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import os
import math
import numpy as np
import torch
import random
import time
import scipy.io as sio
from utils import utils
from scipy.spatial.transform import Rotation as R
import pytorch3d.transforms

device = 'cuda:0'
# n = torch.zeros((10)).to(device)
d = torch.randint(0,10,(10,),dtype=torch.int).to(device)

# print(n)
# print(d)
tmp = torch.zeros((10),dtype=torch.long,device=device)
# print(torch.where(tmp==True)[0].shape[0])
_tmp = torch.where(tmp==1)
print(_tmp[0].shape[0])
