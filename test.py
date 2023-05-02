import pytorch3d.transforms
import torch

device = 'cuda:0'

goal_rot_1 = torch.tensor((-2.5*torch.pi, 0, 0), dtype=torch.float32).to(device=device)
goal_rot_1 = pytorch3d.transforms.euler_angles_to_matrix(goal_rot_1,"XYZ").to(device=device)

goal_rot_2 = torch.tensor((-0.5*torch.pi, 0, 0), dtype=torch.float32).to(device=device)
goal_rot_2 = pytorch3d.transforms.euler_angles_to_matrix(goal_rot_2,"XYZ").to(device=device)

goal_rot_1 = torch.round(torch.tensor(goal_rot_1), decimals=3)
goal_rot_2 = torch.round(torch.tensor(goal_rot_2), decimals=3)

print(goal_rot_1)
print(goal_rot_2)

print(goal_rot_1 - goal_rot_2)