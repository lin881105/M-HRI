import os
import cv2
import glob
import numpy as np
import imageio

img_root_pth = 'data/peg_insertion/goal_5/2023-06-29-04-05-29/env_00001'

rgb_pth_dict = {}

rgb_pth_dict = sorted(glob.glob(os.path.join(img_root_pth, 'rgb','side', '*.png')))



hand_img_array = []
robot_img_array = []

for rgb_pth in rgb_pth_dict:
    rgb = cv2.imread(rgb_pth)
    rgb = cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
    robot_img_array.append(rgb)
    
imageio.mimsave(os.path.join(img_root_pth,'robot.gif'),robot_img_array,fps=10)




# out = cv2.VideoWriter(f'./demo_video/rgb_depth/all.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, (640, 480 * 2))
# num_frame = 148

# for frame in range(num_frame):
#     out_frame = np.zeros((480 * 4, 640 * 2, 3), dtype=np.uint8)

#     for cnt, camera_id in enumerate(camera_serials[:4]):
#         if cnt == 0:
#             img_array_dict[camera_id][frame] = cv2.rotate(img_array_dict[camera_id][frame], cv2.ROTATE_180)
            
#         out_frame[cnt*480:(cnt+1)*480, :640, :] = img_array_dict[camera_id][frame]

#     for cnt, camera_id in enumerate(camera_serials[4:]):
#         if cnt == 0:
#             img_array_dict[camera_id][frame] = cv2.rotate(img_array_dict[camera_id][frame], cv2.ROTATE_180)

#         out_frame[cnt*480:(cnt+1)*480, 640:, :] = img_array_dict[camera_id][frame]

#     out_frame = cv2.resize(out_frame, (640, 480 * 2), interpolation=cv2.INTER_AREA)

#     out.write(out_frame)

# out.release()