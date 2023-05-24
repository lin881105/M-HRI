import os
import cv2
import glob
import numpy as np

img_root_pth = 'data/goal_1/2023-05-23-11-52-24/env_00004'

rgb_pth_dict = {}
hand_rgb_pth_dict = {}


rgb_pth_dict = sorted(glob.glob(os.path.join(img_root_pth, 'rgb','side', '*.png')))
hand_rgb_pth_dict = sorted(glob.glob(os.path.join(img_root_pth, 'hand_rgb', '*.png')))



img_array = []


diff = len(hand_rgb_pth_dict) - len(rgb_pth_dict)

skip = len(hand_rgb_pth_dict)//diff


print(len(hand_rgb_pth_dict))
print(len(rgb_pth_dict))

print(skip)
# for frame in range(len(hand_rgb_pth_dict)):
#     if frame%skip == 0:
#         delete.append(frame)
# del hand_rgb_pth_dict[delete]

hand_rgb_pth_dict = [frame for i,frame in enumerate(hand_rgb_pth_dict) if i%skip!=0]

print(len(hand_rgb_pth_dict))
print(len(rgb_pth_dict))

for rgb_pth, hand_rgb_pth in zip(rgb_pth_dict, hand_rgb_pth_dict):
    rgb = cv2.imread(rgb_pth)
    hand_rgb = cv2.imread(hand_rgb_pth)

    
    img_array.append(np.hstack([rgb, hand_rgb]))

# img_array_dict[camera_id] = img_array

out = cv2.VideoWriter(f'./out.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 6, (640 * 2, 480))

for img in img_array:
    out.write(img)

out.release()


'''
stack 8-view rgb video together
'''
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