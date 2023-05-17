import os
import cv2
import glob
import numpy as np

camera_serials = ['side', 'in_hand']
img_root_pth = 'data/2023-05-12-15-23-43/env_00002'

rgb_pth_dict = {}
depth_pth_dict = {}
semantic_pth_dict = {}
img_array_dict = {}

for camera_id in camera_serials:
    rgb_pth_dict[camera_id] = sorted(glob.glob(os.path.join(img_root_pth, 'rgb', camera_id, '*.png')))
    depth_pth_dict[camera_id] = sorted(glob.glob(os.path.join(img_root_pth, 'depth', camera_id, '*.npy')))
    semantic_pth_dict[camera_id] = sorted(glob.glob(os.path.join(img_root_pth, 'semantic', camera_id, '*.npy')))

for camera_id in camera_serials:
    img_array = []

    for rgb_pth, depth_pth, semantic_pth in zip(rgb_pth_dict[camera_id], depth_pth_dict[camera_id], semantic_pth_dict[camera_id]):
        rgb = cv2.imread(rgb_pth)

        depth = np.load(depth_pth) / (-1)
        depth[depth > 10] = 0
        depth = depth / np.max(depth) * 255
        depth = depth.astype(np.uint8)
        depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)

        semantic = np.load(semantic_pth)
        semantic = semantic / np.max(semantic) * 255
        semantic = semantic.astype(np.uint8)
        semantic = cv2.cvtColor(semantic, cv2.COLOR_GRAY2BGR)
        
        img_array.append(np.hstack([rgb, depth, semantic]))

    img_array_dict[camera_id] = img_array

    out = cv2.VideoWriter(f'./{camera_id}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 6, (640 * 3, 480))
    
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