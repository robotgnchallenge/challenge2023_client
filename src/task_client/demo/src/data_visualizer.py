from sys import argv

import numpy as np
import cv2

from constant import obj_idx_colour, obj_name


def visualize(file_name, seg_map, bbox=None, bbox_idx=None):
    """_summary_

    Args:
        seg_map (_type_): segmentation map
        bbox (_type_, optional): 2D list, bbox in format [left, top, right, bottom]
    """
    seg_img = np.zeros((seg_map.shape[0], seg_map.shape[1], 3))
    for i in range(seg_map.shape[0]):
        for j in range(seg_map.shape[1]):
            color = obj_idx_colour[seg_map[i][j]]
            seg_img[i][j] = [color[2], color[1], color[0]]

    cv2.imwrite(file_name.replace('.npy', '.png'), seg_img)

    if (bbox):
        img = cv2.imread(file_name.replace('.npy', '.png'))
        for i in range(len(bbox)):
            cv2.rectangle(img, (bbox[i][1], bbox[i][0]),
                          (bbox[i][3], bbox[i][2]), (0, 255, 0), 1)
            cv2.putText(img, obj_name[bbox_idx[i]], (bbox[i][1], bbox[i][0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0))
        cv2.imwrite(file_name.replace('.npy', '.png'), img)


if __name__ == '__main__':
    file_name = argv[1]
    data = np.load(file_name, allow_pickle=True).item()
    visualize(file_name, data['seg_hand']['data'], data['bbox'],
              data['bbox_idx'])
