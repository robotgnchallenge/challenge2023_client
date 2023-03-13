import os
import sys
import argparse
import numpy as np
import time
import glob
import cv2
import math
import rospy
import tf

import tensorflow.compat.v1 as tensorflow

tensorflow.disable_eager_execution()
physical_devices = tensorflow.config.experimental.list_physical_devices('GPU')
tensorflow.config.experimental.set_memory_growth(physical_devices[0], True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR))
from config_utils import load_config
from data import regularize_pc_point_count, depth2pc, load_available_input_data, subscribe_pc_ros

from contact_grasp_estimator import GraspEstimator
from visualization_utils import visualize_grasps, show_image

from transforms3d.quaternions import mat2quat, quat2mat


def euler_from_quaternion(x, y, z, w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z  # in radians


def pose_4x4_to_pos_quat(pose):
    """
	Convert pose, 4x4 format into pos and quat

	Args:
	    pose: numpy array, 4x4
	Returns:
		pos: length-3 position
	    quat: length-4 quaternion

	"""
    mat = pose[:3, :3]
    quat = mat2quat(mat)
    pos = np.zeros([3])
    pos[0] = pose[0, 3]
    pos[1] = pose[1, 3]
    pos[2] = pose[2, 3]
    return pos, quat


def grasp_nms(trans, quat, score, trans_thresh, rot_thresh):
    select_gsp = []
    idx = score.argsort()[::-1]

    while (idx.size > 0):
        cur = idx[0]
        select_gsp.append(cur)

        trans_diff = np.zeros((idx.shape[0] - 1))
        rot_diff = np.zeros((idx.shape[0] - 1, 3))
        for i in range(1, idx.shape[0]):
            trans_diff[i - 1] = np.linalg.norm(trans[cur] - trans[idx[i]])
            rot_diff[i -
                     1] = 2 * np.arccos(abs(np.dot(quat[cur], quat[idx[i]])))

        trans_update = np.where(trans_diff >= trans_thresh)
        roll_update = np.where(rot_diff[:, 0] >= rot_thresh)
        pitch_update = np.where(rot_diff[:, 1] >= rot_thresh)
        yaw_update = np.where(rot_diff[:, 1] >= rot_thresh)

        filt_idx = np.intersect1d(
            trans_update,
            np.union1d(np.union1d(roll_update, pitch_update), yaw_update))

        idx = idx[filt_idx + 1]

    return select_gsp


def inference(cam_data,
              global_config,
              checkpoint_dir,
              input_paths,
              K=None,
              local_regions=True,
              skip_border_objects=False,
              filter_grasps=True,
              segmap_id=None,
              z_range=[0.2, 1.8],
              forward_passes=1,
              tf_pub=False):
    # rgb = cam_data['rgb']
    pc_full = cam_data['pc_full']
    # pc_colors = cam_data['pc_colors']

    #Initial tf ros
    if tf_pub:
        rospy.init_node('grasp_pose_publisher', anonymous=True)
        rate = rospy.Rate(1.0)
        br = tf.TransformBroadcaster()
        listener = tf.TransformListener()

    grasp_estimator = GraspEstimator(global_config)
    grasp_estimator.build_network()

    # Add ops to save and restore all the variables.
    saver = tensorflow.train.Saver(save_relative_paths=True)

    # Create a session
    config = tensorflow.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.6
    sess = tensorflow.Session(config=config)

    # Load weights
    grasp_estimator.load_weights(sess, saver, checkpoint_dir, mode='test')

    os.makedirs('results', exist_ok=True)
    pc_segments = {}

    print('Generating Grasps...')
    pred_grasps_cam, scores, contact_pts, _ = grasp_estimator.predict_scene_grasps(
        sess,
        pc_full,
        pc_segments=pc_segments,
        local_regions=local_regions,
        filter_grasps=filter_grasps,
        forward_passes=forward_passes)

    grasp_num = pred_grasps_cam[-1].shape[0]
    gsp_trans = np.zeros((grasp_num, 3))
    gsp_quat = np.zeros((grasp_num, 4))
    gsp_euler = np.zeros((grasp_num, 3))

    for i in range(grasp_num):
        gsp_trans[i], gsp_quat[i] = pose_4x4_to_pos_quat(
            pred_grasps_cam[-1][i])
        gsp_euler[i] = euler_from_quaternion(gsp_quat[i][0], gsp_quat[i][1],
                                             gsp_quat[i][2], gsp_quat[i][3])

    return pred_grasps_cam


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ckpt_dir',
        default='../../checkpoints/scene_test_2048_bs3_hor_sigma_001',
        help='Log dir [default: checkpoints/scene_test_2048_bs3_hor_sigma_001]'
    )
    parser.add_argument(
        '--np_path',
        default='test_data/7.npy',
        help=
        'Input data: npz/npy file with keys either "depth" & camera matrix "K" or just point cloud "pc" in meters. Optionally, a 2D "segmap"'
    )
    parser.add_argument('--png_path',
                        default='',
                        help='Input data: depth map png in meters')
    parser.add_argument(
        '--K',
        default=None,
        help='Flat Camera Matrix, pass as "[fx, 0, cx, 0, fy, cy, 0, 0 ,1]"')
    parser.add_argument('--z_range',
                        default=[0.2, 0.6],
                        help='Z value threshold to crop the input point cloud')
    parser.add_argument('--local_regions',
                        action='store_true',
                        default=False,
                        help='Crop 3D local regions around given segments.')
    parser.add_argument('--filter_grasps',
                        action='store_true',
                        default=False,
                        help='Filter grasp contacts according to segmap.')
    parser.add_argument(
        '--skip_border_objects',
        action='store_true',
        default=False,
        help=
        'When extracting local_regions, ignore segments at depth map boundary.'
    )
    parser.add_argument(
        '--forward_passes',
        type=int,
        default=1,
        help=
        'Run multiple parallel forward passes to mesh_utils more potential contact points.'
    )
    parser.add_argument('--segmap_id',
                        type=int,
                        default=0,
                        help='Only return grasps of the given object id')
    parser.add_argument('--arg_configs',
                        nargs="*",
                        type=str,
                        default=[],
                        help='overwrite config parameters')
    parser.add_argument('--tf_pub', default=False, help='grasp pose publisher')
    parser.add_argument('--mode', default='ros', help='grasp pose publisher')
    FLAGS = parser.parse_args()

    global_config = load_config(FLAGS.ckpt_dir,
                                batch_size=FLAGS.forward_passes,
                                arg_configs=FLAGS.arg_configs)

    print(str(global_config))
    print('pid: %s' % (str(os.getpid())))

    inference(global_config,
              FLAGS.ckpt_dir,
              FLAGS.np_path if not FLAGS.png_path else FLAGS.png_path,
              z_range=eval(str(FLAGS.z_range)),
              K=FLAGS.K,
              local_regions=FLAGS.local_regions,
              filter_grasps=FLAGS.filter_grasps,
              segmap_id=FLAGS.segmap_id,
              forward_passes=FLAGS.forward_passes,
              skip_border_objects=FLAGS.skip_border_objects)
