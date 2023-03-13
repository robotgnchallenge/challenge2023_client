# -- coding: utf-8 --
import argparse


def get_args():
    parser = argparse.ArgumentParser(
        description='Goal-Oriented-Semantic-Exploration')

    # Environment, dataset and episode specifications
    parser.add_argument('-efw', '--env_frame_width', type=int, default=640,
                        help='Frame width (default:640)')
    parser.add_argument('-efh', '--env_frame_height', type=int, default=480,
                        help='Frame height (default:480)')

    parser.add_argument('--map_resolution', type=int, default=5)

    # parse arguments
    args, unknow = parser.parse_known_args()

    return args
