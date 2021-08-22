# The edge_smooth.py is from taki0112/CartoonGAN-Tensorflow https://github.com/taki0112/CartoonGAN-Tensorflow#2-do-edge_smooth
import numpy as np
import cv2, os, argparse
from glob import glob
from tqdm import tqdm
from shutil import copyfile


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def copy_img(source_dir_list, target_dir_list):
    # check_folder(os.path.join(target_dir))

    for target_dir in target_dir_list:
        check_folder(os.path.join(target_dir, 'smooth_HD'))
        check_folder(os.path.join(target_dir, 'style_HD'))
        for dirPath, dirNames, fileNames in os.walk(target_dir):
            for f in fileNames:
                file_path = os.path.join(dirPath, f)
                # whether file exist or not
                for source_dir in source_dir_list:
                    origin_path = file_path.replace(target_dir, source_dir)
                    if 'smooth/' in origin_path or 'style/' in origin_path:
                        origin_path1 = origin_path.replace('smooth/', 'smooth_HD/')
                        target_file_path1 = file_path.replace('smooth/', 'smooth_HD/')
                        origin_path2 = origin_path.replace('style/', 'style_HD/')
                        target_file_path2 = file_path.replace('style/', 'style_HD/')

                    image1 = cv2.imread(origin_path1)
                    image2 = cv2.imread(origin_path2)
                    if os.path.isfile(origin_path) and image1 is not None and image2 is not None:
                        # print(file_path)
                        # print(origin_path)
                        # print(target_file_path)
                        copyfile(origin_path1, target_file_path1)
                        copyfile(origin_path2, target_file_path2)


"""main"""
def main():
    source_dir_list = ['../style_dataset/CG', '../style_dataset/safebooru']
    target_dir_list = ['../style_dataset/cluster_labels/0', '../style_dataset/cluster_labels/1',
                       '../style_dataset/cluster_labels/2', '../style_dataset/cluster_labels/3']
    copy_img(source_dir_list, target_dir_list)


if __name__ == '__main__':
    main()
