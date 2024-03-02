#!/usr/bin/env python3
"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Mayank Deshpande (msdeshp4@umd.edu)
M.Eng. in Robotics,
University of Maryland, College Park

Vikram Setty (msdeshp4@umd.edu)
M.Eng. in Robotics,
University of Maryland, College Park

Vinay Lanka (msdeshp4@umd.edu)
M.Eng. in Robotics,
University of Maryland, College Park
"""


# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)

import cv2
import os
import sys
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import numpy as np
import time
import torchvision.transforms as tf
import argparse
from Network.Network import *
import shutil
import string
import math as m
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import torch



# Don't generate pyc codes
sys.dont_write_bytecode = True


def generate_test_data():

    i = 4
    img = cv2.imread('Phase2/Data/Train/' + str(i) + '.jpg',0)
    I = cv2.resize(img, (320, 240))
    x,y = img.shape
    width = 128
    height = 128
    perturbation = 16
    patch_x = random.randint(perturbation, x-width-perturbation)
    patch_y = random.randint(perturbation, y-height-perturbation)
    patch_A = I[patch_x : patch_x + width, patch_y : patch_y + height]
    patch_a_corners = np.array([[patch_y, patch_x], [patch_y, patch_x + width], [patch_y + height, patch_x + width], [patch_y + height, patch_x]], dtype=np.float32)
    # plt.imshow(patch_A)
    # plt.show()
    
    p1 = np.array([[0, 0], [0, x], [y, x], [y, 0]]).reshape(-1,1,2)

    pert_x = np.array(random.sample(range(-perturbation, perturbation), 4)).reshape(4,1)
    pert_y = np.array(random.sample(range(-perturbation, perturbation), 4)).reshape(4,1)
    pert_mat = np.hstack([pert_x, pert_y])

    warped_corners = patch_a_corners + pert_mat

    H_AB = cv2.getPerspectiveTransform(np.float32(patch_a_corners), np.float32(warped_corners))
    H_BA = np.linalg.inv(H_AB)
    p2  = cv2.perspectiveTransform(np.float32(p1), H_BA)
    [xmin, ymin] = np.int32(p2.min(axis=0).ravel())
    [xmax, ymax] = np.int32(p2.max(axis=0).ravel())
    i = [-xmin,-ymin]
    T = np.array([[1,0,i[0]],[0,1,i[1]],[0,0,1]])
    warp_img = cv2.warpPerspective(I, T.dot(H_BA), (xmax-xmin, ymax-ymin),flags = cv2.INTER_LINEAR)
    warped_patch = warp_img[patch_x + i[1] : patch_x + width + i[1], patch_y + i[0] : patch_y + height + i[0]]
    # plt.imshow(warp_img)
    # plt.show()
    plt.imshow(warped_patch)
    plt.show()
    return patch_A, warped_patch, patch_a_corners, warped_corners, warp_img, i

def input(img1, img2):
    img1 = np.expand_dims(img1, 2)
    img2 = np.expand_dims(img2, 2)
    transformImg=tf.Compose([tf.ToTensor()])
    I = np.concatenate((img1, img2), axis = 2)
    I = np.transpose(I, (0, 2, 1))
    I = transformImg(I)
    I = I.unsqueeze(0)
    return I

def Test_operation(Img, ModelPath):
    model = HomographyModel()
    CheckPoint = torch.load(ModelPath)
    model.load_state_dict(CheckPoint['model_state_dict'])
    model.eval()
    H4PT = model(Img)
    H4PT = H4PT.squeeze(0)
    return H4PT

def predicted_patch(H4PT_pred, patch_a_corners):
    pred_warped_corners = patch_a_corners - H4PT_pred
    return pred_warped_corners

def corners(img, warped_corners, pred_warped_corners, t):
    cv2.polylines(img, [warped_corners + t], isClosed = True, color = (255,0,0), thickness = 2)
    cv2.polylines(img, [pred_warped_corners + t], isClosed = True, color = (0,255,0), thickness = 2)
    cv2.imshow('img', img)
    cv2.waitKey()

def main():

    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        "--ModelPath",
        dest="ModelPath",
        default="Phase2/Checkpoints/2a4900model.ckpt",
        help="Path to load latest model from, Default:ModelPath",
    )
    Args = Parser.parse_args()
    ModelPath = Args.ModelPath

    patch_A, warped_patch, patch_a_corners, warped_corners, warped_img, t = generate_test_data()
    input_img = input(patch_A, warped_patch)
    H4PT_pred = Test_operation(input_img, ModelPath)

    H4PT_pred = H4PT_pred.detach().numpy()
    H4PT_pred = np.reshape(H4PT_pred, (4,2), order = 'F')
    pred_warped_corners = predicted_patch(H4PT_pred, patch_a_corners)
    pred_warped_corners = pred_warped_corners.astype(int)
    corners(warped_img, warped_corners, pred_warped_corners, t)


if __name__ == "__main__":
    main()