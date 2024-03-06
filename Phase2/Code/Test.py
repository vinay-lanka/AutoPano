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
from Network.Unsup_Network import *
import shutil
import string
import math as m
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import torch
from copy import deepcopy

# if torch.cuda.is_available():
#     device = torch.device("cuda")
# else:
#     device = torch.device("cpu")
# print(f"Running on device: {device}")

# Don't generate pyc codes
sys.dont_write_bytecode = True

# def GenerateBatch(BasePath, State, Model):
#     PaPbBatch = []
#     GroundTruthBatch = []
#     if Model == "UnSup":
#         PaCornersBatch = []
#         PaBatch = []
#         PbBatch = []
#     ImageNum = 0
#     patches_a_path = os.path.join(BasePath, f"Synthetic{State}/patch_A")
#     patches_b_path = os.path.join(BasePath, f"Synthetic{State}/patch_B")
#     image_path = os.path.join(BasePath, f"Synthetic{State}/Grayscale")
#     GTs_path = os.path.join(BasePath, f"Synthetic{State}/H4Pt/")
#     if Model == "UnSup":
#         Pa_base_corners_path = os.path.join(BasePath, f"Synthetic{State}/patch_A_corners/")
#     if os.path.exists(patches_a_path): 
#         image_list = os.listdir(patches_a_path)
#     else:
#         raise Exception ("Patch A Directory doesn't exist")
    
#     RandIdx = random.randint(1, len(image_list)-1)

#     patch_a_path = os.path.join(patches_a_path,image_list[RandIdx])
#     patch_b_path = os.path.join(patches_b_path,image_list[RandIdx])
#     GT_path = GTs_path + image_list[RandIdx][:-4] + f".txt"
#     if Model == "UnSup":
#         Pa_corners_path = Pa_base_corners_path + image_list[RandIdx][:-4] + f".txt"

#     patch_A = cv2.imread(patch_a_path, cv2.IMREAD_GRAYSCALE)
#     patch_B = cv2.imread(patch_b_path, cv2.IMREAD_GRAYSCALE)

#     patch_A = np.expand_dims(patch_A, 2)
#     patch_B = np.expand_dims(patch_B, 2)
#     Pa_Pb = np.concatenate((patch_A, patch_B), axis = 2)
#     GT = np.genfromtxt(GT_path, delimiter=',')
#     PaPbBatch.append(torch.from_numpy(Pa_Pb))
#     GroundTruthBatch.append(torch.from_numpy(GT).to(device))

#     if Model == "UnSup":
#         Pa_corners = np.genfromtxt(Pa_corners_path, delimiter=',').flatten(order = 'F')
#         PaCornersBatch.append(torch.from_numpy(Pa_corners).to(device))
#         patch_A = np.squeeze(patch_A)
#         patch_A = np.expand_dims(patch_A, axis=0)
#         patch_B = np.squeeze(patch_B)
#         patch_B = np.expand_dims(patch_B, axis=0)
#         PaBatch.append(torch.from_numpy(patch_A))
#         PbBatch.append(torch.from_numpy(patch_B))


#     PaPbBatch = torch.stack(PaPbBatch)
#     PaPbBatch = PaPbBatch.to(device,dtype=torch.float32)

#     img = cv2.imread()

#     if Model == "UnSup":
#         PaBatch = torch.stack(PaBatch)
#         PaBatch = PaBatch.to(device,dtype=torch.float32)
#         PbBatch = torch.stack(PbBatch)
#         PbBatch = PbBatch.to(device,dtype=torch.float32)
#         return PaPbBatch, torch.stack(PaCornersBatch), PaBatch, PbBatch
#     return PaPbBatch, torch.stack(GroundTruthBatch), torch.stack(PaCornersBatch)

def generate_test_data(index):
    img = cv2.imread('Phase2/Data/Test/' + str(index) + '.jpg')
    I = deepcopy(img)
    I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    I = cv2.resize(I, (320, 240))
    x,y = I.shape
    width = 128
    height = 128
    perturbation = 16
    patch_x = random.randint(perturbation, x-width-perturbation)
    patch_y = random.randint(perturbation, y-height-perturbation)
    patch_A = I[patch_x : patch_x + width, patch_y : patch_y + height]
    patch_a_corners = np.array([[patch_y, patch_x], 
                                [patch_y, patch_x + width], 
                                [patch_y + height, patch_x + width], 
                                [patch_y + height, patch_x]], dtype=np.float32)

    p1 = np.array([[0, 0], [0, x], [y, x], [y, 0]]).reshape(-1,1,2)

    pert_x = np.array(random.sample(range(-perturbation, perturbation), 4)).reshape(4,1)
    pert_y = np.array(random.sample(range(-perturbation, perturbation), 4)).reshape(4,1)
    pert_mat = np.hstack([pert_x, pert_y])

    warped_corners = patch_a_corners + pert_mat

    H4Pt = np.float32(warped_corners) - patch_a_corners 
    H4Pt = H4Pt.flatten(order = 'F')

    H_AB = cv2.getPerspectiveTransform(np.float32(patch_a_corners), np.float32(warped_corners))
    H_BA = np.linalg.inv(H_AB)
    p2  = cv2.perspectiveTransform(np.float32(p1), H_BA)
    [xmin, ymin] = np.int32(p2.min(axis=0).ravel())
    [xmax, ymax] = np.int32(p2.max(axis=0).ravel())
    i = [-xmin,-ymin]
    T = np.array([[1,0,i[0]],[0,1,i[1]],[0,0,1]])
    warp_img = cv2.warpPerspective(I, T.dot(H_BA), (xmax-xmin, ymax-ymin),flags = cv2.INTER_LINEAR)
    
    warped_patch = warp_img[patch_x + i[1] : patch_x + width + i[1], patch_y + i[0] : patch_y + height + i[0]]
    
    #Norm
    patch_A = np.float32(patch_A)
    patch_A=(patch_A-np.mean(patch_A))/255
    patch_B = np.float32(warped_patch)
    patch_B=(patch_B-np.mean(patch_B))/255
    
    return patch_A, patch_B, patch_a_corners, warped_corners, img, i, H4Pt

def input(img1, img2):
    img1 = np.expand_dims(img1, 2)
    img2 = np.expand_dims(img2, 2)
    I = torch.stack([torch.from_numpy(np.concatenate((img1, img2), axis = 2))]).to(dtype=torch.float32)
    return I

def SupTest_operation(Img, ModelPath):
    model = SupHomographyModel()
    CheckPoint = torch.load(ModelPath)
    model.load_state_dict(CheckPoint['model_state_dict'])
    # model.to(device)
    model.eval()
    H4PT = model.model(Img)
    return H4PT

def UnSupTest_operation(PatchA, PaPbB, Corners, ModelPath):
    model = UnSupHomographyModel()
    CheckPoint = torch.load(ModelPath)
    model.load_state_dict(CheckPoint['model_state_dict'])
    # model.to(device)
    model.eval()
    Pred = model.model(PatchA, PaPbB, Corners)
    return Pred

def predicted_patch(H4PT_pred, patch_a_corners):
    # pred_warped_corners = patch_a_corners - H4PT_pred
    pred_warped_corners = patch_a_corners + H4PT_pred
    return pred_warped_corners

def corners(img, warped_corners, pred_warped_corners, t):
    warped_corners = warped_corners.astype('int32')
    plt.imshow(img)
    plt.show()
    # pred_warped_corners = np.roll(pred_warped_corners, 2)
    cv2.polylines(img, [warped_corners + t], isClosed = True, color = (255,0,0), thickness = 2)
    cv2.polylines(img, [pred_warped_corners + t], isClosed = True, color = (0,255,0), thickness = 2)
    plt.imshow(img)
    plt.show()
    # cv2.imshow('img', img)
    # cv2.waitKey()

def main():

    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        "--Epochs",
        dest="Epochs",
        default=25,
        help="Number of epochs run"
    )
    Args = Parser.parse_args()
    Epochs = Args.Epochs

    SupModelPath = "./Phase2/Checkpoints/Sup" + str(Epochs-1) + "model.ckpt"
    UnSupModelPath = "./Phase2/Checkpoints/UnSup" + str(14) + "model.ckpt"

    #Supervised Testing
    BasePath = './Phase2/Data/'
    # PaPb, tGT, Ca = GenerateBatch(BasePath, "Test", "Sup")
    patch_A, warped_patch, patch_a_corners, warped_corners, warped_img, t, H4pt = generate_test_data(1)
    plt.imshow(warped_patch, cmap='gray')
    plt.show()
    input_img = input(patch_A, warped_patch)
    SupH4PT_pred = SupTest_operation(input_img, SupModelPath)

    patcha_corners = torch.stack([torch.from_numpy(np.resize(patch_a_corners,(1,8))).to(torch.double)])
    print(patcha_corners)
    patch_A = torch.stack([torch.from_numpy(np.expand_dims(patch_A, axis=0)).to(torch.float)])
    UnSup_pred = UnSupTest_operation(patch_A, input_img, patcha_corners, UnSupModelPath)
    print(UnSup_pred.shape)
    UnSup_pred = UnSup_pred.detach().numpy()
    plt.imshow(UnSup_pred[0][0], cmap='gray')
    plt.show()
    # print(SupH4PT_pred)
    SupH4PT_pred = SupH4PT_pred.detach().numpy()
    # # H4PT_pred = H4Pt
    SupH4PT_pred = np.reshape(SupH4PT_pred, (4,2), order = 'F')
    pred_warped_corners = predicted_patch(SupH4PT_pred, patch_a_corners)
    pred_warped_corners = pred_warped_corners.astype(int)
    corners(warped_img, warped_corners, pred_warped_corners, t)


if __name__ == "__main__":
    main()