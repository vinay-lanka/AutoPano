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

import sys

import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import kornia as k

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Running on device: {device}")


def UnSupLossFn(batch_b_pred, patch_b):
    criterion = nn.L1Loss()
    loss = criterion(batch_b_pred, patch_b)

    return loss

class UnSupNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(16 * 16 * 128, 1024)
        self.fc2 = nn.Linear(1024, 8)

    def forward(self, x):

        mini_batch_size = x.shape[0]
        dim_x = x.shape[1]
        dim_y = x.shape[2]
        depth = x.shape[3]

        x = x.view(torch.Size([mini_batch_size, depth, dim_x, dim_y]))

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool(x)
        x = self.conv4(x)
        x = self.conv4(x)
        x = self.dropout(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.fc2(x)
        return x

class TensorDLT(nn.Module):
    def init(self) -> None:
        '''
        H_4pt_X is H_4pt for batch X
        H4pt is the predicted H_4pt by the homography net, use it to calculate the corners in the predicted/ warped image
        C_a are the corner points of the patch in Image A or in this case the training image
        '''
        super().init()

    def tensorDLT(self, H_4pt_X, patch_a_corners):
        H = torch.tensor([])
        for H4pt in H_4pt_X:
            Cb = patch_a_corners + H4pt
            A = []
            b = []
            for i in range(0,8,2): #since there are 4 corner pairs
                Ai = [[0, 0, 0, -patch_a_corners[i], -patch_a_corners[i+1], -1, Cb[i+1]*patch_a_corners[i], Cb[i+1]*patch_a_corners[i+1]]]
                Ai.append([patch_a_corners[i], patch_a_corners[i+1], 1, 0, 0, 0, -Cb[i]*patch_a_corners[i], -Cb[i]*patch_a_corners[i+1]])
                A.extend(Ai)

                bi = [-Cb[i+1],-Cb[i]]
                b.extend(bi)

            A = torch.tensor(A).to(device)
            b = torch.tensor(b).to(device)
            print(A)
            h = torch.dot(torch.inverse(A), b)
            H = torch.cat(H,h.reshape(1,-1), axis=0)
        H = H[1:,:]
        print(H.shape)
        return H

    def forward(self,H_4pt_X, patch_a_corners):
        return self.tensorDLT(H_4pt_X, patch_a_corners)

def unsupervised_HomographyNet(patch_set, orig_img, patch_a_corners): # orig_img is original resized image, patch set is set of both image patches that we get after I = np.concatenate((patch_A, patch_B), axis = 2)
    h4pt_pred = UnSupNet(patch_set)

    print("h4pt_pred", h4pt_pred)
    corners_b_pred = (torch.sub(patch_a_corners,h4pt_pred))

    patch_a_corners = patch_a_corners.reshape(-1,4,2)
    corners_b_pred = corners_b_pred.reshape(-1,4,2)
    print("batch_corners_b_pred", corners_b_pred)
    print("batch_corners", patch_a_corners)

    # h_pred = (k.geometry.homography.find_homography_dlt(patch_a_corners, corners_b_pred, weights=None) )
    h_pred = (TensorDLT.tensorDLT(corners_b_pred, patch_a_corners) )
    print(h_pred)
    h_pred_inv = torch.inverse(h_pred)


    patch_b_pred = k.geometry.transform.warp_perspective(orig_img, h_pred_inv, dsize = (128,128),
                                                                    mode='bilinear', padding_mode='zeros', 
                                                                    align_corners=True, fill_value=torch.zeros(3))
    
    return patch_b_pred