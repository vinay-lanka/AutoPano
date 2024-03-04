#!/usr/bin/env python3

"""
RBE/CS Fall 2022: Classicorners_al and Deep Learning Approaches for
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


# Code starts here:

import numpy as np
import cv2
import os
from copy import deepcopy
import random
import shutil
from tqdm import tqdm


# Add any python libraries here
def generate_train_dataset(model_type = 'any'):
    Num_Trainset = 5000
    count = 0
    gray_img = os.path.join("Phase2/Data/train_dataset/Grayscorners_ale_image")
    if os.path.exists(gray_img):
        shutil.rmtree(gray_img)
        os.mkdir(gray_img)
    else:
        os.mkdir(gray_img)
    patch_a = os.path.join("Phase2/Data/train_dataset/patch_a")
    if os.path.exists(patch_a):
        shutil.rmtree(patch_a)
        os.mkdir(patch_a)
    else:
        os.mkdir(patch_a)

    warped_PF = os.path.join("Phase2/Data/train_dataset/patches_warped")
    if os.path.exists(warped_PF):
        shutil.rmtree(warped_PF)
        os.mkdir(warped_PF)
    else:
        os.mkdir(warped_PF)

    h4pt = os.path.join("Phase2/Data/train_dataset/H_4points")
    if os.path.exists(h4pt):
        shutil.rmtree(h4pt)
        os.mkdir(h4pt)
    else:
        os.mkdir(h4pt)

    for i in tqdm(range(1,Num_Trainset+1)):

        count += 1
        # Read the image
        img = cv2.imread('Phase2/Data/Train/' + str(i) + '.jpg')
        I = deepcopy(img)

        # Resize and convert the image to grayscorners_ale and save the image in the 
        I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
        I = cv2.resize(I, (320, 240))
        gray_filename = gray_img + "/" + str(count) + ".png"
        cv2.imwrite(gray_filename, I)

        # Extract a random patch from the image
        width = 128
        height = 128
        perturbation = 16
        x,y = I.shape

        patch_x = random.randint(perturbation, x-width-perturbation)
        patch_y = random.randint(perturbation, y-height-perturbation)

        patch_A = I[patch_x : patch_x + width, patch_y : patch_y + height]
        patch_a_corners = np.array([[patch_y, patch_x], [patch_y, patch_x + width], [patch_y + height, patch_x + width], [patch_y + height, patch_x]], dtype=np.float32)
        

        p1 = np.array([[0, 0], [0, x], [y, x], [y, 0]]).reshape(-1,1,2)
        filename_a = patch_a + "/" + str(count) + ".png"
        cv2.imwrite(filename_a, patch_A)

        # Get another set of corners by applying random perturbation to the patch A corner and find this displacement matrix
        pert_x = np.array(random.sample(range(-perturbation, perturbation), 4)).reshape(4,1)
        pert_y = np.array(random.sample(range(-perturbation, perturbation), 4)).reshape(4,1)
        pert_mat = np.hstack([pert_x, pert_y])

        patch_b_corners = patch_a_corners + pert_mat
        patch_b_corners = np.array(patch_b_corners,dtype =np.float32)
        H_4point = np.subtract(patch_b_corners,patch_a_corners)
        H_4point = H_4point.flatten(order = 'F')
        h4_filename = h4pt + "/" + str(count) + ".txt"
        np.savetxt(h4_filename, H_4point, delimiter = ",")

        # Flatten the patch A corners and save theem
        # flattened_patch_a_corners = patch_a_corners.flatten(order = 'F')
        # corners_a_filename = corners_a + "/" + str(count) + ".txt"
        # np.savetxt(corners_a_filename, flattened_patch_a_corners, delimiter = ",")

        # Get the Homography and the warped image
        H_AB = cv2.getPerspectiveTransform(patch_a_corners, patch_b_corners)
        H_BA = np.linalg.inv(H_AB)

        p2  = cv2.perspectiveTransform(np.float32(p1), H_BA)
        [xmin, ymin] = np.int32(p2.min(axis=0).ravel())
        [xmax, ymax] = np.int32(p2.max(axis=0).ravel())
        t = [-xmin,-ymin]
        T = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate
        warp_img = cv2.warpPerspective(I, T.dot(H_BA), (xmax-xmin, ymax-ymin),flags = cv2.INTER_LINEAR)
        warped_patch = warp_img[patch_x + t[1] : patch_x + width + t[1], patch_y + t[0] : patch_y + height + t[0]]
        
        filename_b = warped_PF + "/" + str(count) + ".png"
        cv2.imwrite(filename_b, warped_patch)

def generate_val_dataset(model_type = 'any'):
    Num_Trainset = 1000
    count = 0

    patch_a = os.path.join("Phase2/Data/val_dataset/patch_a")
    if os.path.exists(patch_a):
        shutil.rmtree(patch_a)
        os.mkdir(patch_a)
    else:
        os.mkdir(patch_a)

    warped_PF = os.path.join("Phase2/Data/val_dataset/patches_warped")
    if os.path.exists(warped_PF):
        shutil.rmtree(warped_PF)
        os.mkdir(warped_PF)
    else:
        os.mkdir(warped_PF)

    h4pt = os.path.join("Phase2/Data/val_dataset/H_4points")
    if os.path.exists(h4pt):
        shutil.rmtree(h4pt)
        os.mkdir(h4pt)
    else:
        os.mkdir(h4pt)

    print("Starting to generate the validation set")

    for i in tqdm(range(1,Num_Trainset+1)):

        count += 1
        # Read the image
        img = cv2.imread('Phase2/Data/Val/' + str(i) + '.jpg')
        I = deepcopy(img)

        # Resize and convert the image to grayscorners_ale and save the image in the 
        I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
        I = cv2.resize(I, (320, 240))
        # I = (I-np.mean(I))/255

        # Extract a random patch from the image
        width = 128
        height = 128
        perturbation = 16
        x,y = I.shape

        patch_x = random.randint(perturbation, x-width-perturbation)
        patch_y = random.randint(perturbation, y-height-perturbation)

        patch_A = I[patch_x : patch_x + width, patch_y : patch_y + height]
        patch_a_corners = np.array([[patch_y, patch_x], [patch_y, patch_x + width], [patch_y + height, patch_x + width], [patch_y + height, patch_x]], dtype=np.float32)
        

        p1 = np.array([[0, 0], [0, x], [y, x], [y, 0]]).reshape(-1,1,2)
        filename_a = patch_a + "/" + str(count) + ".png"
        cv2.imwrite(filename_a, patch_A)

        # Get another set of corners by applying random perturbation to the patch A corner and find this displacement matrix
        pert_x = np.array(random.sample(range(-perturbation, perturbation), 4)).reshape(4,1)
        pert_y = np.array(random.sample(range(-perturbation, perturbation), 4)).reshape(4,1)
        pert_mat = np.hstack([pert_x, pert_y])

        patch_b_corners = patch_a_corners + pert_mat
        patch_b_corners = np.array(patch_b_corners,dtype =np.float32)
        H_4point = np.subtract(patch_b_corners,patch_a_corners)
        H_4point = H_4point.flatten(order = 'F')
        h4_filename = h4pt + "/" + str(count) + ".txt"
        np.savetxt(h4_filename, H_4point, delimiter = ",")

        # Get the Homography and the warped image
        H_AB = cv2.getPerspectiveTransform(patch_a_corners, patch_b_corners)
        H_BA = np.linalg.inv(H_AB)

        p2  = cv2.perspectiveTransform(np.float32(p1), H_BA)
        [xmin, ymin] = np.int32(p2.min(axis=0).ravel())
        [xmax, ymax] = np.int32(p2.max(axis=0).ravel())
        t = [-xmin,-ymin]
        Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate
        warp_img = cv2.warpPerspective(I, Ht.dot(H_BA), (xmax-xmin, ymax-ymin),flags = cv2.INTER_LINEAR)
        warped_patch = warp_img[patch_x + t[1] : patch_x + width + t[1], patch_y + t[0] : patch_y + height + t[0]]
        
        filename_b = warped_PF + "/" + str(count) + ".png"
        cv2.imwrite(filename_b, warped_patch)


def main():
    # Add any Command Line arguments here
    # Parser = argparse.ArgumentParser()
    # Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')

    # Args = Parser.parse_args()
    # NumFeatures = Args.NumFeatures
    # generate_train_dataset()
    generate_val_dataset()          


    """
    Read a set of images for Panorama stitching
    """

    """
	Obtain Homography using Deep Learning Model (Supervised and Unsupervised)
	"""

    """
	Image Warping + Blending
	Save Panorama output as mypano.png
	"""


if __name__ == "__main__":
    main()