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
# termcolor, do (pip install termcolor)

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms as tf
from torch.optim import AdamW, SGD, Adam
from Network.Network import *
from Wrapper import *
import cv2
import sys
import os
import numpy as np
import random
import skimage
import PIL
import os
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import numpy as np
import time
from Misc.MiscUtils import *
from Misc.DataUtils import *
from torchvision.transforms import ToTensor
import argparse
import shutil
import string
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from termcolor import colored, cprint
import math as m
from random import choice
from tqdm import tqdm

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Running on device: {device}")

class YourDataset(Dataset):

    def __init__(self, x_train, y_train, transform=None):
        self.x_train = x_train
        self.y_train = y_train
        self.transform = transform
    
    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):

        x = self.x_train[idx]
        y = self.y_train[idx]

        if self.transform:
            x = self.transform(x)

        return x, y

def GenerateBatch(BasePath, MiniBatchSize, Val): #DirNamesTrain, TrainCoordinates, ImageSize,
    """
    Inputs:
    BasePath - Path to COCO folder without "/" at the end
    DirNamesTrain - Variable with Subfolder paths to train files
    NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
    TrainCoordinates - Coordinatess corresponding to Train
    NOTE that TrainCoordinates can be replaced by Val/TestCoordinatess for generating batch corresponding to validation (held-out testing in this case)/testing
    ImageSize - Size of the Image
    MiniBatchSize is the size of the MiniBatch
    Outputs:
    I1Batch - Batch of images
    CoordinatesBatch - Batch of coordinates
    """
    I1Batch = []
    CoordinatesBatch = []
    # ImageNum = 0
    # while ImageNum < MiniBatchSize:
        # ImageNum += 1

        # Generate random image
    print("Generating Data!!")
    orignal_img_fpath = os.path.join(BasePath, f"train_dataset/patch_a")
    warped_img_fpath = os.path.join(BasePath, f"train_dataset/patches_warped")
    if os.path.exists(orignal_img_fpath): 
        image_list = os.listdir(orignal_img_fpath)
    else:
        raise Exception ("Directory Image1 doesn't exist")
    
    # RandIdx = random.randint(1, len(image_list)-1)
    orig_path = []
    warped_path = []
    for i in range(len(image_list)):
        orig_img_path = os.path.join(orignal_img_fpath,image_list[i])
        warped_img_path = os.path.join(warped_img_fpath,image_list[i])
        orig_path.append(orig_img_path)
        warped_path.append(warped_img_path)

    img1 = [cv2.imread(i, cv2.IMREAD_GRAYSCALE) for i in orig_path]
    img2 = [cv2.imread(i, cv2.IMREAD_GRAYSCALE) for i in warped_path]
    origset = np.array(img1)
    warpedset = np.array(img2)

    for i in range(0, len(origset)):
        img1 = origset[i]
        img1 = np.expand_dims(img1, 2)
        img2 = warpedset[i]
        img2 = np.expand_dims(img2, 2)
        IMG = np.concatenate((img1, img2), axis = 2)
        I1Batch.append(torch.from_numpy(IMG))
        label = np.genfromtxt('Phase2/Data/train_dataset/H_4points/' + str(i+1) + '.txt', delimiter=',')
        CoordinatesBatch.append(torch.from_numpy(label))
        # Append All Images and Mask
        # I = np.transpose(I, (2, 0, 1))
        # I1Batch.append(torch.from_numpy(IMG))
        # CoordinatesBatch.append(torch.from_numpy(label))

    # Generating validation batch
    if Val:
        val_patch_path = os.path.join(BasePath, f"val_dataset/patch_a")
        val_warped_path = os.path.join(BasePath, f"val_dataset/patches_warped")
        if os.path.exists(val_patch_path): 
            imgs_list = os.listdir(val_patch_path)
        else:
            raise Exception ("Directory Image1 doesn't exist")
        
        images1_path = []
        images2_path = []
        for i in range(len(imgs_list)):
        # for i in range(2):
            # print(imgs_list[i])
            I1 = os.path.join(val_patch_path,imgs_list[i])
            I2 = os.path.join(val_warped_path,imgs_list[i])
            images1_path.append(I1)
            images2_path.append(I2)

        image1 = [cv2.imread(i,0) for i in images1_path]
        image2 = [cv2.imread(i,0) for i in images2_path]
        trainsetA = np.array(image1)
        trainsetB = np.array(image2)
        val_batch = []
        val_labels = []
        count = 0
        for i in range(0,len(trainsetA)):

            count+=1
            img1 = trainsetA[i]
            img1 = np.expand_dims(img1, 2)
            img2 = trainsetB[i]
            img2 = np.expand_dims(img2, 2)
            img = np.concatenate((img1, img2), axis = 2)
            # print(img.shape)
            # img = np.transpose(img, (2, 0, 1))
            val_batch.append(torch.from_numpy(img))
            # print(imgs_list[i+1])
            val_label = np.genfromtxt('Phase2/Data/val_dataset/H_4points/' + imgs_list[i][:-4] + '.txt', delimiter=',')
            val_labels.append(torch.from_numpy(val_label).to(device))
            
    
        # print("returning val")
        val_batch = torch.stack(val_batch)
        val_batch = val_batch.to(torch.float32)
        val_dataset = [val_batch, torch.stack(val_labels)]
        val_dataset = YourDataset(val_batch, torch.stack(val_labels))
        val_loader = DataLoader(val_dataset, batch_size=1)
        return val_loader #val_batch, torch.stack(val_labels)
    else:
        # print("returning train")
        I1Batch = torch.stack(I1Batch)
        I1Batch = I1Batch.to(torch.float32)
        # train_dataset = [I1Batch, torch.stack(CoordinatesBatch)]
        train_dataset = YourDataset(I1Batch, torch.stack(CoordinatesBatch))
        train_loader = DataLoader(train_dataset, batch_size=MiniBatchSize, shuffle = True)
        return train_loader #I1Batch, torch.stack(CoordinatesBatch)


def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
    """
    Prints all stats with all arguments
    """
    print("Number of Epochs Training will run for " + str(NumEpochs))
    print("Factor of reduction in training data is " + str(DivTrain))
    print("Mini Batch Size " + str(MiniBatchSize))
    print("Number of Training Images " + str(NumTrainSamples))
    if LatestFile is not None:
        print("Loading latest checkpoint with the name " + LatestFile)


def TrainOperation(
    DirNamesTrain,
    TrainCoordinates,
    NumTrainSamples,
    NumEpochs,
    MiniBatchSize,
    SaveCheckPoint,
    CheckPointPath,
    DivTrain,
    LatestFile,
    BasePath,
    LogsPath,
    ModelType,
):
    """
    Inputs:
    ImgPH is the Input Image placeholder
    DirNamesTrain - Variable with Subfolder paths to train files
    TrainCoordinates - Coordinates corresponding to Train/Test
    NumTrainSamples - length(Train)
    NumEpochs - Number of passes through the Train data
    MiniBatchSize is the size of the MiniBatch
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    CheckPointPath - Path to save checkpoints/model
    DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of dataor for debugging code
    LatestFile - Latest checkpointfile to continue training
    BasePath - Path to COCO folder without "/" at the end
    LogsPath - Path to save Tensorboard Logs
        ModelType - Supervised or Unsupervised Model
    Outputs:
    Saves Trained network in CheckPointPath and Logs to LogsPath
    """
    # Predict output with forward pass
    model = HomographyModel()

    Optimizer = Adam(model.parameters(), lr=0.005)

    # Tensorboard
    # Create a summary to monitor loss tensor
    Writer = SummaryWriter(LogsPath)

    if LatestFile is not None:
        CheckPoint = torch.load(CheckPointPath + LatestFile + ".ckpt")
        # Extract only numbers from the name
        StartEpoch = int("".join(c for c in LatestFile.split("a")[0] if c.isdigit()))
        model.load_state_dict(CheckPoint["model_state_dict"])
        print("Loaded latest checkpoint with the name " + LatestFile + "....")
    else:
        StartEpoch = 0
        print("New model initialized....")
    train_loader = GenerateBatch(
            BasePath, MiniBatchSize, Val=False
        )
    train_loss = []
    validation_loss = []
    for Epochs in tqdm(range(StartEpoch, NumEpochs)):
        # NumIterationsPerEpoch = int(NumTrainSamples / MiniBatchSize / DivTrain)
        # for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
        
        # Predict output with forward pass
        model.train()
        train_losses = []
        # print(I1Batch.shape)
        PerEpochCounter = 0
        for Batch in train_loader:
            PerEpochCounter += 1
            # print(Batch.shape)
            Image, Label = Batch
            PredicatedCoordinatesBatch = model(Image)
            LossThisBatch = LossFn(PredicatedCoordinatesBatch, Label)
            train_losses.append(LossThisBatch)
            Optimizer.zero_grad()
            LossThisBatch.backward()
            Optimizer.step()

            del I1Batch
            del CoordinatesBatch
            del PredicatedCoordinatesBatch
            torch.cuda.empty_cache()
            # Save checkpoint every some SaveCheckPoint's iterations
            if PerEpochCounter % SaveCheckPoint == 0:

                # Save the Model learnt in this epoch
                SaveName = (
                    CheckPointPath
                    + str(Epochs)
                    + "a"
                    + str(PerEpochCounter)
                    + "model.ckpt"
                )

                torch.save(
                    {
                        "epoch": Epochs,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": Optimizer.state_dict(),
                        "loss": LossThisBatch,
                    },
                    SaveName,
                )
                print("\n" + SaveName + " Model Saved...")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            val_loader = GenerateBatch(
                BasePath, MiniBatchSize=1, Val=True
            )
            result = [model.validation(val_batch, val_labels) for val_batch, val_labels in val_loader]
            result = model.validation_epoch_end(result)
            result['train_loss'] = torch.stack(train_losses).mean().item()
            validation_loss.append(result['val_loss'])
            train_loss.append(result['train_loss'])
            # result = model.validation(val_batch, val_labels)

                val_image = torch.stack([val_image]).to(device)
                val_label = torch.stack([val_label]).to(device)
                loss = model.validation(val_image, val_label)
                # print(loss)
                val_loss+=loss
            val_loss = val_loss / len(val_labels)
        print("Validation loss - ", val_loss)
        # Write validation losses to tensorboard
        Writer.add_scalar(
            "LossEveryEpoch",
            val_loss,
            # result["val_loss"],
            Epochs,
        )
        # If you don't flush the tensorboard doesn't update until a lot of iterations!
        Writer.flush()

        # Save model every epoch
        SaveName = CheckPointPath + str(Epochs) + "model.ckpt"
        torch.save(
            {
                "epoch": Epochs,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": Optimizer.state_dict(),
                "loss": LossThisBatch,
            },
            SaveName,
        )
        print("\n" + SaveName + " Model Saved...")
    return train_loss, validation_loss

def plot_train_losses(train):
    plt.plot(train, '-x', label =  'TrainSet')
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.legend(ncol=2, loc="upper right")
    plt.title('Loss vs. No. of epochs')
    plt.savefig("loss.png")

def main():
    """
    Inputs:
    # None
    # Outputs:
    # Runs the Training and testing code based on the Flag
    #"""
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        "--BasePath",
        default='Phase2/Data/',
        help="Base path of images, Default:/home/lening/workspace/rbe549/YourDirectoryID_p1/Phase2/Data",
    )
    Parser.add_argument(
        "--CheckPointPath",
        default="./Phase2/Checkpoints/",
        help="Path to save Checkpoints, Default: ../Checkpoints/",
    )

    Parser.add_argument(
        "--ModelType",
        default="Unsup",
        help="Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Unsup",
    )
    Parser.add_argument(
        "--NumEpochs",
        type=int,
        default=10,
        help="Number of Epochs to Train for, Default:10",
    )
    Parser.add_argument(
        "--DivTrain",
        type=int,
        default=1,
        help="Factor to reduce Train data by per epoch, Default:1",
    )
    Parser.add_argument(
        "--MiniBatchSize",
        type=int,
        default=64,
        help="Size of the MiniBatch to use, Default:1",
    )
    Parser.add_argument(
        "--LoadCheckPoint",
        type=int,
        default=0,
        help="Load Model from latest Checkpoint from CheckPointsPath?, Default:0",
    )
    Parser.add_argument(
        "--LogsPath",
        default="Phase2/Logs/",
        help="Path to save Logs for Tensorboard, Default=Logs/",
    )

    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    BasePath = Args.BasePath
    DivTrain = float(Args.DivTrain)
    MiniBatchSize = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    CheckPointPath = Args.CheckPointPath
    LogsPath = Args.LogsPath
    ModelType = Args.ModelType

    # Setup all needed parameters including file reading
    (
        DirNamesTrain,
        SaveCheckPoint,
        ImageSize,
        NumTrainSamples,
        TrainCoordinates,
        NumClasses,
    ) = SetupAll(BasePath, CheckPointPath)

    # Find Latest Checkpoint File
    if LoadCheckPoint == 1:
        LatestFile = FindLatestModel(CheckPointPath)
    else:
        LatestFile = None

    # Pretty print stats
    PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)

    train_loss, val_loss = TrainOperation(
        DirNamesTrain,
        TrainCoordinates,
        NumTrainSamples,
        NumEpochs,
        MiniBatchSize,
        SaveCheckPoint,
        CheckPointPath,
        DivTrain,
        LatestFile,
        BasePath,
        LogsPath,
        ModelType,
    )
    plot_train_losses(train_loss)

if __name__ == "__main__":
    main()