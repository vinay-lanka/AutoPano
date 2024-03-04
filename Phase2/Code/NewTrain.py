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

def GenerateBatch(BasePath, MiniBatchSize, State):
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
    ImageBatch - Batch of images
    GroundTruthBatch - Batch of coordinates
    """
    ImageBatch = []
    GroundTruthBatch = []
    ImageNum = 0
    patches_a_path = os.path.join(BasePath, f"Synthetic{State}/patch_A")
    patches_b_path = os.path.join(BasePath, f"Synthetic{State}/patch_B")
    GTs_path = os.path.join(BasePath, f"Synthetic{State}/H4Pt/")
    if os.path.exists(patches_a_path): 
        image_list = os.listdir(patches_a_path)
    else:
        raise Exception ("Patch A Directory doesn't exist")
    
    while ImageNum < MiniBatchSize:
        ImageNum += 1
        RandIdx = random.randint(1, len(image_list)-1)
        # print(len(image_list))
        if State == "Train":
            patch_a_path = os.path.join(patches_a_path,image_list[RandIdx])
            patch_b_path = os.path.join(patches_b_path,image_list[RandIdx])
            GT_path = GTs_path + image_list[RandIdx][:-4] + f".txt"
        elif State == "Val":
            # print(ImageNum)
            patch_a_path = os.path.join(patches_a_path,image_list[ImageNum-1])
            patch_b_path = os.path.join(patches_b_path,image_list[ImageNum-1])
            GT_path = GTs_path + image_list[ImageNum-1][:-4] + f".txt"

        patch_A = cv2.imread(patch_a_path, cv2.IMREAD_GRAYSCALE)
        patch_B = cv2.imread(patch_b_path, cv2.IMREAD_GRAYSCALE)

        patch_A = np.expand_dims(patch_A, 2)
        patch_B = np.expand_dims(patch_B, 2)
        I = np.concatenate((patch_A, patch_B), axis = 2)
        GT = np.genfromtxt(GT_path, delimiter=',')

        ImageBatch.append(torch.from_numpy(I))
        GroundTruthBatch.append(torch.from_numpy(GT).to(device))

    ImageBatch = torch.stack(ImageBatch)
    ImageBatch = ImageBatch.to(device,dtype=torch.float32)
    return ImageBatch, torch.stack(GroundTruthBatch)


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
    if ModelType == 'Sup':
        # Predict output with forward pass
        model = SupHomographyModel()
    else:
        model = UnSupHomographyModel()

    Optimizer = Adam(model.parameters(), lr=0.0001)

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

    #Enable cuda
    model.to(device)

    for Epochs in tqdm(range(StartEpoch, NumEpochs)):
        NumIterationsPerEpoch = int(NumTrainSamples / MiniBatchSize / DivTrain)
        for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):

            if ModelType == 'Sup':
                I1Batch, CoordinatesBatch = GenerateBatch(
                    BasePath, MiniBatchSize, "Train"
                )

            I1Batch.to(device)
            CoordinatesBatch.to(device)
            # Predict output with forward pass
            model.train()

            PredicatedCoordinatesBatch = model(I1Batch)
            LossThisBatch = SupLossFn(PredicatedCoordinatesBatch, CoordinatesBatch)
            # print(LossThisBatch)

            Optimizer.zero_grad()
            LossThisBatch.backward()
            Optimizer.step()

            del I1Batch
            del CoordinatesBatch
            del PredicatedCoordinatesBatch
            torch.cuda.empty_cache()
            # Save checkpoint every some SaveCheckPoint's iterations
            if PerEpochCounter % SaveCheckPoint == 0:
                # Write training losses to tensorboard
                Writer.add_scalar(
                    "LossEveryIter",
                    LossThisBatch,
                    Epochs * NumIterationsPerEpoch + PerEpochCounter,
                )
                # If you don't flush the tensorboard doesn't update until a lot of iterations!
                Writer.flush()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            val_batch, val_labels = GenerateBatch(
                BasePath, 1000, "Val"
            )
            for val_image,val_label in zip(val_batch, val_labels):

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

def plot_train_losses(train):
    plt.plot(train, '-x', label =  'TrainSet')
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.legend(ncol=2, loc="upper right")
    plt.title('Loss vs. No. of epochs')
    plt.savefig("loss.png")

 

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
        default='./Phase2/Data/',
        help="Base path of images, Default:./Phase2/Data/",
    )
    Parser.add_argument(
        "--CheckPointPath",
        default="./Phase2/Checkpoints/",
        help="Path to save Checkpoints, Default: ../Checkpoints/",
    )

    Parser.add_argument(
        "--ModelType",
        default="Sup",
        help="Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Unsup",
    )
    Parser.add_argument(
        "--NumEpochs",
        type=int,
        default=25,
        help="Number of Epochs to Train for, Default:50",
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
        default=32,
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
        default="./Phase2/Logs/",
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

    TrainOperation(
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


if __name__ == "__main__":
    main()