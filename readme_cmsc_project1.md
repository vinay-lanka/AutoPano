# CMSC733 Project 1: MyAutoPano

*Submission by:*
**Vikram Setty (vikrams@umd.edu)**
**Vinay Lanka (vlanka@umd.edu)**
**Mayank Deshpande (msdeshp4@umd.edu)**

This README contains the instructions to run the code for Phase 1 and Phase 2 submitted as a part of the first project of CMSC733 (Computer Processing of Pictorial Information) offerred by the Department of Computer Science at the Unversity of Maryland.

## Phase 1: Traditional Approach
To run the code to generate the Pb-lite edges of images given in a dataset, firstly navigate to the folder with `Wrapper.py` directly accessible and run the following command.
```sh
  python3 Wrapper.py
```
Running this command would generate the final stitched panorama for the first image set of the training folder included with the assignment. This code can however be easily extended by changing the relative path of the image folder in line 278 of `Wrapper.py` by including the correct path in the `path_to_images` variable.

## Phase 2: Deep Learning Approach

To train the network, first we need to generate the synthetic data required to train such homography networks. To do that use the `Wrapper.py` script provided. Make sure you've downloaded both the train and test datasets for this to work and have placed the datasets under the Phase2/Data folder.
```bash
#To generate synthetic data
$ python3 Phase2/Code/Wrapper.py
```

Now to train the network, select the type of network you want to train, both options are given below. For more arguments, check out the `Train.py` file. To test the network, use the `Test.py` file after training both networks. 

Note - For all the files, make sure you have pytorch installed preferably in a CUDA environment for the least errors.

```bash
#To train the supervised network 
$ python3 Phase2/Code/Train.py --NumEpochs 25 --ModelType Sup

#To train the unsupervised network 
$ python3 Phase2/Code/Train.py --NumEpochs 25 --ModelType UnSup

#To test the networks
$ python3 Phase2/Code/Test.py   
```
To change the model architecture, the model imported from `Network.py` and other paths in `Train.py` and `Test.py` would need to be changed accordingly. 

**NOTE**: To run and execute any code (in Phase 1/Phase 2), it is extremely necessary to have appropriate data and result paths. Contact the authors for the datasets.
```bash
```