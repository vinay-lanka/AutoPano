import numpy as np
import cv2
import os
from copy import deepcopy
import random
import shutil
from tqdm import tqdm

patch_size = 128 #square patch so (128,128)
p = 16

def generate_data(img):
    img = deepcopy(img)
    x,y = img.shape

    patch_x = random.randint(p, x-patch_size-p)
    patch_y = random.randint(p, y-patch_size-p)

    patch_A  =   img[patch_x : patch_x + patch_size, patch_y : patch_y + patch_size]    # patch_A extracted from the original Image
    patch_a_corners =   np.array([[patch_y             , patch_x            ],  # Left-Top
                                [patch_y             , patch_x + patch_size ],  # Right-Top
                                [patch_y + patch_size, patch_x + patch_size ],  # Right-Bottom
                                [patch_y + patch_size, patch_x              ]], dtype=np.float32)  # Left-Botton
    p1 = np.array([[0, 0], [0, x], [y, x], [y, 0]]).reshape(-1,1,2)

    pert_x = np.array(random.sample(range(-p, p), 4)).reshape(4,1)
    pert_y = np.array(random.sample(range(-p, p), 4)).reshape(4,1)
    pert_mat = np.hstack([pert_x, pert_y])

    patch_b_corners = patch_a_corners + pert_mat
    patch_b_corners = np.array(patch_b_corners,dtype =np.float32)

    H4Pt = patch_b_corners - patch_a_corners 
    H4Pt = H4Pt.flatten(order = 'F')  

    H_AB = cv2.getPerspectiveTransform(patch_a_corners, patch_b_corners)
    H_BA = np.linalg.inv(H_AB)
    
    p2  = cv2.perspectiveTransform(np.float32(p1), H_BA)
    [xmin, ymin] = np.int32(p2.min(axis=0).ravel())
    [xmax, ymax] = np.int32(p2.max(axis=0).ravel())
    t = [-xmin,-ymin]
    T = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate
    img_warped = cv2.warpPerspective(img, T.dot(H_BA), (xmax-xmin, ymax-ymin),flags = cv2.INTER_LINEAR)
    patch_B  = img_warped[patch_x + t[1] : patch_x + patch_size + t[1], patch_y + t[0] : patch_y + patch_size + t[0]]    # patch_B extracted from the warped Image

    return img_warped, patch_a_corners, patch_b_corners, patch_A, patch_B, H4Pt

def generate_dataset(number_of_files, path, synthetic_data_path):
    #Make paths
    os.makedirs(synthetic_data_path+'Grayscale/', exist_ok=True)
    os.makedirs(synthetic_data_path+'H4Pt/', exist_ok=True)
    os.makedirs(synthetic_data_path+'patch_A/', exist_ok=True)
    os.makedirs(synthetic_data_path+'patch_B/', exist_ok=True)
    os.makedirs(synthetic_data_path+'patch_a_corners/', exist_ok=True)
    for i in range(1,number_of_files+1):
        img = cv2.imread(path+str(i)+'.jpg',cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(synthetic_data_path+'Grayscale/'+str(i)+'.jpg', img)
        img = cv2.resize(img, (320,240))
        img_warped, patch_a_corners, patch_b_corners, patch_A, patch_B, H4Pt = generate_data(img)
        cv2.imwrite(synthetic_data_path+'patch_A/'+str(i)+'.jpg', patch_A)
        cv2.imwrite(synthetic_data_path+'patch_B/'+str(i)+'.jpg', patch_B)
        np.savetxt(synthetic_data_path + 'H4Pt/' + str(i) + ".txt", H4Pt, delimiter = ",")
        np.savetxt(synthetic_data_path + 'patch_a_corners/' + str(i) + ".txt", patch_a_corners, delimiter = ",")
    return


def main():
    train_path = './Phase2/Data/Train/'
    train_synthetic_data_path = './Phase2/Data/SyntheticTrain/'
    os.makedirs(train_synthetic_data_path, exist_ok=True)
    generate_dataset(5000, train_path, train_synthetic_data_path)

    val_path = './Phase2/Data/Val/'
    val_synthetic_data_path = './Phase2/Data/SyntheticVal/'
    os.makedirs(val_synthetic_data_path, exist_ok=True)
    generate_dataset(1000, val_path,val_synthetic_data_path)       
    # TO DO TEST

if __name__ == "__main__":
    main()