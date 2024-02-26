#!/usr/bin/env python3

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project1: MyAutoPano: Phase 1 Starter Code

Author(s): 
Chahat Deep Singh (chahat@terpmail.umd.edu) 
PhD Student in Computer Science,
University of Maryland, College Park

Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

# Code starts here:

import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.feature import peak_local_max
# Add any python libraries here

harris_max_ratio = 0.01
harris_block_size = 2
harris_sobel_aperture = 3
harris_kvalue = 0.001
anms_N_best = 350
descriptor_patch_size = 40
SSD_ratio_threshold = 0.99
num_ransac_iterations = 5000
homography_inlier_threshold = 5

def detect_corners(rgb_img):
      gray_img = cv2.cvtColor(rgb_img.copy(),cv2.COLOR_BGR2GRAY).astype(np.float32)
      corners_info = cv2.cornerHarris(gray_img,harris_block_size,harris_sobel_aperture,harris_kvalue)
      corners_info[corners_info < harris_max_ratio*corners_info.max()] = 0
      corner_locations = np.where(corners_info > 0)
      img_with_corners = rgb_img.copy()
      img_with_corners[corner_locations] = np.array([255,0,0])
      return (corners_info,corner_locations,img_with_corners)

def adaptive_non_max_supression(rgb_img,corners_info):
      strong_corner_coords = peak_local_max(corners_info,min_distance=5)
      #print(strong_corner_coords.shape)
      N_strong = strong_corner_coords.shape[0]
      r_values = np.inf*np.ones((N_strong))
      for i in range(N_strong):
            for j in range(N_strong):
                  x_i, y_i = strong_corner_coords[i]
                  x_j, y_j = strong_corner_coords[j]
                  ED = r_values[i]
                  if corners_info[x_i,y_i] < corners_info[x_j,y_j]:
                        ED = (strong_corner_coords[j,1]-strong_corner_coords[i,1])**2 + (strong_corner_coords[j,0]-strong_corner_coords[i,1])**0
                  if r_values[i] > ED:
                        r_values[i] = ED
      N_best_indices = np.flip(np.argsort(r_values))[0:min(anms_N_best,N_strong)]
      anms_corner_locations = np.array([(int(strong_corner_coords[i][0]),int(strong_corner_coords[i][1])) for i in N_best_indices])
      img_with_anms_corners = rgb_img.copy()
      for i in range(min(anms_N_best,N_strong)):
            #img_with_anms_corners[int(anms_corner_locations[i][0]),int(anms_corner_locations[i][1]),:] = np.array([255,0,0])
            cv2.circle(img_with_anms_corners,(anms_corner_locations[i][1],anms_corner_locations[i][0]),2,(255,0,0),-1)
      return (anms_corner_locations,img_with_anms_corners)

def corner_feature_descriptors(rgb_img,anms_corner_locations):
      gray_img = cv2.cvtColor(rgb_img.copy(),cv2.COLOR_BGR2GRAY)
      feature_descriptor = []
      for loc in anms_corner_locations:
            if(loc[0]<=descriptor_patch_size//2 or loc[0]>=gray_img.shape[0]-descriptor_patch_size//2 or loc[1]<=descriptor_patch_size//2 or loc[1]>=gray_img.shape[1]-descriptor_patch_size//2):
                  continue
            feature_patch = gray_img[loc[0]-descriptor_patch_size//2:loc[0]+descriptor_patch_size//2,loc[1]-descriptor_patch_size//2:loc[1]+descriptor_patch_size//2]
            feature_patch = cv2.GaussianBlur(feature_patch,(3,3),0)
            feature_patch = cv2.resize(feature_patch,(descriptor_patch_size//5,descriptor_patch_size//5),interpolation=cv2.INTER_CUBIC).reshape(-1)
            #print(feature_patch.shape)
            feature_patch = (feature_patch-feature_patch.mean())/feature_patch.std()
            feature_descriptor.append([loc,feature_patch])
      return feature_descriptor

def match_features(img1_features,img2_features):
      matches = []
      for loc1, feature1 in img1_features:
            ssd_array = []
            for loc2, feature2 in img2_features:
                  ssd_array.append(np.sum(np.square(feature1-feature2)))
            best_matches = np.argsort(np.array(ssd_array))
            if ssd_array[best_matches[0]]/ssd_array[best_matches[1]] < SSD_ratio_threshold:
                  matches.append([loc1,img2_features[best_matches[0]][0]])
      return matches

def draw_feature_matches(rgb_img1,rgb_img2,matches):
      # Both images must be of the same sizes to concatenate them side by side
      img_concat = np.concatenate([rgb_img1.copy(),rgb_img2.copy()],axis=1)
      for match in matches:
            loc1, loc2 = match
            loc2[1] += rgb_img1.shape[1]
            #print(img_concat.shape)
            #print(loc1, "       ", loc2)
            #cv2.circle(img_concat,[loc1[1],loc1[0]],2,(0,0,255),-1)
            #cv2.circle(img_concat,[loc2[1],loc2[0]],2,(0,0,255),-1)
            cv2.line(img_concat,[loc1[1],loc1[0]],[loc2[1],loc2[0]],(0,0,255),1)
      plt.imshow(img_concat)
      plt.show()
      

def get_inliers(img1_matches,img2_matches,homography):
      num_inlier_points = 0
      inlier_indices = []
      img1_points_transformed = np.vstack((img1_matches[:,0],img1_matches[:,1],np.ones((1,img1_matches.shape[0]))))
      img1_points_transformed = np.dot(homography,img1_points_transformed)
      x_transformed = img1_points_transformed[0,:]/(img1_points_transformed[2,:]+np.exp(-7))
      y_transformed = img1_points_transformed[1,:]/(img1_points_transformed[2,:]+np.exp(-7))
      img1_points_transformed = np.array([x_transformed,y_transformed]).T
      #print(img1_points_transformed.shape,"   ",img2_matches.shape)
      for i in range(img1_matches.shape[0]):
            if np.linalg.norm(img1_points_transformed[i]-img2_matches[i]) < homography_inlier_threshold:
                  num_inlier_points += 1
                  inlier_indices.append(i)
      return num_inlier_points, inlier_indices

def RANSAC(matches):
      img1_matches = np.array(matches)[:,0,:]
      img2_matches = np.array(matches)[:,1,:]
      best_inlier_indices = []
      max_inliers = 0
      best_homography = None
      for i in range(num_ransac_iterations):
            random_points = np.random.choice(len(matches),size=4)
            img1_points = img1_matches[random_points].astype(np.float32)
            img2_points = img2_matches[random_points].astype(np.float32)
            homography = cv2.getPerspectiveTransform(img1_points,img2_points)
            num_inlier_points, inlier_indices = get_inliers(img1_matches,img2_matches,homography)
            if num_inlier_points > max_inliers or max_inliers == 0:
                  max_inliers = num_inlier_points
                  best_inlier_indices = inlier_indices
                  best_homography = homography
      img1_points = img1_matches[best_inlier_indices].astype(np.float32)
      img2_points = img2_matches[best_inlier_indices].astype(np.float32)
      best_homography,_ = cv2.findHomography(img1_points,img2_points)
      #print(best_homography)
      return best_homography, best_inlier_indices

def homography_wrapper(rgb_img1,rgb_img2):
     corner_info1, corner_loc1, corner_img1 = detect_corners(rgb_img1)
     anms_corner_locs1, anms_corners_img1 = adaptive_non_max_supression(rgb_img1,corner_info1)
     features1 = corner_feature_descriptors(rgb_img1,anms_corner_locs1)
     corner_info2, corner_loc2, corner_img2 = detect_corners(rgb_img2)
     anms_corner_locs2, anms_corners_img2 = adaptive_non_max_supression(rgb_img2,corner_info2)
     features2 = corner_feature_descriptors(rgb_img2,anms_corner_locs2)
     matches = match_features(features1,features2)
     best_homography, best_inlier_indices = RANSAC(matches)
     ransac_matches = []
     for index in best_inlier_indices:
          ransac_matches.append(matches[index])
     #draw_feature_matches(anms_corners_img1,anms_corners_img2,ransac_matches)
     #print(len(best_inlier_indices))
     return best_homography

def warp_image(rgb_img1,rgb_img2,best_homography):
      h,w,c = rgb_img1.shape
      old_corners = np.array([[0,0,1],[0,h,1],[w,0,1],[w,h,1]]).T.astype(np.float32)
      new_corners = np.dot(best_homography,old_corners)
      X = new_corners[0]
      Y = new_corners[1]
      Z = new_corners[2]
      w_min = min(X/Z)
      w_max = max(X/Z)
      h_min = min(Y/Z)
      h_max = max(Y/Z)
      warping_correction_matrix = np.array([[1,0,-w_min],[0,1,abs(h_min)],[0,0,1]]).astype(np.float32)
      new_homography = np.dot(warping_correction_matrix,best_homography)
      new_w = int(w_max-w_min) + rgb_img2.shape[0] + 1
      new_h = int(h_max+abs(h_min)) + rgb_img2.shape[1] + 1
      warped_img1 = cv2.warpPerspective(rgb_img1.copy(),new_homography,dsize=(new_w,new_h))
      plt.imshow(warped_img1)
      plt.show()
      return warped_img1, int(abs(w_min)), int(abs(h_min))

def stitch_images(images):
      base_img = images[0].copy()
      for img in images[1:]:
            best_homography = homography_wrapper(base_img,img)
            new_img, w_min, h_min = warp_image(base_img,img,best_homography)
            for i in range(img.shape[0]):
                  for j in range(img.shape[1]):
                        new_img[w_min+i,h_min+j,:] = img[i,j,:]
                        #if np.linalg.norm(new_img[w_min+i,h_min+j,:]-np.array([0,0,0])) < 0.001:
                        #      new_img[w_min+i,h_min+j,:] = img[i,j,:]
            base_img = new_img
      base_img = cv2.resize(base_img,(1280,1024),interpolation=cv2.INTER_CUBIC)
      base_img = cv2.GaussianBlur(base_img,(3,3),0)
      plt.imshow(base_img)
      plt.show()

# def stitch_images(images):
#       output_img = np.zeros((1080,630,3))
#       warping_correction_matrix = np.array([[1,0,100],[0,1,150],[0,0,1]]).astype(np.float32)
#       warped_imgs = []
#       base_img = images[0]
#       base_img = cv2.resize(base_img,(800,800),interpolation=cv2.INTER_CUBIC)
#       warped_imgs.append(cv2.warpPerspective(base_img,warping_correction_matrix,(1080,630)))
#       for img in images[1:]:
#           img = cv2.resize(img,(800,800),interpolation=cv2.INTER_CUBIC)
#           best_homography = homography_wrapper(base_img,img)
#           H = np.dot(warping_correction_matrix,np.linalg.inv(best_homography))
#           warped_imgs.append(cv2.warpPerspective(img,H,(1080,630)))
#       for i in range(output_img.shape[0]):
#             for j in range(output_img.shape[1]):
#                   for img in warped_imgs:
#                         if not np.array_equal(img[i,j,:],np.array([0,0,0])):
#                               output_img[i,j,:] = img[i,j,:]
#       cv2.GaussianBlur(output_img,(3,3),0)
#       plt.imshow(output_img)
#       plt.show()
                              
       
      
                  
def main():
	# Add any Command Line arguments here
    # Parser = argparse.ArgumentParser()
    # Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')
    
    # Args = Parser.parse_args()
    # NumFeatures = Args.NumFeatures

	"""
	Read a set of images for Panorama stitching
	"""

	"""
	Corner Detection
	Save Corner detection output as corners.png
	"""

	"""
	Perform ANMS: Adaptive Non-Maximal Suppression
	Save ANMS output as anms.png
	"""

	"""
	Feature Descriptors
	Save Feature Descriptor output as FD.png
	"""

	"""
	Feature Matching
	Save Feature Matching output as matching.png
	"""


	"""
	Refine: RANSAC, Estimate Homography
	"""


	"""
	Image Warping + Blending
	Save Panorama output as mypano.png
	"""

    
if __name__ == '__main__':
    img1 = cv2.imread("Phase1/Data/Train/Set2/1.jpg")
    img2 = cv2.imread("Phase1/Data/Train/Set2/2.jpg")
    img3 = cv2.imread("Phase1/Data/Train/Set2/3.jpg")
    img4 = cv2.imread("Phase1/Data/Train/Set3/4.jpg")
    img5 = cv2.imread("Phase1/Data/Train/Set3/5.jpg")
    img6 = cv2.imread("Phase1/Data/Train/Set3/6.jpg")
    img7 = cv2.imread("Phase1/Data/Train/Set3/7.jpg")
    img8 = cv2.imread("Phase1/Data/Train/Set3/8.jpg")
    imgs = [img1,img2]
    #print(img1.shape)
    info,loc,corners_img = detect_corners(img1)
    locs,anms_corners_img1 = adaptive_non_max_supression(img1,info)
    features1 = corner_feature_descriptors(img1,locs)
#     img2 = cv2.imread("Phase1/Data/Train/Set1/2.jpg")
    img2 = cv2.resize(img2,[img1.shape[1],img1.shape[0]],cv2.INTER_CUBIC)
    #print(img2.shape)
    info,loc,corners_img = detect_corners(img2)
    locs,anms_corners_img2 = adaptive_non_max_supression(img2,info)
    features2 = corner_feature_descriptors(img2,locs)
    matches = match_features(features1,features2)
    #print(len(features1))
    #print(len(features2))
    #print(len(matches))
    #draw_feature_matches(anms_corners_img1,anms_corners_img2,matches)
    best_homography, best_inlier_indices = RANSAC(matches)
    #print(best_homography)
    ransac_matches = []
    for index in best_inlier_indices:
          ransac_matches.append(matches[index])
#     print(len(ransac_matches))
#     draw_feature_matches(anms_corners_img1,anms_corners_img2,ransac_matches)
    warp_image(img1,img2,best_homography)
    stitch_images(imgs)
    
 
