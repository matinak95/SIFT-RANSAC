import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import sys




def sift_generator(img, img_id):

  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  sift = cv2.SIFT_create()
  keypoints, descriptors = sift.detectAndCompute(img, None)

  # sketching the detected key points
  sift_image = cv2.drawKeypoints(gray, keypoints, np.array([]), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
  #cv2.drawKeypoints(gray, keypoints, img)

  
  # # show the image
  cv2.imshow('image', sift_image)
  cv2.waitKey()


  # save the image
  cv2.imwrite("Results/"+img_id+"_general_sift.jpg", sift_image)

  return keypoints, descriptors

def matcher(desc_s,desc_d):


  bf = cv2.BFMatcher()

  matches = bf.knnMatch(desc_s, desc_d, k=2)


  # Apply ratio test
  good = []
  for m,n in matches:
      if m.distance < 0.7*n.distance:
          good.append(m)
  
  return good

def homographer(kp1, kp2, src_img, dst_img, good):
  MIN_MATCH_COUNT = 1

  if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w,d = src_img.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    dst_img = cv2.polylines(dst_img,[np.int32(dst)],True,255,3, cv2.LINE_AA)

  else:
      print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
      matchesMask = None
      sys.exit(0)
  
  return dst_img, matchesMask







def main():

    img_dir = './HW3_Data'
  
    img_list = os.listdir(img_dir)

    for img_path in img_list:

        """
        Load the image file here through cv2.imread
        """
        img_id = img_path[:-4]
        if img_id[0]=='d': continue
        img_name = os.path.join(img_dir, img_path)
        img_src = cv2.imread(img_name)          
        key_src, desc_src = sift_generator(img_src, img_id)
        
        for dest_path in img_list:
          dest_id = dest_path[:-4]
          if dest_id[0]=='s': continue

          dest_name = os.path.join(img_dir, dest_path)
          img_dest = cv2.imread(dest_name)          
          key_dst, desc_dst = sift_generator(img_dest, dest_id)



          key_dst, desc_dst = sift_generator(img_dest, dest_id)

          good = matcher(desc_src, desc_dst)

          img_match = cv2.drawMatches(img_src,key_src,img_dest,key_dst,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

          cv2.imwrite("Results/src_"+img_id[-1]+"_dst_"+dest_id[-1]+"_all_matches.jpg", img_match)
          # cv2.imshow("image", img_match)
          # cv2.waitKey()

          img_dest, matchesMask = homographer(key_src, key_dst, img_src, img_dest, good)

          draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                            singlePointColor = None,
                            matchesMask = matchesMask, # draw only inliers
                            flags = 2)
          final_img = cv2.drawMatches(img_src,key_src,img_dest,key_dst,good,None,**draw_params)
          cv2.imwrite("Results/src_"+img_id[-1]+"_dst_"+dest_id[-1]+"_ransac_homography.jpg", final_img)
          
          # cv2.imshow('image', final_img)
          # cv2.waitKey()

if __name__== "__main__":
  main()
  
