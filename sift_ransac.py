import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import sys




def sift_generator(img, img_id):

  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  sift = cv2.SIFT_create()
  keypoints, descriptors = sift.detectAndCompute(img, None)
  print("Number of SIFT features in {}: {}".format(img_id,len(keypoints)))

  # sketching the detected key points
  sift_image = cv2.drawKeypoints(gray, keypoints, np.array([]), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

  # save the image
  cv2.imwrite("Results/"+img_id+"_general_sift.jpg", sift_image)

  return keypoints, descriptors

def matcher(desc_s,desc_d):

  bf = cv2.BFMatcher()

  matches = bf.knnMatch(desc_s, desc_d, k=2)

  print("Number of Matches: {}".format(len(matches)))


  # Apply ratio test
  good = []
  for m,n in matches:
      if m.distance < 0.7*n.distance:
          good.append(m)
  
  print("Number of Good Matches: {}".format(len(good)))
  good = sorted(good, key = lambda x:x.distance)
  return good

def homographer(kp1, kp2, src_img, dst_img, good):
  MIN_MATCH_COUNT = 1

  if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

    print("M Matrix is: {}".format(M))

    


    matchesMask = mask.ravel().tolist()
    print("Number of RANSAC Matches: {}".format(sum(matchesMask)))

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
        
        seen=[]

        """
        Load the image file here through cv2.imread
        """
        img_id = img_path[:-4]

        if img_id[0]=='d' or img_id in seen: continue
        seen.append(img_id)
        img_name = os.path.join(img_dir, img_path)
        img_src = cv2.imread(img_name)          
        key_src, desc_src = sift_generator(img_src, img_id)
        
        for dest_path in img_list:
          dest_id = dest_path[:-4]
          if dest_id[0]=='s' or dest_id in seen: continue
          seen.append(dest_id)

          dest_name = os.path.join(img_dir, dest_path)
          img_dest = cv2.imread(dest_name)          
          key_dst, desc_dst = sift_generator(img_dest, dest_id)

          good = matcher(desc_src, desc_dst)

          img_match = cv2.drawMatches(img_src,key_src,img_dest,key_dst,good,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
          img_match_show = cv2.drawMatches(img_src,key_src,img_dest,key_dst,good[0:21],None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

          cv2.imwrite("Results/src_"+img_id[-1]+"_dst_"+dest_id[-1]+"_all_matches.jpg", img_match_show)

          img_dest, matchesMask = homographer(key_src, key_dst, img_src, img_dest, good)

          for i in range(len(matchesMask)):
            if sum(matchesMask[0:i])==10:
              index=i
              break

          

          draw_params = dict(
                            singlePointColor = None,
                            matchesMask = matchesMask[0:index], # draw only inliers
                            flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
          final_img = cv2.drawMatches(img_src,key_src,img_dest,key_dst,good[0:index],None,**draw_params)
          cv2.imwrite("Results/src_"+img_id[-1]+"_dst_"+dest_id[-1]+"_ransac_homography.jpg", final_img)
          


if __name__== "__main__":
  main()
  
