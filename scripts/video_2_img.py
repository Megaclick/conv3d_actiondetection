import cv2
import os 
vidcap = cv2.VideoCapture('../1.mp4')
success,image = vidcap.read()
count = 0
path_dir = '../phone_dataset/test3'

if not os.path.exists(path_dir):
    os.makedirs(path_dir)
while success:
  cv2.imwrite(path_dir+"/frame%d.jpg" % count, image)     # save frame as JPEG file      
  success,image = vidcap.read()
  print('frame:{}'.format(count), success)
  count += 1