import os 
import sys 
import random
import cv2 
from scipy import ndimage 

def rnd_brightness(image):

	img = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

	img[:,:,2] = num_rand*img[:,:,2]
	new_img = cv2.cvtColor(img,cv2.COLOR_HSV2BGR)
	return new_img

def rnd_scal(image):
	rotated_img = ndimage.rotate(image,num_rand_rotate,reshape=False)
	return rotated_img

if len(sys.argv)!=5:
	print('origen - destino - fun')
	print(len(sys.argv))
	sys.exit()

if len(sys.argv) == 5:

	for total in range(int(sys.argv[4])):

		path_dir = '../phone_dataset/{}/'.format(str(int(sys.argv[1])+total))
		path_out = '../phone_dataset/{}/'.format(str(int(sys.argv[2])+total))
		num_rand = random.uniform(0.3,0.7)
		num_rand_rotate = random.uniform(-0.7,0.7)*10


		if not os.path.exists(path_out):
		    os.makedirs(path_out)
		    print('created a folder in '+path_out)
		if sys.argv[3] == str(0):    
			for i in range(30):
				image = cv2.imread(path_dir + 'frame{}.jpg'.format(i))
				new_img = rnd_brightness(image)
				cv2.imwrite(path_out+'frame{}.jpg'.format(i), new_img)  
			print('ended random bright' + str(num_rand))
		elif sys.argv[3] == str(1):
			for i in range(30):
				image = cv2.imread(path_dir + 'frame{}.jpg'.format(i))
				new_img = rnd_scal(image)
				cv2.imwrite(path_out+'frame{}.jpg'.format(i), new_img)  
			print('ended random rotation' + str(num_rand_rotate))

		else: 
			print('carpeta_origen - carpeta_destino - funcion - cantidad iteraciones')
			print(len(sys.argv))
			sys.exit()
