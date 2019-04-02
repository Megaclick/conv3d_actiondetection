import os
import quicksort
import sys

def fun1(num):
	file_path = '../phone_dataset/{}/'.format(str(num))
	count = 0
	file_name = 'e-000'
	num_list = []
	for file in os.listdir(file_path):
		a,b = file.split('.')
		c,d = a.split('-')
		num_list.append(int(d))
		#os.rename(file_path+file,file_path+'frame{}.jpg'.format(count))
		count +=1
	print(num_list)
	quicksort.quickSort(num_list)
	print(num_list)
	count = 0
	for i in num_list:
		if num_list[count] <10:
			os.rename(file_path+'e-0000{}.jpg'.format(i),file_path+'frame{}.jpg'.format(count))
		elif num_list[count] <100:
			os.rename(file_path+'e-000{}.jpg'.format(i),file_path+'frame{}.jpg'.format(count))
		else:
			os.rename(file_path+'e-00{}.jpg'.format(i),file_path+'frame{}.jpg'.format(count))
		count +=1
def fun2(num):
	file_path = '../phone_dataset/{}/'.format(str(num))
	count = 0
	file_name = 'e-000'
	num_list = []
	for file in os.listdir(file_path):
		a,b = file.split('.')
		c,d = a.split('e')
		num_list.append(int(d))
		#os.rename(file_path+file,file_path+'frame{}.jpg'.format(count))
		count +=1
	print(num_list)
	quicksort.quickSort(num_list)
	print(num_list)
	count = 0
	for i in num_list:
		if num_list[count] <10:
			os.rename(file_path+'frame{}.jpg'.format(i),file_path+'frame{}.jpg'.format(count))
		elif num_list[count] <100:
			os.rename(file_path+'frame{}.jpg'.format(i),file_path+'frame{}.jpg'.format(count))
		else:
			os.rename(file_path+'frame{}.jpg'.format(i),file_path+'frame{}.jpg'.format(count))
		count +=1
def fun3(num):
	file_path = '../phone_dataset/{}/'.format(str(num))
	count = 0
	num_list = []
	prev_name = ''
	for file in os.listdir(file_path):
	    a,b = file.split('.')
	    c,d = a.split('ms-')
	    num_list.append(d)
	    #os.rename(file_path+file,file_path+'frame{}.jpg'.format(count))
	    count +=1
	    if count == 1: 
	        prev_name = c
	print(num_list)
	quicksort.quickSort(num_list)
	print(num_list)
	print(c)
	count = 0
	for i in num_list:
	    if int(num_list[count]) <100000:
	        os.rename(file_path+c+'ms-{}.jpg'.format(i),file_path+'frame{}.jpg'.format(count))
	        count +=1

if len(sys.argv)!=3:
	print('gg')
	print(len(sys.argv))
	sys.exit()
else:
	if sys.argv[1] == 'fun1':
		fun1(sys.argv[2])
	elif sys.argv[1] == 'fun2':
		fun2(sys.argv[2])
	elif sys.argv[1] == 'fun3':
		fun3(sys.argv[2])
	else:
		print('gg')
