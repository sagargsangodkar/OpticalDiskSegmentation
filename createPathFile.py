import os

path = './gt_c/'
fid = open('path_gt_list.txt','w')

for files in sorted(os.listdir(path)):
	#print(files)
	fid.write(path+files+'\n')

fid.close()
