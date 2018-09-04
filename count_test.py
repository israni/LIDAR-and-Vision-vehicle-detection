#! /usr/bin/python3
from glob import glob
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import IPython
import parse_lidar
import csv
import label_image
import imageio
import random
import time
import utils


CAR_WORDS = ['car']



files = glob('../rob599_dataset_deploy/test/*/*_image.jpg')
files.sort()

graph = label_image.load_graph('./retrained_graph.pb')
labels = label_image.load_labels('./retrained_labels.txt')

fig1 = plt.figure(1, figsize=(16, 9))
fig3 = plt.figure(3, figsize=(10,10))

with open('./outfile.txt','w') as f:
    f.write('guid/image,N\n')

plt.ion()

for i in range(len(files)):
    if i%1 == 0:
        print("Trial ", i, 'out of ', len(files))
        
    imgpath = files[i]
    img = plt.imread(imgpath)

    fig1.clear()
    fig3.clear()

    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.imshow(img)
    

    xyz = utils.get_lidar(imgpath)
    xyz = parse_lidar.densify_lidar(xyz)
    proj = utils.get_camera_projection(imgpath)


    #==============================================================================
    #   Analyze lidar data, pulling out images of point clusters that might be cars
    #==============================================================================
    imgs_of_interest, clusters = parse_lidar.get_imgs_and_clusters(img, xyz, np.array(proj))
    
    for cluster in clusters:
        utils.plot_img_lidar(ax1, cluster, proj)

    car_count = 0
    num_fig = len(imgs_of_interest)

    for i in range(num_fig):
        tmp_imgpath = '/tmp/img.jpg'
        imageio.imwrite(tmp_imgpath, imgs_of_interest[i])

        #=========================================================
        #    Classify image using tensorflow
        #=========================================================
        label = labels[label_image.classify_image(graph, tmp_imgpath)]

        for word in CAR_WORDS:
            if label.find(word) >= 0:
                car_count += 1
                break


        ax = fig3.add_subplot(np.ceil(np.sqrt(num_fig)), np.ceil(np.sqrt(num_fig)), i+1)
        ax.imshow(imgs_of_interest[i])

        
        for word in CAR_WORDS:
            if label.find(word) >= 0:
                found = True
                label = word.capitalize()
                break
        
        ax.set_title(label)
        ax.set_axis_off()
    plt.show()
    plt.pause(0.1)


    with open('./outfile.txt','a') as f:
        f.write(imgpath[30:-10] + ',' + str(car_count) + '\n')



