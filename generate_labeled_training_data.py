#! /usr/bin/python3


## Generates labeled images using the lidar segementation and ground truth bounding boxes


from glob import glob
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import IPython
import parse_lidar
import utils
import csv
import classify_image
import imageio
import random
import time
import os
from scipy.spatial import Delaunay, ConvexHull


CAR_WORDS = ['minivan', 'sports car', 'car,', 'cab', 'taxi', 'convertible', 'limo',
             'jeep', 'landrover', 'R.V.', 'go-kart', 'dustcart', 'pickup',
             'snowplow', 'cassette player', 'Model T', 'ricksha', 'rickshaw']
             # 'moving van', #This one is iffy. sometimes random stuff is moving vans


# classes = ['Unknown', 'Compacts', 'Sedans', 'SUVs', 'Coupes',
#            'Muscle', 'SportsClassics', 'Sports', 'Super', 'Motorcycles',
#            'OffRoad', 'Industrial', 'Utility', 'Vans', 'Cycles',
#            'Boats', 'Helicopters', 'Planes', 'Service', 'Emergency',
#            'Military', 'Commercial', 'Trains']

SAVE_DIRECTORY = os.path.join('..','labeled_training_data')

LABELS_FILE = os.path.join(SAVE_DIRECTORY, 'labels.txt')

def get_labeled_outfile(image_path, i):
    """
    Returns the path where the image should be written
    """
    segments = os.path.normpath(imgpath).split(os.sep)
    img_number = segments[-1][0:4]
    outdir = os.path.join(SAVE_DIRECTORY, segments[-2], img_number)
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, str(i) + '.jpg')
    return outfile

def number_in_box(points, box):
    """
    Returns the number of given points that are within the given box

    The method used is to find the delaunay triangulation of the box, 
     then check if the simplex exists for each point
    """
    return np.sum(Delaunay(box).find_simplex(points) >= 0)

def classify_using_bounding_boxes(lidar, bboxes):
    """
    Returns the class of the img based on the bounding box
    """
    for bbox in bboxes:
        ignore_in_eval = bbox[10]
        if ignore_in_eval:
            continue

        vert3D, _, _ = utils.unpack_bbox(bbox, expansion = 0.3)
        vert3D = vert3D.transpose()
        num_in_bbox = number_in_box(lidar, vert3D)
        # print("num_in_box: ", num_in_bbox, "\t len(lidar): ", len(lidar))
        if num_in_bbox > 100 or num_in_bbox > len(lidar)/10:
            return "car"
    # IPython.embed()

    return "not"
        




files = glob('../rob599_dataset_deploy/trainval/*/*_image.jpg')
# files = glob('../rob599_dataset_deploy/trainval/e95739d4-4eeb-4087-b22f-851964073287/0025_image.jpg')

# files.sort()
# random.shuffle(files)
# classify_image.create_graph()
fig1 = plt.figure(1, figsize=(16, 9))
fig2 = plt.figure(2, figsize=(8, 8))

fig3 = plt.figure(3, figsize=(10,10))

with open(LABELS_FILE, 'w') as f:
    f.write('filepath, label\n')


plt.ion()

for i in range(len(files)):
    
    if i%100 == 0:
        print("Trial ", i, 'out of ', len(files))
        
    imgpath = files[i]
    # print(imgpath)
    img = plt.imread(imgpath)

    xyz = utils.get_lidar(imgpath)
    proj = utils.get_camera_projection(imgpath)
    bboxes = utils.get_bboxes(imgpath)


    fig1.clear()
    fig2.clear()
    fig3.clear()

    ax2 = Axes3D(fig2)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')


    ax1 = fig1.add_subplot(1, 1, 1)


    ax1.imshow(img)



    imgs_of_interest, pois = parse_lidar.get_imgs_and_clusters(img, xyz, np.array(proj))
    utils.plot_img_pois(ax1, pois, proj)
    ax1.axis('scaled')
    utils.plot_img_bboxes(ax1, bboxes, proj)

    for poi in pois:
        ax2.scatter(poi[:,0], poi[:,1], poi[:,2], marker='.', s=1)
    utils.plot_3d_bboxes(ax2, bboxes)


    car_count = 0
    num_fig = len(imgs_of_interest)
    

    for i in range(num_fig):
        saved_imgpath = get_labeled_outfile(imgpath, i)
        imageio.imwrite(saved_imgpath, imgs_of_interest[i])
        # label = classify_image.run_inference_on_image_path(saved_imgpath)
        label = classify_using_bounding_boxes(pois[i], bboxes)

        with open(LABELS_FILE, 'a') as f:
            f.write(saved_imgpath + ',' + label + '\n')

        ax = fig3.add_subplot(np.ceil(np.sqrt(num_fig)),np.ceil(np.sqrt(num_fig)),i+1)
        ax.imshow(imgs_of_interest[i])
        
        # for word in CAR_WORDS:
        #     if label.find(word) >= 0:
        #         found = True
        #         label = word.capitalize()
        #         break
        
        ax.set_title(label)
        ax.set_axis_off()

    plt.show()
    plt.pause(0.1)
    # IPython.embed()


    # with open('./outfile.txt','a') as f:
    #     f.write(imgpath[30:-10] + ',' + str(car_count) + '\n')



