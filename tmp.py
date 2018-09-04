#! /usr/bin/python3
from glob import glob
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import IPython
import parse_lidar
import csv
import classify_image
import label_image
import imageio
import utils


# CAR_WORDS = ['minivan', 'sports car', 'car,', 'cab', 'taxi', 'convertible', 'limo',
#              'jeep', 'landrover', 'R.V.']

CAR_WORDS = ['car']





classes = ['Unknown', 'Compacts', 'Sedans', 'SUVs', 'Coupes',
           'Muscle', 'SportsClassics', 'Sports', 'Super', 'Motorcycles',
           'OffRoad', 'Industrial', 'Utility', 'Vans', 'Cycles',
           'Boats', 'Helicopters', 'Planes', 'Service', 'Emergency',
           'Military', 'Commercial', 'Trains']

# files = glob('deploy/*/*/*_image.jpg')
files = glob('../rob599_dataset_deploy/trainval/*/*_image.jpg')
# files = glob('../rob599_dataset_deploy/test/0815cc1e-9a0c-4875-a5ca-784ef1a32bba/0008_image.jpg')
# files = glob('../rob599_dataset_deploy/trainval/94835623-ca6c-4ac4-82d4-0e63c1b7c16a/0079_image.jpg')


num_cars = {}
with open('../rob599_dataset_deploy/trainval/num_cars.csv') as true_cars_file:
    reader = csv.reader(true_cars_file, delimiter=',')
    next(reader)
    for row in reader:
        num_cars[row[0]] = int(row[1])

# IPython.embed()

idx = np.random.randint(0, len(files))
snapshot = files[idx]
print(snapshot)

img = plt.imread(snapshot)

xyz = utils.get_lidar(snapshot)
proj = utils.get_camera_projection(snapshot)
bboxes = utils.get_bboxes(snapshot)


xyz = parse_lidar.densify_lidar(xyz)


# centers = parse_lidar.get_points_of_interest(xyz)
imgs_of_interest = parse_lidar.get_potential_car_images(img, xyz, np.array(proj))
clusters = parse_lidar.get_points_of_interest(xyz)
inliers = parse_lidar.lidar_mask(xyz)
not_lines = parse_lidar.mask_out_long_smooth_lines(xyz)
not_hor = parse_lidar.mask_out_horizontal(xyz)
# line_mask = parse_lidar.mask_out_long_smooth_lines(xyz)
# IPython.embed()
# uv = uv[:, inliers]





fig1 = plt.figure(1, figsize=(16, 9))
ax1 = fig1.add_subplot(1, 1, 1)
ax1.imshow(img)
for cluster in clusters:
    utils.plot_img_lidar(ax1, cluster, proj)
# utils.plot_img_lidar(ax1, xyz[inliers, :], proj)
# utils.plot_img_lidar(ax1, xyz[np.logical_not(not_hor), :], proj)



# ax1.scatter(uv[0, :], uv[1, :], marker='.', s=1)
# ax1.scatter(uv[0, line_mask], uv[1, line_mask], marker='.', s=2)
ax1.axis('scaled')
fig1.tight_layout()



fig2 = plt.figure(2, figsize=(8, 8))
ax2 = Axes3D(fig2)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')

# classify_image.create_graph()
graph = label_image.load_graph('./retrained_graph.pb')
labels = label_image.load_labels('./retrained_labels.txt')

car_count = 0


fig3 = plt.figure(3, figsize=(10,10))
num_fig = len(imgs_of_interest)
for i in range(num_fig):
    ax = fig3.add_subplot(np.ceil(np.sqrt(num_fig)),np.ceil(np.sqrt(num_fig)),i+1)
    ax.imshow(imgs_of_interest[i])

    imgpath = '/tmp/img.jpg'
    imageio.imwrite(imgpath, imgs_of_interest[i])

    # IPython.embed()
    
    # label = classify_image.run_inference_on_image_path(imgpath)
    label = labels[label_image.classify_image(graph, imgpath)]
    found = False
    for word in CAR_WORDS:
        if label.find(word) >= 0:
            found = True
            label = word.capitalize()
            car_count += 1
            break
    # IPython.embed()
    ax.set_title(label)
    ax.set_axis_off()


# IPython.embed()
# ax2.scatter(centers[:,0], centers[:,1], centers[:,2], marker='x', s=100)

step = 5
# ax2.scatter(xyz[0, ::step], xyz[1, ::step], xyz[2, ::step], \
#     c=clr[::step], marker='.', s=1)
# ax2.scatter(xyz[0, inliers], xyz[1, inliers], xyz[2, inliers], marker='.', s=1)
print(len(clusters), "clusters found")
for cluster in clusters:
    # IPython.embed()
    ax2.scatter(cluster[:,0], cluster[:,1], cluster[:,2], marker='.', s=1)

utils.plot_img_bboxes(ax1, bboxes, proj)
utils.plot_3d_bboxes(ax2, bboxes)

id = snapshot[0:-10]
id = id[34:]

try:
    print('ground truth: ', num_cars[id])
except:
    pass

    
print('we count:     ', car_count)
plt.show()


