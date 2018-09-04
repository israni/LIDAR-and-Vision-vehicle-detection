#! /usr/bin/python3

from glob import glob
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import IPython


def get_lidar(imagepath):
    xyz = np.array(np.memmap(imagepath.replace('_image.jpg', '_cloud.bin'), dtype=np.float32))
    xyz.resize([3, xyz.size // 3])
    return xyz.transpose()

def get_camera_projection(imagepath):
    proj = np.array(np.memmap(imagepath.replace('_image.jpg', '_proj.bin'), dtype=np.float32))
    proj.resize([3, proj.size // 3])
    return proj

def get_bboxes(imagepath):
    
    try:
        #Convert to array as soon as possible. Otherwise potential data corruption
        bboxes = np.memmap(imagepath.replace('_image.jpg', '_bbox.bin'),
                                    dtype=np.float32)

    except:
        print('[*] bbox not found.')
        bboxes = np.array([], dtype=np.float32)
    bboxes.resize([bboxes.size // 11, 11])
    return np.copy(np.array(bboxes))
    


def rot(n, theta):
    n = n / np.linalg.norm(n, 2)
    K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    return np.identity(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K



def unpack_bbox(bbox, expansion=0.0):
    bbox = np.copy(bbox)
    n = bbox[0:3]
    theta = np.linalg.norm(n)
    n /= theta
    R = rot(n, theta)
    t = bbox[3:6]

    # size of the bbox
    sz = bbox[6:9] + expansion
    vert_3D, edges = get_bbox(-sz / 2, sz / 2)
    vert_3D = R @ vert_3D + t[:, np.newaxis]
    return vert_3D, edges, t



def get_bbox(p0, p1):
    '''
    Input:
    *   p0, p1
        (3)
        Corners of a bounding box represented in the body frame.

    Output:
    *   v
        (3, 8)
        Vertices of the bounding box represented in the body frame.
    *   e
        (2, 14)
        Edges of the bounding box. The first 2 edges indicate the `front` side
        of the box.
    '''
    v = np.array([[p0[0], p0[0], p0[0], p0[0], p1[0], p1[0], p1[0], p1[0]],
                  [p0[1], p0[1], p1[1], p1[1], p0[1], p0[1], p1[1], p1[1]],
                  [p0[2], p1[2], p0[2], p1[2], p0[2], p1[2], p0[2], p1[2]]])
    e = np.array([[2, 3, 0, 0, 3, 3, 0, 1, 2, 3, 4, 4, 7, 7],
                  [7, 6, 1, 2, 1, 2, 4, 5, 6, 7, 5, 6, 5, 6]], dtype=np.uint8)

    return v, e

def plot_img_bboxes(ax, bboxes, proj):
    colors = ['C{:d}'.format(i) for i in range(10)]
    for k, b in enumerate(bboxes):
        vert_3D, edges, t = unpack_bbox(b)
        
        vert_2D = proj @ np.vstack([vert_3D, np.ones(8)])
        vert_2D = vert_2D / vert_2D[2, :]
        
        clr = colors[k % len(colors)]
        ignore_in_eval = bool(b[10])
        if ignore_in_eval:
            continue
        for e in edges.T:
            ax.plot(vert_2D[0, e], vert_2D[1, e], color=clr)

def plot_img_lidar(ax, xyz, proj):

    uv = proj @ np.vstack([xyz.transpose(), np.ones_like(xyz[:,0])])
    uv = uv / uv[2, :]
    clr = np.linalg.norm(xyz, axis=1)
    ax.scatter(uv[0, :], uv[1, :], c=clr, marker='.', s=1)

def plot_img_pois(ax, pois, proj):
    for points in pois:
        plot_img_lidar(ax, points, proj)

def plot_3d_bboxes(ax, bboxes):
    classes = ['Unknown', 'Compacts', 'Sedans', 'SUVs', 'Coupes',
           'Muscle', 'SportsClassics', 'Sports', 'Super', 'Motorcycles',
           'OffRoad', 'Industrial', 'Utility', 'Vans', 'Cycles',
           'Boats', 'Helicopters', 'Planes', 'Service', 'Emergency',
           'Military', 'Commercial', 'Trains']

    colors = ['C{:d}'.format(i) for i in range(10)]
    for k, b in enumerate(bboxes):
        vert_3D, edges, t = unpack_bbox(b)
        
        
        clr = colors[k % len(colors)]
        for e in edges.T:
            ax.plot(vert_3D[0, e], vert_3D[1, e], vert_3D[2, e], color=clr)
        # IPython.embed()
        # ax.scatter(vert_3D[0,:], vert_3D[1,:], vert_3D[2,:])

        c = classes[int(b[9])]
        ignore_in_eval = bool(b[10])
        if ignore_in_eval:
            ax.text(t[0], t[1], t[2], c, color='w')
        else:
            ax.text(t[0], t[1], t[2], c)

        ax.auto_scale_xyz([-40, 40], [-40, 40], [0, 80])
        ax.view_init(elev=-30, azim=-90)

    for e in np.identity(3):
        ax.plot([0, e[0]], [0, e[1]], [0, e[2]], color=e)


