# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os

from scipy import spatial

from utils1 import *
import math
import numpy as np
from pyntcloud import PyntCloud
import open3d as o3d
import matplotlib.pyplot as plt
from pandas import DataFrame
import copy
import numpy.linalg as la
import scipy.spatial as spt
from PIL import Image
from PIL import ImageDraw
from dxfwrite import DXFEngine as dxf
import time

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # 导入数据
    time_start=time.clock()
    pcd_whole_model = o3d.io.read_point_cloud(r'..\..\data\origin.pcd')
    # o3d.visualization.draw_geometries([pcd_whole_model])
    downpcd = pcd_whole_model.voxel_down_sample(voxel_size=0.05)
    uni_down_model = downpcd.uniform_down_sample(every_k_points=5)
    cl, ind = downpcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=15.0)
    display_inlier_outlier(downpcd, ind)
    inlier_cloud = downpcd.select_by_index(ind)
    coor = np.asarray(inlier_cloud.points)
    w, v = PCA(coor)
    # 在原点云中画图
    point_cloud_vector1 = v[:, 0]  # 点云主方向对应的向量，第一主成分
    point_cloud_vector2 = v[:, 1]  # 点云主方向对应的向量，第二主成分
    point_cloud_vector = v[:, 0:2]  # 点云主方向与次方向
    point_cloud_vector3 = np.cross(point_cloud_vector1, point_cloud_vector2)
    print('the main orientation of this pointcloud is: ', point_cloud_vector1)
    print('the main orientation of this pointcloud is: ', point_cloud_vector2)
    # print(type(point_cloud_vector2))
    # print('the main orientation of this pointcloud is: ', n3)
    scale = 50
    n1 = np.c_[point_cloud_vector1[0] * scale, point_cloud_vector1[1] * scale + 55, point_cloud_vector1[2] * scale + 70]
    n2 = np.c_[point_cloud_vector2[0] * scale, point_cloud_vector2[1] * scale + 55, point_cloud_vector2[2] * scale + 70]
    n3 = np.c_[point_cloud_vector3[0] * scale, point_cloud_vector3[1] * scale + 55, point_cloud_vector3[2] * scale + 70]
    print(n1[0])
    point = [[0, 55, 70], n1[0], n2[0], n3[0]]  # 画点：原点、第一主成分、第二主成分
    lines = [[0, 1], [0, 2], [0, 3]]  # 画出三点之间连线
    colors = [[1, 0, 0], [0, 0, 0], [0, 1, 0]]
    # 构造open3d中的LineSet对象，用于主成分显示
    line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(point), lines=o3d.utility.Vector2iVector(lines))
    line_set.colors = o3d.utility.Vector3dVector(colors)
    coor2, coor_proj = get_projection(coor, point_cloud_vector3, z0=np.array([[0], [0], [1]]))
    for i in range(len(coor2)):
        coor2[i, 2] = 70
    coor2_r_df = DataFrame(coor2)
    coor2_r_df.columns = ['x', 'y', 'z']  # 给选取到的数据 附上标题
    coor2_r_pynt = PyntCloud(coor2_r_df)  # 将points的数据 存到结构体中
    projection2_r1 = coor2_r_pynt.to_instance("open3d", mesh=False)  # 实例化
    for i in range(len(coor2)):
        coor2[i, 2] = 70
    coor2_r_df = DataFrame(coor2)
    coor2_r_df.columns = ['x', 'y', 'z']  # 给选取到的数据 附上标题
    coor2_r_pynt = PyntCloud(coor2_r_df)  # 将points的数据 存到结构体中
    projection2_r1 = coor2_r_pynt.to_instance("open3d", mesh=False)  # 实例化
    # o3d.visualization.draw_geometries([inlier_cloud, line_set, projection2_r1])
    # o3d.visualization.draw_geometries([projection2_r1])
    # 剖面
    point_cloud_2d_a = np.asarray(inlier_cloud.points)[:, 0:2]
    pointa11 = np.asarray(pcd_whole_model.points)
    point_cloud_2d_all = pointa11[:, 0:2]
    point_cloud_raw = np.genfromtxt(r'..\..\data\pou.txt',
                                    delimiter=" ")  # 为 xyz的 N*3矩阵
    coor_pou = point_cloud_raw[:, 0:3]
    side_range = (-50, 10)  # left-most to right-most
    fwd_range = (-10, 35)  # back-most to forward-most

    x_points = coor_pou[:, 0]
    y_points = coor_pou[:, 1]
    z_points = coor_pou[:, 2]
    print(x_points)

    f_filt = np.logical_and((x_points > fwd_range[0]), (x_points < fwd_range[1]))
    s_filt = np.logical_and((y_points > -side_range[1]), (y_points < -side_range[0]))
    filter = np.logical_and(f_filt, s_filt)
    indices = np.argwhere(filter).flatten()

    x_points = x_points[indices]
    y_points = y_points[indices]
    z_points = z_points[indices]

    res = 0.01  # 调参
    x_img = (-y_points / res).astype(np.int32)
    y_img = (-x_points / res).astype(np.int32)

    x_img -= int(np.floor(side_range[0] / res))
    y_img += int(np.ceil(fwd_range[1] / res))

    height_range = (-2, 0.5)

    pixel_values = np.clip(a=z_points,
                           a_min=height_range[0],
                           a_max=height_range[1])


    def scale_to_255(a, min, max, dtype=np.uint8):
        """ Scales an array of values from specified min, max range to 0-255
            Optionally specify the data type of the output (default is uint8)
        """
        return (((a - min) / float(max - min)) * 255).astype(dtype)


    pixel_values = scale_to_255(pixel_values, min=height_range[0], max=height_range[1])

    x_max = 1 + int((side_range[1] - side_range[0]) / res)
    y_max = 1 + int((fwd_range[1] - fwd_range[0]) / res)
    im = np.zeros([y_max, x_max], dtype=np.uint8)

    im[y_img, x_img] = pixel_values
    v0 = [-0.368, -0.92982579]
    edge = cv2.Canny(im, 50, 150)  # 画边框
    lines = cv2.HoughLines(edge, 1, np.pi / 180, 450)
    # image = Image.fromarray(edge)
    # draw = ImageDraw.Draw(image)
    # print(type(lines))
    rho0 = 0
    theta0 = 0
    for line in lines:
        rho, theta = line[0]
        rho0, theta0 = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = rho * a
        y0 = rho * b
        x1 = int(x0 + 50000 * (-b))  # 这里的1000是为了求延长线，其他数值也可以
        y1 = int(y0 + 50000 * a)
        x2 = int(x0 - 50000 * (-b))
        y2 = int(y0 - 50000 * a)
        #     cv2.line(im,(x1,y1),(x2,y2),(0,255,0),2)
    #     draw.line((x1, y1, x2, y2), 'cyan', width=7)
    # image.show()

theta = -math.pi / 2
rotation_m = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
point_cloud_r1 = np.dot(coor_pou[:, 0:2], rotation_m)
# point_cloud_r1=point_translate(point_cloud_r1)
point_cloud = point_rotation(point_cloud_r1, v0)
# plt.figure(figsize=(9, 5), dpi=100)
# plt.scatter(point_cloud[:, 0], point_cloud[:, 1], s=0.1)
# plt.show()

prefix = "..\..\data\segmentation_door7\\"
file_list = os.listdir(r'..\..\data\segmentation_door7')
list = []
for file in file_list:
    dir = prefix + file
    pcd = o3d.io.read_point_cloud(dir)
    points = np.asarray(pcd.points)
    points = points[:, 0:3]
    if (np.min(points[:, 2]) > 79.5):
        continue
    if (np.max(points[:, 2]) > 81.6 or np.max(points[:, 2]) < 81.1):
        continue
    coor_2d = points[:, 0:2]
    # two points which are fruthest apart will occur as vertices of the convex hull
    candidates = coor_2d[spatial.ConvexHull(coor_2d).vertices]

    # get distances between each pair of candidate points
    dist_mat = spatial.distance_matrix(candidates, candidates)
    if (np.max(dist_mat) < 0.5 or np.max(dist_mat) > 2.0):
        continue

    list.append(dir)

door_locs = []
door_width = []
for dir in list:
    pcd = o3d.io.read_point_cloud(dir)
    coor = np.asarray(pcd.points)[:, 0:2]
    x_min = np.min(coor[:, 0])
    x_max = np.max(coor[:, 0])
    y_min = np.min(coor[:, 1])
    y_max = np.max(coor[:, 1])
    flag = 0
    if y_max - y_min > x_max - x_min:
        flag = 1
    door_locs.append([(x_min + x_max) / 2, (y_min + y_max) / 2, flag])
    #     print(y_max-y_min,x_max-x_min)
    plt.scatter(coor[:, 0], coor[:, 1])
    candidates = coor[spatial.ConvexHull(coor).vertices]
    dist_mat = spatial.distance_matrix(candidates, candidates)
    if np.max(dist_mat) <= 0.9:
        door_width.append(np.max(dist_mat))
# print(door_locs)
pcd = o3d.io.read_point_cloud(r'..\..\data\segmentation_door7\\door36.pcd')
coor = np.asarray(pcd.points)[:, 0:2]
x_min = np.min(coor[:, 0])
x_max = np.max(coor[:, 0])
y_min = np.min(coor[:, 1])
y_max = np.max(coor[:, 1])
flag = 0
if y_max - y_min > x_max - x_min:
    flag = 1
door_locs.append([(x_min + x_max) / 2, (y_min + y_max) / 2, flag])
np.savetxt(r'..\..\data\door_locs7.txt', np.c_[door_locs],fmt='%f',delimiter=' ')

door1 = np.genfromtxt(r'..\..\data\door_locs7.txt', delimiter=" ")
door2 = np.dot(door1[:, 0:2], rotation_m)
# door2=point_translate(door2)
door2 = point_rotation(door2, v0)
point_cloud = np.array(point_cloud)
x_min = np.min(point_cloud[:, 0])
y_min = np.min(point_cloud[:, 1])
x_max = np.max(point_cloud[:, 0])
y_max = np.max(point_cloud[:, 1])
point_cloud = np.c_[point_cloud[:, 0] - x_min, point_cloud[:, 1] - y_min]
door2 = np.c_[door2[:, 0] - x_min, door2[:, 1] - y_min, door1[:, 2]]

a, c = oriented_search(point_cloud, 0.1, 1)
boundary3, boundary_index = boundary_coarse(a, c, 1)
corner1 = find_corner_3(c, boundary_index, width=1, height=4, direct=1, denoise=0)

# plt.figure(figsize=(9, 5), dpi=100)
# for ii in range(len(corner1)):
#     corner2 = corner1[ii]
#     corner2 = np.array(corner2)
#     #     print(corner2)
#     corner2 = corner2.astype(float)
#     points = 0.1 * corner2
#     for i in range(len(points) - 1):
#         plt.plot([points[i][0], points[i + 1][0]],
#                  [points[i][1], points[i + 1][1]], linewidth=1, c='black')
#     plt.plot([points[0][0], points[len(points) - 1][0]],
#              [points[0][1], points[len(points) - 1][1]], linewidth=1, c='black')
# plt.scatter(door2[:, 0], door2[:, 1], marker='*', c='lime')
# plt.show()

drawing = dxf.drawing('floor_plan.dxf')
drawing.add_layer('wall', color=7)
drawing.add_layer('door', color=3)
for ii in range(len(corner1)):
    corner2 = corner1[ii]
    corner2 = np.array(corner2)
    corner2 = corner2.astype(float)
    points = 0.1 * corner2 * 1000
    for i in range(len(points)):
        j = (i + 1) % (len(points))
        line1 = dxf.line((points[i][0], points[i][1]), (points[j][0], points[j][1]))
        line1['layer'] = 'wall'
        line1['color'] = 256
        line1['thickness'] = 1.0
        drawing.add(line1)
block = dxf.block(name='DOOR-01', basepoint=(500, 0))
block.add(dxf.line((0, 0), (0, -900)))
block.add(dxf.line((0, -900), (100, -900)))
block.add(dxf.line((100, -900), (100, 0)))
block.add(dxf.arc(900, (100, 0), 270, 360))
drawing.blocks.add(block)
# blockref = dxf.insert(blockname='DOOR-01', insert=(door3[0][0], door3[0][1])) # create a block-reference
# drawing.add(blockref) # add block-reference to drawing
# drawing.save()
block = dxf.block(name='DOOR-02', basepoint=(0, -500))
block.add(dxf.line((0, 0), (900, 0)))
block.add(dxf.line((900, 0), (900, -100)))
block.add(dxf.line((900, -100), (0, -100)))
block.add(dxf.arc(900, (0, -100), 270, 360))
drawing.blocks.add(block)
for idx in range(len(door2)):
    if door1[idx][2] == 0:
        blockref = dxf.insert(blockname='DOOR-02', insert=(door2[idx][0] * 1000, door2[idx][1] * 1000))
    else:
        blockref = dxf.insert(blockname='DOOR-01', insert=(door2[idx][0] * 1000, door2[idx][1] * 1000))
    blockref['layer'] = 'door'
    drawing.add(blockref)
drawing.save()
time_end=time.clock()
print("Floor plan is successfully generated!")
print("程序运行时间：%as",time_end-time_start)
