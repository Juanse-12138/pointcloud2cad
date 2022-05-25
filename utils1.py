import math

import cv2
import numpy as np
from dxfwrite.dimlines import dimstyles
from pyntcloud import PyntCloud
import open3d as o3d
import matplotlib.pyplot as plt
from pandas import DataFrame
import copy
import numpy.linalg as la
import scipy.spatial as spt
import random


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    # o3d.visualization.draw_geometries([inlier_cloud,outlier_cloud])


def PCA(data, correlation=False, sort=True):
    average_data = np.mean(data, axis=0)  # 求 NX3 向量的均值
    decentration_matrix = data - average_data  # 去中心化
    H = np.dot(decentration_matrix.T, decentration_matrix)  # 求解协方差矩阵 H
    eigenvectors, eigenvalues, eigenvectors_T = np.linalg.svd(H)  # SVD求解特征值、特征向量
    # 屏蔽结束

    if sort:
        sort = eigenvalues.argsort()[::-1]  # 降序排列
        eigenvalues = eigenvalues[sort]  # 索引
        eigenvectors = eigenvectors[:, sort]

    return eigenvalues, eigenvectors


def get_projection(coor, n1, z0=np.array([[0], [0], [1]])):
    #     function:get the projection of the pointcloud
    #     coor--------点云坐标
    #     n1----------投影平面法向量
    #     z0----------需要旋转到的坐标平面，默认为xoy平面
    n2 = np.array([[n1[0]], [n1[1]], [n1[2]]])
    n2_t = n2.T
    t = -np.dot(coor, n2)
    coor2 = np.dot(t, n2_t) + coor
    coor2_df = DataFrame(coor2)
    coor2_df.columns = ['x', 'y', 'z']  # 给选取到的数据 附上标题
    coor2_pynt = PyntCloud(coor2_df)  # 将points的数据 存到结构体中
    projection2 = coor2_pynt.to_instance("open3d", mesh=False)
    v_pivot = np.array([n2[1] * z0[2] - n2[2] * z0[1], n2[2] * z0[0] - n2[0] * z0[2], n2[0] * z0[1] - n2[1] * z0[0]])
    lv = np.sqrt(np.dot(v_pivot.T, v_pivot))
    # print(lv)
    v_pivot = v_pivot / lv
    ln2 = np.sqrt(np.dot(n2.T, n2))
    lz0 = np.sqrt(np.dot(z0.T, z0))
    dianji2 = np.dot(n2.T, z0)
    cos_1 = dianji2 / (ln2 * lz0)
    angle_rad0 = np.arccos(cos_1)
    sin_1 = np.sin(angle_rad0)
    I = np.identity(3)
    A1 = np.array([[v_pivot[0, 0] * v_pivot[0, 0], v_pivot[0, 0] * v_pivot[1, 0], v_pivot[0, 0] * v_pivot[2, 0]],
                   [v_pivot[1, 0] * v_pivot[0, 0], v_pivot[1, 0] * v_pivot[1, 0], v_pivot[1, 0] * v_pivot[2, 0]],
                   [v_pivot[2, 0] * v_pivot[0, 0], v_pivot[2, 0] * v_pivot[1, 0], v_pivot[2, 0] * v_pivot[2, 0]]])
    A2 = np.array(
        [[0, -v_pivot[2, 0], v_pivot[1, 0]], [v_pivot[2, 0], 0, -v_pivot[0, 0]], [-v_pivot[1, 0], v_pivot[0, 0], 0]])
    M = A1 + cos_1 * (I - A1) + sin_1 * A2
    coor2_r = np.dot(coor2, M.T)
    coor2_r_df = DataFrame(coor2_r)
    coor2_r_df.columns = ['x', 'y', 'z']  # 给选取到的数据 附上标题
    coor2_r_pynt = PyntCloud(coor2_r_df)  # 将points的数据 存到结构体中
    projection2_r1 = coor2_r_pynt.to_instance("open3d", mesh=False)  # 实例化
    return coor2_r, projection2_r1


def point_translate(point_cloud):
    point_cloud=np.array(point_cloud)
    x_min = np.min(point_cloud[:, 0])
    y_min = np.min(point_cloud[:, 1])
    x_max = np.max(point_cloud[:, 0])
    y_max = np.max(point_cloud[:, 1])
    # print(x_min,x_max,y_min,y_max)
    point_cloud = np.c_[point_cloud[:, 0] - x_min, point_cloud[:, 1] - y_min]
    return point_cloud


def point_rotation(points, n1):  # 二维点集旋转 ----输入点集和旋转的向量，按原点旋转
    n0 = np.array([0, 1])
    n1 = np.array(n1)
    l1 = np.sqrt(n1.dot(n1))
    l0 = np.sqrt(n0.dot(n0))
    cos_r = n1.dot(n0) / (l1 * l0)
    theta = np.arccos(cos_r)
    rotation_m = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    points=np.asarray(points)[:,0:2]
    return np.dot(points,rotation_m)


def point_scatter(point_cloud,wid=5,hei=9):
    point_cloud=np.array(point_cloud)
    x_max = np.max(point_cloud[:, 0])
    y_max = np.max(point_cloud[:, 1])
    x_min = np.min(point_cloud[:, 0])
    y_min = np.min(point_cloud[:, 1])
    x_range=x_max-x_min
    y_range=y_max-y_min
    plt.figure(figsize=(wid,hei),dpi=100)
    plt.scatter(point_cloud[:,0], point_cloud[:,1],s=0.1)
    plt.show()


def oriented_search(point_cloud_2d, space=0.05, dense=0):
    # 参数定义
    # space = 0.05  # 设置网络划分的大小
    # dense = 0  # 设置网格密度大小,即多少个点表示该网格存在
    point_cloud_2d=np.array(point_cloud_2d)
    x_max=np.max(point_cloud_2d[:,0])
    y_max=np.max(point_cloud_2d[:,1])
    x_min=np.min(point_cloud_2d[:,0])
    y_min=np.min(point_cloud_2d[:,1])
    point_cloud_2d = point_cloud_2d[point_cloud_2d[: ,0].argsort()]
    point_is_exist = []
    point_divided=[]
    start = 0
    start_x = 0
    list1 = []
    list21=[]
    list22=[]
    for x in np.arange(x_min-space, x_max+space, space):
        list1.clear()
        list22.clear()
        while start_x<point_cloud_2d.shape[0] and x + space >= point_cloud_2d[start_x,0] >= x:
            start_x += 1
        arr = np.array(point_cloud_2d[start:start_x,:])
        start = start_x
        arr = arr[arr[: ,1].argsort()]
        start_y = 0
        for y in np.arange(y_min-space, y_max+space, space):
            list21.clear()
            count = 0
            while start_y<arr.shape[0] and y + space >= arr[start_y,1] >= y:
                list21.append(arr[start_y])
                count += 1
                start_y += 1
            if count > dense:
                list1.append(True)
            else:
                list1.append(False)
            list22.append(copy.deepcopy(list21))
        point_is_exist.append(copy.deepcopy(list1))
        point_divided.append(copy.deepcopy(list22))
    return point_is_exist,point_divided


def boundary_coarse(point_exist, point_divide,dense=5):
    boundary_index = []
    boundary_c = []
    for i in range(1, len(point_exist) - 1):
        for j in range(1, len(point_exist[0]) - 1):
            if point_exist[i][j]:
                if not (point_exist[i][j + 1] and point_exist[i + 1][j + 1] and point_exist[i + 1][j] and
                        point_exist[i + 1][j - 1] and point_exist[i][j - 1] and point_exist[i - 1][j - 1] and
                        point_exist[i - 1][j] and point_exist[i - 1][j + 1]):
                    boundary_index.append([i, j])
    for i in range(0, len(boundary_index)):  # 对于每一个符合边界条件特征的区域的索引值
        arr = point_divide[boundary_index[i][0]][boundary_index[i][1]]
#         rang=min(dense,len(arr))
        rang=len(arr)
        for j in range(0, rang):
            #index=random.randint(0,len(arr)-1)
            boundary_c.append(arr[j])
    return boundary_c,boundary_index


def path_track(point_divided, boundary_index, boundary_value, i, j, width=1, height=5, direct=1, denoise=0):
    m = len(point_divided)
    n = len(point_divided[0])
    k = max(width, height)
    direction = direct  # 1--向右，2--向上，3--向下，4--向左
    corner_points = []
    count = 0
    corner_points.append([i - k, j - k])

    while count < len(boundary_index):
        upward = boundary_value[i + 1:i + height + 1, j - width:j + width + 1]
        upcount = np.sum(upward == 1)
        downward = boundary_value[i - height:i, j - width:j + width + 1]
        downcount = np.sum(downward == 1)
        leftward = boundary_value[i - width:i + width + 1, j - height:j]
        leftcount = np.sum(leftward == 1)
        rightward = boundary_value[i - width:i + width + 1, j + 1:j + height]
        rightcount = np.sum(rightward == 1)
        # if upcount < width and downcount < width and leftcount < width and rightcount < width: break
        if upcount < 1 and downcount < 1 and leftcount < 1 and rightcount < 1: break
        if direction == 1:  # 向右
            if i < m + k - 1 and upcount > denoise:
                i += 1
                for jj in range(j - width, j + width + 1):
                    boundary_value[i][jj] = 2
            else:
                corner_points.append([i - k, j - k])
                direction = 2 if leftcount > rightcount else 3
        elif direction == 4:  # 向左
            if i > k and downcount > denoise:
                i -= 1
                for jj in range(j - width, j + width + 1):
                    boundary_value[i][jj] = 2
            else:
                corner_points.append([i - k, j - k])
                direction = 2 if leftcount > rightcount else 3
        elif direction == 2:  # 向下
            if j > k and leftcount > denoise:
                j -= 1
                for ii in range(i - width, i + width + 1):
                    boundary_value[ii][j] = 2
            else:
                corner_points.append([i - k, j - k])
                direction = 1 if upcount > downcount else 4
        else:
            if j < n + k - 1 and rightcount > denoise:
                j += 1
                for ii in range(i - width, i + width + 1):
                    boundary_value[ii][j] = 2
            else:
                corner_points.append([i - k, j - k])
                direction = 1 if upcount > downcount else 4
        count += 1
    return corner_points


def find_corner_3(point_divided, boundary_index, width=1, height=5, direct=1, denoise=0):
    m = len(point_divided)
    n = len(point_divided[0])
    k = max(width, height)
    dx = [-1, 0, 1, 1, 1, 0, -1, -1]
    dy = [1, 1, 1, 0, -1, -1, -1, 0]
    boundary_value = np.zeros((m + 2 * k, n + 2 * k))
    for a in range(len(boundary_index)):
        arr = boundary_index[a]
        boundary_value[arr[0] + k, arr[1] + k] = 1
    boundary_index = np.array(boundary_index)
    boundary_index = boundary_index[boundary_index[:, 1].argsort()]
    i = boundary_index[-1][0] + k
    j = boundary_index[-1][1] + k
    boundary_value[i][j] = 2
    # 视为左下角角点，取xmin, ymin
    # arr = point_divided[i - k][j - k]
    # point_index = arr.min(axis=0)
    # point = arr[0]
    corner_points_all = []
    corner_points_all.append(
        path_track(point_divided, boundary_index, boundary_value, i, j, width, height, direct, denoise))
    #     while np.sum(boundary_value == 1) > 0:
    boundary_index2 = []
    for ii in range(len(boundary_index)):
        arr = boundary_index[ii]
        if boundary_value[arr[0] + k, arr[1] + k] == 1:
            boundary_index2.append(arr)
    boundary_index = copy.deepcopy(boundary_index2)
    boundary_index = np.array(boundary_index)
    # boundary_index = boundary_index[boundary_index[:, 1].argsort()]
    counts=np.bincount(boundary_index[:, 0])
    i=np.argmax(counts)
    j_arr=np.where(boundary_index[:, 0]==i)
    j=boundary_index[j_arr[0][0],1]+k
    i=boundary_index[j_arr[0][0],0]+k
    boundary_value[i][j] = 2
    corner_points_all.append(
        path_track(point_divided, boundary_index, boundary_value, i, j, width, height, 3, denoise))

    return corner_points_all