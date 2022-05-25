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


def oriented_search(point_cloud_2d, space=0.05, dense=0):
    # 参数定义
    # space = 0.05  # 设置网络划分的大小
    # dense = 0  # 设置网格密度大小,即多少个点表示该网格存在
    point_cloud_2d = np.array(point_cloud_2d)
    x_max = np.max(point_cloud_2d[:, 0])
    y_max = np.max(point_cloud_2d[:, 1])
    x_min = np.min(point_cloud_2d[:, 0])
    y_min = np.min(point_cloud_2d[:, 1])
    point_cloud_2d = point_cloud_2d[point_cloud_2d[:, 0].argsort()]
    point_is_exist = []
    point_divided = []
    start = 0
    start_x = 0
    list1 = []
    list21 = []
    list22 = []
    for x in np.arange(x_min - space, x_max + space, space):
        list1.clear()
        list22.clear()
        while start_x < point_cloud_2d.shape[0] and x + space >= point_cloud_2d[start_x, 0] >= x:
            start_x += 1
        arr = np.array(point_cloud_2d[start:start_x, :])
        start = start_x
        arr = arr[arr[:, 1].argsort()]
        start_y = 0
        for y in np.arange(y_min - space, y_max + space, space):
            list21.clear()
            count = 0
            while start_y < arr.shape[0] and y + space >= arr[start_y, 1] >= y:
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
    return point_is_exist, point_divided


def boundary_coarse(point_exist, point_divide, dense=5):
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
        rang = min(dense, len(arr))
        for j in range(0, rang):
            # index=random.randint(0,len(arr)-1)
            boundary_c.append(arr[j])
    return boundary_c, boundary_index


def boundary_coarse2(point_exist, point_divide, dense=5):
    boundary_index = []
    boundary_c = []
    dx = [-1, 0, 1, 1, 1, 0, -1, -1]
    dy = [1, 1, 1, 0, -1, -1, -1, 0]
    cnt0 = 3
    for i in range(1, len(point_exist) - 1):
        for j in range(1, len(point_exist[0]) - 1):
            if point_exist[i][j]:
                cnt = 0
                for dd in range(8):
                    if not (point_exist[dx[dd]][dy[dd]]):
                        cnt += 1
                if cnt >= cnt0:
                    boundary_index.append([i, j])
    for i in range(0, len(boundary_index)):  # 对于每一个符合边界条件特征的区域的索引值
        arr = point_divide[boundary_index[i][0]][boundary_index[i][1]]
        rang = min(dense, len(arr))
        for j in range(0, rang):
            # index=random.randint(0,len(arr)-1)
            boundary_c.append(arr[j])
    return boundary_c, boundary_index


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
    # print(M)
    coor2_r = np.dot(coor2, M.T)
    # print(coor2_r)
    coor2_r_df = DataFrame(coor2_r)
    coor2_r_df.columns = ['x', 'y', 'z']  # 给选取到的数据 附上标题
    coor2_r_pynt = PyntCloud(coor2_r_df)  # 将points的数据 存到结构体中
    projection2_r1 = coor2_r_pynt.to_instance("open3d", mesh=False)  # 实例化
    o3d.visualization.draw_geometries([projection2_r1])
    return coor2_r, projection2_r1


# 离群点移除
def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


# 对一个二维的数组np.array显示3d点云
def array2show(point_cloud_raw0):
    point_cloud_raw = DataFrame(point_cloud_raw0[:, 0:3])  # 选取每一列 的 第0个元素到第二个元素   [0,3)
    point_cloud_raw.columns = ['x', 'y', 'z']  # 给选取到的数据 附上标题
    point_cloud_pynt = PyntCloud(point_cloud_raw)  # 将points的数据 存到结构体中
    wall1 = point_cloud_pynt.to_instance("open3d", mesh=False)  # 实例化
    o3d.visualization.draw_geometries([wall1])  # 显示原始点云


# 霍夫变换
def hough_line(point_cloud):  # 传入的pointcloud是一个numpy.ndarray
    rho_ = 0.05  # 设置rou变化的比率
    x_min = np.min(point_cloud[:, 0])
    y_min = np.min(point_cloud[:, 1])
    x_max = np.max(point_cloud[:, 0])
    y_max = np.max(point_cloud[:, 1])
    point_cloud = np.c_[point_cloud[:, 0] - x_min, point_cloud[:, 1] - y_min]
    thetas = np.deg2rad(np.arange(0, 180))
    diag_len = int(round(math.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2)))
    rhos = np.linspace(-diag_len, diag_len, 2 * diag_len)
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint8)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    for i in range(len(point_cloud)):
        x = point_cloud[i][0]
        y = point_cloud[i][1]
        for j in range(len(thetas)):
            rho = round((x * cos_t[j] + y * sin_t[j] + diag_len) / 5)
            if isinstance(rho, int):
                accumulator[rho, j] += 1
            else:
                accumulator[int(rho), j] += 1
    x_min = np.min(point_cloud[:, 0])
    y_min = np.min(point_cloud[:, 1])
    x_max = np.max(point_cloud[:, 0])
    y_max = np.max(point_cloud[:, 1])
    #     show_hough_line(point_cloud, x_min, x_max, y_min, y_max)
    return accumulator, thetas, rhos


# 在open3d中画直线
def show_hough_line(point_cloud, x_min, x_max, y_min, y_max, vote=1000):
    accumulator, thetas, rhos = hough_line(point_cloud)
    all_points = []
    lines = []
    number = 0
    for i in range(rhos):
        for j in range(thetas):
            if accumulator[i, j] > vote:
                theta = thetas[j]
                rho = rhos[i]
                draw_point = []
                x1 = (rho - y_min * np.sin(theta)) / np.cos(theta)
                x2 = (rho - y_max * np.sin(theta)) / np.cos(theta)
                y1 = (rho - x_min * np.cos(theta)) / np.sin(theta)
                y2 = (rho - x_min * np.cos(theta)) / np.sin(theta)
                if x_min <= x1 <= x_max:
                    draw_point.append([x1, y_min, 0])
                if x_min <= x2 <= x_max:
                    draw_point.append([x2, y_max, 0])
                if y_min <= y1 <= y_max:
                    draw_point.append([x_min, y1, 0])
                if y_min <= y2 <= y_max:
                    draw_point.append([x_max, y2, 0])
                if len(draw_point) > 2:
                    for _ in range(2):
                        all_points.append(draw_point[_])
                lines.append([2 * number, 2 * number + 1])
                number += 1
    line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(all_points),
                                    lines=o3d.utility.Vector2iVector(lines))
    colors = [[1, 0, 0] * len(lines)]
    line_set.colors = o3d.utility.Vector3dVector(colors)
    # lines = [[0, 1], [0, 2]]  # 画出三点之间连线
    # colors = [[1, 0, 0], [0, 0, 0]]
    point_cloud_raw = DataFrame(point_cloud[:, 0:3])  # 选取每一列 的 第0个元素到第二个元素   [0,3)
    point_cloud_raw.columns = ['x', 'y', 'z']  # 给选取到的数据 附上标题
    point_cloud_pynt = PyntCloud(point_cloud_raw)  # 将points的数据 存到结构体中
    profile = point_cloud_pynt.to_instance("open3d", mesh=False)  # 实例化
    o3d.visualization.draw_geometries([profile, line_set])  #


# 顺时针排列点
def arrange_point_clockwise(point):
    point = np.array(point)
    point = point[:, 0:2]
    point = point[point[:, 0].argsort()]
    x_min = point[0, 0]
    y_x_min = point[0, 1]
    point_k = []
    for i in range(len(point)):
        if point[i, 0] > x_min:
            k = (point[i, 1] - y_x_min) / (point[i, 0] - x_min)
            point_k.append([point[i, 0], point[i, 1], k])
    point_k = np.array(point_k)
    point_k = point_k[point_k[:, 2].argsort()]
    point_k = np.insert(point_k, 0, [x_min, y_x_min, 0], axis=0)
    return point_k[:, 0:2]


def k_curvature(point, k=38):
    point = arrange_point_clockwise(point)
    n = len(point)
    kappa_set = []
    for i in range(len(point)):
        index1 = i - k if i >= k else n + i - k
        index2 = i + k if (i + k) < n else i + k - n
        x = [point[index1, 0], point[i, 0], point[index2, 0]]
        y = [point[index1, 1], point[i, 1], point[index2, 1]]

        t_a = la.norm([x[1] - x[0], y[1] - y[0]])
        t_b = la.norm([x[2] - x[1], y[2] - y[1]])

        M = np.array([
            [1, -t_a, t_a ** 2],
            [1, 0, 0],
            [1, t_b, t_b ** 2]
        ])

        a = np.matmul(la.inv(M), x)
        b = np.matmul(la.inv(M), y)

        kappa = 2 * (a[2] * b[1] - b[2] * a[1]) / (a[1] ** 2. + b[1] ** 2.) ** 1.5
        kappa_set.append(kappa)
        # return kappa, [b[1], -a[1]] / np.sqrt(a[1] ** 2. + b[1] ** 2.)
        return kappa_set


def sort_contour(points):
    points = np.array(points)
    if len(points) == 0:
        return 0
    sorted_set_from_start = []
    sorted_set_from_end = []
    p_start = points[0]
    sorted_set_from_start.append(p_start)
    points = np.delete(points, 0, 0)
    # kt = spt.KDTree(data=p_start, leafsize=10)
    ckt = spt.cKDTree(points)
    d, index = ckt.query(p_start)
    p_end = points[index]
    sorted_set_from_end.append(p_end)
    points = np.delete(points, index, 0)
    start_or_end = True
    while len(points) != 0:
        if start_or_end is True:
            point1 = p_start
            point2 = p_end
            # kt = spt.KDTree(data=point, leafsize=10)
            ckt = spt.cKDTree(points)
            d1, index = ckt.query(point1)
            point = points[index]
            d2 = math.sqrt((point2[0] - point[0]) ** 2 + (point2[1] - point[1]) ** 2)
            points = np.delete(points, index, 0)
            if d1 >= d2:
                p_start = point
                sorted_set_from_start.append(point)
            else:
                start_or_end = False
                p_end = point
                sorted_set_from_end.append(point)
        else:
            point2 = p_start
            point1 = p_end
            # kt = spt.KDTree(data=point, leafsize=10)
            ckt = spt.cKDTree(points)
            d1, index = ckt.query(point1)
            point = points[index]
            d2 = math.sqrt((point2[0] - point[0]) ** 2 + (point2[1] - point[1]) ** 2)
            points = np.delete(points, index, 0)
            if d1 >= d2:
                p_end = point
                sorted_set_from_end.append(point)
            else:
                start_or_end = True
                p_start = point
                sorted_set_from_start.append(point)
    sorted_set_from_end.reverse()
    # sorted_set_from_start.extend(sorted_set_from_end)
    print(len(sorted_set_from_start), len(sorted_set_from_end))
    return sorted_set_from_start


def plot_point2line(points):  # 输入的是二维的列表
    plt.figure(figsize=(4, 15), dpi=100)
    for i in range(len(points) - 1):
        plt.plot([points[i][0], points[i + 1][0]],
                 [points[i][1], points[i + 1][1]], linewidth=1)
    plt.plot([points[0][0], points[len(points) - 1][0]],
             [points[0][1], points[len(points) - 1][1]], linewidth=1)


# 沿点进行角点寻找
def find_corner_v1(points, k=20):  # 输入经过排序的点集
    corner_points = []
    points = np.array(points)
    k0 = k
    ref = min(k0, len(points))
    point = points[0]
    corner_points.append(point)
    points_ref = points[1:k0]
    accumulator, thetas, _ = hough_line(points_ref)
    r, _ = np.where(accumulator == np.max(accumulator))
    theta = thetas[r]
    theta_ref = theta
    points = np.delete(points, [0, ref], 0)
    return 0


def point_translate(point_cloud):  # 默认将点的坐标都平移到第一象限，针对二维的点
    x_min = np.min(point_cloud[:, 0])
    y_min = np.min(point_cloud[:, 1])
    x_max = np.max(point_cloud[:, 0])
    y_max = np.max(point_cloud[:, 1])
    # print(x_min,x_max,y_min,y_max)
    point_cloud = np.c_[point_cloud[:, 0] - x_min, point_cloud[:, 1] - y_min]
    return point_cloud


def point_rotation(points, n1):  # 二维点集旋转 ----输入点集和旋转的向量，按原点逆时针旋转
    n0 = np.array([0, 1])
    n1 = np.array(n1)
    l1 = np.sqrt(n1.dot(n1))
    l0 = np.sqrt(n0.dot(n0))
    cos_r = n1.dot(n0) / (l1 * l0)
    theta = np.arccos(cos_r)
    rotation_m = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return np.dot(points, rotation_m)


def find_corner(point_divided, boundary_index, width=1, height=5, direct=1, denoise=0):
    m = len(point_divided)
    n = len(point_divided[0])
    k = max(width, height)
    boundary_value = np.zeros((m + 2 * k, n + 2 * k))
    for a in range(len(boundary_index)):
        arr = boundary_index[a]
        boundary_value[arr[0] + k, arr[1] + k] = 1
    corner_points = []
    boundary_index = np.array(boundary_index)
    boundary_index = boundary_index[boundary_index[:, 1].argsort()]
    # j0=np.where(boundary_index==np.max(boundary_index,))
    i = boundary_index[-1][0] + k
    j = boundary_index[-1][1] + k
    boundary_value[i][j] = 2
    # 视为左下角角点，取xmin, ymin
    # arr = point_divided[i - k][j - k]
    # point_index = arr.min(axis=0)
    # point = arr[0]
    corner_points.append([i - k, j - k])

    direction = direct  # 1--向右，2--向上，3--向下，4--向左
    count = 0
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
        elif direction == 2:
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


def find_corner_2(point_divided, boundary_index, width=1, height=5, direct=1, denoise=0):
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
            cntss = 0
            for idx in range(8):
                if boundary_value[arr[0] + k + dx[idx], arr[1] + k + dy[idx]] == 1:
                    cntss += 1
            if cntss > 1:
                boundary_index2.append(arr)
    boundary_index = copy.deepcopy(boundary_index2)
    boundary_index = np.array(boundary_index)
    boundary_index = boundary_index[boundary_index[:, 1].argsort()]
    i = boundary_index[-1][0] + k
    j = boundary_index[-1][1] + k
    boundary_value[i][j] = 2
    corner_points_all.append(
        path_track(point_divided, boundary_index, boundary_value, i, j, width, height, direct, denoise))

    return corner_points_all


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
    corner_points_all = [
        path_track(point_divided, boundary_index, boundary_value, i, j, width, height, direct, denoise)]
    #     while np.sum(boundary_value == 1) > 0:
    boundary_index2 = []
    for ii in range(len(boundary_index)):
        arr = boundary_index[ii]
        if boundary_value[arr[0] + k, arr[1] + k] == 1:
            boundary_index2.append(arr)
    boundary_index = copy.deepcopy(boundary_index2)
    boundary_index = np.array(boundary_index)
    # boundary_index = boundary_index[boundary_index[:, 1].argsort()]
    counts = np.bincount(boundary_index[:, 0])
    i = np.argmax(counts)
    j_arr = np.where(boundary_index[:, 0] == i)
    j = boundary_index[j_arr[0][0], 1] + k
    i = boundary_index[j_arr[0][0], 0] + k
    boundary_value[i][j] = 2
    corner_points_all.append(
        path_track(point_divided, boundary_index, boundary_value, i, j, width, height, 3, denoise))

    return corner_points_all


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
            if j < m + k - 1 and rightcount > denoise:
                j += 1
                for ii in range(i - width, i + width + 1):
                    boundary_value[ii][j] = 2
            else:
                corner_points.append([i - k, j - k])
                direction = 1 if upcount > downcount else 4
        count += 1
    return corner_points


# 绘制散点图
def point_scatter(point_cloud):
    x_max = np.max(point_cloud[:, 0])
    y_max = np.max(point_cloud[:, 1])
    x_min = np.min(point_cloud[:, 0])
    y_min = np.min(point_cloud[:, 1])
    x_range = x_max - x_min
    y_range = y_max - y_min
    plt.figure(figsize=(x_range, y_range), dpi=100)
    plt.scatter(point_cloud[:, 0], point_cloud[:, 1])
    plt.show()


def plot_point2line2(corner1, width=10, height=10):  # 输入的是二维的列表
    plt.figure(figsize=(width, height), dpi=100)
    for ii in range(len(corner1)):
        corner2 = corner1[ii]
        corner2 = np.array(corner2)
        #     print(corner2)
        corner2 = corner2.astype(float)
        points = 0.1 * corner2
        for i in range(len(points) - 1):
            plt.plot([points[i][0], points[i + 1][0]],
                     [points[i][1], points[i + 1][1]], linewidth=2)
            plt.plot([points[0][0], points[len(points) - 1][0]],
                     [points[0][1], points[len(points) - 1][1]], linewidth=2)


def region_growing(points, theta_th, curvature_th, min_cluster_size, max_cluster_size, number_of_neighbour):
    clusters = []
    while (len(points) > min_cluster_size):
        cluster = []
        seed = random.randint(0, len(points))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        [k, indices, _] = pcd_tree.search_knn_vector_3d(pcd.points[seed], number_of_neighbour)
        indices


corner1 = []
from dxfwrite import DXFEngine as dxf

drawing = dxf.drawing('result.dxf')
drawing.add_layer('wall', color=1, linetype='Solid')
for ii in range(len(corner1)):
    corner2 = corner1[ii]
    corner2 = np.array(corner2)
    corner2 = corner2.astype(float)
    points = 0.1 * corner2
    for i in range(len(points)):
        j = (i + 1) % (len(points) - 1)
        line1 = dxf.line((points[i][0], points[i][1]), (points[j][0], points[j][0]))
        line1['layer'] = 'wall'
        line1['color'] = 256
        line1['thickness'] = 1.0
        drawing.add(line1)
        dimstyles.new("dots", tick="DIMTICK_DOT", scale=1., roundval=2, textabove=.5)
    dxf.block()
drawing.save()