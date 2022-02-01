import pointcloud
import plane
from typing import List, Dict, Optional, Tuple, Any, Type
from collections import defaultdict, namedtuple
import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def len_pcd(pcd:o3d.geometry.PointCloud) -> float:
    return len(pcd.points)

def volume_pcd(pcd:o3d.geometry.PointCloud) -> float:
    hull = ConvexHull(np.asarray(pcd.points))
    return hull.volume

def area_pcd(pcd:o3d.geometry.PointCloud) -> float:
    hull = ConvexHull(np.asarray(pcd.points))
    return hull.area

def clustering_pcd_return_pcd_clusters(pcd: o3d.geometry.PointCloud, eps: float = 0.02,
                                       min_points: int = 10, remove_noise: bool = True) -> \
        Tuple[List[o3d.geometry.PointCloud], Optional[o3d.geometry.PointCloud]]:
    """
    Clusterize a pointcloud and returns the list of the clusters as smaller pointclouds. If *remove_noise*
    is True, a tuple with two elements is returned: the first element is the previous mentioned list,
    and the second one the pointcloud containing the noise.

    :param pcd: The pointcloud to clusterize.
    :type pcd: o3d.geometry.PointCloud
    :param eps: Maximum distance between points of a cluster. Default value is 0.02.
    :type eps: float
    :param min_points: Minimum number of points in a cluster. Default value is 10.
    :type min_points: int
    :param remove_noise: If the points that are noise are returned.
    :type remove_noise: bool
    :return: List of clusters and, optionally, noise.
    :rtype: Tuple[List[o3d.geometry.PointCloud], Optional[o3d.geometry.PointCloud]]

    :Example:

    ::

        >>> import findroom
        >>> import clusteringroom
        >>> import os
        >>> import open3d as o3d
        >>> home_dir = os.getenv("HOME")
        >>> pcd_filename = home_dir + "/Lantegi/FEBRERO/Kubic_v2 - e1 - 2021-01-26-102719/voxelized2cm.pcd"
        >>> pcd = o3d.io.read_point_cloud(pcd_filename)
        >>> plane_thickness = 0.05
        >>> inside_room, raw_limits_room, limits_room, plane_models = findroom.find_room_in_hololens_pointcloud(pcd, distance_threshold=0.05, plane_thickness= plane_thickness)
        >>> list_clusters = clusteringroom.clustering_pcd_return_pcd_clusters(inside_room, eps = 0.8, min_points = 100)
        >>> list_clusters, noise_cloud = clusteringroom.clustering_pcd_return_pcd_clusters(inside_room, eps = 0.8, min_points = 100, remove_noise = False)
    """
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))
    # dictionary with the list of indices for each label
    dict_label_to_indices = defaultdict(list)
    for index, label in enumerate(labels):
        dict_label_to_indices[label].append(index)
    noise_pcd = o3d.geometry.PointCloud()
    list_clusters = list()
    for label in dict_label_to_indices.keys():
        if label != -1:
            pcd_temp = pcd.select_by_index(indices=dict_label_to_indices[label])
            list_clusters.append(pcd_temp)
        if label == -1:
            noise_pcd = pcd.select_by_index(indices=dict_label_to_indices[label])
    if remove_noise:
        return list_clusters
    else:
        return list_clusters, noise_pcd

def sort_pcd_list_by_area_of_convex_hull(pcd_list: List[o3d.geometry.PointCloud], reverse: bool = True) \
        -> List[o3d.geometry.PointCloud]:
    """

    :param pcd_list:
    :param reverse:
    :return:

    :Example:

    ::

        >>> import findroom
        >>> import clusteringroom
        >>> import os
        >>> import open3d as o3d
        >>> home_dir = os.getenv("HOME")
        >>> pcd_filename = home_dir + "/Lantegi/FEBRERO/Kubic_v2 - e1 - 2021-01-26-102719/voxelized2cm.pcd"
        >>> pcd = o3d.io.read_point_cloud(pcd_filename)
        >>> plane_thickness = 0.05
        >>> inside_room, raw_limits_room, limits_room, plane_models = findroom.find_room_in_hololens_pointcloud(pcd, distance_threshold=0.05, plane_thickness= plane_thickness)
        >>> list_clusters = clusteringroom.clustering_pcd_return_pcd_clusters(inside_room, eps = 0.1, min_points = 100)
        >>> list_clusters = clusteringroom.sort_pcd_list_by_area_of_convex_hull(list_clusters)
        >>> o3d.visualization.draw_geometries([list_clusters[0]])
    """
    pcd_list.sort(key=area_pcd, reverse=reverse)
    return pcd_list

def sort_pcd_list_by_volume_of_convex_hull(pcd_list: List[o3d.geometry.PointCloud], reverse: bool = True) \
        -> List[o3d.geometry.PointCloud]:
    pcd_list.sort(key=volume_pcd, reverse=reverse)
    return pcd_list

#@profile
def find_clusters_in_ceiling(pcd_list: List[o3d.geometry.PointCloud],
                             planes: Dict[str, Type[plane.PlaneIndoor]],
                             tolerance: float = 0.1) -> List[o3d.geometry.PointCloud]:
    """

    :param pcd_list:
    :param limits_room:
    :param tolerance:
    :return:

    :Example:

    ::

        >>> import findroom
        >>> import clusteringroom
        >>> import os
        >>> import open3d as o3d
        >>> home_dir = os.getenv("HOME")
        >>> pcd_filename = home_dir + "/Lantegi/FEBRERO/Kubic_v2 - e1 - 2021-01-26-102719/voxelized2cm.pcd"
        >>> pcd = o3d.io.read_point_cloud(pcd_filename)
        >>> plane_thickness = 0.05
        >>> inside_room, raw_limits_room, limits_room, plane_models = findroom.find_room_in_hololens_pointcloud(pcd, distance_threshold=0.05, plane_thickness= plane_thickness)
        >>> list_clusters = clusteringroom.clustering_pcd_return_pcd_clusters(inside_room, eps = 0.1, min_points = 100)
        >>> clusters_in_ceiling = clusteringroom.find_clusters_in_ceiling(list_clusters, planes, tolerance = 0.1)
        >>> o3d.visualization.draw_geometries(clusters_in_ceiling)
    """
    list_clusters_in_ceiling = list()
    for pcd in pcd_list:
        distance_to_ceiling = pointcloud.get_distance_between_pointcloud_and_plane(pcd, planes["ceiling"])
        if distance_to_ceiling <= tolerance:
            list_clusters_in_ceiling.append(o3d.geometry.PointCloud(pcd))
    return list_clusters_in_ceiling

#@profile
def find_clusters_in_floor(pcd_list: List[o3d.geometry.PointCloud],
                           planes: Dict[str, Type[plane.PlaneIndoor]],
                           tolerance: float = 0.1) -> List[o3d.geometry.PointCloud]:
    """

    :param pcd_list:
    :param limits_room:
    :param tolerance:
    :return:

    :Example:

    ::

        >>> import findroom
        >>> import clusteringroom
        >>> import os
        >>> import open3d as o3d
        >>> home_dir = os.getenv("HOME")
        >>> pcd_filename = home_dir + "/Lantegi/FEBRERO/Kubic_v2 - e1 - 2021-01-26-102719/voxelized2cm.pcd"
        >>> pcd = o3d.io.read_point_cloud(pcd_filename)
        >>> plane_thickness = 0.05
        >>> inside_room, raw_limits_room, limits_room, plane_models = findroom.find_room_in_hololens_pointcloud(pcd, distance_threshold=0.05, plane_thickness= plane_thickness)
        >>> list_clusters = clusteringroom.clustering_pcd_return_pcd_clusters(inside_room, eps = 0.1, min_points = 100)
        >>> clusters_in_floor = clusteringroom.find_clusters_in_floor(list_clusters, planes, tolerance = 0.1)
        >>> o3d.visualization.draw_geometries(clusters_in_floor)
    """
    list_clusters_in_floor = list()
    for pcd in pcd_list:
        distance_to_floor = pointcloud.get_distance_between_pointcloud_and_plane(pcd, planes["floor"])
        if distance_to_floor <= tolerance:
            list_clusters_in_floor.append(o3d.geometry.PointCloud(pcd))
    return list_clusters_in_floor

#@profile
def close_to_walls(pcd: o3d.geometry.PointCloud, planes: Dict[str, Type[plane.PlaneIndoor]],
                   tolerance: float = 0.1) -> bool:
    return pointcloud.get_distance_between_pointcloud_and_plane(pcd, planes["wall_1_1"]) <= tolerance or \
           pointcloud.get_distance_between_pointcloud_and_plane(pcd, planes["wall_1_2"]) <= tolerance or \
           pointcloud.get_distance_between_pointcloud_and_plane(pcd, planes["wall_2_1"]) <= tolerance or \
           pointcloud.get_distance_between_pointcloud_and_plane(pcd, planes["wall_2_2"]) <= tolerance

#@profile
def find_clusters_in_wall(pcd_list: List[o3d.geometry.PointCloud],
                          planes: Dict[str, Type[plane.PlaneIndoor]],
                          tolerance: float = 0.1) -> List[o3d.geometry.PointCloud]:
    """

    :param pcd_list:
    :param limits_room:
    :param tolerance:
    :return:

    :Example:

    ::

        >>> import findroom
        >>> import clusteringroom
        >>> import os
        >>> import open3d as o3d
        >>> home_dir = os.getenv("HOME")
        >>> pcd_filename = home_dir + "/Lantegi/FEBRERO/Kubic_v2 - e1 - 2021-01-26-102719/voxelized2cm.pcd"
        >>> pcd = o3d.io.read_point_cloud(pcd_filename)
        >>> plane_thickness = 0.05
        >>> inside_room, raw_limits_room, limits_room, plane_models = findroom.find_room_in_hololens_pointcloud(pcd, distance_threshold=0.05, plane_thickness= plane_thickness)
        >>> list_clusters = clusteringroom.clustering_pcd_return_pcd_clusters(inside_room, eps = 0.1, min_points = 100)
        >>> clusters_in_wall = clusteringroom.find_clusters_in_wall(list_clusters, planes, tolerance = 0.1)
        >>> o3d.visualization.draw_geometries(clusters_in_wall)
    """
    list_clusters_in_wall = list()
    for pcd in pcd_list:
        if close_to_walls(pcd, planes, tolerance):
            list_clusters_in_wall.append(o3d.geometry.PointCloud(pcd))
    return list_clusters_in_wall

#@profile
def find_clusters_column_candidates(pcd_list: List[o3d.geometry.PointCloud],
                                    planes: Dict[str, Type[plane.PlaneIndoor]],
                                    tolerance: float = 0.1) -> List[o3d.geometry.PointCloud]:
    """

    :param pcd_list:
    :param limits_room:
    :param tolerance:
    :return:

    :Example:

    ::

        >>> import findroom
        >>> import clusteringroom
        >>> import os
        >>> import open3d as o3d
        >>> home_dir = os.getenv("HOME")
        >>> pcd_filename = home_dir + "/Lantegi/FEBRERO/Kubic_v2 - e1 - 2021-01-26-102719/voxelized2cm.pcd"
        >>> pcd = o3d.io.read_point_cloud(pcd_filename)
        >>> plane_thickness = 0.05
        >>> inside_room, raw_limits_room, limits_room, plane_models = findroom.find_room_in_hololens_pointcloud(pcd, distance_threshold=0.05, plane_thickness= plane_thickness)
        >>> list_clusters = clusteringroom.clustering_pcd_return_pcd_clusters(inside_room, eps = 0.1, min_points = 100)
        >>> clusters_in_column_candidates = clusteringroom.find_clusters_column_candidates(list_clusters, planes, tolerance = 0.1)
        >>> o3d.visualization.draw_geometries(clusters_in_column_candidates)
    """
    list_clusters_in_column_candidates = list()
    for pcd in pcd_list:
        distance_to_ceiling = pointcloud.get_distance_between_pointcloud_and_plane(pcd, planes["ceiling"])
        distance_to_floor = pointcloud.get_distance_between_pointcloud_and_plane(pcd, planes["floor"])
        if distance_to_ceiling <= tolerance and distance_to_floor <= tolerance:
            list_clusters_in_column_candidates.append(o3d.geometry.PointCloud(pcd))
    return list_clusters_in_column_candidates

#@profile
def find_clusters_open_door_candidates(pcd_list: List[o3d.geometry.PointCloud],
                                       planes: Dict[str, Type[plane.PlaneIndoor]],
                                       tolerance: float = 0.1) -> List[o3d.geometry.PointCloud]:
    """

    :param pcd_list:
    :param limits_room:
    :param tolerance:
    :return:

    :Example:

    ::

        >>> import findroom
        >>> import clusteringroom
        >>> import os
        >>> import open3d as o3d
        >>> home_dir = os.getenv("HOME")
        >>> pcd_filename = home_dir + "/Lantegi/FEBRERO/Kubic_v2 - e1 - 2021-01-26-102719/voxelized2cm.pcd"
        >>> pcd = o3d.io.read_point_cloud(pcd_filename)
        >>> plane_thickness = 0.05
        >>> inside_room, raw_limits_room, limits_room, plane_models = findroom.find_room_in_hololens_pointcloud(pcd, distance_threshold=0.05, plane_thickness= plane_thickness)
        >>> list_clusters = clusteringroom.clustering_pcd_return_pcd_clusters(inside_room, eps = 0.1, min_points = 100)
        >>> clusters_in_open_door_candidates = clusteringroom.find_clusters_open_door_candidates(list_clusters, planes, tolerance = 0.1)
        >>> o3d.visualization.draw_geometries(clusters_in_open_door_candidates)
    """
    list_clusters_in_open_door_candidates = list()
    for pcd in pcd_list:
        if area_pcd(pcd) > 1.0 and pointcloud.get_distance_between_pointcloud_and_plane(pcd, planes["ceiling"]) >= 1.0:
            distance_to_floor = pointcloud.get_distance_between_pointcloud_and_plane(pcd, planes["floor"])
            is_close_to_walls = close_to_walls(pcd, planes, tolerance)
            if distance_to_floor <= tolerance and is_close_to_walls:
                list_clusters_in_open_door_candidates.append(o3d.geometry.PointCloud(pcd))
    return list_clusters_in_open_door_candidates

#@profile
def find_clusters_to_remove(pcd_list: List[o3d.geometry.PointCloud],
                            planes: Dict[str, Type[plane.PlaneIndoor]],
                            tolerance: float = 0.1) -> List[o3d.geometry.PointCloud]:
    """
    Clusters to remove because they are not close to the walls or the ceiling; everything we
    are interested in have to be close to the ceiling or the walls.

    :param pcd_list:
    :param limits_room:
    :param tolerance:
    :return:

    :Example:

    ::

        >>> import findroom
        >>> import clusteringroom
        >>> import os
        >>> import open3d as o3d
        >>> home_dir = os.getenv("HOME")
        >>> pcd_filename = home_dir + "/Lantegi/FEBRERO/Kubic_v2 - e1 - 2021-01-26-102719/voxelized2cm.pcd"
        >>> pcd = o3d.io.read_point_cloud(pcd_filename)
        >>> plane_thickness = 0.05
        >>> inside_room, raw_limits_room, limits_room, plane_models = findroom.find_room_in_hololens_pointcloud(pcd, distance_threshold=0.05, plane_thickness= plane_thickness)
        >>> list_clusters = clusteringroom.clustering_pcd_return_pcd_clusters(inside_room, eps = 0.1, min_points = 100)
        >>> clusters_in_open_door_candidates = clusteringroom.find_clusters_open_door_candidates(list_clusters, planes, tolerance = 0.1)
        >>> o3d.visualization.draw_geometries(clusters_in_open_door_candidates)
    """
    list_clusters_to_remove = list()
    indices = list()
    for index, pcd in enumerate(pcd_list):
        if pointcloud.get_distance_between_pointcloud_and_plane(pcd, planes["ceiling"]) >= 0.5 \
                and not close_to_walls(pcd, planes, tolerance=tolerance):
            list_clusters_to_remove.append(o3d.geometry.PointCloud(pcd))
            indices.append(index)
    return list_clusters_to_remove, indices

def paint_list_of_clusters(pcd_list: List[o3d.geometry.PointCloud]) -> List[o3d.geometry.PointCloud]:
    plt.get_cmap("tab20")
    list_clusters_painted = list()
    for index, pcd in enumerate(pcd_list):
        new_pcd = o3d.geometry.PointCloud(pcd)
        new_pcd.paint_uniform_color(plt.get_cmap("tab20")(index % 20)[:-1])
        list_clusters_painted.append(new_pcd)
    return list_clusters_painted

def return_clusters_not_in_indices_list(pcd_list: List[o3d.geometry.PointCloud], indices: List[int]) \
        -> List[o3d.geometry.PointCloud]:
    list_clusters = list()
    for index, pcd in enumerate(pcd_list):
        new_pcd = o3d.geometry.PointCloud(pcd)
        if index not in indices:
            list_clusters.append(new_pcd)
    return list_clusters

def from_list_clusters_to_pointcloud(pcd_list: List[o3d.geometry.PointCloud]) \
        -> o3d.geometry.PointCloud:
    new_pcd = o3d.geometry.PointCloud()
    for pcd in pcd_list:
        new_pcd = new_pcd + pcd
    return new_pcd

def paint_red_clusters_in_indices_list_and_grey_the_others(pcd_list: List[o3d.geometry.PointCloud], indices: List[int]) \
        -> List[o3d.geometry.PointCloud]:
    list_clusters = list()
    for index, pcd in enumerate(pcd_list):
        new_pcd = o3d.geometry.PointCloud(pcd)
        if index in indices:
            new_pcd.paint_uniform_color([1, 0, 0])
        else:
            new_pcd.paint_uniform_color([0.5, 0.5, 0.5])
        list_clusters.append(new_pcd)
    return list_clusters

def clustering_inside_room(pcd: o3d.geometry.PointCloud) -> Dict[str, o3d.geometry.PointCloud]:
    pass

# def get_clustering_value_and_labels(data: np.ndarray, k: int) -> float:
#     scaler = StandardScaler()
#     scaler.fit(data)
#     data = scaler.transform(data)
#     kmeans_model = KMeans(n_clusters=k, random_state=1).fit(data)
#     labels = kmeans_model.labels_
#     v = metrics.silhouette_score(data, labels, metric='euclidean')
#     return v, labels

#@profile
def find_best_initial_sublist_for_clustering_and_cluster_it(l: List[Any], k: int = 2, max_length: int = None) -> List[List[Any]]:
    # it compares with clustering with k + 1 and checks which one is better
    # looks for the maximum drop in the sum of v(k) and v(k)-v(k+1), so it combines
    # how good is the clustering with how better than the alternative is that same clustering
    # l = [[1.3],[1.4],[1.5],[3.4],[3.5],[3.6],[3.7], [2.5], [3.5]]
    # len(l) > k + 1 for silhouette to work

    def get_clustering_value_and_labels(data: np.ndarray, k: int) -> float:
        scaler = StandardScaler()
        scaler.fit(data)
        data_transformed = scaler.transform(data)
        kmeans_model = KMeans(n_clusters=k, random_state=1, algorithm="elkan", max_iter=10).fit(data_transformed)
        labels = kmeans_model.labels_
        v = metrics.silhouette_score(data, labels, metric='euclidean')
        return v, labels

    if max_length is None:
        max_length = len(l)
    else:
        max_length = min(len(l), max_length)
    all_v  = list()
    all_labels = list()
    all_v_k_1  = list()
    all_labels_k_1 = list()
    for i in range(k+2, max_length+1):
        data = np.array(l[:i])
        v, labels = get_clustering_value_and_labels(data, k)
        v_k_1, labels_k_1 = get_clustering_value_and_labels(data, k+1)
        # print (v, labels)
        all_v.append(v)
        all_labels.append(labels)
        all_v_k_1.append(v_k_1)
        all_labels_k_1.append(labels_k_1)
    a = np.diff(all_v)
    # print(a)
    improvement_over_k_1 = np.array(all_v) - np.array(all_v_k_1)
    # print(improvement_over_k_1)
    total_advantage_of_k = np.array(all_v) + improvement_over_k_1
    # print(total_advantage_of_k)
    drop_in_advantage = -np.diff(total_advantage_of_k)
    # print(drop_in_advantage)
    index_max_drop = np.argmax(drop_in_advantage)
    return l[:index_max_drop + k + 2], all_labels[index_max_drop]

def get_ceiling_and_floor_heights_from_two_lists(l1: List[float], l2: List[float], step: float) \
        -> Tuple[float, float]:
    # those values are max_bounds, we have to find the minimum of the ceiling
    # and the maximum of the floor
    if np.mean(l1) > np.mean(l2):
        ceiling_height = np.min(l1) - step # because it is a max_bound
        floor_height = np.max(l2)
    else:
        ceiling_height = np.min(l2) - step # because it is a max_bound
        floor_height = np.max(l1)
    return ceiling_height, floor_height

#@profile
def get_ceiling_and_floor(pcd: o3d.geometry.PointCloud, axis: int = 2, step: float = 0.02, max_length: int = 50):
    Data = namedtuple("Data", "list_slice_values cluster_1 cluster_2 ceiling_height floor_height")
    which_slices, list_slice_values_ordered = pointcloud.cut_into_disjoint_slices_by_axis_pointcloud(pcd, axis, step)
    # list_slice_values = np.array([[pcd_slice.get_max_bound()[axis]] for pcd_slice in list_slices])
    #list_slice_values = list_slice_values.reshape(-1, 1)
    # print(list_slice_values)
    values, labels = find_best_initial_sublist_for_clustering_and_cluster_it(list_slice_values_ordered, k = 2, max_length = max_length)
    values = values.ravel()
    cluster_1 = [item[0] for item in zip(values, labels) if item[1] == 0]
    cluster_2 = [item[0] for item in zip(values, labels) if item[1] == 1]
    ceiling_height, floor_height = get_ceiling_and_floor_heights_from_two_lists(cluster_1, cluster_2, step)
    return Data(list_slice_values_ordered, cluster_1, cluster_2, ceiling_height, floor_height)








