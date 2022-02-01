"""
In this module there are several functions to find a room in a pointcloud. It builds upon
`plane<plane>_`, and assumes that the vertical direction of the pointcloud is the
Y axis, with positive values going upwards.
"""

import open3d as o3d
import numpy as np
import math
from typing import Tuple, List, Dict, Optional, Any, Type
import pickle as pkl
import os
from scipy.spatial import ConvexHull
import indoor3d.plane as indplane
import indoor3d.pointcloud as pointcloud

def save_room_context(pcd: o3d.geometry.PointCloud,
                      pcd_cropped: o3d.geometry.PointCloud, pcd_not_cropped: o3d.geometry.PointCloud,
                      inside_room: o3d.geometry.PointCloud, limits_room: Dict[str, o3d.geometry.PointCloud],
                      # raw_limits_room: Dict[str, o3d.geometry.PointCloud],
                      pcd_ceiling: o3d.geometry.PointCloud, pcd_floor: o3d.geometry.PointCloud,
                      pcd_wall_1_1: o3d.geometry.PointCloud, pcd_wall_1_2: o3d.geometry.PointCloud,
                      pcd_wall_2_1: o3d.geometry.PointCloud, pcd_wall_2_2: o3d.geometry.PointCloud,
                      plane_models: Dict[str, Tuple[float, float, float, float]],
                      plane_thickness: float, directory: str) -> None:
    """
    Saves to disk the information about a room: the inside, the ceiling, floor, walls, the plane
    models as well as the thickness of the pointclouds related to planes.

    :param inside_room: The inside of the room.
    :type inside_room: o3d.geometry.PointCloud
    :param limits_room: The limits of the room: ceiling, floor and walls.
    :type limits_room: Dict[str, o3d.geometry.PointCloud]
    :param plane_models: The plane models corresponsind to the room limits.
    :type plane_models: Dict[str, Tuple[float, float, float, float]]
    :param plane_thickness: The thickness of the poinclouds corresponding to the limits of the room.
    :type plane_thickness: float
    :param directory: The directory where the data will be stored.
    :type directory: str
    :return: Nothing.
    :rtype: NoneType

    :Example:

    ::

        >>> import findroom
        >>> import os
        >>> import open3d as o3d
        >>> home_dir = os.getenv("HOME")
        >>> pcd_filename = home_dir + "/Lantegi/FEBRERO/Kubic_v2 - e1 - 2021-01-26-102719/voxelized2cm.pcd"
        >>> pcd = o3d.io.read_point_cloud(pcd_filename)
        >>> plane_thickness = 0.05
        >>> inside_room, raw_limits_room, limits_room, plane_models = findroom.find_room_in_hololens_pointcloud(pcd, distance_threshold=0.05, plane_thickness= plane_thickness)
        >>> directory = home_dir + "/Github/Lantegi/Code/Open3D/room_context"
        >>> findroom.save_room_context(inside_room, limits_room, raw_limits_room, plane_models, plane_thickness, directory)

    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    o3d.io.write_point_cloud(directory + "/pcd.pcd", pcd)
    o3d.io.write_point_cloud(directory + "/pcd_cropped.pcd", pcd_cropped)
    o3d.io.write_point_cloud(directory + "/pcd_not_cropped.pcd", pcd_not_cropped)
    o3d.io.write_point_cloud(directory + "/inside_room.pcd", inside_room)
    o3d.io.write_point_cloud(directory + "/ceiling.pcd", limits_room["ceiling"])
    o3d.io.write_point_cloud(directory + "/floor.pcd", limits_room["floor"])
    o3d.io.write_point_cloud(directory + "/wall_1_1.pcd", limits_room["wall_1_1"])
    o3d.io.write_point_cloud(directory + "/wall_1_2.pcd", limits_room["wall_1_2"])
    o3d.io.write_point_cloud(directory + "/wall_2_1.pcd", limits_room["wall_2_1"])
    o3d.io.write_point_cloud(directory + "/wall_2_2.pcd", limits_room["wall_2_2"])
    o3d.io.write_point_cloud(directory + "/raw_ceiling.pcd", pcd_ceiling)
    o3d.io.write_point_cloud(directory + "/raw_floor.pcd", pcd_floor)
    o3d.io.write_point_cloud(directory + "/raw_wall_1_1.pcd", pcd_wall_1_1)
    o3d.io.write_point_cloud(directory + "/raw_wall_1_2.pcd", pcd_wall_1_2)
    o3d.io.write_point_cloud(directory + "/raw_wall_2_1.pcd", pcd_wall_2_1)
    o3d.io.write_point_cloud(directory + "/raw_wall_2_2.pcd", pcd_wall_2_2)
    room_context = {"plane_models": plane_models, "plane_thickness": plane_thickness}
    pkl.dump(room_context, open(directory + "/room_context.pkl", "wb"))


def load_room_context(directory: str) -> Dict[str, Any]:
    """
    Loads the room context stored in a directory: the inside of the room, the limits (ceiling, floor, walls),
    the plane models and the plane thickness.

    :param directory: Directory where the data is stored.
    :type directory: str
    :return: All the context data previously stored.
    :rtype: Dict[str, Any]

    :Example:

    ::

        >>> room_context = findroom.load_room_context("room_context")
    """
    pcd = o3d.io.read_point_cloud(directory + "/pcd.pcd")
    inside_room = o3d.io.read_point_cloud(directory + "/inside_room.pcd")
    pcd_cropped = o3d.io.read_point_cloud(directory + "/pcd_cropped.pcd")
    pcd_not_cropped = o3d.io.read_point_cloud(directory + "/pcd_not_cropped.pcd")
    limits_room = dict()
    limits_room["ceiling"] = o3d.io.read_point_cloud(directory + "/ceiling.pcd")
    limits_room["floor"] = o3d.io.read_point_cloud(directory + "/floor.pcd")
    limits_room["wall_1_1"] = o3d.io.read_point_cloud(directory + "/wall_1_1.pcd")
    limits_room["wall_1_2"] = o3d.io.read_point_cloud(directory + "/wall_1_2.pcd")
    limits_room["wall_2_1"] = o3d.io.read_point_cloud(directory + "/wall_2_1.pcd")
    limits_room["wall_2_2"] = o3d.io.read_point_cloud(directory + "/wall_2_2.pcd")
    raw_limits_room = dict()
    raw_limits_room["ceiling"] = o3d.io.read_point_cloud(directory + "/raw_ceiling.pcd")
    raw_limits_room["floor"] = o3d.io.read_point_cloud(directory + "/raw_floor.pcd")
    raw_limits_room["wall_1_1"] = o3d.io.read_point_cloud(directory + "/raw_wall_1_1.pcd")
    raw_limits_room["wall_1_2"] = o3d.io.read_point_cloud(directory + "/raw_wall_1_2.pcd")
    raw_limits_room["wall_2_1"] = o3d.io.read_point_cloud(directory + "/raw_wall_2_1.pcd")
    raw_limits_room["wall_2_2"] = o3d.io.read_point_cloud(directory + "/raw_wall_2_2.pcd")
    room_context = pkl.load(open(directory + "/room_context.pkl", "rb"))
    return {"pcd": pcd, "pcd_cropped": pcd_cropped, "pcd_not_cropped": pcd_not_cropped,
            "inside_room": inside_room, "limits_room": limits_room,
            "raw_limits_room": raw_limits_room,
            "plane_models": room_context["plane_models"],
            "plane_thickness": room_context["plane_thickness"]}


def get_line_set_from_room_limit_points(
        limit_points: Dict[str, Tuple[Tuple[float, float, float], Tuple[float, float, float],
                                      Tuple[float, float, float], Tuple[float, float, float]]],
                    color: Optional[Tuple[float, float, float]] = None ) -> o3d.geometry.LineSet:
    """
    Returns the set of lines that draw the cuboid of the room limit points.

    :param limit_points: The points that limit the room and form a cuboid.
    :type limit_points: Dict[str, Tuple[Tuple[float, float, float], Tuple[float, float, float],
                                      Tuple[float, float, float], Tuple[float, float, float]]]
    :param color: Color of the lines, defaults to red, parameter value is (1, 0, 0).
    :type color: Optional[Tuple[float, float, float]]
    :return: A set of lines that draw a cuboid.
    :rtype: o3d.geometry.LineSet

    :Example:

    ::

        >>> import findroom
        >>> import os
        >>> import open3d as o3d
        >>> home_dir = os.getenv("HOME")
        >>> pcd_filename = home_dir + "/Lantegi/FEBRERO/Kubic_v2 - e1 - 2021-01-26-102719/voxelized2cm.pcd"
        >>> pcd = o3d.io.read_point_cloud(pcd_filename)
        >>> room_planes = findroom.find_room_planes_in_hololens_pointcloud(pcd, distance_threshold=0.05)
        >>> limit_points = findroom.find_limit_points_from_room_planes(room_planes)
        >>> limit_points
        {'ceiling': ((-2.659084167584312, 1.663165942978962, -12.059679580097264),
        (0.04595579610447797, 1.7270522547819955, -5.768675983794973),
        (-6.274718120791219, 1.6609373095351838, -3.0201432659553724),
        (-9.109512043352122, 1.5932584443969449, -9.765668008237839)),
        'floor': ((-2.565693580945881, -1.5989279880871063, -12.080538184884569),
        (0.13783230321479978, -1.633257582540433, -5.800215740209161),
        (-6.337506317681437, -1.6103617016636906, -2.9846542023329574),
        (-9.17057479435457, -1.5738354771285867, -9.731963293688295))}
        >>> line_set = findroom.get_line_set_from_room_limit_points(limit_points)
        >>> line_set
        LineSet with 12 lines.
        >>> o3d.visualization.draw_geometries([pcd, line_set])
    """
    if color is None:
        color = (1, 0, 0)
    points_ceiling = [list(point) for point in limit_points["ceiling"]]
    points_floor = [list(point) for point in limit_points["floor"]]
    points_room = points_ceiling + points_floor
    return pointcloud.get_line_set_from_cuboid_points(points_room, color)


def find_limit_points_from_room_planes(dict_planes: Dict[str, Type[indplane.PlaneIndoor]]) -> \
        Dict[str, Tuple[Tuple[float, float, float], Tuple[float, float, float],
                        Tuple[float, float, float], Tuple[float, float, float]]]:
    """
    Finds the eight points that mark the limits of the room. It returns a dictionary with two entries:
    "floor" and "ceiling", each of them with four points, corresponding to the limit points of the floor
    or the ceiling, respectively.

    :param dict_planes: The planes that delimit the room.
    :type dict_planes: Dict[str, Type[indplane.PlaneIndoor]])
    :return: A dictionary with two entries, "ceiling" and "floor", where each entry has four points.
    :rtype: Dict[str, Tuple[Tuple[float, float, float], Tuple[float, float, float],
                        Tuple[float, float, float], Tuple[float, float, float]]]

    :Example:

    ::

        >>> import findroom
        >>> import os
        >>> import open3d as o3d
        >>> home_dir = os.getenv("HOME")
        >>> pcd_filename = home_dir + "/Lantegi/FEBRERO/Kubic_v2 - e1 - 2021-01-26-102719/voxelized2cm.pcd"
        >>> pcd = o3d.io.read_point_cloud(pcd_filename)
        >>> room_planes = findroom.find_room_planes_in_hololens_pointcloud(pcd, distance_threshold=0.05)
        >>> limit_points = findroom.find_limit_points_from_room_planes(room_planes)
        >>> limit_points
        {'ceiling': ((-2.659084167584312, 1.663165942978962, -12.059679580097264),
        (0.04595579610447797, 1.7270522547819955, -5.768675983794973),
        (-6.274718120791219, 1.6609373095351838, -3.0201432659553724),
        (-9.109512043352122, 1.5932584443969449, -9.765668008237839)),
        'floor': ((-2.565693580945881, -1.5989279880871063, -12.080538184884569),
        (0.13783230321479978, -1.633257582540433, -5.800215740209161),
        (-6.337506317681437, -1.6103617016636906, -2.9846542023329574),
        (-9.17057479435457, -1.5738354771285867, -9.731963293688295))}
        >>> line_set = findroom.get_line_set_from_room_limit_points(limit_points)
        >>> o3d.visualization.draw_geometries([pcd, line_set])
    """
    dict_limit_points = {}
    limit_ceiling_1 = indplane.get_point_intersection_of_three_planes(dict_planes["ceiling"], dict_planes["wall_1_1"],
                                                                        dict_planes["wall_2_1"])
    limit_ceiling_2 = indplane.get_point_intersection_of_three_planes(dict_planes["ceiling"], dict_planes["wall_1_1"],
                                                                        dict_planes["wall_2_2"])
    limit_ceiling_3 = indplane.get_point_intersection_of_three_planes(dict_planes["ceiling"], dict_planes["wall_1_2"],
                                                                        dict_planes["wall_2_2"])
    limit_ceiling_4 = indplane.get_point_intersection_of_three_planes(dict_planes["ceiling"], dict_planes["wall_1_2"],
                                                                        dict_planes["wall_2_1"])
    dict_limit_points["ceiling"] = (limit_ceiling_1, limit_ceiling_2, limit_ceiling_3, limit_ceiling_4)
    limit_floor_1 = indplane.get_point_intersection_of_three_planes(dict_planes["floor"], dict_planes["wall_1_1"],
                                                                      dict_planes["wall_2_1"])
    limit_floor_2 = indplane.get_point_intersection_of_three_planes(dict_planes["floor"], dict_planes["wall_1_1"],
                                                                      dict_planes["wall_2_2"])
    limit_floor_3 = indplane.get_point_intersection_of_three_planes(dict_planes["floor"], dict_planes["wall_1_2"],
                                                                      dict_planes["wall_2_2"])
    limit_floor_4 = indplane.get_point_intersection_of_three_planes(dict_planes["floor"], dict_planes["wall_1_2"],
                                                                      dict_planes["wall_2_1"])
    dict_limit_points["floor"] = (limit_floor_1, limit_floor_2, limit_floor_3, limit_floor_4)
    dict_limit_points["wall_1_1"] = (limit_ceiling_1, limit_ceiling_2, limit_floor_1, limit_floor_2)
    dict_limit_points["wall_1_2"] = (limit_ceiling_3, limit_ceiling_4, limit_floor_3, limit_floor_4)
    dict_limit_points["wall_2_1"] = (limit_ceiling_1, limit_ceiling_4, limit_floor_1, limit_floor_4)
    dict_limit_points["wall_2_2"] = (limit_ceiling_2, limit_ceiling_3, limit_floor_2, limit_floor_3)
    return dict_limit_points

#@profile
def find_room_planes_in_hololens_pointcloud(pcd: o3d.geometry.PointCloud,
                                            y_ceiling_floor: Tuple[float, float] = None,
                                            distance_threshold: float = 0.01,
                                            ransac_n: int = 3,
                                            num_iterations: int = 1000,
                                            max_planes: int = 25,
                                            max_tolerance_degrees: float = 10,
                                            debug: bool = False) -> \
        Optional[Dict[str, Type[indplane.PlaneIndoor]]]:
    """
    Finds the room planes in a pointcloud. It returns None if no room is found. The dict entries are:
    "ceiling", "floor", "wall_1_1", "wall_1_2", "wall_2_1", "wall_2_2". Wall_1_1 and wall_1_2 are
    parallel to each other, likewise for wall_2_1 and wall_2_2. Wall_1_1 and wall_1_2
    are perpendicular to wall_2_1 and wall_2_2.

    :param pcd: Pointcloud where the room should be located.
    :type pcd: o3d.geometry.PointCloud
    :param distance_threshold: Maximum distance from the plane model, default is 0.01.
    :type distance_threshold: float
    :param ransac_n: Minimum number of points for a plane, default is 3.
    :type ransac_n: int
    :param num_iterations: Maximum number of iterations in the RANSAC method, default is 1000.
    :type num_iterations: int
    :param max_planes: Maximum number of planes to search for, default is 15.
    :type max_planes: int
    :param max_tolerance_degrees: Tolerance for parallel and perpendicular condition, default is 10 degrees.
    :type max_tolerance_degrees: float
    :param debug: if debug information is shown, default is False.
    :type debug: bool
    :return: Room planes if found, None otherwise.
    :rtype: Optional[Dict[str, Tuple[float, float, float, float]]]

    :Example:

    ::

        >>> import findroom
        >>> import os
        >>> import open3d as o3d
        >>> home_dir = os.getenv("HOME")
        >>> pcd_filename = home_dir + "/Lantegi/FEBRERO/Kubic_v2 - e1 - 2021-01-26-102719/voxelized2cm.pcd"
        >>> pcd = o3d.io.read_point_cloud(pcd_filename)
        >>> room_planes = findroom.find_room_planes_in_hololens_pointcloud(pcd, distance_threshold=0.05)
        >>> room_planes
        {'ceiling': array([-9.27806923e-03,  9.99956900e-01,  3.40712719e-04, -1.71247288e+00]),
        'floor': array([1.33089689e-03, 9.99997699e-01, 1.68252695e-03, 1.62352110e+00]),
        'wall_1_1': array([ 0.38325152, -0.03832315,  0.92284864, 12.17910712]),
        'wall_1_2': array([0.37658693, 0.03685206, 0.925648  , 5.23732535]),
        'wall_2_1': array([ 0.92027216,  0.04608669, -0.38855523, -2.28982772]),
        'wall_2_2': array([ 0.9197419 , -0.01003106, -0.39239548,  4.60987813])}
    """
    current_pcd = o3d.geometry.PointCloud(pcd)
    num_current_iterations = 1
    if debug:
        list_inlier_clouds = []
        list_outlier_clouds_files = []
    if y_ceiling_floor is not None:
        y_ceiling = y_ceiling_floor[0]
        y_floor = y_ceiling_floor[1]
        list_planes = [indplane.PlaneIndoor((0, 1, 0, -y_ceiling)), indplane.PlaneIndoor((0, 1, 0, -y_floor))]


    else:
        list_planes = []
    room_planes = None
    # first we extract six planes, the minimum
    while len(current_pcd.points) > ransac_n and num_current_iterations <= max_planes and room_planes is None:
        if debug:
            print(f"Current PCD points: {len(current_pcd.points)}")
        plane_model, inliers = current_pcd.segment_plane(distance_threshold=distance_threshold,
                                                         ransac_n=ransac_n,
                                                         num_iterations=num_iterations)
        list_planes.append(indplane.PlaneIndoor(plane_model))
        if debug:
            print(f"Plane equation: {plane_model.A:.2f}x + {plane_model.B:.2f}y + {plane_model.C:.2f}z + {plane_model.D:.2f} = 0")
            print(f"Number of points in the above plane: {len(inliers)}")
        if debug:
            inlier_cloud = current_pcd.select_by_index(inliers)
            inlier_cloud_file = f"/tmp/inlier_{num_current_iterations-1}.pcd"
            o3d.io.write_point_cloud(inlier_cloud_file, inlier_cloud)
            list_inlier_clouds.append(inlier_cloud)
        outlier_cloud = current_pcd.select_by_index(inliers, invert=True)
        if debug:
            outlier_cloud_file = f"/tmp/outlier_{num_current_iterations - 1}.pcd"
            o3d.io.write_point_cloud(outlier_cloud_file, outlier_cloud)
            list_outlier_clouds_files.append(outlier_cloud_file)
        current_pcd = o3d.geometry.PointCloud(outlier_cloud)
        print(list_planes)
        room_planes = find_room_planes(list_planes, max_tolerance_degrees=max_tolerance_degrees, debug=debug)
        print(num_current_iterations)
        num_current_iterations += 1
    return room_planes

def get_negative_of_room_limits(room_limits: Dict[str, o3d.geometry.PointCloud], size: int = 10000,
                                threshold: float = 0.05, color: Tuple[float, float, float] = None) -> \
        Dict[str, o3d.geometry.PointCloud]:
    """

    :param room_limits:
    :return:

    :Example:

    ::

        >>> import findroom
        >>> import os
        >>> import open3d as o3d
        >>> home_dir = os.getenv("HOME")
        >>> pcd_filename = home_dir + "/Lantegi/FEBRERO/Kubic_v2 - e1 - 2021-01-26-102719/voxelized2cm.pcd"
        >>> pcd = o3d.io.read_point_cloud(pcd_filename)
        >>> inside_room, raw_limits_room, limits_room, plane_models = findroom.find_room_in_hololens_pointcloud(pcd, distance_threshold=0.05, plane_thickness= 0.05)
        >>> dict_negatives = findroom.get_negative_of_room_limits(limits_room)
        >>> o3d.visualization.draw_geometries([dict_negatives["ceiling"], dict_negatives["floor"], dict_negatives["wall_1_1"], dict_negatives["wall_1_2"], dict_negatives["wall_2_1"], dict_negatives["wall_2_2"]])
    """
    dict_negatives = dict()
    dict_negatives["ceiling"] = pointcloud.get_pointcloud_negative_of_pointcloud(room_limits["ceiling"], size, threshold, color)
    dict_negatives["floor"] = pointcloud.get_pointcloud_negative_of_pointcloud(room_limits["floor"], size, threshold, color)
    dict_negatives["wall_1_1"] = pointcloud.get_pointcloud_negative_of_pointcloud(room_limits["wall_1_1"], size, threshold, color)
    dict_negatives["wall_1_2"] = pointcloud.get_pointcloud_negative_of_pointcloud(room_limits["wall_1_2"], size, threshold, color)
    dict_negatives["wall_2_1"] = pointcloud.get_pointcloud_negative_of_pointcloud(room_limits["wall_2_1"], size, threshold, color)
    dict_negatives["wall_2_2"] = pointcloud.get_pointcloud_negative_of_pointcloud(room_limits["wall_2_2"], size, threshold, color)
    return dict_negatives

def get_raw_limits_in_order(pcd: o3d.geometry.PointCloud, dict_planes: Dict[str, Tuple[float, float, float, float]],
                            plane_thickness: float = 0.05, precise = "ceiling") -> Dict[str, o3d.geometry.PointCloud]:
    dict_raw_limits = {}
    if precise == "wall_2_1":
        dict_clouds = pointcloud.get_partition_of_pointcloud_by_quasi_parallel_planes_with_thickness(pcd, dict_planes["ceiling"],
                                                                                                          dict_planes["floor"],
                                                                                                          plane_thickness)
        dict_raw_limits["ceiling"] = o3d.geometry.PointCloud(dict_clouds["in_plane_1"])
        dict_raw_limits["floor"] = o3d.geometry.PointCloud(dict_clouds["in_plane_2"])
        next_pcd = o3d.geometry.PointCloud(dict_clouds["middle"])
        # second, the first pair of walls
        dict_clouds = pointcloud.get_partition_of_pointcloud_by_quasi_parallel_planes_with_thickness(next_pcd, dict_planes["wall_1_1"],
                                                                                                          dict_planes["wall_1_2"],
                                                                                                          plane_thickness)
        dict_raw_limits["wall_1_1"] = o3d.geometry.PointCloud(dict_clouds["in_plane_1"])
        dict_raw_limits["wall_1_2"] = o3d.geometry.PointCloud(dict_clouds["in_plane_2"])
        next_pcd = o3d.geometry.PointCloud(dict_clouds["middle"])
        # third, the second pair of walls
        dict_clouds = pointcloud.get_partition_of_pointcloud_by_quasi_parallel_planes_with_thickness(next_pcd, dict_planes["wall_2_1"],
                                                                                                          dict_planes["wall_2_2"],
                                                                                                          plane_thickness)
        dict_raw_limits["wall_2_1"] = o3d.geometry.PointCloud(dict_clouds["in_plane_1"])
        dict_raw_limits["wall_2_2"] = o3d.geometry.PointCloud(dict_clouds["in_plane_2"])
    if precise == "wall_1_1":
        dict_clouds = pointcloud.get_partition_of_pointcloud_by_quasi_parallel_planes_with_thickness(pcd, dict_planes["ceiling"],
                                                                                                          dict_planes["floor"],
                                                                                                          plane_thickness)
        dict_raw_limits["ceiling"] = o3d.geometry.PointCloud(dict_clouds["in_plane_1"])
        dict_raw_limits["floor"] = o3d.geometry.PointCloud(dict_clouds["in_plane_2"])
        next_pcd = o3d.geometry.PointCloud(dict_clouds["middle"])
        # second, the first pair of walls
        dict_clouds = pointcloud.get_partition_of_pointcloud_by_quasi_parallel_planes_with_thickness(next_pcd, dict_planes["wall_2_1"],
                                                                                                          dict_planes["wall_2_2"],
                                                                                                          plane_thickness)
        dict_raw_limits["wall_2_1"] = o3d.geometry.PointCloud(dict_clouds["in_plane_1"])
        dict_raw_limits["wall_2_2"] = o3d.geometry.PointCloud(dict_clouds["in_plane_2"])
        next_pcd = o3d.geometry.PointCloud(dict_clouds["middle"])
        # third, the second pair of walls
        dict_clouds = pointcloud.get_partition_of_pointcloud_by_quasi_parallel_planes_with_thickness(next_pcd, dict_planes["wall_1_1"],
                                                                                                          dict_planes["wall_1_2"],
                                                                                                          plane_thickness)
        dict_raw_limits["wall_1_1"] = o3d.geometry.PointCloud(dict_clouds["in_plane_1"])
        dict_raw_limits["wall_1_2"] = o3d.geometry.PointCloud(dict_clouds["in_plane_2"])
    if precise == "ceiling":
        dict_clouds = pointcloud.get_partition_of_pointcloud_by_quasi_parallel_planes_with_thickness(pcd, dict_planes["wall_1_1"],
                                                                                                          dict_planes["wall_1_2"],
                                                                                                          plane_thickness)
        dict_raw_limits["wall_1_1"] = o3d.geometry.PointCloud(dict_clouds["in_plane_1"])
        dict_raw_limits["wall_1_2"] = o3d.geometry.PointCloud(dict_clouds["in_plane_2"])
        next_pcd = o3d.geometry.PointCloud(dict_clouds["middle"])
        # second, the first pair of walls
        dict_clouds = pointcloud.get_partition_of_pointcloud_by_quasi_parallel_planes_with_thickness(next_pcd, dict_planes["wall_2_1"],
                                                                                                          dict_planes["wall_2_2"],
                                                                                                          plane_thickness)
        dict_raw_limits["wall_2_1"] = o3d.geometry.PointCloud(dict_clouds["in_plane_1"])
        dict_raw_limits["wall_2_2"] = o3d.geometry.PointCloud(dict_clouds["in_plane_2"])
        next_pcd = o3d.geometry.PointCloud(dict_clouds["middle"])
        # third, the second pair of walls
        dict_clouds = pointcloud.get_partition_of_pointcloud_by_quasi_parallel_planes_with_thickness(next_pcd, dict_planes["ceiling"],
                                                                                                          dict_planes["floor"],
                                                                                                          plane_thickness)
        dict_raw_limits["ceiling"] = o3d.geometry.PointCloud(dict_clouds["in_plane_1"])
        dict_raw_limits["floor"] = o3d.geometry.PointCloud(dict_clouds["in_plane_2"])
    return dict_clouds["middle"], dict_raw_limits


# return the middle (inside) of the room
#@profile
def find_room_in_hololens_pointcloud(pcd: o3d.geometry.PointCloud,
                                     y_ceiling_floor: Tuple[float, float] = None,
                                     given_planes: Dict[str, Tuple[float, float, float, float]] = None,
                                     distance_threshold: float = 0.01,
                                     ransac_n: int = 3,
                                     num_iterations: int = 1000,
                                     max_planes: int = 15,
                                     max_tolerance_degrees: float = 10,
                                     plane_thickness: float = 0.05, precise: str = "ceiling", debug: bool = False) -> \
        Tuple[
            o3d.geometry.PointCloud, Dict[str, o3d.geometry.PointCloud], Dict[str, o3d.geometry.PointCloud], Dict[str, Tuple[float, float, float, float]]]:
    """
    Finds the inside of a room in the HOLOLENS pointcloud. It also returns a dictionary with the
    pointclouds that surround the room. The pointclouds are addressed as "ceiling", "floor", "wall_1_1",
    "wall_1_2", "wall_2_1" and "wall_2_2", and the room planes as well. This pointclouds could be bigger than
    the real limits, due to RANSAC adding points beyond the room. Therefore, there are two dictionaries: the first one
    is the here defined, and the second one is after constraining to the convex hull of the eight points
    where the six planes intersect.

    :param pcd: Pointcloud where the room should be located.
    :type pcd: o3d.geometry.PointCloud
    :param distance_threshold: Maximum distance from the plane model, default is 0.01.
    :type distance_threshold: float
    :param ransac_n: Minimum number of points for a plane, default is 3.
    :type ransac_n: int
    :param num_iterations: Maximum number of iterations in the RANSAC method, default is 1000.
    :type num_iterations: int
    :param max_planes: Maximum number of planes to search for, default is 15.
    :type max_planes: int
    :param max_tolerance_degrees: Tolerance for parallel and perpendicular condition, default is 10 degrees.
    :type max_tolerance_degrees: float
    :param plane_thickness: Maximum distance to the plane, default is 0.05.
    :type plane_thickness: float
    :return: The inside room and a dictionary with the six point clouds that delimit it, as well as the plane models.
    :rtype: Tuple[o3d.geometry.PointCloud, Dict[str, o3d.geometry.PointCloud], Dict[str, o3d.geometry.PointCloud], Dict[str, Tuple[float, float, float, float]]]

    :Example:

    ::

        >>> import findroom
        >>> import os
        >>> import open3d as o3d
        >>> home_dir = os.getenv("HOME")
        >>> pcd_filename = home_dir + "/Lantegi/FEBRERO/Kubic_v2 - e1 - 2021-01-26-102719/voxelized2cm.pcd"
        >>> pcd = o3d.io.read_point_cloud(pcd_filename)
        >>> inside_room, raw_limits_room, limits_room, plane_models = findroom.find_room_in_hololens_pointcloud(pcd, distance_threshold=0.05, plane_thickness= 0.05)
        >>> inside_room
        PointCloud with 159002 points.
        >>> limits_room
        {'ceiling': PointCloud with 44070 points.,
        'floor': PointCloud with 254870 points.,
        'wall_1_1': PointCloud with 82133 points.,
        'wall_1_2': PointCloud with 61470 points.,
        'wall_2_1': PointCloud with 26954 points.,
        'wall_2_2': PointCloud with 63098 points.}
        >>> plane_models
        {'ceiling': array([-0.01165034,  0.99701198, -0.07636349, -2.14752472]),
        'floor': array([0.00290915, 0.99999014, 0.0033559 , 1.64111861]),
        'wall_1_1': array([ 0.91992695,  0.01246937, -0.39189147, -2.36333332]),
        'wall_1_2': array([ 0.91880986, -0.0078184 , -0.39462299,  4.58650518]),
        'wall_2_1': array([ 0.34378658,  0.10654374,  0.93298404, 12.13518052]),
        'wall_2_2': array([0.39340657, 0.01036559, 0.91930617, 5.26312062])}
        >>> o3d.visualization.draw_geometries([inside_room])
        >>> o3d.visualization.draw_geometries([raw_limits_room["ceiling"], raw_limits_room["floor"], raw_limits_room["wall_1_1"], raw_limits_room["wall_1_2"], raw_limits_room["wall_2_1"], raw_limits_room["wall_2_2"]])
        >>> o3d.visualization.draw_geometries([limits_room["ceiling"], limits_room["floor"], limits_room["wall_1_1"], limits_room["wall_1_2"], limits_room["wall_2_1"], limits_room["wall_2_2"]])
    """
    if given_planes is not None:
        dict_planes = given_planes
    else:
        dict_planes = find_room_planes_in_hololens_pointcloud(pcd, y_ceiling_floor, distance_threshold, ransac_n,
                                                          num_iterations, max_planes,
                                                          max_tolerance_degrees, debug)
        if dict_planes is None:
            return None

    inside_room, dict_raw_limits = get_raw_limits_in_order(pcd, dict_planes, plane_thickness, precise=precise)
    limit_points = find_limit_points_from_room_planes(dict_planes)
    pcd_just_covering_the_inside = o3d.geometry.PointCloud()
    eight_limit_points = [limit_points["ceiling"][0], limit_points["ceiling"][1], limit_points["ceiling"][2],
                          limit_points["ceiling"][3], limit_points["floor"][0], limit_points["floor"][1],
                          limit_points["floor"][2], limit_points["floor"][3]]
    pcd_just_covering_the_inside.points = o3d.utility.Vector3dVector(eight_limit_points)
    hull = ConvexHull(np.asarray(pcd_just_covering_the_inside.points))
    dict_limits = {}
    dict_limits["ceiling"] = pointcloud.get_pointcloud_after_substracting_convex_hull(dict_raw_limits["ceiling"], hull, reverse = True)
    dict_limits["floor"] = pointcloud.get_pointcloud_after_substracting_convex_hull(dict_raw_limits["floor"], hull, reverse = True)
    dict_limits["wall_1_1"] = pointcloud.get_pointcloud_after_substracting_convex_hull(dict_raw_limits["wall_1_1"], hull, reverse = True)
    dict_limits["wall_1_2"] = pointcloud.get_pointcloud_after_substracting_convex_hull(dict_raw_limits["wall_1_2"], hull, reverse = True)
    dict_limits["wall_2_1"] = pointcloud.get_pointcloud_after_substracting_convex_hull(dict_raw_limits["wall_2_1"], hull, reverse = True)
    dict_limits["wall_2_2"] = pointcloud.get_pointcloud_after_substracting_convex_hull(dict_raw_limits["wall_2_2"], hull, reverse = True)
    return inside_room, dict_raw_limits, dict_limits, dict_planes


def is_potential_ceiling_or_floor(plane: Type[indplane.PlaneIndoor],
                                  max_tolerance_degrees: float = 10, axis: str = "Y") -> bool:
    """
    Returns True if the plane model is a potential ceiling or floor, False otherwise. It checks if the
    plane model is perpendicular to the Y axis up to some tolerance, default is 10 degrees.

    :param plane: Plane.
    :type plane: Type[indplane.PlaneIndoor]
    :param max_tolerance_degrees: Tolerance to perfect alignment in degrees.
    :type max_tolerance_degrees: float
    :return: True if it is a potential ceiling or floor, False otherwise.

    :Example:

    ::

        >>> import findroom
        >>> plane = indplane.PlaneIndoor((0, 0, 1, 5))
        >>> findroom.is_potential_ceiling_or_floor(plane)
        False
        >>> plane = indplane.PlaneIndoor((0, 1, 0, 5))
        >>> findroom.is_potential_ceiling_or_floor(plane)
        True
        >>> plane = indplane.PlaneIndoor((0.1, 1, 0.1, 5))
        >>> findroom.is_potential_ceiling_or_floor(plane)
        True
        >>> findroom.is_potential_ceiling_or_floor(plane, max_tolerance_degrees=5)
        False
    """
    max_tolerance_radians = max_tolerance_degrees * math.pi / 180
    plane_angle = indplane.get_angle_between_plane_and_axis(plane, axis)
    return abs((math.pi / 2) - abs(plane_angle)) <= max_tolerance_radians


def is_potential_wall(plane: Type[indplane.PlaneIndoor],
                      max_tolerance_degrees: float = 10, axis = "Y") -> bool:
    """
    Returns True if the plane model is a potential wall, False otherwise. It checks if the
    plane model is perpendicular to the Y = 0 plane up to some tolerance, default is 10 degrees.

    :param plane: Plane.
    :type plane: Type[indplane.PlaneIndoor]
    :param max_tolerance_degrees: Tolerance to perfect alignment in degrees.
    :type max_tolerance_degrees: float
    :return: True if it is a potential ceiling or floor, False otherwise.

    :Example:

    ::

        >>> import findroom
        >>> plane = indplane.PlaneIndoor((0, 1, 0, 5))
        >>> findroom.is_potential_wall(plane)
        False
        >>> plane = indplane.PlaneIndoor((0, 0, 1, 5))
        >>> findroom.is_potential_wall(plane)
        True
        >>> plane = indplane.PlaneIndoor((0.1, 0.1, 1, 5))
        >>> findroom.is_potential_wall(plane)
        True
        >>> findroom.is_potential_wall(plane, max_tolerance_degrees=5)
        False
    """

    max_tolerance_radians = max_tolerance_degrees * math.pi / 180
    plane_angle = indplane.get_angle_plane_with_plane_axis_equal_0(plane, axis)
    return abs((math.pi / 2) - abs(plane_angle)) <= max_tolerance_radians


def get_ceiling_and_floor_from_candidates(list_planes: List[Type[indplane.PlaneIndoor]]) \
        -> Dict[str, Type[indplane.PlaneIndoor]]:
    """
    Returns the best guesses for ceiling and floor from the list of candidates. For each plane model
    the point P in the plane closest to the origin is computed. The ceiling is the plane model
    with the highest value of the Y coordinate of P. The floor is computed likewise for the lowest value
    of the Y coordinate of P. It requires a minimum of two candidates.

    :param list_planes: List of plane models with floor and ceiling candidates.
    :type list_planes: List[Type[indplane.PlaneIndoor]]
    :return: Dictionary with two keys, "ceiling_model" and "floor_model".
    :rtype: Dict[str, Type[indplane.PlaneIndoor]]

    :Example:

    ::

        >>> import findroom
        >>> plane1 = indplane.PlaneIndoor((0, 1, 0, 5))
        >>> plane2 = indplane.PlaneIndoor((0.1, 3, 0.1, 5))
        >>> plane3 = indplane.PlaneIndoor((0, 11, 0, 5))
        >>> plane4 = indplane.PlaneIndoor((0.1, 7, 0.1, 5))
        >>> list_planes = [plane1, plane2, plane3, plane4]
        >>> dict_models = findroom.get_ceiling_and_floor_from_candidates(list_planes)
        >>> dict_models
        {'ceiling_model': (0, 11, 0, 5), 'floor_model': (0, 1, 0, 5)}
    """
    index_ceiling = -1
    index_floor = -1
    dist_ceiling = 0
    dist_floor = 0
    y_ceiling = -1000
    y_floor = 1000
    for index, plane_model in enumerate(list_planes):
        p = indplane.get_point_on_plane_closest_to_the_origin(list_planes[index])
        # print(f"Plane Model: {list_plane_models[index]}\n")
        # print(f"Point points on plane closest to the origin: {N(p)}\n")
        d = indplane.get_distance_between_plane_and_origin(list_planes[index])
        # if p[1] >= y_ceiling and d > dist_ceiling:
        if p[1] >= y_ceiling:
            index_ceiling = index
            dist_ceiling = d
            y_ceiling = p[1]
        # if p[1] < y_floor and d > dist_floor:
        if p[1] < y_floor:
            index_floor = index
            dist_floor = d
            y_floor = p[1]
    return {"ceiling_model": list_planes[index_ceiling], "floor_model": list_planes[index_floor]}


# check if walls are parallel to each other in pairs and return the four walls
# TODO check minimum distances between parallel walls.
def return_four_walls_from_candidates(list_planes: List[Type[indplane.PlaneIndoor]],
                                      max_tolerance_degrees: float = 10, debug: bool = False) -> \
        Optional[Tuple[Type[indplane.PlaneIndoor], Type[indplane.PlaneIndoor],
                       Type[indplane.PlaneIndoor], Type[indplane.PlaneIndoor]]]:
    """
    Given a list of plane models candidates to walls, it extracts four of them, such that the first
    and second are parallel, the third and four as well, and these two pairs are perpendicular to
    each other. If no candidates under these conditions are found, it returns None.

    :param list_planes: List of all the candidate plane for potential walls.
    :type list_planes: List[Type[indplane.PlaneIndoor]]
    :param max_tolerance_degrees: Tolerance for parallel and perpendicular conditions, default to 10 degrees.
    :type max_tolerance_degrees: float
    :param debug: if debug information is shown, default is False.
    :type debug: bool
    :return: The four walls.
    :rtype: Optional[Tuple[Type[indplane.PlaneIndoor], Type[indplane.PlaneIndoor],
                       Type[indplane.PlaneIndoor], Type[indplane.PlaneIndoor]]]

    :Example:

    ::

        >>> import findroom
        >>> p1 = indplane.PlaneIndoor((1, 2, 3, 4))
        >>> p2 = indplane.PlaneIndoor((3, 6.1, 9.2, 12))
        >>> p3 = indplane.PlaneIndoor((3, 1, -1.67, 10))
        >>> p4 = indplane.PlaneIndoor((6.1, 2.1, -3.34, -35))
        >>> p5 = indplane.PlaneIndoor((-6.33, 10.67, -5, 0))
        >>> p6 = indplane.PlaneIndoor((-6.33, 10.67, -5, 15))
        >>> p7 = indplane.PlaneIndoor((1, 1, 1, 17))
        >>> p8 = indplane.PlaneIndoor((-1, -67, 567, 12))
        >>> list_plane_models = [p1, p7, p8, p6, p5, p4, p3, p2]
        >>> findroom.return_four_walls_from_candidates(list_planes)
        ((1, 2, 3, 4), (3, 6.1, 9.2, 12), (-6.33, 10.67, -5, 15), (-6.33, 10.67, -5, 0))
    """
    # we have to find a set of four such that there are two pairs of parallel planes
    # it is kind of finding equivalence classes
    if debug:
        print("Entering in return_four_walls")
        for current_plane in list_planes:
            print(current_plane)

    list_of_parallel_classes = indplane.get_partition_of_list_of_planes_by_parallelism(list_planes, max_tolerance_degrees)

    if debug:
        for parallel_class in list_of_parallel_classes:
            print("Parallel class:")
            for current_plane in parallel_class:
                print(current_plane)

    # remove parallel classes with less than two elements
    list_of_parallel_classes = [parallel_class for parallel_class in list_of_parallel_classes
                                if len(parallel_class) >= 2]
    if len(list_of_parallel_classes) < 2:
        return None
    # now find the first two classes that are perpendicular to each other
    perpendicular_classes = indplane.get_lists_of_planes_perpendicular_to_each_other_in_partition_first_two(
        list_of_parallel_classes,
        max_tolerance_degrees=max_tolerance_degrees,
        min_percentage=0)
    if perpendicular_classes is None:
        return None
    return perpendicular_classes[0][0], perpendicular_classes[0][1], \
           perpendicular_classes[1][0], perpendicular_classes[1][1]


# check if the list of plane_models define a room, and return the ceiling, floor and four walls
def find_room_planes(planes: List[Type[indplane.PlaneIndoor]],
                     max_tolerance_degrees: float = 10, minimum_distance: float = 2.0,
                     debug: bool = False) -> \
        Optional[Dict[str, Type[indplane.PlaneIndoor]]]:
    """
    Returns the ceiling, floor and four walls plane models corresponding to a room from a list of plane
    models. It supposes that the ceiling is +Y. The room has to be wider than *minimum_distance* from
    wall to wall (or ceiling to floor). These distances are computed from the limit points of the room,
    as the planes ar enot going to be exactly parallel. If no good candidates are found it returns None.
    The dict entries are: "ceiling", "floor", "wall_1_1", "wall_1_2", "wall_2_1", "wall_2_2". Wall_1_1 and
    wall_1_2 are parallel to each other, likewise for wall_2_1 and wall_2_2. Wall_1_1 and wall_1_2
    are perpendicular to wall_2_1 and wall_2_2.

    :param planes: List of plane models from which extract the room.
    :type planes: List[Type[indplane.PlaneIndoor]]
    :param max_tolerance_degrees: Tolerance for parallel and perpendicular conditions, default is 10 degrees.
    :type max_tolerance_degrees: float
    :param minimum_distance: Minimum distance between the walls of the room. Default value is 2.0.
    :type minimum_distance: float
    :param debug: if debug information is shown, default is False.
    :type debug: bool
    :return: Dictionary with ceiling, floor and walls plane models. None if no suitable candidates are found. The keys are "ceiling", "floor", "wall_1_1", "wall_1_2", "wall_2_1", "wall_2_2".
    :rtype: Optional[Dict[str, Type[indplane.PlaneIndoor]]]

    :Example:

    ::

        >>> import findroom
        >>> p1 = indplane.PlaneIndoor((1, 2, 3, 4))
        >>> p2 = indplane.PlaneIndoor((3, 6.1, 9.2, 12))
        >>> p3 = indplane.PlaneIndoor((3, 1, -1.67, 10))
        >>> p4 = indplane.PlaneIndoor((6.1, 2.1, -3.34, -35))
        >>> p5 = indplane.PlaneIndoor((-6.33, 10.67, -5, 0))
        >>> p6 = indplane.PlaneIndoor((-6.33, 10.67, -5, 15))
        >>> p7 = indplane.PlaneIndoor((1, 1, 1, 17))
        >>> p8 = indplane.PlaneIndoor((-1, -67, 567, 12))
        >>> list_plane_models = [p1, p7, p8, p6, p5, p4, p3, p2]
        >>> room_planes = findroom.find_room_planes(list_plane_models)
        >>> type(room_planes)
        <class 'NoneType'>
        >>> p1 = indplane.PlaneIndoor((0, 2, 0, 4))
        >>> p2 = indplane.PlaneIndoor((0.1, 3.1, 0.1, -10))
        >>> p3 = indplane.PlaneIndoor((3, 0, 0, 10))
        >>> p4 = indplane.PlaneIndoor((6, 0.1, -0.2, -35))
        >>> p5 = indplane.PlaneIndoor((0, 0, -5, 0))
        >>> p6 = indplane.PlaneIndoor((1, 0.3, 35, 15))
        >>> p7 = indplane.PlaneIndoor((1, 1, 1, 17))
        >>> p8 = indplane.PlaneIndoor((-1, -67, 567, 12))
        >>> list_plane_models = [p1, p7, p8, p6, p5, p4, p3, p2]
        >>> room_planes = findroom.find_room_planes(list_plane_models)
        >>> room_planes
        {'ceiling': (0.1, 3.1, 0.1, -10), 'floor': (0, 2, 0, 4), 'wall_1_1': (-1, -67, 567, 12), 'wall_1_2': (1, 0.3, 35, 15), 'wall_2_1': (6, 0.1, -0.2, -35), 'wall_2_2': (3, 0, 0, 10)}
    """
    if len(planes) < 6:
        return None
    list_potential_ceiling_or_floor = []
    list_potential_walls = []
    for current_plane in planes:
        if is_potential_ceiling_or_floor(current_plane, max_tolerance_degrees=max_tolerance_degrees):
            list_potential_ceiling_or_floor.append(current_plane)
        if is_potential_wall(current_plane, max_tolerance_degrees=max_tolerance_degrees):
            list_potential_walls.append(current_plane)
    if len(list_potential_ceiling_or_floor) < 2:
        return None
    if len(list_potential_walls) < 4:
        return None
    dict_models = \
        get_ceiling_and_floor_from_candidates(list_potential_ceiling_or_floor)
    ceiling_model = dict_models["ceiling_model"]
    floor_model = dict_models["floor_model"]
    if debug:
        print("list_potential_ceiling_or_floor")
        for current_plane in list_potential_ceiling_or_floor:
            print(current_plane)
        print("Ceiling:", ceiling_model)
        print("Floor:", floor_model)
        print("list_potential_walls")
        for current_plane in list_potential_walls:
            print(current_plane)
    walls = return_four_walls_from_candidates(list_potential_walls, max_tolerance_degrees=max_tolerance_degrees,
                                              debug=debug)
    if walls is None:
        return None
    if debug:
        print("Plane ceiling found")
        print(ceiling_model)
        print("Plane floor found")
        print(floor_model)
    dict_planes = {"ceiling": ceiling_model, "floor": floor_model, "wall_1_1": walls[0], "wall_1_2": walls[1],
                   "wall_2_1": walls[2], "wall_2_2": walls[3]}
    # check if the room is wide enough
    limit_points = find_limit_points_from_room_planes(dict_planes)
    if is_the_room_wide_enough(limit_points, minimum_distance):
        return dict_planes
    else:
        return None

def is_the_room_wide_enough(limit_points: Dict[str, Tuple[Tuple[float, float, float], Tuple[float, float, float],
                        Tuple[float, float, float], Tuple[float, float, float]]],
                            minimum_distance: float = 2.0) -> bool:
    ceiling_points = limit_points["ceiling"]
    first_distance_ceiling = np.linalg.norm(np.array(ceiling_points[0])-np.array(ceiling_points[1]))
    second_distance_ceiling = np.linalg.norm(np.array(ceiling_points[1])-np.array(ceiling_points[2]))
    distance_ceiling_floor = np.linalg.norm(np.array(ceiling_points[0])-np.array(limit_points["floor"][0]))
    return first_distance_ceiling >= minimum_distance and \
           second_distance_ceiling >= minimum_distance and \
           distance_ceiling_floor >= minimum_distance


def get_numpy_array_from_limit_points(limit_points: Dict[str, Tuple[Tuple[float, float, float], Tuple[float, float, float],
                        Tuple[float, float, float], Tuple[float, float, float]]]) -> np.ndarray:
    """
    Returns the numpy array consisting of the eight points of *limit_points*, in the same order.

    :param limit_points: The limit points of a room.
    :type limit_points:  Dict[str, Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]]
    :return: The limit points in a numpy array.
    :rtype: np.arrays

    :Example:

    ::

        >>> import findroom
        >>> import os
        >>> import open3d as o3d
        >>> home_dir = os.getenv("HOME")
        >>> pcd_filename = home_dir + "/Lantegi/FEBRERO/Kubic_v2 - e1 - 2021-01-26-102719/voxelized2cm.pcd"
        >>> pcd = o3d.io.read_point_cloud(pcd_filename)
        >>> room_planes = findroom.find_room_planes_in_hololens_pointcloud(pcd, distance_threshold=0.05)
        >>> limit_points = findroom.find_limit_points_from_room_planes(room_planes)
        >>> limit_points
        {'ceiling': ((-2.585664972152478, 1.648248855898352, -12.045802064601098),
        (-9.04968448875847, 1.4275304795309494, -9.482337437906097),
        (-6.292824877357577, 1.5998825167397532, -3.0435533997041655),
        (0.05990474820833932, 1.814499449765199, -5.784498825826017)),
        'floor': ((-2.5758002843819563, -1.590938542863108, -12.151111774606026),
        (-9.10383490564883, -1.5914027163048543, -9.555297987573882),
        (-6.298988305192846, -1.6104869321951165, -3.000987682956872),
        (0.12377556919950411, -1.6094908703176263, -5.769465054182009))}
        >>> numpy_points = findroom.get_numpy_array_from_limit_points(limit_points)
        >>> numpy_points
        array([[ -2.58566497,   1.64824886, -12.04580206],
            [ -9.04968449,   1.42753048,  -9.48233744],
            [ -6.29282488,   1.59988252,  -3.0435534 ],
            [  0.05990475,   1.81449945,  -5.78449883],
            [ -2.57580028,  -1.59093854, -12.15111177],
            [ -9.10383491,  -1.59140272,  -9.55529799],
            [ -6.29898831,  -1.61048693,  -3.00098768],
            [  0.12377557,  -1.60949087,  -5.76946505]])
    """
    return np.array([limit_points["ceiling"][0], limit_points["ceiling"][1], limit_points["ceiling"][2],
                     limit_points["ceiling"][3], limit_points["floor"][0], limit_points["floor"][1],
                    limit_points["floor"][2], limit_points["floor"][3]])

def expand_limit_points(limit_points: Dict[str, Tuple[Tuple[float, float, float], Tuple[float, float, float],
                        Tuple[float, float, float], Tuple[float, float, float]]], size: float = 1.0) -> \
        Dict[str, Tuple[Tuple[float, float, float], Tuple[float, float, float],
                        Tuple[float, float, float], Tuple[float, float, float]]]:
    """
    Expands the limit points outwards, in such a way that the new points are *size* away from the corresponding
    original points.

    :param limit_points: The original limit points.
    :type limit_points: Dict[str, Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]]
    :param size: The linear size of the outward expansion.
    :type size: float
    :return: A new dict of limit points.
    :rtype: Dict[str, Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]]

    :Example:

    ::

        >>> import findroom
        >>> import os
        >>> import open3d as o3d
        >>> home_dir = os.getenv("HOME")
        >>> pcd_filename = home_dir + "/Lantegi/FEBRERO/Kubic_v2 - e1 - 2021-01-26-102719/voxelized2cm.pcd"
        >>> pcd = o3d.io.read_point_cloud(pcd_filename)
        >>> room_planes = findroom.find_room_planes_in_hololens_pointcloud(pcd, distance_threshold=0.05)
        >>> limit_points = findroom.find_limit_points_from_room_planes(room_planes)
        >>> limit_points
        {'ceiling': ((-2.585664972152478, 1.648248855898352, -12.045802064601098),
        (-9.04968448875847, 1.4275304795309494, -9.482337437906097),
        (-6.292824877357577, 1.5998825167397532, -3.0435533997041655),
        (0.05990474820833932, 1.814499449765199, -5.784498825826017)),
        'floor': ((-2.5758002843819563, -1.590938542863108, -12.151111774606026),
        (-9.10383490564883, -1.5914027163048543, -9.555297987573882),
        (-6.298988305192846, -1.6104869321951165, -3.000987682956872),
        (0.12377556919950411, -1.6094908703176263, -5.769465054182009))}
        >>> new_limit_points = findroom.expand_limit_points(limit_points, 0.1)
        {'ceiling': ((-2.548759669438483, 1.6803939311098945, -12.133006954207314),
        (-9.138653759653101, 1.4550220445271118, -9.518788537896897),
        (-6.3282154375796535, 1.630653589043257, -2.9552321148603635),
        (0.1469263663650855, 1.8491812055463839, -5.7495072012664785)),
        'floor': ((-2.5393076585122065, -1.6218755411966865, -12.238925158501868),
        (-9.191666204357292, -1.6217446462056169, -9.592244303453574),
        (-6.334158773016039, -1.641588482952323, -2.9126942419003887),
        (0.2120085732577106, -1.640646653455662, -5.734191066997115))}
    """

    new_limit_points = {}
    np_points = get_numpy_array_from_limit_points(limit_points)
    centroid = tuple(np.mean(np_points, axis=0))
    new_ceiling_0 = vector.get_point_further_along_direction(centroid, limit_points["ceiling"][0], size)
    new_ceiling_1 = vector.get_point_further_along_direction(centroid, limit_points["ceiling"][1], size)
    new_ceiling_2 = vector.get_point_further_along_direction(centroid, limit_points["ceiling"][2], size)
    new_ceiling_3 = vector.get_point_further_along_direction(centroid, limit_points["ceiling"][3], size)
    new_floor_0 = vector.get_point_further_along_direction(centroid, limit_points["floor"][0], size)
    new_floor_1 = vector.get_point_further_along_direction(centroid, limit_points["floor"][1], size)
    new_floor_2 = vector.get_point_further_along_direction(centroid, limit_points["floor"][2], size)
    new_floor_3 = vector.get_point_further_along_direction(centroid, limit_points["floor"][3], size)
    new_limit_points["ceiling"] = (new_ceiling_0, new_ceiling_1, new_ceiling_2, new_ceiling_3)
    new_limit_points["floor"] = (new_floor_0, new_floor_1, new_floor_2, new_floor_3)
    return new_limit_points

def get_inside_list_of_points(pcd: o3d.geometry.PointCloud, list_of_points: np.ndarray) -> o3d.geometry.PointCloud:
    hull = ConvexHull(list_of_points)
    inside_limit_points = pointcloud.get_pointcloud_after_substracting_convex_hull(pcd, hull, reverse=True)
    return inside_limit_points

def get_inside_limit_points(pcd: o3d.geometry.PointCloud, limit_points:  Dict[str, Tuple[Tuple[float, float, float], Tuple[float, float, float],
                        Tuple[float, float, float], Tuple[float, float, float]]]) -> o3d.geometry.PointCloud:
    """
    Returns the pointcloud that lies inside some limit points. The typical usage is to extend the limit points
    found when searching for the room, in order to try to remove false holes in walls.

    :param pcd: The original pointcloud.
    :type pcd: o3d.geometry.PointCloud
    :param limit_points: The limit points that delimit the space the select.
    :type limit_points: Dict[str, Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]]
    :return: The inside pointcloud.
    :rtpye: o3d.geometry.PointCloud

    :Example:

    ::

        >>> import findroom
        >>> import os
        >>> import open3d as o3d
        >>> home_dir = os.getenv("HOME")
        >>> pcd_filename = home_dir + "/Lantegi/FEBRERO/Kubic_v2 - e1 - 2021-01-26-102719/voxelized2cm.pcd"
        >>> pcd = o3d.io.read_point_cloud(pcd_filename)
        >>> room_planes = findroom.find_room_planes_in_hololens_pointcloud(pcd, distance_threshold=0.05)
        >>> limit_points = findroom.find_limit_points_from_room_planes(room_planes)
        >>> pcd_inside = findroom.get_inside_limit_points(pcd, limit_points)
        >>> o3d.visualization.draw_geometries([pcd_inside])
        >>> new_limit_points = findroom.expand_limit_points(limit_points, 0.5)
        >>> pcd_inside_expanded = findroom.get_inside_limit_points(pcd, new_limit_points)
        >>> o3d.visualization.draw_geometries([pcd_inside_expanded])
    """
    np_points = get_numpy_array_from_limit_points(limit_points)
    hull = ConvexHull(np_points)
    inside_limit_points = pointcloud.get_pointcloud_after_substracting_convex_hull(pcd, hull, reverse=True)
    return inside_limit_points

def change_axes_abel_to_axes_hololens(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    np_points = np.asarray(pcd.points)
    new_np_points = np.asarray([[point[1], point[2], point[0]] for point in np_points])
    pcd_results = o3d.geometry.PointCloud(pcd)
    pcd_results.points = o3d.utility.Vector3dVector(new_np_points)
    return pcd_results

def extract_boundary_slice(pcd: o3d.geometry.PointCloud,
                           plane: Type[indplane.PlaneIndoor],
                           limit_points: Tuple[Tuple[float, float, float],
                                               Tuple[float, float, float],
                                               Tuple[float, float, float],
                                               Tuple[float, float, float]],
                           plane_thickness = 0.05) -> o3d.geometry.PointCloud:
    normal_vector = plane.normal
    limit_convex_hull = list()
    size = normal_vector * plane_thickness
    print(size)
    for point in limit_points:
        new_point_positive = (point[0] + size[0], point[1] + size[1], point[2] + size[2])
        limit_convex_hull.append(new_point_positive)
        new_point_negative = (point[0] - size[0], point[1] - size[1], point[2] - size[2])
        limit_convex_hull.append(new_point_negative)
    np_points = np.asarray(limit_convex_hull)
    # print(limit_convex_hull)
    print(limit_convex_hull[0])
    # print(np_points)
    return get_inside_list_of_points(pcd, np_points)

def extract_boundaries_slices(pcd: o3d.geometry.PointCloud,
                              planes: Dict[str, Type[indplane.PlaneIndoor]],
                              plane_thickness: float = 0.05) -> Dict[str, o3d.geometry.PointCloud]:
    limit_points = find_limit_points_from_room_planes(planes)
    slices_dict = {}
    for name, plane_model in planes.items():
        slices_dict[name] = extract_boundary_slice(pcd, plane_model, limit_points[name], plane_thickness)
    return slices_dict
