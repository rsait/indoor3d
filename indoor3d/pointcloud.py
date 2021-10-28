"""
In this module we embed all the functions related to general geometry questions.
We use the module `vg <https://pypi.org/project/vg/>`_, which is useful for some operations
with vectors. The *plane_model* refers to four values (a, b, c, d) which
corresponds to the equation form Ax + By + Cz + D = 0. The normal to a plane is (A, B, C).
"""

import open3d as o3d
import numpy as np
import math
from sympy import Point3D, Plane
from typing import List, Tuple, Dict, Union, Optional, Iterable, NamedTuple
from typing import Type
import plane
from scipy.spatial import ConvexHull
import scipy
from numpy.random import default_rng
from collections import Counter

numeric = Union[int, float, np.number]

def get_distance_between_pointcloud_and_plane(pcd: o3d.geometry.PointCloud, plane: Type[plane.PlaneRSAIT]) -> float:
    """
    Returns the distance between the pointcloud and the plane.

    :param pcd: Pointcloud.
    :type pcd: o3d.geometry.PointCloud
    :param plane: Plane model.
    :type plane: Type[plane.PlaneRSAIT]
    :return: Distance between the pointcloud and the plane.
    :rtype: float

    :Example:

    ::

        >>> import pointcloudrsait
        >>> import plane
        >>> import open3d as o3d
        >>> mesh_box = o3d.geometry.TriangleMesh.create_box(width=1.0, height=5.0, depth=1.0)
        >>> pcd = mesh_box.sample_points_uniformly(number_of_points = 10000, seed = 42)
        >>> plane = plane.PlaneRSAIT((1, 1, 1, 1))
        >>> distance = pointcloudrsait.get_distance_between_pointcloud_and_plane(pcd, plane)
        >>> distance
        0.5886230622926497
    """
    a, b, c, d = plane.A, plane.B, plane.C, plane.D
    pcd_points = np.asarray(pcd.points)
    points_transposed = pcd_points.transpose()
    sign = points_transposed[0] * a + points_transposed[1] * b + points_transposed[2] * c + d
    numerator = np.abs(sign)
    denominator = math.sqrt(a * a + b * b + c * c)
    return np.min(np.abs(numerator / denominator))




def get_distance_between_pointcloud_and_point(pcd: o3d.geometry.PointCloud, point: np.ndarray) -> float:
    """
    Returns the distance between the pointcloud and the point.

    :param pcd: Pointcloud.
    :type pcd: o3d.geometry.PointCloud
    :param point: Point.
    :type point: np.ndarray
    :return: Distance between the pointcloud and the point.
    :rtype: float

    :Example:

    ::

        >>> import pointcloudrsait
        >>> import open3d as o3d
        >>> mesh_box = o3d.geometry.TriangleMesh.create_box(width=1.0, height=5.0, depth=1.0)
        >>> pcd = mesh_box.sample_points_uniformly(number_of_points = 10000, seed = 42)
        >>> distance = pointcloudrsait.get_distance_between_pointcloud_and_point(pcd, [10, 10, 10])
        >>> distance
        13.703526637666375
    """
    pcd_point = o3d.geometry.PointCloud()
    pcd_point.points = o3d.utility.Vector3dVector(np.array([point]))
    distances = pcd.compute_point_cloud_distance(pcd_point)
    return min(distances)

#@profile
# return the points in the plane, and in the two sides of the plane
# tolerance is the maximum distance of the points in the plane to the plane
def get_partition_of_pointcloud_by_plane_with_thickness(pcd: o3d.geometry.PointCloud,
                                                        plane: Type[plane.PlaneRSAIT],
                                                        plane_thickness: float = 0,
                                                        debug: bool = False) -> Dict[str, o3d.geometry.PointCloud]:
    '''
    Given a point cloud and a plane, returns the points which are in one side, in the other,
    or in the plane, if supposed it has some thickness. Such thickness is twice the **plane_thickness**
    parameter, because it is computed once for each side. It returns three elements:
    the points in the "positive" set, those such that the signed distance is positive;
    the points in the "negative" set, those whose signed distance is negative; and the "in_plane" set,
    those which are closer to the plane that the **plane_thickness** parameter. The elements are returned
    in a dictionary with "positive", "negative" and "in_plane" as keys.

    :param pcd: Pointcloud to be segmented.
    :type pcd: o3d.geometry.PointCloud
    :param plane_model: Plane to partition the space.
    :type plane_model: Type[plane.PlaneRSAIT]
    :param plane_thickness: Maximum distance from the plane to be considered for the pointcloud "in_plane".
    :type plane_thickness: float
    :param debug: if debug information is shown, default is False.
    :type debug: bool
    :return: Dictionary of pointclouds as values and with three keys: "positive", "negative" and "in_plane".
    :rtype: Dict[str, o3d.geometry.PointCloud]

    :Example:

    ::

        >>> import pointcloudrsait
        >>> import plane
        >>> import open3d as o3d
        >>> # import os
        >>> # home_dir = os.getenv("HOME")
        >>> # pcd = o3d.io.read_point_cloud(home_dir + "/Github/Lantegi/Code/Open3D/gui/skull.ply")
        >>> pcd = o3d.io.read_point_cloud("Code/Open3D/gui/skull.ply")
        >>> plane = plane.PlaneRSAIT((0, 0, 1, 0))
        >>> dict_clouds = pointcloudrsait.get_partition_of_pointcloud_by_plane_with_thickness(pcd, plane, plane_thickness = 10)
        >>> dict_clouds
        {'in_plane': PointCloud with 14374 points., 'positive': PointCloud with 57124 points., 'negative': PointCloud with 61511 points.}
    '''
    in_plane = o3d.geometry.PointCloud()
    positive = o3d.geometry.PointCloud()
    negative = o3d.geometry.PointCloud()

    pcd_points = np.asarray(pcd.points)
    pcd_colors = np.asarray(pcd.colors)
    dist, sign = plane.get_distance_and_sign_between_plane_and_points(plane, pcd_points)
    list_points_in_plane = pcd_points[dist <= plane_thickness]
    list_colors_in_plane = pcd_colors[dist <= plane_thickness]
    positive_condition = np.logical_and(dist > plane_thickness, sign > 0)
    list_points_in_positive = pcd_points[positive_condition]
    list_colors_in_positive = pcd_colors[positive_condition]
    negative_condition = np.logical_and(dist > plane_thickness, sign <= 0)
    list_points_in_negative = pcd_points[negative_condition]
    list_colors_in_negative = pcd_colors[negative_condition]

    in_plane.points = o3d.utility.Vector3dVector(list_points_in_plane)
    in_plane.colors = o3d.utility.Vector3dVector(list_colors_in_plane)
    positive.points = o3d.utility.Vector3dVector(list_points_in_positive)
    positive.colors = o3d.utility.Vector3dVector(list_colors_in_positive)
    negative.points = o3d.utility.Vector3dVector(list_points_in_negative)
    negative.colors = o3d.utility.Vector3dVector(list_colors_in_negative)
    if debug:
        print("find_sides_of_plane")
        print("in_plane")
        print(len(in_plane.points))
        print("positive")
        print(len(positive.points))
        print("negative")
        print(len(negative.points))

    return {"in_plane": in_plane, "positive": positive, "negative": negative}

#@profile
def get_partition_of_pointcloud_by_quasi_parallel_planes_with_thickness(pcd: o3d.geometry.PointCloud,
                                                                        plane_1: Type[plane.PlaneRSAIT],
                                                                        plane_2: Type[plane.PlaneRSAIT],
                                                                        plane_thickness: float = 0) -> Dict[str, o3d.geometry.PointCloud]:
    """
    Given a point cloud and two parallel (up to some tolerance not defined here) plane models,
    returns the points which are in plane_1, in plane_2, in one side of both, in the other,
    or in the middle. The plane is supposed to have some thickness. Such thickness is twice
    the **plane_thickness** parameter, because it is computed once for each side.
    It returns five elements: the points in the "in_plane_1" set, those which are closer to the plane
    that the **tolerance** parameter; same for points in the "in_plane_2" set; the points in the "middle"
    of the two planes; the points is "one_side"; and the points in the "other_side". These elements
    (pointclouds) are returned in a dictionary with "in_plane_1", "in_plane_2", "middle", "one_side"
    and "other_side" as keys.

    :param pcd: Pointcloud to be segmented.
    :type pcd: o3d.geometry.PointCloud
    :param plane_1: Plane model to partition the space.
    :type plane_1: Type[plane.PlaneRSAIT]
    :param plane_2: Plane model to partition the space.
    :type plane_2: Type[plane.PlaneRSAIT]
    :param plane_thickness: Maximum distance from the plane to be considered for the pointcloud "in_plane".
    :type plane_thickness: float
    :return: Dictionary of pointclouds as values and with five keys: "in_plane_1", "in_plane_2", "middle", "one_side" and "other_side".
    :rtype: Dict[str, o3d.geometry.PointCloud]

    :Example:

    ::

        >>> import pointcloudrsait
        >>> import plane
        >>> import open3d as o3d
        >>> # import os
        >>> # home_dir = os.getenv("HOME")
        >>> # pcd = o3d.io.read_point_cloud(home_dir + "/Github/Lantegi/Code/Open3D/gui/skull.ply")
        >>> pcd = o3d.io.read_point_cloud("Code/Open3D/gui/skull.ply")
        >>> plane_1 = plane.PlaneRSAIT((0, 0, 1, 0))
        >>> plane_2 = plane.PlaneRSAIT((0, 0, 1, 30))
        >>> dict_clouds = pointcloudrsait.get_partition_of_pointcloud_by_quasi_parallel_planes_with_thickness(pcd, plane_1, plane_2, plane_thickness = 5)
        >>> dict_clouds
        {'one_side': PointCloud with 60690 points., 'other_side': PointCloud with 42574 points., 'middle': PointCloud with 14660 points., 'in_plane_1': PointCloud with 7192 points., 'in_plane_2': PointCloud with 7893 points.}
    """
    # first, we find the space partitions by each plane
    dict_clouds_1 = get_partition_of_pointcloud_by_plane_with_thickness(pcd, plane_1, plane_thickness=plane_thickness)
    dict_clouds_2 = get_partition_of_pointcloud_by_plane_with_thickness(pcd, plane_2, plane_thickness=plane_thickness)
    point_in_plane_2 = plane.get_point_on_plane_closest_to_the_origin(plane_2)
    # we look for the sign of plane_2 with respect to plane_1
    distance, sign = plane.get_distance_and_sign_between_plane_and_point(plane_1, point_in_plane_2)
    # and then we divide the space according to that sign; the "good half" is the same sign than plane_2
    if sign > 0:
        one_side, next_pcd = dict_clouds_1["negative"], dict_clouds_1["positive"]
    else:
        one_side, next_pcd = dict_clouds_1["positive"], dict_clouds_1["negative"]
    point_in_plane_1 = plane.get_point_on_plane_closest_to_the_origin(plane_1)
    # we look for the sign of plane_1 with respect to plane_2
    distance, sign = plane.get_distance_and_sign_between_plane_and_point(plane_2, point_in_plane_1)
    # new partition of the previous "good half"
    dict_clouds_2 = get_partition_of_pointcloud_by_plane_with_thickness(next_pcd, plane_2, plane_thickness=plane_thickness)
    if sign > 0:
        other_side, middle = dict_clouds_2["negative"], dict_clouds_2["positive"]
    else:
        other_side, middle = dict_clouds_2["positive"], dict_clouds_2["negative"]
    dict_clouds = dict()
    dict_clouds["one_side"] = one_side
    dict_clouds["other_side"] = other_side
    dict_clouds["middle"] = middle
    dict_clouds["in_plane_1"] = dict_clouds_1["in_plane"]
    dict_clouds["in_plane_2"] = dict_clouds_2["in_plane"]
    return dict_clouds

def check_points_inside_convex_hull(points: Iterable[Iterable[float]], hull: scipy.spatial.ConvexHull) -> np.ndarray:
    """
    Checks if the points in the list are inside the convex hull. It returns True in that case,
    False otherwise. It does not work ok with points just in the hull.

    :param points: List of points.
    :type points: Iterable[Iterable[float]]
    :param hull: Convex hull to check against.
    :param hull: scipy.spatial.ConvexHull
    :return: List of booleans, one for each point, True if the point lies inside the convex hull, False otherwise.
    :rtype: np.ndarray

    :Example:

    ::

        >>> import pointcloudrsait
        >>> import numpy as np
        >>> from scipy.spatial import ConvexHull
        >>> points = np.array([[-2.65908417, 1.66316594, -12.05967958], [0.0459558, 1.72705225, -5.76867598], [ -6.27471812,   1.66093731,  -3.02014327], [-9.10951204, 1.59325844, -9.76566801], [-2.56569358, -1.59892799, -12.08053818], [0.1378323 , -1.63325758, -5.80021574], [-6.33750632, -1.6103617 , -2.9846542], [-9.17057479, -1.57383548, -9.73196329], [-4.49166262, 0.0285039, -7.65144228]])
        >>> hull = ConvexHull(points)
        >>> pointcloudrsait.check_points_inside_convex_hull([(points[2] + points[3])/2], hull)
        array([ True])
        >>> pointcloudrsait.check_points_inside_convex_hull([[0,0,0]], hull)
        array([False])
    """
    A = hull.equations[:, 0:-1]
    b = np.transpose(np.array([hull.equations[:, -1]]))
    return np.all((A @ np.transpose(points)) <= np.tile(-b, (1, len(points))), axis=0)

def get_pointcloud_after_substracting_convex_hull(pcd: o3d.geometry.PointCloud, substract: scipy.spatial.qhull.ConvexHull,
                                                  reverse: bool = False) \
        -> o3d.geometry.PointCloud:
    """
    Substracts a hull from a pointcloud. It removes all the points of the pointcloud that are
    inside the convex hull. The points in the surface of the hull remain. If *reverse* is True, it
    removes the points outside the convex hull.

    :param pcd: Pointcloud to process.
    :type pcd: o3d.geometry.PointCloud
    :param substract: Convex hull for checking if it is inside it.
    :type substract: scipy.spatial.qhull.ConvexHull
    :param reverse: If True, remove the points outside the hull. Default value is False.
    :type reverse: bool
    :return: Result of substracting the hull to the pointcloud
    :rtype: o3d.geometry.PointCloud

    :Example:

    ::

        >>> import pointcloudrsait
        >>> import open3d as o3d
        >>> from scipy.spatial import ConvexHull
        >>> mesh_box = o3d.geometry.TriangleMesh.create_box(width=1.0, height=5.0, depth=1.0)
        >>> pcd = mesh_box.sample_points_uniformly(number_of_points = 10000, seed = 42)
        >>> # pcd.paint_uniform_color((0,0,1))
        >>> points = [[0,0,0], [0,0,1], [0,1,0], [0,1,1], [1,0,0], [1,0,1], [1,1,0], [1,1,1]]
        >>> hull = ConvexHull(points)
        >>> pcd_minus_hull = pointcloudrsait.get_pointcloud_after_substracting_convex_hull(pcd, hull)
        >>> pcd_inside_hull = pointcloudrsait.get_pointcloud_after_substracting_convex_hull(pcd, hull, reverse = True)
        >>> # pcd_inside_hull.paint_uniform_color((1,0,0))
        >>> # o3d.visualization.draw_geometries([pcd_inside_hull, pcd_minus_hull])
        >>> pcd_minus_hull
        PointCloud with 7778 points.
        >>> pcd_inside_hull
        PointCloud with 2222 points.
    """
    pcd_points = np.asarray(pcd.points)
    if not reverse:
        points_outside_hull = pcd_points[np.invert(check_points_inside_convex_hull(pcd_points, substract))]
        pcd_outside = o3d.geometry.PointCloud()
        pcd_outside.points = o3d.utility.Vector3dVector(points_outside_hull)
        if len(pcd.colors) > 0:
            pcd_colors = np.asarray(pcd.colors)
            colors_outside_hull = pcd_colors[np.invert(check_points_inside_convex_hull(pcd_points, substract))]
            pcd_outside.colors = o3d.utility.Vector3dVector(colors_outside_hull)
        return pcd_outside
    else:
        points_inside_hull = pcd_points[check_points_inside_convex_hull(pcd_points, substract)]
        pcd_inside = o3d.geometry.PointCloud()
        pcd_inside.points = o3d.utility.Vector3dVector(points_inside_hull)
        if len(pcd.colors) > 0:
            pcd_colors = np.asarray(pcd.colors)
            colors_inside_hull = pcd_colors[check_points_inside_convex_hull(pcd_points, substract)]
            pcd_inside.colors = o3d.utility.Vector3dVector(colors_inside_hull)
        return pcd_inside

def get_pointcloud_uniform_sample_inside_convex_hull_of_pointcloud(pcd: o3d.geometry.PointCloud, size: int = 10000, seed: int = None,
                                                                   color: Tuple[float, float, float] = None) -> o3d.geometry.PointCloud:
    """
    Returns a pointcloud that is composed of points sampled inside the convex hull of another pointcloud.

    :param pcd: The input pointcloud.
    :type pcd: o3d.geometry.PointCloud
    :param size: Number of points to be sampled.
    :type size: int
    :param seed: Seed of the random number generator.
    :type seed: Optional[int]
    :param color: Color of the sampled pointcloud. Defaults to None, green inside the code.
    :type color: Tuple[float, float, float]
    :return: The sampled pointcloud.
    :rtype: o3d.geometry.PointCloud

    :Example:

    ::

        >>> import pointcloudrsait
        >>> import open3d as o3d
        >>> import numpy as np
        >>> points = np.array([[-2.65908417, 1.66316594, -12.05967958], [0.0459558, 1.72705225, -5.76867598], [ -6.27471812,   1.66093731,  -3.02014327], [-9.10951204, 1.59325844, -9.76566801], [-2.56569358, -1.59892799, -12.08053818], [0.1378323 , -1.63325758, -5.80021574], [-6.33750632, -1.6103617 , -2.9846542], [-9.17057479, -1.57383548, -9.73196329], [-4.49166262, 0.0285039, -7.65144228]])
        >>> pcd = o3d.geometry.PointCloud()
        >>> pcd.points = o3d.utility.Vector3dVector(points)
        >>> pcd_inside = pointcloudrsait.get_pointcloud_uniform_sample_inside_convex_hull_of_pointcloud(pcd, seed = 42)
        >>> # o3d.visualization.draw_geometries([pcd_inside])
        >>> pcd_inside.points[0]
        array([-3.98803751,  0.84332722, -4.3866304 ])
    """
    if color is None:
        color = (0, 1, 0)
    hull = ConvexHull(np.asarray(pcd.points))
    oriented_bounding_box = pcd.get_oriented_bounding_box()
    points_in_oriented_box = get_points_uniform_sampled_inside_oriented_box(oriented_bounding_box, size, seed = seed)
    points_inside_hull = points_in_oriented_box[check_points_inside_convex_hull(points_in_oriented_box, hull)]
    pcd_inside = o3d.geometry.PointCloud()
    pcd_inside.points = o3d.utility.Vector3dVector(points_inside_hull)
    pcd_inside.paint_uniform_color(color)
    return pcd_inside

def get_points_uniform_sampled_inside_oriented_box(box: o3d.geometry.OrientedBoundingBox, size: int = 1, seed = None) -> np.ndarray:
    """
    Returns an array of points inside the oriented box.

    :param box: The input oriented box.
    :type box: o3d.geometry.OrientedBoundingBox
    :param size: Number of points to be sampled.
    :type size: int
    :param seed: Seed of the random number generator.
    :type seed: Optional[int]
    :return: Points sampled inside the oriented box.
    :rtype: np.ndarray

    :Example:

    ::

        >>> import pointcloudrsait
        >>> import open3d as o3d
        >>> import numpy as np
        >>> points = np.array([[-2.65908417, 1.66316594, -12.05967958], [0.0459558, 1.72705225, -5.76867598], [ -6.27471812,   1.66093731,  -3.02014327], [-9.10951204, 1.59325844, -9.76566801], [-2.56569358, -1.59892799, -12.08053818], [0.1378323 , -1.63325758, -5.80021574], [-6.33750632, -1.6103617 , -2.9846542], [-9.17057479, -1.57383548, -9.73196329], [-4.49166262, 0.0285039, -7.65144228]])
        >>> pcd = o3d.geometry.PointCloud()
        >>> pcd.points = o3d.utility.Vector3dVector(points)
        >>> oriented_bounding_box = pcd.get_oriented_bounding_box()
        >>> points_in_oriented_box = pointcloudrsait.get_points_uniform_sampled_inside_oriented_box(oriented_bounding_box, size = 2, seed = 42)
        >>> for point in points_in_oriented_box: print (point)
        [-4.82733002 -1.33178146 -3.41651872]
        [-6.26834682  1.62343098 -6.66719078]
    """
    rng = default_rng(seed)
    x_coords = rng.uniform(low=0, high=box.extent[0], size=size)
    y_coords = rng.uniform(low=0, high=box.extent[1], size=size)
    z_coords = rng.uniform(low=0, high=box.extent[2], size=size)
    x_coords = x_coords - box.extent[0] / 2
    y_coords = y_coords - box.extent[1] / 2
    z_coords = z_coords - box.extent[2] / 2
    w_coords = np.ones(size)
    affine_matrix = np.matrix(np.vstack((np.hstack((box.R, box.center[:, None])), [0, 0, 0, 1])))
    all_points = np.matrix(np.row_stack([x_coords, y_coords, z_coords, w_coords]))
    all_points_transformed = np.array((affine_matrix * all_points).T[:, :3])
    return all_points_transformed


def get_points_uniform_sampled_inside_axis_aligned_box(min_x: float, max_x: float, min_y: float, max_y: float,
                                                       min_z: float, max_z: float, size: int = 1, seed: int = None) -> np.ndarray:
    """
    Returns an array of points inside the axis-aligned box delimited by the input parameters.

    :param seed:
    :param min_x: Minimum value of X coordinate for the box.
    :type min_x: float
    :param max_x: Maximum value of X coordinate for the box.
    :type max_x: float
    :param min_y: Minimum value of Y coordinate for the box.
    :type min_y: float
    :param max_y: Maximum value of Y coordinate for the box.
    :type max_y: float
    :param min_z: Minimum value of Z coordinate for the box.
    :type min_z: float
    :param max_z: Maximum value of Z coordinate for the box.
    :type max_z: float
    :param size: Number of points to sample, default to 1.
    :type size: int
    :return: Points sampled inside the box.
    :rtype: np.ndarray

    :Example:

    ::

        >>> import pointcloudrsait
        >>> A = pointcloudrsait.get_points_uniform_sampled_inside_axis_aligned_box(0, 1, 0, 1, 0, 1, size=10, seed = 42)
        >>> for point in A: print(point)
        [0.77395605 0.37079802 0.75808774]
        [0.43887844 0.92676499 0.35452597]
        [0.85859792 0.64386512 0.97069802]
        [0.69736803 0.82276161 0.89312112]
        [0.09417735 0.4434142  0.7783835 ]
        [0.97562235 0.22723872 0.19463871]
        [0.7611397  0.55458479 0.466721  ]
        [0.78606431 0.06381726 0.04380377]
        [0.12811363 0.82763117 0.15428949]
        [0.45038594 0.6316644  0.68304895]
    """
    rng = default_rng(seed)
    x_coords = rng.uniform(low=min_x, high=max_x, size=size)
    y_coords = rng.uniform(low=min_y, high=max_y, size=size)
    z_coords = rng.uniform(low=min_z, high=max_z, size=size)
    return np.column_stack([x_coords, y_coords, z_coords])

def get_pointcloud_after_substracting_point_cloud(pcd: o3d.geometry.PointCloud, substract: o3d.geometry.PointCloud,
                                                  threshold: float = 0.05) -> o3d.geometry.PointCloud:
    """
    Substracts one pointcloud from another. It removes all the points of the first pointcloud that are
    closer than *threshold* to some point of the second pointcloud.

    :param pcd: Pointcloud to substract from.
    :type pcd: o3d.geometry.PointCloud
    :param substract: Pointcloud to substract.
    :type substract: o3d.geometry.PointCloud
    :param threshold: If a point of the first pointcloud is closer to some point of the second pointcloud than this value, the point is removed.
    :type threshold: float
    :return: The results after substracting the second pointcloud from the first pointcloud.
    :rtype: o3d.geometry.PointCloud

    :Example:

    ::

        >>> import pointcloudrsait
        >>> import open3d as o3d
        >>> mesh_box = o3d.geometry.TriangleMesh.create_box(width=1.0, height=5.0, depth=1.0)
        >>> pcd_1 = mesh_box.sample_points_uniformly(number_of_points = 10000, seed = 42)
        >>> mesh_box = o3d.geometry.TriangleMesh.create_box(width=1.5, height=4.0, depth=0.5)
        >>> pcd_2 = mesh_box.sample_points_uniformly(number_of_points = 10000, seed = 42)
        >>> # pcd_1.paint_uniform_color([1, 0, 0])
        >>> # pcd_2.paint_uniform_color([0, 1, 0])
        >>> pcd_1_minus_pcd_2 = pointcloudrsait.get_pointcloud_after_substracting_point_cloud(pcd_1, pcd_2, threshold = 0.02)
        >>> pcd_1_minus_pcd_2
        PointCloud with 5832 points.
        >>> # o3d.visualization.draw_geometries([pcd_1_minus_pcd_2])
        >>> pcd_2_minus_pcd_1 = pointcloudrsait.get_pointcloud_after_substracting_point_cloud(pcd_2, pcd_1, threshold = 0.02)
        >>> pcd_2_minus_pcd_1
        PointCloud with 4726 points.
        >>> # o3d.visualization.draw_geometries([pcd_2_minus_pcd_1])
    """

    def aux_func(x, y, z):
        [_, _, d] = pcd_tree.search_knn_vector_3d([x, y, z], knn=1)
        return d[0]

    pcd_tree = o3d.geometry.KDTreeFlann(substract)
    points = np.asarray(pcd.points)
    if len(pcd.colors) == 0:
        remaining_points = [point for point in points if
                            aux_func(point[0], point[1], point[2]) > threshold]
        pcd_result = o3d.geometry.PointCloud()
        pcd_result.points = o3d.utility.Vector3dVector(np.asarray(remaining_points))
        return pcd_result
    colors = np.asarray(pcd.colors)
    remaining_points_and_colors = [(point, color) for point, color in zip(points, colors) if
                                   aux_func(point[0], point[1], point[2]) > threshold]
    remaining_points = [item[0] for item in remaining_points_and_colors]
    remaining_colors = [item[1] for item in remaining_points_and_colors]
    pcd_result = o3d.geometry.PointCloud()
    pcd_result.points = o3d.utility.Vector3dVector(np.asarray(remaining_points))
    pcd_result.colors = o3d.utility.Vector3dVector(np.asarray(remaining_colors))
    return pcd_result

def get_line_set_from_cuboid_points(points: Iterable[Iterable[float]],
                                    color: Optional[Tuple[float, float, float]] = None) -> o3d.geometry.LineSet:
    """
    Returns the set of lines corresponding to a cuboid.

    :param points: The points that delimit the cuboid. These points are supposed to be two opposite faces, first the four of one face and then the corresponding points in the other face. The points of a face are given in clockwise or counterclockwise order.
    :type points: Iterable[Iterable[float]]
    :param color: Color of the lines, defaults to red, parameter value is (1, 0, 0).
    :type color: Optional[Tuple[float, float, float]]
    :return: A set of lines that draw a cuboid.
    :rtype: o3d.geometry.LineSet

    :Example:

    ::

        >>> import pointcloudrsait
        >>> import open3d as o3d
        >>> points = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]
        >>> line_set = pointcloudrsait.get_line_set_from_cuboid_points(points, color = (0, 1, 0))
        >>> # o3d.visualization.draw_geometries([line_set])
        >>> line_set
        LineSet with 12 lines.
    """
    if color is None:
        color = (1, 0, 0)
    lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
    colors = [list(color) for _ in range(len(lines))]
    line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(points),
                                    lines=o3d.utility.Vector2iVector(lines))
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

def create_pointcloud_plane_from_four_points_and_thickness(points: Iterable[Iterable[float]], thickness: float = 0.05,
                                                           size: int = 10000, seed: int = None,
                                                           color: Tuple[float, float, float] = None) -> o3d.geometry.PointCloud:
    """
    Creates a pointcloud from four coplanar points and some thickness. The final thickness is twice such
    thickness parameter.

    :param points: Four coplanar points.
    :type points: Iterable[Iterable[float]]
    :param thickness: The thickness of the pointcloud for each side of the plane.
    :type thickness: float
    :param size: Number of points in the resulting pointcloud.
    :type size: int
    :param seed: Seed of the random number generator.
    :type seed: Optional[int]
    :param color: Color of the lines, defaults to red, parameter value is (1, 0, 0).
    :type color: Optional[Tuple[float, float, float]]
    :return: A pointcloud centered in the plane defined by the input points and with thickness twice the parameter.
    :rtype: o3d.geometry.PointCloud

    :Example:

    ::

        >>> import pointcloudrsait
        >>> import open3d as o3d
        >>> import numpy as np
        >>> points = [[5, 0, 0], [5, 1, 0], [6, 1, 0], [6, 0, 0]]
        >>> pcd = pointcloudrsait.create_pointcloud_plane_from_four_points_and_thickness(points, seed = 42, color=(0, 1, 0))
        >>> pcd.points[0]
        array([5.72072839, 0.77395605, 0.02407728])
        >>> # o3d.visualization.draw_geometries([pcd])
        >>> points = np.array([[-2.65908417, 1.66316594, -12.05967958], [0.0459558, 1.72705225, -5.76867598], [-6.27471812, 1.66093731, -3.02014327], [-9.10951204, 1.59325844, -9.76566801]])
        >>> pcd = pointcloudrsait.create_pointcloud_plane_from_four_points_and_thickness(points, size = 300000, seed = 42, color=(0, 1, 0))
        >>> pcd.points[0]
        array([-2.81447494,  1.69442252, -5.9641852 ])
        >>> # o3d.visualization.draw_geometries([pcd])
    """
    # we need to find four points over and under the quadrilateral, in order to give thickness;
    # these points are computed as normal * thickness from the points.
    if color is None:
        color = (1, 0, 0)
    plane = Plane(Point3D(points[0]), Point3D(points[1]), Point3D(points[2]))
    points = [np.array(point) for point in points]
    real_vector = np.array(
        [float(plane.normal_vector[0]), float(plane.normal_vector[1]), float(plane.normal_vector[2])])
    normal_vector = real_vector / np.linalg.norm(real_vector)
    one_direction_vector = np.array([item * thickness for item in normal_vector])
    # print(one_direction_vector)
    new_points_above = [one_direction_vector + point for point in points]
    new_points_above = [list(point) for point in new_points_above]
    new_points_below = [-one_direction_vector + point for point in points]
    new_points_below = [list(point) for point in new_points_below]
    new_points = new_points_above + new_points_below
    points = [list(point) for point in points]
    all_points = points + new_points
    # print(all_points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    return get_pointcloud_uniform_sample_inside_convex_hull_of_pointcloud(pcd, size=size, seed = seed, color=color)

def get_pointcloud_negative_of_pointcloud(pcd: o3d.geometry.PointCloud, size: int = 10000, threshold: float = 0.05,
                                          seed: int = None,
                                          color: Tuple[float, float, float] = None) -> o3d.geometry.PointCloud:
    """
    Returns the negative of a pointcloud, sampling inside it and removing all the generated points that are close to the original pointcloud.

    :param pcd: Pointcloud.
    :param pcd: o3d.geometry.PointCloud
    :param size: Number of points in the resulting pointcloud.
    :type size: int
    :param threshold: If a point of the first pointcloud is closer to some point of the second pointcloud than this value, the point is removed.
    :type threshold: float
    :param seed: Seed of the random number generator.
    :type seed: Optional[int]
    :param color: Color of the lines, defaults to red, parameter value is (1, 0, 0).
    :type color: Optional[Tuple[float, float, float]]
    :return: The negative of a pointcloud, sampling inside it and removing all the generated points that are close to the original pointcloud.
    :rtype: o3d.geometry.PointCloud

    :Example:

    ::

        >>> import pointcloudrsait
        >>> import open3d as o3d
        >>> # import os
        >>> # home_dir = os.getenv("HOME")
        >>> # pcd = o3d.io.read_point_cloud(home_dir + "/Github/Lantegi/Code/Open3D/gui/skull.ply")
        >>> pcd = o3d.io.read_point_cloud("Code/Open3D/gui/skull.ply")
        >>> negative_pcd = pointcloudrsait.get_pointcloud_negative_of_pointcloud(pcd, size = 10000, seed = 42, threshold = 5)
        >>> # o3d.visualization.draw_geometries([negative_pcd])
        >>> negative_pcd.points[0]
        array([  1.92245891, -90.53361029, -79.71444622])
    """
    pcd_inside = get_pointcloud_uniform_sample_inside_convex_hull_of_pointcloud(pcd, size = size, color = color, seed = seed)
    # o3d.visualization.draw_geometries([pcd_inside])
    return get_pointcloud_after_substracting_point_cloud(pcd_inside, pcd, threshold)

def get_pointcloud_projection_onto_plane_of_pointcloud(pcd: o3d.geometry.PointCloud,
                                                       plane: Type[plane.PlaneRSAIT], percentage: float = 1.0,
                                                       seed: int = None) -> o3d.geometry.PointCloud:
    """
    Computes a projection of a pointcloud into a plane. The number of points of the projection is **percentage** points of the original pointcloud.

    :param pcd: Pointcloud.
    :param pcd: o3d.geometry.PointCloud
    :param plane: Plane.
    :type plane: Type[PlaneRSAIT]
    :param percentage: Percentage of points with respect to the original pointcloud in the projected pointcloud.
    :type percentage: float
    :param seed: Seed of the random number generator.
    :type seed: Optional[int]
    :return: The projection of a pointcloud into a plane.
    :rtype: o3d.geometry.PointCloud

    :Example:

    ::

        >>> import pointcloudrsait
        >>> import plane
        >>> import open3d as o3d
        >>> # import os
        >>> # home_dir = os.getenv("HOME")
        >>> # pcd = o3d.io.read_point_cloud(home_dir + "/Github/Lantegi/Code/Open3D/gui/skull.ply")
        >>> pcd = o3d.io.read_point_cloud("Code/Open3D/gui/skull.ply")
        >>> plane = plane.PlaneRSAIT((1, 2, 3, 4))
        >>> projection = pointcloudrsait.get_pointcloud_projection_onto_plane_of_pointcloud(pcd, plane, percentage = 0.3, seed = 42)
        >>> # o3d.visualization.draw_geometries([projection])
        >>> projection.points[0]
        array([ 29.08498429, -89.64099743,  48.73233686])
    """
    new_pcd = o3d.geometry.PointCloud()
    new_points = list()
    new_colors = list()
    how_many = int(len(pcd.points) * percentage)
    np.random.seed(seed)
    indices = np.random.choice(np.arange(len(pcd.points)), size = how_many, replace = False)
    for index in indices:
        new_point = plane.get_point_projection_onto_plane(plane, pcd.points[index])
        new_points.append(new_point)
        new_colors.append(pcd.colors[index])
    new_pcd.points = o3d.utility.Vector3dVector(new_points)
    new_pcd.colors = o3d.utility.Vector3dVector(new_colors)
    return new_pcd

#@profile
def cut_into_disjoint_slices_by_axis_pointcloud(pcd: o3d.geometry.PointCloud,
                                                axis: int = 2, step: float = 0.02) -> \
        List[o3d.geometry.PointCloud]:
    maxs = pcd.get_max_bound()
    mins = pcd.get_min_bound()
    min_x = mins[0];
    min_y = mins[1];
    min_z = mins[2]
    max_x = maxs[0];
    max_y = maxs[1];
    max_z = maxs[2]
    if axis == 0:
        low_axis = step * math.floor(min_x / step)
        high_axis = step * math.ceil(max_x / step)
    elif axis == 1:
        low_axis = step * math.floor(min_y / step)
        high_axis = step * math.ceil(max_y / step)
    else:
        low_axis = step * math.floor(min_z / step)
        high_axis = step * math.ceil(max_z / step)
    num_slices = round((high_axis - low_axis) / step) + 1
    l_points_slices = list()
    l_colors_slices = list()
    for i in range(num_slices):
        l_points_slices.append([])
        l_colors_slices.append([])
    np_pcd_points = np.asarray(pcd.points)
    which_slices = (np.floor((np_pcd_points[:, axis] - low_axis) / step)).astype(int)
    counter_slices = Counter(which_slices)
    list_slice_values_ordered = np.array([[low_axis + (item[0]+1) * step] for item in counter_slices.most_common()])
    # return which_slices, counter_slices, list_slice_values_ordered
    return which_slices, list_slice_values_ordered

def crop(pcd, cuboid_points):
    points = o3d.utility.Vector3dVector(cuboid_points)
    oriented_bounding_box = o3d.geometry.OrientedBoundingBox.create_from_points(points)
    point_cloud_crop = pcd.crop(oriented_bounding_box)
    return point_cloud_crop, oriented_bounding_box

#@profile
def count_points_in_voxel_in_x_and_z(pcd: o3d.geometry.PointCloud, edge: float = 0.02,
                                     percentile: float = 95) -> None:
    def round_down(x, a):
        return math.floor(x / a) * a
    pcd_points = np.asarray(pcd.points)
    pcd_colors = np.asarray(pcd.colors)
    pcd_points_x_z = pcd_points[:, [0, 2]]
    # voxel_coords = np.floor(pcd_points_x_z / edge) * edge
    # points_in_voxels = Counter(tuple(map(tuple, voxel_coords)))
    # voxel_coords2 = np.floor(pcd_points_x_z / edge)
    # points_in_voxels2 = Counter(tuple(map(tuple, voxel_coords2)))
    voxel_coords = np.floor(pcd_points_x_z / edge)
    voxel_coords = voxel_coords.astype(int)
    voxel_coords = voxel_coords.transpose()
    factor = np.max(voxel_coords[1]) + 1
    voxel_coords = voxel_coords[0]*factor + voxel_coords[1]
    points_in_voxels = Counter(voxel_coords)
    # for point in zip(pcd_points, voxel_coords):
    #     x_coord = round_down(point[0], edge)
    #     z_coord = round_down(point[2], edge)
    #     points_in_voxels[(x_coord, z_coord)] += 1
    threshold_for_points = np.percentile(list(points_in_voxels.values()), percentile)
    # threshold_for_points = np.percentile(list(points_in_voxels3.values()), percentile)
    good_keys = np.array([key for key, value in points_in_voxels.items() if value >= threshold_for_points])
    good_mask = np.isin(voxel_coords, good_keys)
    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(pcd_points[good_mask])
    new_pcd.colors = o3d.utility.Vector3dVector(pcd_colors[good_mask])
    # new_points = list()
    # new_colors = list()
    # for point, color in zip(pcd.points, pcd.colors):
    #     x_coord = round_down(point[0], edge)
    #     z_coord = round_down(point[2], edge)
    #     if points_in_voxels[(x_coord, z_coord)] >= threshold_for_points:
    #         new_points.append([point[0], point[1], point[2]])
    #         new_colors.append([color[0], color[1], color[2]])
    # new_pcd.points = o3d.utility.Vector3dVector(new_points)
    # new_pcd.colors = o3d.utility.Vector3dVector(new_colors)
    return points_in_voxels, new_pcd
