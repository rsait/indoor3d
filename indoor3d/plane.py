"""
In this module we embed all the functions related to general geometry questions.
The *plane_model* refers to four values (a, b, c, d) which
corresponds to the equation form Ax + By + Cz + D = 0. The normal to a plane is (A, B, C).
"""

import numpy as np
import math
import vector
from typing import List, Tuple, Union, Optional, Iterable
from typing import Type

numeric = Union[int, float, np.number]


class PlaneRSAIT:
    """
    My own plane.
    """

    def __init__(self, *args):
        assert (len(args) == 1 or len(args) == 4)
        if len(args) == 1:
            self.A = args[0][0]
            self.B = args[0][1]
            self.C = args[0][2]
            self.D = args[0][3]
        if len(args) == 4:
            self.A = args[0]
            self.B = args[1]
            self.C = args[2]
            self.D = args[3]
        self.normal = vector.get_vector_unit_vector((self.A, self.B, self.C))
        self.normal_not_normalized = np.array([self.A, self.B, self.C])
        self.opposite_normal = -self.normal

    def __repr__(self):
        return f"PlaneRSAIT(A={self.A},B={self.B},C={self.C},D={self.D})"


def get_angle_between_planes(plane_1: Type[PlaneRSAIT],
                             plane_2: Type[PlaneRSAIT]) -> float:
    """
    Finds the angle between two planes. The plane models are instances of MyPlane class, which has four
    attributes (A, B, C, D), corresponding
    to the plane equation in the form Ax + By + Cz + D = 0. Therefore, the angle is computed between the
    normals, denoted by the (A, B, C) vectors. It computes the minimum angle between the normals and one
    normal and the other multiplied by -1, in order to obtain always the minimum between abs(angle) and
    abs(180 - angle).

    :param plane_1: First plane.
    :type plane_1: Type[PlaneRSAIT]
    :param plane_2: Second plane.
    :type plane_2: Type[PlaneRSAIT]
    :return: The angle between the two planes, in radians.
    :rtype: float

    :Example:

    ::

        >>> import planersait
        >>> plane_1 = planersait.PlaneRSAIT((1, 2, 3, 4))
        >>> plane_2 = planersait.PlaneRSAIT((3, 6, -5, 10))
        >>> angle = planersait.get_angle_between_planes(plane_1, plane_2)
        >>> angle
        1.5707963267948966
    """
    angle_between_normals = vector.get_angle_between_vectors(plane_1.normal, plane_2.normal)
    angle_between_normal_and_opposite_normal = vector.get_angle_between_vectors(plane_1.normal,
                                                                                     plane_2.opposite_normal)
    return min(abs(angle_between_normals), abs(angle_between_normal_and_opposite_normal))


def check_planes_paralell(plane_1: Type[PlaneRSAIT],
                          plane_2: Type[PlaneRSAIT],
                          max_tolerance_degrees: float = 10) -> bool:
    '''
    Computes if two planes are parallel, up to some tolerance, expressed in degrees.

    :param plane_1: First plane.
    :type plane_1: Type[PlaneRSAIT]
    :param plane_2: Second plane.
    :type plane_2: Type[PlaneRSAIT]
    :param max_tolerance_degrees: Tolerance for the angle between normals.
    :type max_tolerance_degrees: float
    :return: True if the planes are parallel, False otherwise.
    :rtype: bool

    :Example:

    ::

        >>> import planersait
        >>> plane_1 = planersait.PlaneRSAIT((1, 2, 3, 4))
        >>> plane_2 = planersait.PlaneRSAIT((3, 6, -5, 10))
        >>> planersait.check_planes_paralell(plane_1, plane_2)
        False
        >>> plane_1 = planersait.PlaneRSAIT((1, 2, 3, 4))
        >>> plane_2 = planersait.PlaneRSAIT((-1, -2, -4, 10))
        >>> planersait.check_planes_paralell(plane_1, plane_2) # max_tolerance_degrees is 10 by default.
        True
        >>> planersait.check_planes_paralell(plane_1, plane_2, max_tolerance_degrees = 5)
        False
    '''
    max_angle_radians = max_tolerance_degrees * math.pi / 180
    return abs(get_angle_between_planes(plane_1, plane_2)) < max_angle_radians


def check_planes_perpendicular(plane_1: Type[PlaneRSAIT],
                               plane_2: Type[PlaneRSAIT],
                               max_tolerance_degrees: float = 10) -> bool:
    """
    Computes if two planes are perpendicular, up to some tolerance, expressed in degrees.

    :param plane_1: First plane.
    :type plane_1: Type[PlaneRSAIT]
    :param plane_2: Second plane.
    :type plane_2: Type[PlaneRSAIT]
    :param max_tolerance_degrees: Tolerance for the angle between normals.
    :type max_tolerance_degrees: float
    :return: True if the planes are perpendicular, False otherwise.
    :rtype: bool

    :Example:

    ::

        >>> import planersait
        >>> plane_1 = planersait.PlaneRSAIT((1, 2, 3, 4))
        >>> plane_2 = planersait.PlaneRSAIT((1, 2, 3, 10))
        >>> planersait.check_planes_perpendicular(plane_1, plane_2)
        False
        >>> plane_1 = planersait.PlaneRSAIT((1, 2, 3, 4))
        >>> plane_2 = planersait.PlaneRSAIT((3, 6, -4, 10))
        >>> planersait.check_planes_perpendicular(plane_1, plane_2) # max_tolerance_degrees is 10 by default.
        True
        >>> planersait.check_planes_perpendicular(plane_1, plane_2, max_tolerance_degrees = 5)
        False
    """
    max_angle_radians = max_tolerance_degrees * math.pi / 180
    angle_90_radians = 90 * math.pi / 180
    return abs(get_angle_between_planes(plane_1, plane_2) - angle_90_radians) < max_angle_radians


# https://math.stackexchange.com/questions/1755856/calculate-arbitrary-points-from-a-plane-equation
def get_point_on_plane_closest_to_the_origin(plane: Type[PlaneRSAIT]) -> np.ndarray:
    """
    Returns the point in the plane that is closest to the origin. According to the procedure described
    `here <https://math.stackexchange.com/questions/1755856/calculate-arbitrary-points-from-a-plane-equation>`_.

    :param plane: Plane of interest.
    :type plane: Type[PlaneRSAIT]
    :return: The point on the plane closest to the origin.
    :rtype: np.ndarray

    :Example:

    ::

        >>> import planersait
        >>> plane = planersait.PlaneRSAIT((1, 2, 3, 4))
        >>> point = planersait.get_point_on_plane_closest_to_the_origin(plane)
        >>> point
        array([-0.28571429, -0.57142857, -0.85714286])
    """
    a, b, c, d = plane.A, plane.B, plane.C, plane.D
    d = -d  # for the next formula, which needs Ax + By + Cz = D
    denominator = a * a + b * b + c * c
    point_x = a * d / denominator
    point_y = b * d / denominator
    point_z = c * d / denominator
    return np.array([point_x, point_y, point_z])


def get_distance_between_plane_and_point(plane: Type[PlaneRSAIT],
                                         point: Iterable[float]) -> float:
    """
    Returns the distance between the plane and the point.

    :param plane: Plane.
    :type plane: Type[PlaneRSAIT]
    :param point: Point.
    :type point: Iterable[float]
    :return: Distance between the plane and the point.
    :rtype: float

    :Example:

    ::

        >>> import planersait
        >>> plane = planersait.PlaneRSAIT((1, 2, 3, 4))
        >>> distance = planersait.get_distance_between_plane_and_point(plane, [1, 2, 3])
        >>> distance
        4.810702354423639
    """
    a, b, c, d = plane.A, plane.B, plane.C, plane.D
    x, y, z = point[0], point[1], point[2]
    numerator = abs(x * a + y * b + z * c + d)
    denominator = math.sqrt(a * a + b * b + c * c)
    return abs(numerator / denominator)


def get_distance_between_plane_and_origin(plane: Type[PlaneRSAIT]) -> float:
    """
    Returns the distance between the plane and the origin.

    :param plane: Plane.
    :type plane: Type[PlaneRSAIT]
    :return: Distance between the plane and the origin.
    :rtype: float

    :Example:

    ::

        >>> import planersait
        >>> plane = planersait.PlaneRSAIT((1, 2, 3, 4))
        >>> distance = planersait.get_distance_between_plane_and_origin(plane)
        >>> distance
        1.0690449676496976
    """
    return get_distance_between_plane_and_point(plane, [0, 0, 0])


def get_distance_between_parallel_planes(plane_1: Type[PlaneRSAIT],
                                         plane_2: Type[PlaneRSAIT]) -> float:
    """
    Returns the distance between two parallel planes.

    :param plane_1: First plane.
    :type plane_1: Type[PlaneRSAIT]
    :param plane_2: Second plane.
    :type plane_2: Type[PlaneRSAIT]
    :return: Distance between the two parallel planes.
    :rtype: float

    :Example:

    ::

        >>> import planersait
        >>> plane_1 = planersait.PlaneRSAIT((1, 2, 3, 4))
        >>> plane_2 = planersait.PlaneRSAIT((1, 2, 3, 10))
        >>> distance = planersait.get_distance_between_parallel_planes(plane_1, plane_2)
        >>> distance
        1.6035674514745464
    """
    point_of_plane_1 = get_point_on_plane_closest_to_the_origin(plane_1)
    return get_distance_between_plane_and_point(plane_2, point_of_plane_1)


def get_distance_and_sign_between_plane_and_point(plane: Type[PlaneRSAIT],
                                                  point: Iterable[float]) -> Tuple[float, float]:
    """
    Returns the signed distance between a point and a plane. It returns a tuple with two elements:
    the distance and the sign. The sign is a signed float. Two points with the same sign in their
    signed floats lie in the same side of the plane.

    :param plane: Plane.
    :type plane: Type[PlaneRSAIT]
    :param point: Point.
    :type point: Iterable[float]
    :return: Tuple of two elements: distance and sign.
    :rtype: Tuple[float]

    :Example:

    ::

        >>> import planersait
        >>> plane = planersait.PlaneRSAIT((1, 2, 3, 4))
        >>> distance, sign = planersait.get_distance_and_sign_between_plane_and_point(plane, [1, 2, 3])
        >>> distance
        4.810702354423639
        >>> sign
        18
    """
    a, b, c, d = plane.A, plane.B, plane.C, plane.D
    x, y, z = point[0], point[1], point[2]
    sign = x * a + y * b + z * c + d
    numerator = abs(sign)
    denominator = math.sqrt(a * a + b * b + c * c)
    return abs(numerator / denominator), sign


# https://www.superprof.co.uk/resources/academic/maths/analytical-geometry/distance/angle-between-line-and-plane.html
def get_angle_between_plane_and_axis(plane: Type[PlaneRSAIT], axis: str) -> float:
    '''
    Returns the angle in radians between the plane model and the line corresponding to the Y axis. According to the procedure described
    `here <https://www.superprof.co.uk/resources/academic/maths/analytical-geometry/distance/angle-between-line-and-plane.html>`_.

    :param plane: Plane.
    :type plane: Type[PlaneRSAIT]
    :return: Angle between the plane and the Y axis.
    :rtype: float

    :Example:

    ::

        >>> import planersait
        >>> plane = planersait.PlaneRSAIT((1, 2, 3, 4))
        >>> angle = planersait.get_angle_between_plane_and_axis(plane, axis = "X")
        >>> angle
        0.2705497629785729
        >>> angle = planersait.get_angle_between_plane_and_axis(plane, axis = "Y")
        >>> angle
        0.5639426413606289
        >>> angle = planersait.get_angle_between_plane_and_axis(plane, axis = "Z")
        >>> angle
        0.9302740141154721
        >>> plane = planersait.PlaneRSAIT((0.1, 0.1, 1, 4))
        >>> angle = planersait.get_angle_between_plane_and_axis(plane, axis = "X")
        >>> angle
        0.09917726107940236
        >>> angle = planersait.get_angle_between_plane_and_axis(plane, axis = "Y")
        >>> angle
        0.09917726107940236
        >>> angle = planersait.get_angle_between_plane_and_axis(plane, axis = "Z")
        >>> angle
        1.430306625041376
    '''
    axis_param = axis.upper()
    assert (axis_param in {"X", "Y", "Z"})
    if axis_param == "X":
        line_vector = np.array([1, 0, 0])
    elif axis_param == "Y":
        line_vector = np.array([0, 1, 0])
    else:
        line_vector = np.array([0, 0, 1])
    numerator = abs(np.dot(line_vector, plane.normal))
    denominator = np.linalg.norm(plane.normal) * np.linalg.norm(line_vector)
    angle = np.arcsin(numerator / denominator)
    return angle


def get_angle_plane_with_plane_axis_equal_0(plane: Type[PlaneRSAIT], axis: str) -> float:
    """
    Returns the angle in radians between the plane model and the plane X = 0 or Y = 0 or Z = 0.

    :param plane: Plane.
    :type plane: Type[PlaneRSAIT]
    :return: Angle between the plane model and the plane X = 0 or Y = 0 or Z = 0.
    :rtype: float

    :Example:

    ::

        >>> import planersait
        >>> plane = planersait.PlaneRSAIT((1, 2, 3, 4))
        >>> angle = planersait.get_angle_plane_with_plane_axis_equal_0(plane, axis = "X")
        >>> angle
        1.3002465638163236
        >>> angle = planersait.get_angle_plane_with_plane_axis_equal_0(plane, axis = "Y")
        >>> angle
        1.0068536854342678
        >>> angle = planersait.get_angle_plane_with_plane_axis_equal_0(plane, axis = "Z")
        >>> angle
        0.6405223126794245
        >>> plane = planersait.PlaneRSAIT((0.1, 0.1, 1, 4))
        >>> angle = planersait.get_angle_plane_with_plane_axis_equal_0(plane, axis = "X")
        >>> angle
        1.4716190657154942
        >>> angle = planersait.get_angle_plane_with_plane_axis_equal_0(plane, axis = "Y")
        >>> angle
        1.4716190657154942
        >>> angle = planersait.get_angle_plane_with_plane_axis_equal_0(plane, axis = "Z")
        >>> angle
        0.1404897017535205
        >>> plane = planersait.PlaneRSAIT((0, 0, 1, 4))
        >>> angle = planersait.get_angle_plane_with_plane_axis_equal_0(plane, axis = "X")
        >>> angle
        1.5707963267948966
        >>> angle = planersait.get_angle_plane_with_plane_axis_equal_0(plane, axis = "Y")
        >>> angle
        1.5707963267948966
        >>> angle = planersait.get_angle_plane_with_plane_axis_equal_0(plane, axis = "Z")
        >>> angle
        0.0
    """

    axis_param = axis.upper()
    assert (axis_param in {"X", "Y", "Z"})
    if axis_param == "X":
        plane_axis = PlaneRSAIT((1, 0, 0, 0))
    elif axis_param == "Y":
        plane_axis = PlaneRSAIT((0, 1, 0, 0))
    else:
        plane_axis = PlaneRSAIT((0, 0, 1, 0))
    return get_angle_between_planes(plane, plane_axis)


def get_plane_parallel_to_given_plane_through_a_point(plane: Type[PlaneRSAIT], point: Iterable[float]) -> Type[
    PlaneRSAIT]:
    """
    Returns the plane that is parallel to the given plane and passes through the given point.

    :param plane: Plane.
    :type plane: Type[PlaneRSAIT]
    :param point: Point.
    :type point: Iterable[float]
    :return: Plane parallel to the given plane and passing through the given point.
    :rtype: Type[PlaneRSAIT]

    :Example:

    ::

        >>> import planersait
        >>> plane = planersait.PlaneRSAIT((1, 2, 3, 4))
        >>> plane_parallel = planersait.get_plane_parallel_to_given_plane_through_a_point(plane, point = [1,1,1])
        >>> plane_parallel
        PlaneRSAIT(A=1,B=2,C=3,D=-6)
    """
    a, b, c, d = plane.A, plane.B, plane.C, plane.D
    x, y, z = point[0], point[1], point[2]
    new_d = -(a * x + b * y + c * z)
    return PlaneRSAIT((a, b, c, new_d))


def get_point_projection_onto_plane(plane: Type[PlaneRSAIT], point: Iterable[float]) -> np.ndarray:
    """
     Returns the projection of the given point onto the given plane.

     :param plane: Plane.
     :type plane: Type[PlaneRSAIT]
     :param point: Point.
     :type point: Iterable[float]
     :return: Point projection on the plane.
     :rtype: np.ndarray

     :Example:

     ::

         >>> import planersait
         >>> plane = planersait.PlaneRSAIT((1, 2, 3, 4))
         >>> projection = planersait.get_point_projection_onto_plane(plane, point = [1,1,1])
         >>> projection
         array([ 0.28571429, -0.42857143, -1.14285714])
     """
    point_plane = get_point_on_plane_closest_to_the_origin(plane)
    return np.array(point) - np.dot(np.array(point) - point_plane, plane.normal) * plane.normal


def get_point_intersection_of_three_planes(plane_1: Type[PlaneRSAIT], plane_2: Type[PlaneRSAIT],
                                           plane_3: Type[PlaneRSAIT]) -> np.ndarray:
    """
    Finds the points that is the intersection of three planes. It is assumed the three planes intersect
    in a point.

    :param plane_1: First plane.
    :type plane_1: Type[PlaneRSAIT]
    :param plane_2: Second plane.
    :type plane_2: Type[PlaneRSAIT]
    :param plane_3: Third plane.
    :type plane_3: Type[PlaneRSAIT]
    :return: The point where the three planes meet.
    :rtype: np.ndarray

    :Example:

    ::

        >>> import planersait
        >>> plane_1 = planersait.PlaneRSAIT((1, 0, 0, -1))
        >>> plane_2 = planersait.PlaneRSAIT((0, 1, 0, -1))
        >>> plane_3 = planersait.PlaneRSAIT((0, 0, 1, -1))
        >>> point = planersait.get_point_intersection_of_three_planes(plane_1, plane_2, plane_3)
        >>> point
        array([1., 1., 1.])
    """
    A = np.array([plane_1.normal_not_normalized, plane_2.normal_not_normalized, plane_3.normal_not_normalized])
    B = np.array([-plane_1.D, -plane_2.D, -plane_3.D])
    X = np.linalg.solve(A, B)
    return np.array([X[0], X[1], X[2]])


def get_plane_perpendicular_to_two_planes(plane_1: Type[PlaneRSAIT], plane_2: Type[PlaneRSAIT]) -> Type[PlaneRSAIT]:
    """
    Finds a plane perpendicular to two given planes.

    :param plane_1: First plane.
    :type plane_1: Type[PlaneRSAIT]
    :param plane_model_2: Second plane.
    :type plane_model_2: Type[PlaneRSAIT]
    :return: Plane perpendicular to the input planes.
    :type: Type[PlaneRSAIT]

    :Example:

    ::

        >>> import planersait
        >>> plane_1 = planersait.PlaneRSAIT((1, 2, 3, 4))
        >>> plane_2 = planersait.PlaneRSAIT((3, 1, -5/3, 0))
        >>> new_plane = planersait.get_plane_perpendicular_to_two_planes(plane_1, plane_2)
        >>> new_plane
        PlaneRSAIT(A=-0.4735225469652655,B=0.7975116580467629,C=-0.3738335897094201,D=0)
    """
    v_perp = np.cross(plane_1.normal, plane_2.normal)
    return PlaneRSAIT((v_perp[0], v_perp[1], v_perp[2], 0))


def get_distance_and_sign_between_plane_and_points(plane: Type[PlaneRSAIT], points: Iterable[Iterable[float]]) -> Tuple[
    np.ndarray, np.ndarray]:
    """
    Computes the distance and sign between a plane and a collection of points.

    :param plane: Plane.
    :type plane: Type[PlaneRSAIT]
    :param points: Points.
    :type points: np.ndarray
    :return: Distances and sign between the plane and the points.
    :rtype: Tuple[np.ndarray, np.ndarray]

    :Example:

    ::

        >>> import planersait
        >>> plane = planersait.PlaneRSAIT((1, 1, 1, 1))
        >>> points = [[-1, -1, -1], [0, 0, 0], [1, 1, 1], [2, 2, 2]]
        >>> distances, signs = planersait.get_distance_and_sign_between_plane_and_points(plane, points)
        >>> distances
        array([1.15470054, 0.57735027, 2.30940108, 4.04145188])
        >>> signs
        array([-2,  1,  4,  7])
    """
    a, b, c, d = plane.A, plane.B, plane.C, plane.D
    nppoints = np.array(points)
    points_transposed = nppoints.transpose()
    sign = points_transposed[0] * a + points_transposed[1] * b + points_transposed[2] * c + d
    numerator = np.abs(sign)
    denominator = math.sqrt(a * a + b * b + c * c)
    return np.abs(numerator / denominator), sign


def get_planes_parallel_to_given_distance_of_plane(plane: Type[PlaneRSAIT], distance: float) -> Tuple[
    Type[PlaneRSAIT], Type[PlaneRSAIT]]:
    """
    Returns the two parallel planes to the given plane to the given distance.

    :param plane: Plane.
    :type plane: Type[PlaneRSAIT]
    :param distance: Distance of the parallel planes to the given plane.
    :type distance: float
    :return: The two parallel planes to the given plane to the given distance.
    :rtype: Tuple[Type[PlaneRSAIT], Type[PlaneRSAIT]]

    :Example:

    ::

        >>> import planersait
        >>> plane = planersait.PlaneRSAIT((1, 1, 1, 1))
        >>> distance = 1
        >>> plane_1, plane_2 = planersait.get_planes_parallel_to_given_distance_of_plane(plane, distance = 1)
        >>> plane_1
        PlaneRSAIT(A=1,B=1,C=1,D=-0.7320508075688776)
        >>> plane_2
        PlaneRSAIT(A=1,B=1,C=1,D=2.7320508075688776)
    """
    point_of_plane = get_point_on_plane_closest_to_the_origin(plane)
    normal = plane.normal
    point_of_plane1 = point_of_plane + normal * distance
    point_of_plane2 = point_of_plane - normal * distance
    plane1 = get_plane_parallel_to_given_plane_through_a_point(plane, point_of_plane1)
    plane2 = get_plane_parallel_to_given_plane_through_a_point(plane, point_of_plane2)
    return plane1, plane2


def check_plane_parallel_to_all_planes_in_list(plane: Type[PlaneRSAIT],
                                               list_planes: List[Type[PlaneRSAIT]],
                                               max_tolerance_degrees: float = 10) -> bool:
    """
    Checks if a plane is parallel to all the members of a list. It needs to check all the
    elements of the list because due to the *max_tolerance_degrees* parameter, parallelism is not
    a equivalence relationship. A could be parallel to B, B parallel to C and A not parallel to C
    due to the tolerance. If the list is empty returns True.

    :param plane: Plane.
    :type plane: Type[PlaneRSAIT]
    :param list_planes: List of planes to check for parallelism with the given plane.
    :type list_planes: List[Type[PlaneRSAIT]]
    :param max_tolerance_degrees: Tolerance for the angle between normals.
    :type max_tolerance_degrees: float
    :return: True if the plane model is parallel to all the members of the list, False otherwise.
    :rtype: bool

    :Example:

    ::

        >>> import planersait
        >>> plane = planersait.PlaneRSAIT((1, 2, 3, 4))
        >>> plane_1 = planersait.PlaneRSAIT((2, 4, 6, 12))
        >>> plane_2 = planersait.PlaneRSAIT((-11, -21, -31, 75))
        >>> plane_3 = planersait.PlaneRSAIT((3.2, 6.2, 9.2, 1))
        >>> list_planes = [plane_1, plane_2, plane_3]
        >>> planersait.check_plane_parallel_to_all_planes_in_list(plane, list_planes)
        True
        >>> planersait.check_plane_parallel_to_all_planes_in_list(plane, list_planes, max_tolerance_degrees=0)
        False
    """
    for item in list_planes:
        if not check_planes_paralell(plane, item, max_tolerance_degrees):
            return False
    return True


def get_partition_of_list_of_planes_by_parallelism(list_planes: List[Type[PlaneRSAIT]],
                                                   max_tolerance_degrees: float = 10) -> \
        List[List[Type[PlaneRSAIT]]]:
    """
    Given a list of planes and a tolerance, it returns a list of lists of planes in which
    all the elements in the same sublist are parallel relative to each other according to the tolerance.
    It is possible that this partition is not unique and depends on the order of the initial list.

    :param list_planes: List of planes to partition.
    :type list_planes: List[Type[PlaneRSAIT]]
    :param max_tolerance_degrees: Tolerance for the angle between normals.
    :type max_tolerance_degrees: float
    :return: Partition of the input list into a list of lists of parallel planes.
    :rtype: List[List[Type[PlaneRSAIT]]]

    :Example:

    ::

        >>> import planersait
        >>> plane_1 = planersait.PlaneRSAIT((2, 4, 6, 12))
        >>> plane_2 = planersait.PlaneRSAIT((1, 11, 21, 5))
        >>> plane_3 = planersait.PlaneRSAIT((-11, -21, -31, 75))
        >>> plane_4 = planersait.PlaneRSAIT((-10.3, -110.4, -210.5, 34))
        >>> plane_5 = planersait.PlaneRSAIT((3.2, 6.2, 9.2, 1))
        >>> plane_6 = planersait.PlaneRSAIT((12, 37, 14, 12))
        >>> list_planes = [plane_1, plane_2, plane_3, plane_4, plane_5, plane_6]
        >>> partition = planersait.get_partition_of_list_of_planes_by_parallelism(list_planes)
        >>> for part in partition: print (part)
        [PlaneRSAIT(A=2,B=4,C=6,D=12), PlaneRSAIT(A=-11,B=-21,C=-31,D=75), PlaneRSAIT(A=3.2,B=6.2,C=9.2,D=1)]
        [PlaneRSAIT(A=1,B=11,C=21,D=5), PlaneRSAIT(A=-10.3,B=-110.4,C=-210.5,D=34)]
        [PlaneRSAIT(A=12,B=37,C=14,D=12)]
        >>> partition = planersait.get_partition_of_list_of_planes_by_parallelism(list_planes, max_tolerance_degrees=0.5)
        >>> for part in partition: print (part)
        [PlaneRSAIT(A=2,B=4,C=6,D=12)]
        [PlaneRSAIT(A=1,B=11,C=21,D=5), PlaneRSAIT(A=-10.3,B=-110.4,C=-210.5,D=34)]
        [PlaneRSAIT(A=-11,B=-21,C=-31,D=75), PlaneRSAIT(A=3.2,B=6.2,C=9.2,D=1)]
        [PlaneRSAIT(A=12,B=37,C=14,D=12)]
        >>> partition = planersait.get_partition_of_list_of_planes_by_parallelism(list_planes, max_tolerance_degrees=0)
        >>> for part in partition: print (part)
        [PlaneRSAIT(A=2,B=4,C=6,D=12)]
        [PlaneRSAIT(A=1,B=11,C=21,D=5)]
        [PlaneRSAIT(A=-11,B=-21,C=-31,D=75)]
        [PlaneRSAIT(A=-10.3,B=-110.4,C=-210.5,D=34)]
        [PlaneRSAIT(A=3.2,B=6.2,C=9.2,D=1)]
        [PlaneRSAIT(A=12,B=37,C=14,D=12)]
    """
    list_of_parallel_classes = []
    for plane in list_planes:
        for parallel_class in list_of_parallel_classes:
            if check_plane_parallel_to_all_planes_in_list(plane, parallel_class, max_tolerance_degrees):
                parallel_class.append(plane)
                break
        else:
            list_of_parallel_classes.append([plane])
    return list_of_parallel_classes

def check_plane_perpendicularity_between_lists_of_planes(list_planes_1: List[Type[PlaneRSAIT]],
                                                         list_planes_2: List[Type[PlaneRSAIT]],
                                                         max_tolerance_degrees: float = 10,
                                                         min_percentage: float = 0, return_all_values: bool = False) \
        -> Union[bool, Tuple[bool, List[bool]]]:
    """
    Checks if the percentage of planes perpendicular to each other between two lists is greater than
    some threshold (default is 0), according to some tolerance. It the parameter *return_all_values* is
    set to True, the boolean value of all the checks is returned too.

    :param list_planes_1: First list of planes.
    :type list_planes_1: List[Type[PlaneRSAIT]]
    :param list_planes_2: Second list of planes.
    :type list_planes_2: List[Type[PlaneRSAIT]]
    :param max_tolerance_degrees: Tolerance for the angle between normals, default is 10 degrees.
    :type max_tolerance_degrees: float
    :param min_percentage: Minimum percentage of parallel pairs needed to return True. Default is zero, meaning that ot would return true with just one positive case.
    :type min_percentage: float
    :param return_all_values: If True, all the boolean checks are returned.
    :type return_all_values: bool
    :return: True if the percentage of planes perpendicular to each other between two lists is greater than *min_percentage*, False otherwise. If *return_all_values* is True, all the boolean checks are returned.
    :rtype: Union[bool, Tuple[bool, List[bool]]]

    :Example:

    ::

        >>> import planersait
        >>> list_planes_1 = [planersait.PlaneRSAIT((1, 2, 3, 4)), planersait.PlaneRSAIT((3, 6.1, 9.2, 12))]
        >>> list_planes_2 = [planersait.PlaneRSAIT((3, 6, -4, 10)), planersait.PlaneRSAIT((6.1, 12.3, -8.3, -35))]
        >>> planersait.check_plane_perpendicularity_between_lists_of_planes(list_planes_1, list_planes_2, return_all_values = True)
        (True, [True, True, True, True])
        >>> planersait.check_plane_perpendicularity_between_lists_of_planes(list_planes_1, list_planes_2, max_tolerance_degrees= 2, return_all_values = True)
        (False, [False, False, False, False])
        >>> planersait.check_plane_perpendicularity_between_lists_of_planes(list_planes_1, list_planes_2, max_tolerance_degrees= 5.5, return_all_values = True)
        (True, [False, False, False, True])
        >>> planersait.check_plane_perpendicularity_between_lists_of_planes(list_planes_1, list_planes_2, max_tolerance_degrees= 5.5, min_percentage = 0.5, return_all_values = True)
        (False, [False, False, False, True])
        >>> planersait.check_plane_perpendicularity_between_lists_of_planes(list_planes_1, list_planes_2, max_tolerance_degrees= 5.5, min_percentage = 0.5)
        False
    """

    perpendicularity = [check_planes_perpendicular(p1, p2, max_tolerance_degrees)
                        for p1 in list_planes_1 for p2 in list_planes_2]
    percentage = sum(perpendicularity) / len(perpendicularity)
    above_threshold = (percentage > min_percentage == 0) or (
            percentage >= min_percentage > 0)
    if return_all_values:
        return above_threshold, perpendicularity
    else:
        return above_threshold >= min_percentage

def get_lists_of_planes_perpendicular_to_each_other_in_partition_first_two(partition: List[List[Type[PlaneRSAIT]]],
                                                                           max_tolerance_degrees: float = 10,
                                                                           min_percentage: float = 0) \
        -> Optional[Tuple[List[Type[PlaneRSAIT]], List[Type[PlaneRSAIT]]]]:
    """
    Returns the first two list of planes that are perpendicular to each other in a partition.
    Returns None if nothing found.

    :param partition: List of lists of planes.
    :type partition: List[List[Type[PlaneRSAIT]]]
    :param max_tolerance_degrees: Tolerance for the angle between normals, default is 10 degrees.
    :type max_tolerance_degrees: float
    :param min_percentage: Minimum percentage of parallel pairs needed to return True. Default is zero, meaning that it would return true with just one positive case.
    :type min_percentage: float
    :return: Tuple with two lists of planes perpendicular to each other.
    :rtype: Optional[Tuple[List[Type[PlaneRSAIT]], List[Type[PlaneRSAIT]]]]

    :Example:

    ::

        >>> import planersait
        >>> list_planes_1 = [planersait.PlaneRSAIT((1, 2, 3, 4)), planersait.PlaneRSAIT((3, 6.1, 9.2, 12))]
        >>> list_planes_2 = [planersait.PlaneRSAIT((7, 20, 3, 4)), planersait.PlaneRSAIT((21, 61.6, 9.2, 12))]
        >>> list_planes_3 = [planersait.PlaneRSAIT((3, 6, -4, 10)), planersait.PlaneRSAIT((6.1, 12.3, -8.3, -35))]
        >>> partition = [list_planes_1, list_planes_2, list_planes_3]
        >>> list_planes = planersait.get_lists_of_planes_perpendicular_to_each_other_in_partition_first_two(partition)
        >>> for list_plane in list_planes: print(list_plane)
        [PlaneRSAIT(A=1,B=2,C=3,D=4), PlaneRSAIT(A=3,B=6.1,C=9.2,D=12)]
        [PlaneRSAIT(A=7,B=20,C=3,D=4), PlaneRSAIT(A=21,B=61.6,C=9.2,D=12)]
    """
    for i in range(len(partition) - 1):
        for j in range(i + 1, len(partition)):
            if check_plane_perpendicularity_between_lists_of_planes(partition[i], partition[j],
                                                              max_tolerance_degrees=max_tolerance_degrees,
                                                              min_percentage=min_percentage):
                return partition[i], partition[j]
    return None
