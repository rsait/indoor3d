"""
In this module we embed all the functions related to general geometry questions.
The *plane_model* refers to four values (a, b, c, d) which
corresponds to the equation form Ax + By + Cz + D = 0. The normal to a plane is (A, B, C).
"""

import numpy as np
from typing import Union, Iterable

numeric = Union[int, float, np.number]

def get_vector_1D(v: Iterable[numeric]) -> np.ndarray:
    """
    Returns the np.ndarray vector corresponding to the input parameter.

    :param v: Iterable of numeric values representing a vector.
    :type v: Iterable[numeric]
    :return: The np.ndarray vector corresponding to the input iterable.
    :rtype: np.ndarray

    :Example:

    ::

        >>> import vector
        >>> v = (1,2,3)
        >>> npv = vector.get_vector_1D(v)
        >>> npv
        array([1, 2, 3])
    """
    npv = np.array(v)
    assert(npv.size > 0)
    assert(np.issubdtype(npv.dtype, np.number))
    assert(npv.ndim == 1)
    return npv

def get_vector_unit_vector(v: Iterable[numeric]) -> np.ndarray:
    """
    Returns the unit vector corresponding to the vector represented by the input parameter.

    :param v: Iterable of numeric values representing a vector.
    :type v: Iterable[numeric]
    :return: The unit vector corresponding to the input iterable.
    :rtype: np.ndarray

    :Example:

    ::

        >>> import vector
        >>> v = (1,2,3)
        >>> unit_v = vector.get_vector_unit_vector(v)
        >>> unit_v
        array([0.26726124, 0.53452248, 0.80178373])
    """
    npv = get_vector_1D(v)
    return npv / np.linalg.norm(npv)

def get_angle_between_vectors(v1: Iterable[numeric], v2: Iterable[numeric]) -> float:
    """
    Returns the angle in radians between two vectors.

    :param v1: Iterable of numeric values representing the first vector.
    :type v1: np.ndarray
    :param v2: Iterable of numeric values representing the second vector.
    :type v2: np.ndarray
    :return: The angle between the two vectors, in radians.
    :rtype: float

    :Example:

    ::

        >>> import vector
        >>> v1 = np.array([1,2,3])
        >>> v2 = np.array([3,6,-5])
        >>> angle = vector.get_angle_between_vectors(v1, v2)
        >>> angle
        1.5707963267948966
    """
    npv1 = get_vector_1D(v1)
    npv2 = get_vector_1D(v2)
    assert(npv1.size == npv2.size)
    dot_product = np.dot(get_vector_unit_vector(npv1), get_vector_unit_vector(npv2))
    angle = np.arccos(dot_product)
    return angle

def get_point_further_along_direction(p1: Iterable[float], p2: Iterable[float], distance: float = 1.0) -> np.ndarray:
    """
    Returns a point located at a given distance from the first point along the direction from the first to the second point.

    :param p1: First point.
    :type p1: Iterable[float]
    :param p2: Second point.
    :type p2: Iterable[float]
    :param distance: Distance from the first point.
    :type distance: float
    :return: Point located at a given distance from the first point along the direction from the first to the second point.
    :rtype: np.ndarray

    :Example:

    ::

        >>> import vector
        >>> import numpy as np
        >>> p1 = np.array([1,2,3])
        >>> p2 = np.array([3,6,-5])
        >>> new_point = vector.get_point_further_along_direction(p1, p2, distance=2.0)
        >>> new_point
        array([ 3.43643578,  6.87287156, -6.74574312])
    """
    first_point = np.array(p1)
    second_point = np.array(p2)
    vector_director = second_point - first_point
    module = np.linalg.norm(vector_director)
    new_module = module + distance
    new_vector_director = vector_director * new_module / module
    return first_point + new_vector_director
