import numpy as np
import math


def focal_length_from_two_points(a, b, c, d):
    """
    Brief: focal length from two points
    See appendex 6.1 Focal Length from Two Points
    :param a:
    :param b:
    :param c:
    :param d:
    :return: 0 (failed) or focal lenth in pixels
    """
    c_2 = math.pow(c, 2)  # c^2
    d_2 = math.pow(d, 2)  # d^2
    t1 = 2 * (d_2 * a * b - c_2)
    t2 = math.pow(d_2 * (a + b) - 2 * c, 2) - 4 * (d_2 * a * b - c_2) * (d_2 - 1)

    f = 0
    if t2 < 0:
        print('warning: invalid focal length')
        return f
    assert t2 >= 0
    t3 = 2 * c - d_2 * (a + b) + math.sqrt(t2)

    if t3 == 0:
        print('warning: invalid focal length')
        return f
    assert t3 != 0

    f2 = t1 / t3
    if f2 > 0:
        f = math.sqrt(f2)
    else:
        print('warning: invalid focal length')

    return f

def pan_y_tilt_x(pan, tilt):
    """
    matrix from pan Y along with axis then tilt along with X axis
    m = Q_\phi * Q_\theta in equation (1)
    :param pan:
    :param tilt:
    :return:
    """
    pan = pan * math.pi / 180.0
    tilt = tilt * math.pi / 180.0
    r_tilt = np.asarray([[1, 0, 0],
                          [0, math.cos(tilt), math.sin(tilt)],
                          [0, -math.sin(tilt), math.cos(tilt)]])
    r_pan = np.asarray([[math.cos(pan), 0, -math.sin(pan)],
                        [0, 1, 0],
                        [math.sin(pan), 0, math.cos(pan)]])
    m = r_tilt * r_pan
    return m

def ptz_from_two_point(principal_point, pan_tilt1, pan_tilt2, point1, point2):
    """
    Estimate pan, tilt and zoom from two points
    See Section 3.1 Two-point Algorithm for Data Annotation
    :param principal_point: [u, v], e.g image center, 2 x 1, numpy array
    :param pan_tilt1: pan and tile of point1, unit degress
    :param pan_tilt2: same as pan_tilt1,
    :param point1: point 1 location, unit pixel, 2 x 1, numpy array
    :param point2: same as point1
    :return: None if fail, otherwise, pan_tilt_focal_length
    """
    p1 = point1 - principal_point
    p2 = point2 - principal_point

    a = np.dot(p1, p1)
    b = np.dot(p2, p2)
    c = np.dot(p1, p2)
    z = np.asarray([0, 0, 1])

    pan1, tilt1 = pan_tilt1[0], pan_tilt1[1]
    pan2, tilt2 = pan_tilt2[0], pan_tilt2[1]

    pan_tilt_z = np.matmul(pan_y_tilt_x(pan2 - pan1, tilt2 - tilt1), z)
    d = np.dot(z, pan_tilt_z)

    f = focal_length_from_two_points(a, b, c, d) # this may fail

    if f <= 0:
        print('Warning: estimate focal length failed')
        return None
    ptz = np.zeros(3)
    pi = math.pi
    theta1 = pan1 - math.atan2(p1[0], f) * 180.0 / pi
    theta2 = pan2 - math.atan2(p2[0], f) * 180.0 / pi
    ptz[0] = (theta1 + theta2) / 2.0

    phi1 = tilt1 - math.atan2(p1[1], f) * 180.0 / pi
    phi2 = tilt2 - math.atan2(p2[1], f) * 180.0 / pi
    ptz[1] = (phi1 + phi2) / 2.0
    ptz[2] = f

    return ptz

def pan_tilt_from_principle_point(pp, pp_ptz, p):
    """
    generate pan_tilt sample points using ptz ground truth
    :param pp:  pincipal point
    :param pp_ptz: pincipal point pan, tilt zoom, eg. ptz of the image
    :param p: a point in the image, unit pixel
    :return: return value, pan and tilt
    """
    dx = p[0] - pp[0]
    dy = p[1] - pp[1]
    pan_pp = pp_ptz[0]
    tilt_pp = pp_ptz[1]
    fl = pp_ptz[2]
    delta_pan = math.atan2(dx, fl) * 180 / math.pi
    delta_tilt = math.atan2(dy, fl) * 180 / math.pi
    pt = np.zeros((2, 1))
    pt[0] = pan_pp + delta_pan
    pt[1] = tilt_pp + delta_tilt

    return pt
