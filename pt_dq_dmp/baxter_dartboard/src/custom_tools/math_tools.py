#!/usr/bin/env python

import numpy as np
from copy import deepcopy as dcp
from dual_quaternions import DualQuaternion
from pyquaternion import Quaternion as Quat


def dx_dt(t, x):
    """
    Differentiate x w_i.r.t. t

    :param numpy.ndarray t: Time vector, (m)
    :param numpy.ndarray x: Data vector, (m, dim)
    :return: Differentiated data vector, (m, dim)
    :rtype: numpy.ndarray or List[DualQuaternion]
    """
    flag = x.dtype == DualQuaternion
    if flag:
        x = dql_to_npa(x)
    # Timestep vector
    dt = np.diff(t, axis=0).reshape((-1, 1))
    # Differentiation
    dx = np.divide(np.diff(x, axis=0), dt)
    dx = np.append(dx, [dx[-1, :]], axis=0)
    if flag:
        dx = npa_to_dql(dx)
    return dx


def dq_log(dq):
    """
    Dual Quaternion Logarithm (SE(3) -> se(3))

    :param DualQuaternion dq: Dual Quaternion
    :return: Dual Quaternion Logarithm
    :rtype: DualQuaternion
    """
    log = dcp(dq)
    log.q_r = 2 * Quat.log(dq.q_r)
    log.q_d = 2 * (dq.q_d * dq.q_r.inverse)
    return log


def dq_exp(dq):
    """
    Dual Quaternion Exponential (se(3) -> SE(3))

    :param DualQuaternion dq: Dual Quaternion
    :return: Dual Quaternion Exponential
    :rtype: DualQuaternion
    """
    exp = dcp(dq)
    exp.q_r = Quat.exp(.5 * dq.q_r)
    exp.q_d = (.5 * dq.q_d) * Quat.exp(.5 * dq.q_r)
    return exp


def next_dq_from_twist(dt, dq, tw):
    """
    Next Dual Quaternion Pose from Current Pose, Current Twist, and Timestep

    :param float dt: Timestep
    :param DualQuaternion dq: Current Pose
    :param DualQuaternion tw: Current Twist
    :return: Next Pose
    :rtype: DualQuaternion
    """
    dq_n = dq * dq_exp(dt * tw)
    return dq_n


def twist_from_dq_diff(dt, p_dq, dq):
    """
    Current Twist from Previous Pose, Current Pose, and Timestep

    :param float dt: Timestep
    :param DualQuaternion p_dq: Previous Pose
    :param DualQuaternion dq: Current Pose
    :return: Current Twist
    :rtype: DualQuaternion
    """
    tw = (1 / dt) * dq_log(p_dq.quaternion_conjugate() * dq)
    return tw


def dql_to_npa(dql):
    """
    Dual Quaternion list to numpy Array

    :param numpy.ndarray dql: Dual Quaternion List
    :return: numpy Array
    :rtype: numpy.ndarray
    """
    npa = np.empty((dql.shape[0], 8))
    for i, dqi in enumerate(dql):
        npa[i] = dqi.dq_array()
    return npa


def npa_to_dql(npa):
    """
    numpy Array to Dual Quaternion list

    :param numpy.ndarray npa: numpy Array
    :return: Dual Quaternion List
    :rtype: numpy.ndarray
    """
    dql = np.empty(npa.shape[0], dtype=DualQuaternion)
    for i, xi in enumerate(npa):
        dql[i] = DualQuaternion.from_dq_array(xi)
    return dql


def twist_from_dq_list(t, dq):
    """
    Twist List from Dual Quaternion list

    :param numpy.ndarray t: Time vector
    :param numpy.ndarray dq: Dual Quaternion list
    :return: Twist List
    :rtype: numpy.ndarray
    """
    tw = np.empty(t.shape[0] - 1, dtype=DualQuaternion)
    dt = np.diff(t)
    for i, (dqi, pdq, dti) in enumerate(zip(dq[1:], dq[0:-1], dt)):
        tw[i] = twist_from_dq_diff(dti, pdq, dqi)
    tw = np.append(tw, [tw[-1]])
    return tw


def edq_from_dq(dq, dqg):
    """
    Dual Quaternion Pose Error w.r.t. Goal Pose

    :param DualQuaternion dq: Current Pose
    :param DualQuaternion dqg: Goal Pose
    :return: Dual Quaternion Pose Error
    :rtype: DualQuaternion
    """
    return dq_log(dq.quaternion_conjugate() * dqg)


def dq_from_edq(edq, dqg):
    """
    Dual Quaternion List from Dual Quaternion Error w_i.r.t. Goal

    :param numpy.ndarray edq: Dual Quaternion Pose Error List
    :param DualQuaternion dqg: Goal Pose
    :return: Dual Quaternion Pose List
    :rtype: numpy.ndarray
    """
    dq = np.empty(edq.shape[0], dtype=DualQuaternion)
    for i, edq_i in enumerate(edq):
        dq[i] = dq_exp(edq_i).quaternion_conjugate() * dqg
    return dq


def quat_rot(a, b):
    """
    Quaternion Rotation from two pure quaternions

    :param Quaternion a: Pure Quaternion a
    :param Quaternion b: Pure Quaternion b
    :return: Quaternion Rotation from a to b
    :rtype: Quaternion
    """
    q = ((a * b).norm - (a * b).conjugate).normalised
    return q


def q_rot_from_vec(vec):
    """
    Quaternion Orientation from Approach Vector

    :param numpy.ndarray vec: Approach Vector (Z Vector of the Body Frame)
    :return: Quaternion Orientation
    :rtype: Quaternion
    """
    vex = np.multiply(np.copy(vec), np.array([1, 1, 0]))
    q_z = quat_rot(Quat(vector=[0, 0, 1]), Quat(vector=vec).normalised)
    q_x = quat_rot(Quat(vector=[1, 0, 0]), Quat(vector=vex).normalised)
    q_r = q_z * q_x
    return q_r


def vel_from_twist(dq, tw):
    """
    Velocity Pure Quaternion from Twist and Pose

    :param DualQuaternion dq: Current Pose
    :param DualQuaternion tw: Current Twist
    :return: Pure Quaternion Velocity
    :rtype: Quaternion
    """
    p = Quat(vector=dq.translation())
    omg = tw.q_r
    # elm = .5 * ((p * omg) - (omg * p))
    elm = (p * omg) - (p * omg).w
    vel = tw.q_d - elm
    return vel


def align_dq_goal(dq, tw=None):
    """
    Align goal throw Pose orientation to goal throw Twist

    :param DualQuaternion dq: Goal Pose
    :param DualQuaternion tw: Goal Twist
    :param List vec: Cartesian vector
    :return: Align Transformation for Goal Pose
    """
    q = dq.q_r
    # vel = vel_from_twist(dq, tw).normalised
    # norm = 1
    # vec = np.array([1, 0, 0])
    # for i in range(3):
    #     vec_ = np.zeros(3)
    #     vec_[i] = 1
    #     norm_ = (Quat(vector=q.rotate(vec_)) - vel).norm
    #     vec = vec if norm < norm_ else vec_
    #     norm = norm if norm < norm_ else norm_
    # q_a = Quat(vector=q.rotate(vec))
    # q_r = quat_rot(q_a, vel)
    # dq_fac = DualQuaternion.from_quat_pose_array(np.append(q_r.elements, [0, 0, 0]))
    # vel_y = Quat(vector=[vel.x, vel.y, 0]).normalised
    # vel_z = Quat(vector=[0, 0, np.abs(vel.z)]).normalised
    # vel_x = vel_y * vel_z
    # vel_x = vel_x - vel_x.w
    vec = q.rotate([0, 1, 0])
    q_a = Quat(vector=vec).normalised
    q_b = Quat(vector=[vec[0], vec[1], 0]).normalised
    q_r = quat_rot(q_a, q_b)
    dq_fac = DualQuaternion.from_quat_pose_array(np.append(q_r.elements, [0, 0, 0]))
    return dq_fac


def main():
    pass


if __name__ == '__main__':
    main()
