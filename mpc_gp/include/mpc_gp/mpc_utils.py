import random


import os
import math
import json
import errno
import shutil
import joblib
import random
import pyquaternion
import numpy as np


def wrap_to_pi(angle):
    """
    :param angle: (float)
    :return: (float) Angle in radian in [-pi, pi]
    """
    while angle > np.pi-0.01:
        angle -= 2.0 * np.pi

    while angle < -np.pi+0.01:
        angle += 2.0 * np.pi

    return angle 


def euler_to_quaternion(roll, pitch, yaw):
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)

    return np.array([qw, qx, qy, qz])



def get_odom_euler(odom):    
    q = pyquaternion.Quaternion(w=odom.pose.pose.orientation.w, x=odom.pose.pose.orientation.x, y=odom.pose.pose.orientation.y, z=odom.pose.pose.orientation.z)
    yaw, pitch, roll = q.yaw_pitch_roll
    return [roll, pitch, yaw]

def quaternion_to_euler(q):
    q = pyquaternion.Quaternion(w=q[0], x=q[1], y=q[2], z=q[3])
    yaw, pitch, roll = q.yaw_pitch_roll
    return [roll, pitch, yaw]


def unit_quat(q):
    """
    Normalizes a quaternion to be unit modulus.
    :param q: 4-dimensional numpy array or CasADi object
    :return: the unit quaternion in the same data format as the original one
    """

    if isinstance(q, np.ndarray):
        # if (q == np.zeros(4)).all():
        #     q = np.array([1, 0, 0, 0])
        q_norm = np.sqrt(np.sum(q ** 2))
    else:
        q_norm = cs.sqrt(cs.sumsqr(q))
    return 1 / q_norm * q

def q_to_rot_mat(q):
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]    
    rot_mat = np.array([
        [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)]])
    return rot_mat


def get_local_vel(odom, is_odom_local_frame = False):
    local_vel = np.array([0.0, 0.0, 0.0])
    if is_odom_local_frame is False: 
        # convert from global to local 
        q_tmp = np.array([odom.pose.pose.orientation.w,odom.pose.pose.orientation.x,odom.pose.pose.orientation.y,odom.pose.pose.orientation.z])
        euler = get_odom_euler(odom)
        rot_mat_ = q_to_rot_mat(q_tmp)
        inv_rot_mat_ = np.linalg.inv(rot_mat_)
        global_vel = np.array([odom.twist.twist.linear.x,odom.twist.twist.linear.y,odom.twist.twist.linear.z])
        local_vel = inv_rot_mat_.dot(global_vel)        
    else:
        local_vel[0] = odom.twist.twist.linear.x
        local_vel[1] = odom.twist.twist.linear.y
        local_vel[2] = odom.twist.twist.linear.z
    return local_vel 