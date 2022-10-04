import csv
from textwrap import wrap
import time
import sys
from matplotlib import pyplot as plt
import numpy as np
import math
import os.path
import pandas as pd

HEIGHT_THRESHOLD = 0.0  # meters
GROUND_HEIGHT_THRESHOLD = -.4  # meters
DT = 0.1
X_LANDMARK = 5.  # meters
Y_LANDMARK = -5.  # meters
EARTH_RADIUS = 6.3781E6  # meters 
 
def load_data(filename):
    """Load data from the csv log

    Parameters:
    filename (str)  -- the name of the csv log

    Returns:
    data (dict)     -- the logged data with data categories as keys
                       and values list of floats
    """
    is_filtered = False
    if os.path.isfile(filename + "_filtered.csv"):
        f = open(filename + "_filtered.csv")
        is_filtered = True
    else:
        f = open(filename + ".csv")

    file_reader = csv.reader(f, delimiter=',')

    # Load data into dictionary with headers as keys
    data = {}
    header = ["X", "Y", "Z", "Time Stamp", "Latitude", "Longitude",
              "Yaw", "Pitch", "Roll", "AccelX", "AccelY", "AccelZ"]
    for h in header:
        data[h] = []

    row_num = 0
    f_log = open("bad_data_log.txt", "w")
    for row in file_reader:
        for h, element in zip(header, row):
            # If got a bad value just use the previous value
            try:
                data[h].append(float(element))
            except ValueError:
                data[h].append(data[h][-1])
                f_log.write(str(row_num) + "\n")

        row_num += 1
    f.close()
    f_log.close()

    return data, is_filtered


def save_data(data, filename):
    """Save data from dictionary to csv

    Parameters:
    filename (str)  -- the name of the csv log
    data (dict)     -- data to log
    """
    header = ["X", "Y", "Z", "Time Stamp", "Latitude", "Longitude",
              "Yaw", "Pitch", "Roll", "AccelX", "AccelY", "AccelZ"]
    f = open(filename, "w")
    num_rows = len(data["X"])
    for i in range(num_rows):
        for h in header:
            f.write(str(data[h][i]) + ",")

        f.write("\n")

    f.close()


def filter_data(data):
    """Filter lidar points based on height and duplicate time stamp

    Parameters:
    data (dict)             -- unfilterd data

    Returns:
    filtered_data (dict)    -- filtered data
    """

    # Remove data that is not above a height threshold to remove
    # ground measurements and remove data below a certain height
    # to remove outliers like random birds in the Linde Field (fuck you birds)
    filter_idx = [idx for idx, ele in enumerate(data["Z"])
                  if ele > GROUND_HEIGHT_THRESHOLD and ele < HEIGHT_THRESHOLD]

    filtered_data = {}
    for key in data.keys():
        filtered_data[key] = [data[key][i] for i in filter_idx]

    # Remove data that at the same time stamp
    ts = filtered_data["Time Stamp"]
    filter_idx = [idx for idx in range(1, len(ts)) if ts[idx] != ts[idx-1]]
    for key in data.keys():
        filtered_data[key] = [filtered_data[key][i] for i in filter_idx]

    return filtered_data


def convert_gps_to_xy(lat_gps, lon_gps, lat_origin, lon_origin):
    """Convert gps coordinates to cartesian with equirectangular projection

    Parameters:
    lat_gps     (float)    -- latitude coordinate
    lon_gps     (float)    -- longitude coordinate
    lat_origin  (float)    -- latitude coordinate of your chosen origin
    lon_origin  (float)    -- longitude coordinate of your chosen origin

    Returns:
    x_gps (float)          -- the converted x coordinate
    y_gps (float)          -- the converted y coordinate
    """
    x_gps = EARTH_RADIUS*(math.pi/180.)*(lon_gps - lon_origin)*math.cos((math.pi/180.)*lat_origin)
    y_gps = EARTH_RADIUS*(math.pi/180.)*(lat_gps - lat_origin)

    return x_gps, y_gps


def wrap_to_pi(angle):
    """Wrap angle data in radians to [-pi, pi]

    Parameters:
    angle (float)   -- unwrapped angle

    Returns:
    angle (float)   -- wrapped angle
    """
    while angle >= math.pi:
        angle -= 2*math.pi

    while angle <= -math.pi:
        angle += 2*math.pi
    return angle


def propogate_state(x_t_prev, u_t):
    """Propogate/predict the state based on chosen motion model

    Parameters:
    x_t_prev (np.array)  -- the previous state estimate
    u_t (np.array)       -- the current control input

    Returns:
    x_bar_t (np.array)   -- the predicted state
    """
    """STUDENT CODE START"""

    x_bar_t = np.array([])
    """STUDENT CODE END"""

    return x_bar_t


def calc_prop_jacobian_x(x_t_prev, u_t):
    """Calculate the Jacobian of your motion model with respect to state

    Parameters:
    x_t_prev (np.array) -- the previous state estimate
    u_t (np.array)      -- the current control input

    Returns:
    G_x_t (np.array)    -- Jacobian of motion model wrt to x
    """
    """STUDENT CODE START"""
    G_x_t = np.zeros((5, 5))  # add shape of matrix
    G_x_t[0,0] = 1
    G_x_t[0,2] = DT
    G_x_t[1,1] = 1
    G_x_t[1,3] = DT
    G_x_t[2,2] = 1
    G_x_t[2,4] = -u_t[0]*math.sin(wrap_to_pi(x_t_prev[4]))*DT
    G_x_t[3,3] = 1
    G_x_t[3,4] = u_t[0]*math.cos(wrap_to_pi((x_t_prev[4])))*DT
    G_x_t[4,4] = 1
    """STUDENT CODE END"""
    return G_x_t


def calc_prop_jacobian_u(x_t_prev, u_t):
    """Calculate the Jacobian of motion model with respect to control input

    Parameters:
    x_t_prev (np.array)     -- the previous state estimate
    u_t (np.array)          -- the current control input

    Returns:
    G_u_t (np.array)        -- Jacobian of motion model wrt to u
    """

    """STUDENT CODE START"""
    G_u_t = np.zeros((5, 2))  # add shape of matrix
    G_u_t[2,0] = math.cos(wrap_to_pi(x_t_prev[4]))*DT
    G_u_t[3,0] = math.sin(wrap_to_pi((x_t_prev[4])))*DT
    G_u_t[4,1] = DT
    """STUDENT CODE END"""

    return G_u_t


def prediction_step(x_t_prev, u_t, sigma_x_t_prev):
    """Compute the prediction of EKF

    Parameters:
    x_t_prev (np.array)         -- the previous state estimate
    u_t (np.array)              -- the control input
    sigma_x_t_prev (np.array)   -- the previous variance estimate

    Returns:
    x_bar_t (np.array)          -- the predicted state estimate of time t
    sigma_x_bar_t (np.array)    -- the predicted variance estimate of time t
    """

    """STUDENT CODE START"""
    # Covariance matrix of control input
    sigma_u_t = np.zeros((5,5))  # add shape of matrix
    G_x = calc_prop_jacobian_x(x_t_prev, u_t)
    G_u = calc_prop_jacobian_u(x_t_prev, u_t)
    # x_bar_t = G_x@x_t_prev + G_u@u_t
    # x_bar_t[4] = wrap_to_pi(x_bar_t[4])
    x_bar_t = np.empty([5,1])
    x_bar_t[0] = x_t_prev[0] + x_t_prev[2]*DT
    x_bar_t[1] = x_t_prev[1] + x_t_prev[3]*DT
    x_bar_t[2] = x_t_prev[2] + u_t[0]*math.cos(wrap_to_pi(x_t_prev[4]))*DT
    x_bar_t[3] = x_t_prev[3] + u_t[0]*math.sin(wrap_to_pi(x_t_prev[4]))*DT
    x_bar_t[4] = wrap_to_pi(wrap_to_pi(x_t_prev[4]) + wrap_to_pi(u_t[1]*DT))
    R = np.identity(2)
    sigma_x_bar_t = G_x@sigma_x_t_prev@G_x.transpose() + G_u@R@G_u.transpose()
    """STUDENT CODE END"""

    return [x_bar_t, sigma_x_bar_t]


def calc_meas_jacobian(x_bar_t):
    """Calculate the Jacobian of your measurment model with respect to state

    Parameters:
    x_bar_t (np.array)  -- the predicted state

    Returns:
    H_t (np.array)      -- Jacobian of measurment model
    """
    """STUDENT CODE START"""
    H_t = np.zeros((3, 5))
    # H_t[0,0] = -math.sin(x_bar_t[4])
    # H_t[0,1] = -math.cos(x_bar_t[4])
    # H_t[0,4] = x_bar_t[1]*math.sin(x_bar_t[4]) - x_bar_t[0]*math.cos(x_bar_t[4])
    # H_t[1,0] = -H_t[0,1]
    # H_t[1,1] = -H_t[0,0]
    # H_t[1,4] = -x_bar_t[1]*math.cos(x_bar_t[4]) - x_bar_t[0]*math.sin(x_bar_t[4])
    # H_t[2,4] = 1
    H_t[0,0] = 1
    H_t[1,1] = 1
    H_t[2,4] = 1
    """STUDENT CODE END"""

    return H_t


def calc_kalman_gain(sigma_x_bar_t, H_t):
    """Calculate the Kalman Gain

    Parameters:
    sigma_x_bar_t (np.array)  -- the predicted state covariance matrix
    H_t (np.array)            -- the measurement Jacobian

    Returns:
    K_t (np.array)            -- Kalman Gain
    """
    """STUDENT CODE START"""
    # Covariance matrix of measurments
    Q = np.identity(3)
   
    K_t = sigma_x_bar_t @ H_t.transpose() @ np.linalg.inv( (H_t @ sigma_x_bar_t @ H_t.transpose() ) + Q)
    """STUDENT CODE END"""

    return K_t


def calc_meas_prediction(x_bar_t):
    """Calculate predicted measurement based on the predicted state

    Parameters:
    x_bar_t (np.array)  -- the predicted state

    Returns:
    z_bar_t (np.array)  -- the predicted measurement
    """

    """STUDENT CODE START"""
    z_bar_t = np.empty([3, 1])
    z_bar_t = calc_meas_jacobian(x_bar_t)@x_bar_t
    """STUDENT CODE END"""

    return z_bar_t


def correction_step(x_bar_t, z_t, sigma_x_bar_t): 
    """Compute the correction of EKF
 
    Parameters:
    x_bar_t       (np.array)    -- the predicted state estimate of time t
    z_t           (np.array)    -- the measured state of time t
    sigma_x_bar_t (np.array)    -- the predicted variance of time t

    Returns:
    x_est_t       (np.array)    -- the filtered state estimate of time t
    sigma_x_est_t (np.array)    -- the filtered variance estimate of time t
    """

    """STUDENT CODE START"""
    H_t = calc_meas_jacobian(x_bar_t)
    K_t = calc_kalman_gain(sigma_x_bar_t, H_t) 
    z_bar_t = calc_meas_prediction(x_bar_t)
    z_minus = z_t - z_bar_t
    z_minus[2] = wrap_to_pi(z_minus[2])
    x_est_t = x_bar_t + K_t@(z_minus)
    # x_est_t[4] = wrap_to_pi(x_est_t[4])
    sigma_x_est_t = (np.identity(5) - K_t@H_t)@sigma_x_bar_t
    """STUDENT CODE END"""
    return [x_est_t, sigma_x_est_t]


def getYawVel(yawCurr, yawPrev):
    return wrap_to_pi((yawCurr) - (yawPrev))/DT

  
state_est_t_prev = [1.92, -0.1, 0.88, -0.01, -0.03]
var_est_t_prev = [[ 0.13, -0., 0.09, -0., -0. ],
[-0., 0.08, -0., 0.02, -0.],
[0.09, -0., 0.14, -0.01, -0.],
[-0., 0.02, -0., 0.01, -0.],
[ 0., 0., 0., 0., 0.01]]
u_t = [0.51, 0.13]
z_t = [2.02, -0.07, 0.04]
 

state_pred_t, var_pred_t = prediction_step(state_est_t_prev, u_t, var_est_t_prev)
print("x_bar_t")
print(state_pred_t)
print("sigma_bar_t")
print(var_pred_t)

state_est_t, var_est_t = correction_step(state_pred_t,
                                                 z_t,
                                                 var_pred_t)
