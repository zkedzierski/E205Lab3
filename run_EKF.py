"""
Author: Andrew Q. Pham
Email: apham@g.hmc.edu
Date of Creation: 2/26/20
Description:
    Extended Kalman Filter implementation to filtering localization estimate
    This code is for teaching purposes for HMC ENGR205 System Simulation Lab 3
    Student code version with parts omitted.
"""

import csv
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
    G_x_t[2,4] = -u_t[0]*math.sin(x_t_prev[4])*DT
    G_x_t[3,3] = 1
    G_x_t[3,4] = u_t[0]*math.cos(x_t_prev[4])*DT
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
    G_u_t[2,0] = math.cos(x_t_prev[4])*DT
    G_u_t[3,0] = math.sin(x_t_prev[4])*DT
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
    x_bar_t[4] = ((x_t_prev[4]) + (u_t[1]*DT))
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
    z_bar_t = np.ones([3, 1])
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
    x_est_t = x_bar_t + K_t@(z_t - z_bar_t)
    x_est_t[2] = wrap_to_pi(x_est_t[2])
    sigma_x_est_t = (np.identity(5) - K_t@H_t)@sigma_x_bar_t
    """STUDENT CODE END"""
    return [x_est_t, sigma_x_est_t]


def getYawVel(yawPrev, yawCurr):
    return (yawCurr - yawPrev)/DT

def main():
    """Run a EKF on logged data from IMU and LiDAR moving in a box formation around a landmark"""

    # filepath = "./logs/"
    filename = "/Users/kevinkong/Documents/E205/E205Lab3/2020_2_26__16_59_7_filtered"
    data, is_filtered = load_data(filename)

    # # Save filtered data so don't have to process unfiltered data everytime
    # if not is_filtered:
    #     data = filter_data(data)
    #     save_data(f_data, filename)

    # Load data into variables
    x_lidar = data["X"]
    y_lidar = data["Y"]
    z_lidar = data["Z"]
    time_stamps = data["Time Stamp"]
    lat_gps = data["Latitude"]
    lon_gps = data["Longitude"]
    yaw_lidar = data["Yaw"]
    pitch_lidar = data["Pitch"]
    roll_lidar = data["Roll"]
    x_ddot = data["AccelX"]
    y_ddot = data["AccelY"] # TO DO: MOVING WINDOW AVERAGE
    lat_origin = lat_gps[0]
    lon_origin = lon_gps[0]
    
    for t in range(len(time_stamps)):
            yaw_lidar[t] = wrap_to_pi(yaw_lidar[t] *  math.pi /180)

    #  Initialize filter
    """STUDENT CODE START"""
    N = 5 # number of states

    """INITIAL GUESS HARDCODED"""
    [x_intial, y_intial] = convert_gps_to_xy(lat_origin, lon_origin, lat_origin, lon_origin)  
    state_est_t_prev = np.empty([N, 1])
    state_est_t_prev[0,0] = x_intial
    state_est_t_prev[1,0] = y_intial
    state_est_t_prev[2,0] = 0
    state_est_t_prev[3,0] = 0
    state_est_t_prev[4,0] = 0
    var_est_t_prev = np.identity(N)
    
    state_estimates = np.empty((N, len(time_stamps)))
    for i in range(N):
        if i == 4:
            state_estimates[i, 0] = wrap_to_pi(state_est_t_prev[i, 0])
        else:
            state_estimates[i, 0] = state_est_t_prev[i, 0]

    covariance_estimates = np.empty((N, N, len(time_stamps) ))
    #####
    for i in range(N):
        for j in range(N):
            covariance_estimates[i, j, 0] = var_est_t_prev[i, j]
    #####

    gps_estimates = np.empty((2, len(time_stamps)))

    [gps_estimate_x, gps_estimate_y] = convert_gps_to_xy(lat_origin, lon_origin, lat_origin, lon_origin)

    gps_estimates[0,0] = gps_estimate_x
    gps_estimates[1,0] = gps_estimate_y
    """STUDENT CODE END"""

    # Moving Average over 3
    xdd_df = pd.DataFrame(x_ddot, columns = ["xdd"])
    x_moving_avg = xdd_df["xdd"].rolling(3).mean()
    x_moving_avg[0] = 0
    x_moving_avg[1] = 0
    counter = 0

    #  Run filter over data
    for t, _ in enumerate(time_stamps):
        # Get control input
        """STUDENT CODE START"""
        # Justin's hacky code: may not work
        if counter < 2:
            counter += 1
            continue
     
        # Input
        u_t = np.empty([2, 1])
        u_t[0] = x_moving_avg[t]
        u_t[1] = getYawVel(yaw_lidar[t-1], yaw_lidar[t])
        """STUDENT CODE END"""

        # Prediction Step
        state_pred_t, var_pred_t = prediction_step(state_est_t_prev, u_t, var_est_t_prev)

        # Get measurement
        """STUDENT CODE START"""
        z_t = np.empty([3, 1])
        z_t[0] = 5 - (y_lidar[t]* math.cos(wrap_to_pi(yaw_lidar[t])) + x_lidar[t]*math.sin(wrap_to_pi(yaw_lidar[t])))
        z_t[1] = -5 - (y_lidar[t]* math.sin(wrap_to_pi(yaw_lidar[t])) - x_lidar[t]*math.cos(wrap_to_pi(yaw_lidar[t])))
        z_t[2] = wrap_to_pi(yaw_lidar[t])
        """STUDENT CODE END"""

        # Correction Step
        state_est_t, var_est_t = correction_step(state_pred_t,
                                                 z_t,
                                                 var_pred_t)
        #  For clarity sake/teaching purposes, we explicitly update t->(t-1)
        state_est_t_prev = state_est_t
        var_est_t_prev = var_est_t

        # Log Data
     
        # state_estimates[:, t] = state_est_t
        for i in range(N):
            state_estimates[i,t] = state_est_t[i]

        covariance_estimates[:, :, t] = var_est_t

        x_gps, y_gps = convert_gps_to_xy(lat_gps=lat_gps[t],
                                         lon_gps=lon_gps[t],
                                         lat_origin=lat_origin,
                                         lon_origin=lon_origin)
        gps_estimates[:, t] = np.array([x_gps, y_gps])
        # gps_estimates[0,t] = x_gps
        # gps_estimates[1,t] = y_gps


    """STUDENT CODE START"""
    # Plot or print results here
    plt.plot(gps_estimates[0],gps_estimates[1])
    plt.plot(state_estimates[0,:], state_estimates[1,:])
    plt.xlabel("X Coord (m)")
    plt.ylabel("Y Coord (m)")
    plt.title("Real vs Estimated Path")
    plt.show()
    """STUDENT CODE END"""
    return 0


if __name__ == "__main__":
    main()