import os
import glob
import pandas as pd
import numpy as np
from IPython.display import display

import utils
calculate_angle = utils.calculate_angle
calculate_distance = utils.calculate_distance

CUSTOM_POSE_CONNECTIONS = frozenset([(11, 12), (11, 23), (12, 24), (23, 24), (23, 25),
                              (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
                              (29, 31), (30, 32), (27, 31), (28, 32)])

def load_data_into_df(filepaths):
  dataframes = []
  for filepath in filepaths:
    dataframes.append(pd.read_csv(filepath))

  return pd.concat(dataframes, ignore_index=True)

def build_feature_vectors(df):
  df = df.rename(columns={'Unnamed: 0': 'frame_number'})
  df = df.drop(['frame_number', 'frame_width', 'frame_height'], axis=1)

  for i in range(33):
    df = df.drop(['body_'+str(i)], axis=1)

  exclude_landmarks = [1,2,3,4,5,6,7,8,9,10,13,14,15,16,17,18,19,20,21,22]
  for i in exclude_landmarks:
    df = df.drop(['x_'+str(i),'y_'+str(i),'z_'+str(i),'visibility_'+str(i)], axis=1)

  df['left_knee_angle'] = df.apply(lambda x: calculate_angle([x['x_23'], x['y_23']], 
                                    [x['x_25'], x['y_25']], [x['x_27'], x['y_27']]), axis=1)
  df['left_hip_angle'] = df.apply(lambda x: calculate_angle([x['x_11'], x['y_11']], 
                                    [x['x_23'], x['y_23']], [x['x_25'], x['y_25']]), axis=1)
  df['left_foot_index_angle'] = df.apply(lambda x: calculate_angle([x['x_29'], x['y_29']], 
                                    [x['x_31'], x['y_31']], [x['x_31']-0.1, x['y_31']]), axis=1)
  df['right_knee_angle'] = df.apply(lambda x: calculate_angle([x['x_24'], x['y_24']], 
                                    [x['x_26'], x['y_26']], [x['x_28'], x['y_28']]), axis=1)
  df['right_hip_angle'] = df.apply(lambda x: calculate_angle([x['x_12'], x['y_12']], 
                                    [x['x_24'], x['y_24']], [x['x_26'], x['y_26']]), axis=1)
  df['right_foot_index_angle'] = df.apply(lambda x: calculate_angle([x['x_30'], x['y_30']], 
                                    [x['x_32'], x['y_32']], [x['x_32']-0.1, x['y_32']]), axis=1)
  
  # df['left_knee_to_right_knee'] = df.apply(lambda x: calculate_distance([x['x_25'], x['y_25']],[x['x_26'], x['y_26']]), axis=1)
  # df['left_hip_to_right_hip'] = df.apply(lambda x: calculate_distance([x['x_23'], x['y_23']],[x['x_24'], x['y_24']]), axis=1)
  # df['left_hip_to_left_shoulder'] = df.apply(lambda x: calculate_distance([x['x_11'], x['y_11']],[x['x_23'], x['y_23']]), axis=1)
  # df['right_hip_to_right_shoulder'] = df.apply(lambda x: calculate_distance([x['x_12'], x['y_12']],[x['x_24'], x['y_24']]), axis=1)
  # df['left_shoulder_to_right_shoulder'] = df.apply(lambda x: calculate_distance([x['x_11'], x['y_11']],[x['x_12'], x['y_12']]), axis=1)
  
  df['hip_width_to_knee_width_ratio'] = df.apply(lambda x: calculate_distance([x['x_25'], x['y_25']], 
                                          [x['x_26'], x['y_26']])/calculate_distance([x['x_23'], x['y_23']], 
                                          [x['x_24'], x['y_24']]), axis=1)

  return df

def preprocess_data(filepaths):
  return build_feature_vectors(load_data_into_df(filepaths))

def main():
  df = preprocess_data([os.path.normpath(i) for i in 
                        glob.glob('D:\Documents\CS 198\Data Collection\Dataset\Extracted Data/**/*.csv', 
                        recursive = True)])
  display(df.head())
  return

if __name__ == "__main__":
  main()