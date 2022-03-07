import pandas as pd
import glob
import os
import numpy as np
from IPython.display import display

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

import cv2
import mediapipe as mp
import numpy as np
import math
# import custom_drawing_utils as mp_drawing
import utils
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
calculate_angle = utils.calculate_angle

CUSTOM_POSE_CONNECTIONS = frozenset([(11, 12), (11, 23), (12, 24), (23, 24), (23, 25),
                              (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
                              (29, 31), (30, 32), (27, 31), (28, 32)])

## Load raw extracted data from csv files into dataframes
squat_csv_filepaths = [os.path.normpath(i) for i in glob.glob('D:\Documents\CS 198\Data Collection\Dataset\Extracted Data/**/*.csv', recursive = True)]

squat_class_dataframes = []
for filepath in squat_csv_filepaths:
  squat_class_dataframes.append(pd.read_csv(filepath))

compiled_dataframe = pd.concat(squat_class_dataframes, ignore_index=True)
compiled_dataframe = compiled_dataframe.rename(columns={'Unnamed: 0': 'frame_number'})

compiled_dataframe_copy = compiled_dataframe
compiled_dataframe_copy = compiled_dataframe_copy.drop(['frame_number'], axis=1)

for i in range(33):
    compiled_dataframe_copy = compiled_dataframe_copy.drop(['body_'+str(i)], axis=1)

display(compiled_dataframe_copy)

## Create SVM classifier with raw extracted data

# Define X and y for testing and training
y = compiled_dataframe_copy['target']

# Drop G3 and the target variable, Quality
X = compiled_dataframe_copy.drop(['target'], axis=1)  
print(X.shape, y.shape)

# Split the data for testing and training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=0)

scaler = StandardScaler()

# we fit the train data
scaler.fit(X_train)

# scaling the train data
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Fitting a support vector machine
model = SVC(random_state=0)
print("SVC parameters:", model.get_params())

model.fit(X_train, y_train)

# Compute model predictions
model_predictions = model.predict(X_test)

orig_acc_score_SVM = accuracy_score(y_test, model_predictions)
print("Accuracy:", orig_acc_score_SVM)

# Plot confusion matrix
cm = confusion_matrix(y_test, model_predictions, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()  
print("Figure 3. Confusion Matrix of SVM Model")
plt.show() 

# Print classification report
print("Table 6. Classification Report of SVM Model")
print(classification_report(y_test, model_predictions))


cap = cv2.VideoCapture("D:/Documents/CS 198/Data Collection/Dataset/Segmented Videos/01/bo/bo_05.mp4")
frame_num = 0
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
      frame50 = utils.rescale_frame(frame, 50)

      # Recolor image to RGB
      image = cv2.cvtColor(frame50, cv2.COLOR_BGR2RGB)
      image.flags.writeable = False

      # Make detection
      results = pose.process(image)

      # Recolor back to BGR
      image.flags.writeable = True
      # Render detections
      mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

      try:
        # Extract landmarks
        landmarks = results.pose_landmarks.landmark
        
        landmark_data = []
        for j in range(33):
          landmark_data += [landmarks[j].x, landmarks[j].y, landmarks[j].visibility]

        

        # custom_landmarks = [results.pose_landmarks.landmark[i] for i in [0, 11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]]
        # cv2.imwrite('D:/Documents/CS 198/Data Collection/Dataset/Extracted Frames/Frame'+str(frame_num)+'.jpg', image)
        # frame_num += 1
        
        # left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        # left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        # left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        # left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        # left_heel = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y]
        # left_foot_index = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]

        # right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        # right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        # right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        # right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        # right_heel = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y]
        # right_foot_index = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]

        # left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        # left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
        # left_foot_index_angle = calculate_angle(left_heel, left_foot_index, [left_foot_index[0]-0.1, left_foot_index[1]])
        
        # right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
        # right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
        # right_foot_index_angle = calculate_angle(right_heel, right_foot_index, [right_foot_index[0]-0.1, right_foot_index[1]])

        # left_knee_to_right_knee = utils.calculate_distance(left_knee, right_knee)
        # # print(left_knee_to_right_knee)

        # cv2.putText(image, str(int(left_knee_angle)), 
        #                         tuple(np.multiply(left_knee, [int(frame50.shape[1]), int(frame50.shape[0])]).astype(int)), 
        #                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
        #                               )
        # cv2.putText(image, str(int(right_knee_angle)), 
        #               tuple(np.multiply(right_knee, [int(frame50.shape[1]), int(frame50.shape[0])]).astype(int)), 
        #               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
        #                     )
        # cv2.putText(image, str(int(left_hip_angle)), 
        #                        tuple(np.multiply(left_hip, [int(frame50.shape[1]), int(frame50.shape[0])]).astype(int)), 
        #                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
        #                             )
        # cv2.putText(image, str(int(right_hip_angle)), 
        #                 tuple(np.multiply(right_hip, [int(frame50.shape[1]), int(frame50.shape[0])]).astype(int)), 
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
        #                             )
        # cv2.putText(image, str(int(left_foot_index_angle)), 
        #                 tuple(np.multiply(left_foot_index, [int(frame50.shape[1]), int(frame50.shape[0])]).astype(int)), 
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
        #                             )
        # cv2.putText(image, str(int(right_foot_index_angle)), 
        #                 tuple(np.multiply(right_foot_index, [int(frame50.shape[1]), int(frame50.shape[0])]).astype(int)), 
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
        #                             )

      except:
        pass
      input_data = landmark_data
      feedback = model.predict(scaler.transform([input_data]))
      cv2.rectangle(image, (0,0), (80,73), (245,117,16), -1)

      cv2.putText(image, 'FEEDBACK', (0,12), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
      cv2.putText(image, str(feedback[0]), 
                  (0,60), 
                  cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

      cv2.imshow('Video', image)
      key = cv2.waitKey(20)
      if key == ord('q'):
        break
    else:
      break

cap.release()
cv2.destroyAllWindows()