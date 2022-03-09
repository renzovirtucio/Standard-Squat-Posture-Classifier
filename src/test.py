import cv2
import joblib
import mediapipe as mp
import utils
import models
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
calculate_angle = utils.calculate_angle
calculate_distance = utils.calculate_distance

def test_model(model, scaler):
  cap = cv2.VideoCapture("D:/Documents/CS 198/Data Collection/Dataset/Segmented Videos/01/ak/ak_01.mp4")
  frame_num = 0
  with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
      ret, frame = cap.read()
      if ret == True:
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        # Render detections
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
          # Extract landmarks
          landmarks = results.pose_landmarks.landmark
          
          landmark_data = [frame.shape[0], frame.shape[1]]
          custom_landmarks = [0, 11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
          for j in custom_landmarks:
            landmark_data += [landmarks[j].x, landmarks[j].y, landmarks[j].z, landmarks[j].visibility]

          # cv2.imwrite('D:/Documents/CS 198/Data Collection/Dataset/Extracted Frames/Frame'+str(frame_num)+'.jpg', image)
          # frame_num += 1
          
          left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
          left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
          left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
          left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
          left_heel = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y]
          left_foot_index = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]

          right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
          right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
          right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
          right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
          right_heel = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y]
          right_foot_index = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]

          left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
          left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
          left_foot_index_angle = calculate_angle(left_heel, left_foot_index, [left_foot_index[0]-0.1, left_foot_index[1]])
          
          right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
          right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
          right_foot_index_angle = calculate_angle(right_heel, right_foot_index, [right_foot_index[0]-0.1, right_foot_index[1]])

          left_knee_to_right_knee = calculate_distance(left_knee, right_knee)
          left_hip_to_right_hip = calculate_distance(left_hip, right_hip)
          hip_width_to_knee_width_ratio = left_knee_to_right_knee/left_hip_to_right_hip

          cv2.putText(image, str(int(left_knee_angle)), 
                                  tuple(np.multiply(left_knee, [int(frame.shape[1]), int(frame.shape[0])]).astype(int)), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                        )
          cv2.putText(image, str(int(right_knee_angle)), 
                        tuple(np.multiply(right_knee, [int(frame.shape[1]), int(frame.shape[0])]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                              )
          cv2.putText(image, str(int(left_hip_angle)), 
                                 tuple(np.multiply(left_hip, [int(frame.shape[1]), int(frame.shape[0])]).astype(int)), 
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                      )
          cv2.putText(image, str(int(right_hip_angle)), 
                          tuple(np.multiply(right_hip, [int(frame.shape[1]), int(frame.shape[0])]).astype(int)), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                      )
          cv2.putText(image, str(int(left_foot_index_angle)), 
                          tuple(np.multiply(left_foot_index, [int(frame.shape[1]), int(frame.shape[0])]).astype(int)), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                      )
          cv2.putText(image, str(int(right_foot_index_angle)), 
                          tuple(np.multiply(right_foot_index, [int(frame.shape[1]), int(frame.shape[0])]).astype(int)), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                      )
        except:
          pass
        input_data = landmark_data + [left_knee_angle, left_hip_angle, left_foot_index_angle, right_knee_angle, 
                      right_hip_angle, right_foot_index_angle, left_knee_to_right_knee, left_hip_to_right_hip,
                      hip_width_to_knee_width_ratio]
        feedback = model.predict(scaler.transform([input_data]))
        cv2.rectangle(image, (0,0), (150,80), (255,255,255), -1)

        cv2.putText(image, 'PREDICTED CLASS:', (0,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(feedback[0]), 
                    (0,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 2, cv2.LINE_AA)

        cv2.imshow('Video Feed', image)
        key = cv2.waitKey(20)
        if key == ord('q'):
          break
      else:
        break

  cap.release()
  cv2.destroyAllWindows()

def main():
  model_filename = "initial_svm.sav"
  model, scaler = joblib.load(model_filename)
  test_model(model, scaler)
  return

if __name__ == "__main__":
  main()