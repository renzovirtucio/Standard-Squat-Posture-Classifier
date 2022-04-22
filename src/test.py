import cv2
import joblib
import mediapipe as mp
import utils
import numpy as np
from utils import calculate_angle, calculate_distance, rescale_frame
# import custom_drawing_utils as mp_drawing
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def test_model(clf):
  squat_classes = ["acc", "ak", "bo", "hs", "kvg", "kvr"]
  for i in squat_classes:
    # cap = cv2.VideoCapture("D:/Documents/CS 198/Data Collection/Test/"+i+".mp4")
    cap = cv2.VideoCapture("D:/Documents/CS 198/Data Collection/Dataset/Raw Videos/14/"+i+".mp4")
    # cap = cv2.VideoCapture("D:/Documents/CS 198/Data Collection/Dataset/Segmented Videos/01/"+i+"/"+i+"_05.mp4")
    # cap = cv2.VideoCapture("C:/Users/Renzo Virtucio/Downloads/acc-zoomedout.mp4")
    # cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    frame_num = 0
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
      while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
          # frame = rescale_frame(frame, 50)
          # Recolor image to RGB
          image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          image.flags.writeable = False

          # Make detection
          results = pose.process(image)

          # Recolor back to BGR
          image.flags.writeable = True
          # Render detections
          # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
          image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
          input_data = []
          try:
            # Extract landmarks
            landmarks = results.pose_landmarks.landmark
            
            custom_landmarks = [0, 11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
            # for j in custom_landmarks:
            #   input_data += [landmarks[j].x, landmarks[j].y, landmarks[j].z, landmarks[j].visibility]
            
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
            left_ankle_angle = calculate_angle(left_knee, left_ankle, left_foot_index)
            # left_foot_index_angle = calculate_angle(left_heel, left_foot_index, [left_foot_index[0]-0.1, left_foot_index[1]])
            
            right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
            right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
            right_ankle_angle = calculate_angle(right_knee, right_ankle, right_foot_index)
            # right_foot_index_angle = calculate_angle(right_heel, right_foot_index, [right_foot_index[0]-0.1, right_foot_index[1]])

            left_knee_to_right_knee = calculate_distance(left_knee, right_knee)
            left_hip_to_right_hip = calculate_distance(left_hip, right_hip)
            left_hip_to_left_shoulder = calculate_distance(left_hip, left_shoulder)/left_hip_to_right_hip
            right_hip_to_right_shoulder = calculate_distance(right_hip, right_shoulder)/left_hip_to_right_hip
            left_shoulder_to_right_shoulder = calculate_distance(left_shoulder, right_shoulder)/left_hip_to_right_hip

            left_hip_to_left_knee = calculate_distance([landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, 
                                      landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y], 
                                      [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, 
                                      landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y])/left_hip_to_right_hip
            right_hip_to_right_knee = calculate_distance([landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, 
                                      landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y],
                                      [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, 
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y])/left_hip_to_right_hip
            left_knee_to_left_ankle = calculate_distance([landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, 
                                      landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y],
                                      [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, 
                                      landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y])/left_hip_to_right_hip
            right_knee_to_right_ankle = calculate_distance([landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, 
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y],
                                      [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, 
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y])/left_hip_to_right_hip

            left_ankle_to_left_heel = calculate_distance([landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, 
                                      landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y],
                                      [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x, 
                                      landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y])/left_hip_to_right_hip
            left_heel_to_left_foot_index = calculate_distance([landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x, 
                                      landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y],
                                      [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x, 
                                      landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y])/left_hip_to_right_hip
            left_foot_index_to_left_ankle = calculate_distance([landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x, 
                                      landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y],
                                      [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, 
                                      landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y])/left_hip_to_right_hip

            right_ankle_to_right_heel = calculate_distance([landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, 
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y],
                                      [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x, 
                                      landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y])/left_hip_to_right_hip
            right_heel_to_right_foot_index = calculate_distance([landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x, 
                                      landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y],
                                      [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x, 
                                      landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y])/left_hip_to_right_hip
            right_foot_index_to_right_ankle = calculate_distance([landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x, 
                                      landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y],
                                      [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, 
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y])/left_hip_to_right_hip
            
            hip_width_to_knee_width_ratio = left_knee_to_right_knee/left_hip_to_right_hip

            input_data += [left_knee_angle, left_hip_angle, left_ankle_angle, right_knee_angle, 
                        right_hip_angle, right_ankle_angle, left_hip_to_right_hip, left_hip_to_left_shoulder, 
                        right_hip_to_right_shoulder, left_shoulder_to_right_shoulder, 
                        left_hip_to_left_knee, right_hip_to_right_knee, left_knee_to_left_ankle, 
                        right_knee_to_right_ankle, left_ankle_to_left_heel, left_heel_to_left_foot_index, 
                        left_foot_index_to_left_ankle, right_ankle_to_right_heel, right_heel_to_right_foot_index, 
                        right_foot_index_to_right_ankle, hip_width_to_knee_width_ratio]

            # cv2.putText(image, str(int(left_knee_angle)), 
            #                         tuple(np.multiply(left_knee, [int(frame.shape[1]), int(frame.shape[0])]).astype(int)), 
            #                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
            #                               )
            # cv2.putText(image, str(int(right_knee_angle)), 
            #               tuple(np.multiply(right_knee, [int(frame.shape[1]), int(frame.shape[0])]).astype(int)), 
            #               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
            #                     )
            # cv2.putText(image, str(int(left_hip_angle)), 
            #                        tuple(np.multiply(left_hip, [int(frame.shape[1]), int(frame.shape[0])]).astype(int)), 
            #                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
            #                             )
            # cv2.putText(image, str(int(right_hip_angle)), 
            #                 tuple(np.multiply(right_hip, [int(frame.shape[1]), int(frame.shape[0])]).astype(int)), 
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
            #                             )
            # cv2.putText(image, str(int(left_ankle_angle)), 
            #                 tuple(np.multiply(left_ankle, [int(frame.shape[1]), int(frame.shape[0])]).astype(int)), 
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
            #                             )
            # cv2.putText(image, str(int(right_ankle_angle)), 
            #                 tuple(np.multiply(right_ankle, [int(frame.shape[1]), int(frame.shape[0])]).astype(int)), 
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
            #                             )
          except:
            pass
        
          # if frame_num == 0: print(input_data)

          if len(input_data) > 0:
            feedback = clf.predict([input_data])
            if (feedback != "others"):
              cv2.rectangle(image, (0,0), (225,75), (255,255,255), -1)

              pred_color = (0,177,64) if str(feedback[0]) == "acc" else (43,75,238)

              cv2.putText(image, 'PREDICTED SQUAT POSTURE:', (0,12), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
              cv2.putText(image, str(feedback[0]), 
                          (0,60), 
                          cv2.FONT_HERSHEY_SIMPLEX, 2, pred_color, 2, cv2.LINE_AA)
          
          # Save frame to .jpg
          # cv2.imwrite('D:/Documents/CS 198/Data Collection/Dataset/Extracted Frames/Frame'+str(frame_num)+'.jpg', image)
          frame_num += 1
          
          cv2.imshow(i+".mp4", image)
          key = cv2.waitKey(20)
          if key == ord('q'):
            break
        else:
          break

    cap.release()
    cv2.destroyAllWindows()

  return

def main():
  clf_filename = "../assets/svm_clf.sav"
  clf = joblib.load(clf_filename)
  test_model(clf)
  return

if __name__ == "__main__":
  main()