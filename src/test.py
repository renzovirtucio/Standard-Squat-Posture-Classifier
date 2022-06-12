import cv2
import joblib
import mediapipe as mp
import numpy as np
from utils import calculate_angle, calculate_distance, resize_with_aspect_ratio
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def test_model(clf):
  cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
  
  with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
      ret, frame = cap.read()

      if ret == True:
        frame = resize_with_aspect_ratio(frame, width=800)
        image = frame
        frame2 = np.full((frame.shape[0], 300, frame.shape[2]), 255, dtype=frame.dtype)

        # Make detection
        results = pose.process(image)

        input_data = []
        try:
          # Extract landmarks
          landmarks = results.pose_landmarks.landmark
          
          # Build feature vector
          left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
          left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
          left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
          left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
          left_foot_index = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]

          right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
          right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
          right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
          right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
          right_foot_index = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]

          left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
          left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
          left_ankle_angle = calculate_angle(left_knee, left_ankle, left_foot_index)
          
          right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
          right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
          right_ankle_angle = calculate_angle(right_knee, right_ankle, right_foot_index)

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

        except:
          pass

        textsize = cv2.getTextSize("REPORT", cv2.FONT_HERSHEY_SIMPLEX, 0.75, 1)[0]
        centerx = (frame2.shape[1] - textsize[0]) // 2
        cv2.putText(frame2, "REPORT", (centerx, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(frame2, 'YOUR SQUAT POSTURE:', (0,62), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(frame2, 'FEEDBACK:', (0,132), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

        if len(input_data) > 0:
          feedback = clf.predict([input_data])

          if (feedback != "others"):
            posture = {
              "acc": "Acceptable",
              "ak": "Anterior Knees",
              "bo": "Bent Over",
              "hs": "Half Squat",
              "kvg": "Knee Valgus",
              "kvr": "Knee Varus"
            }

            comment = {
              "Acceptable": "Great job!",
              "Anterior Knees": "Your knees have\nmoved forward\ntoo much.",
              "Bent Over": "Try to keep your\nback straight.",
              "Half Squat": "Go lower!",
              "Knee Valgus": "Your knees are\npointing inside.",
              "Knee Varus": "Your knees are\npointing outside."
            }

            feedback = posture[str(feedback[0])]

            pred_color = (0,177,64) if feedback == "Acceptable" else (43,75,238)

            
            cv2.putText(frame2, feedback, 
                        (0,100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.25, pred_color, 2, cv2.LINE_AA)

            y0, dy = 160, 25
            for j, line in enumerate(comment[feedback].split('\n')):
                y = y0 + j*dy
                cv2.putText(frame2, line, (0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, pred_color, 2, cv2.LINE_AA)

        image = np.hstack((frame2, cv2.flip(image, 1)))
        window_name = "Webcam Stream"
        cv2.imshow(window_name, image)
        key = cv2.waitKey(1)
        if key == ord('q') or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
          break
      else:
        break

  cap.release()
  cv2.destroyAllWindows()

  return

def main():
  clf_filename = "../assets/clf.sav"
  clf = joblib.load(clf_filename)
  test_model(clf)
  return

if __name__ == "__main__":
  main()