import glob
import os
import pandas as pd
import cv2
import mediapipe as mp
import utils
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
calculate_angle = utils.calculate_angle

clear = lambda: os.system('cls')

## Get filepaths of videos according to class
acc_filepaths = [os.path.normpath(i) for i in glob.glob('D:\Documents\CS 198\Data Collection\Dataset\Segmented Videos\**/acc_0?.mp4', 
                   recursive = True)]
ak_filepaths = [os.path.normpath(i) for i in glob.glob('D:\Documents\CS 198\Data Collection\Dataset\Segmented Videos\**/ak_0?.mp4', 
                   recursive = True)]
bo_filepaths = [os.path.normpath(i) for i in glob.glob('D:\Documents\CS 198\Data Collection\Dataset\Segmented Videos\**/bo_0?.mp4', 
                   recursive = True)]
hs_filepaths = [os.path.normpath(i) for i in glob.glob('D:\Documents\CS 198\Data Collection\Dataset\Segmented Videos\**/hs_0?.mp4', 
                   recursive = True)]
kvg_filepaths = [os.path.normpath(i) for i in glob.glob('D:\Documents\CS 198\Data Collection\Dataset\Segmented Videos\**/kvg_0?.mp4', 
                   recursive = True)]
kvr_filepaths = [os.path.normpath(i) for i in glob.glob('D:\Documents\CS 198\Data Collection\Dataset\Segmented Videos\**/kvr_0?.mp4', 
                   recursive = True)]
compiled_filepaths = [acc_filepaths, ak_filepaths, bo_filepaths, hs_filepaths, kvg_filepaths, kvr_filepaths]

## Define dataframe headers
headers = ["frame_height", "frame_width"]
for i in range(33):
  headers += ["body_" + str(i), "x_" + str(i), "y_" + str(i), "z_" + str(i), "visibility_" + str(i)]
target_classes = ["acc", "ak", "bo", "hs", "kvg", "kvr"]

## Extract body landmark data per video
for i in range(len(compiled_filepaths)):
  k = 0

  for filepath in compiled_filepaths[i]:
    k += 1
    clear()
    print(str(i+1) + " out of " + str(len(compiled_filepaths)) + ": Extracting data from " + filepath + "...")
    
    repetition = []
    cap = cv2.VideoCapture(filepath)
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
          mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
          image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

          try:
            # Extract landmarks
            landmarks = results.pose_landmarks.landmark
            landmark_data = [frame.shape[0], frame.shape[1]]
            for j in range(33):
              landmark_data += [j, landmarks[j].x, landmarks[j].y, landmarks[j].z, landmarks[j].visibility]
            
            repetition.append(landmark_data)

          except:
            pass

        else:
          break
    
    df = pd.DataFrame(repetition, columns=headers)
    df['target'] = target_classes[i]
    file_dest = "D:/Documents/CS 198/Data Collection/Dataset/Extracted Data/" + target_classes[i] + "/" + target_classes[i] + "_" + str(k) + ".csv"
    df.to_csv(path_or_buf=file_dest)

    cap.release()
    cv2.destroyAllWindows()