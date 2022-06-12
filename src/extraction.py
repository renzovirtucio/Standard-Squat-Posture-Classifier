import glob
import os
import pandas as pd
import cv2
import mediapipe as mp
import utils
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

clear = lambda: os.system('cls')

def extract_raw_data():
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

def extract_frames():
  directories = [os.path.normpath(i) for i in glob.glob('D:/Documents/CS 198/Data Collection/Dataset/Segmented Videos/*', 
                    recursive = True)]
  
  squat_classes = ["acc", "ak", "bo", "hs", "kvg", "kvr"]

  for dir in directories:
    for squat_class in squat_classes:
      for i in range(1, 6):
        new_dir = os.path.join(dir, squat_class, squat_class + "_0" + str(i))
        os.mkdir(new_dir)
        print(new_dir)
        cap = cv2.VideoCapture(new_dir+".mp4")
        frame_num = 1
        
        while(cap.isOpened()):
            ret, frame = cap.read()
            
            # This condition prevents from infinite looping
            # incase video ends.
            if ret == False:
                break
            
            # Save Frame by Frame into disk using imwrite method
            cv2.imwrite(os.path.join(new_dir, 'Frame'+str(frame_num)+'.jpg'), frame)
            frame_num += 1
        
        cap.release()
        cv2.destroyAllWindows()

  return

def extract_raw_data_v2():
  ## Get filepaths of videos according to class
  acc_filepaths = [os.path.normpath(i) for i in glob.glob('D:\Documents\CS 198\Data Collection\Dataset\Segmented Videos\*/acc/*/*.jpg', 
                    recursive = True)]
  ak_filepaths = [os.path.normpath(i) for i in glob.glob('D:\Documents\CS 198\Data Collection\Dataset\Segmented Videos\*/ak/*/*.jpg', 
                    recursive = True)]
  bo_filepaths = [os.path.normpath(i) for i in glob.glob('D:\Documents\CS 198\Data Collection\Dataset\Segmented Videos\*/bo/*/*.jpg', 
                    recursive = True)]
  hs_filepaths = [os.path.normpath(i) for i in glob.glob('D:\Documents\CS 198\Data Collection\Dataset\Segmented Videos\*/hs/*/*.jpg', 
                    recursive = True)] + [os.path.normpath(i) for i in glob.glob('D:\Documents\CS 198\Data Collection\Dataset\Segmented Videos\*/*/*/Others/hs/*.jpg', 
                    recursive = True)]
  kvg_filepaths = [os.path.normpath(i) for i in glob.glob('D:\Documents\CS 198\Data Collection\Dataset\Segmented Videos\*/kvg/*/*.jpg', 
                    recursive = True)]
  kvr_filepaths = [os.path.normpath(i) for i in glob.glob('D:\Documents\CS 198\Data Collection\Dataset\Segmented Videos\*/kvr/*/*.jpg', 
                    recursive = True)]
  others_filepaths = [os.path.normpath(i) for i in glob.glob('D:\Documents\CS 198\Data Collection\Dataset\Segmented Videos\*/*/*/Others/*.jpg', 
                    recursive = True)]
  
  compiled_filepaths = [acc_filepaths, ak_filepaths, bo_filepaths, hs_filepaths, kvg_filepaths, kvr_filepaths, others_filepaths]

  print([len(i) for i in compiled_filepaths])

  ## Define dataframe headers
  headers = ["frame_height", "frame_width"]
  for i in range(33):
    headers += ["body_" + str(i), "x_" + str(i), "y_" + str(i), "z_" + str(i), "visibility_" + str(i)]
  target_classes = ["acc", "ak", "bo", "hs", "kvg", "kvr", "others"]

  ## Extract body landmark data per video
  for i in range(len(compiled_filepaths)):
    if i != 0:
      continue

    k = 0

    df_list = []
    for filepath in compiled_filepaths[i]:
      k += 1
      clear()
      print(str(i+1) + " out of " + str(len(compiled_filepaths)) + ": Extracting data from " + filepath + "...")
      
      repetition = []
      with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        # Recolor image to RGB
        frame = cv2.imread(filepath)
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
      
      df = pd.DataFrame(repetition, columns=headers)
      df['target'] = target_classes[i]
      df_list.append(df)
    main_df = pd.concat(df_list, ignore_index=True)
    file_dest = "D:/Documents/CS 198/Data Collection/Dataset/New Extracted Data/" + target_classes[i] + "/" + target_classes[i] + ".csv"
    main_df.to_csv(path_or_buf=file_dest)

  return

def main():
  # extract_raw_data_v2()
  # extract_frames()
  
  t = ["01", "02", "03", "04", "05", "06", "07", "12", "13", "14"]
  total_per_indiv = []

  for i in t: 
    acc_filepaths = [os.path.normpath(i) for i in glob.glob('D:\Documents\CS 198\Data Collection\Dataset\Segmented Videos/'+i+'/acc/*/*.jpg', 
                      recursive = True)]
    ak_filepaths = [os.path.normpath(i) for i in glob.glob('D:\Documents\CS 198\Data Collection\Dataset\Segmented Videos/'+i+'*/ak/*/*.jpg', 
                    recursive = True)]
    bo_filepaths = [os.path.normpath(i) for i in glob.glob('D:\Documents\CS 198\Data Collection\Dataset\Segmented Videos/'+i+'*/bo/*/*.jpg', 
                      recursive = True)]
    hs_filepaths = [os.path.normpath(i) for i in glob.glob('D:\Documents\CS 198\Data Collection\Dataset\Segmented Videos/'+i+'*/hs/*/*.jpg', 
                      recursive = True)]
    kvg_filepaths = [os.path.normpath(i) for i in glob.glob('D:\Documents\CS 198\Data Collection\Dataset\Segmented Videos/'+i+'*/kvg/*/*.jpg', 
                      recursive = True)]
    kvr_filepaths = [os.path.normpath(i) for i in glob.glob('D:\Documents\CS 198\Data Collection\Dataset\Segmented Videos/'+i+'*/kvr/*/*.jpg', 
                      recursive = True)]
    others_filepaths = [os.path.normpath(i) for i in glob.glob('D:\Documents\CS 198\Data Collection\Dataset\Segmented Videos/'+i+'*/*/*/Others/*.jpg', 
                      recursive = True)]

    total_per_indiv.append(len(acc_filepaths)+len(ak_filepaths)+len(bo_filepaths)+len(hs_filepaths)+len(kvg_filepaths)+len(kvr_filepaths)+len(others_filepaths))
    
    print(len(acc_filepaths), len(ak_filepaths), len(bo_filepaths), len(hs_filepaths), len(kvg_filepaths), len(kvr_filepaths), len(others_filepaths), len(acc_filepaths)+len(ak_filepaths)+len(bo_filepaths)+len(hs_filepaths)+len(kvg_filepaths)+len(kvr_filepaths)+len(others_filepaths))
  
  print(sum(total_per_indiv))

  return

if __name__ == "__main__":
  main()