# Standard-Squat-Posture-Classifier

This repository contains the source code and assets of the implementation for the paper, Vision-Based Classification of the Standard Squat Exercise Using MediaPipe Pose (Virtucio, 2022).

`/src/` contains the source code which include the following scripts:
1. `utils.py` - contains the utility functions.
2. `features.py` - contains the procedure of deriving the features from the preprocessed data, i.e., pose data extracted using MediaPipe Pose.
3. `classifier.py` - creates, trains and tests an SVM classifer to classify standard squat postures.
4. `test.py` - uses the saved SVM classifier and processed data for live webcam testing.

`/assets/` contains the assets for the implementation:
1. `clf.sav` - contains the SVM classifer.
2. `processed_data.sav` - contains the processed data, i.e, data with the derived features as its columns.
3. `/assets/extracted-data/` - contains the data extracted using MediaPipe Pose

Important Note: Python packages enlisted in `requirements.txt` must first be installed in your environment before you can execute the scripts.
