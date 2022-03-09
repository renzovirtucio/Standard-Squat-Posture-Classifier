import pandas as pd
import glob
import os
from IPython.display import display

import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
import matplotlib.pyplot as plt
import preprocessing

CUSTOM_POSE_CONNECTIONS = frozenset([(11, 12), (11, 23), (12, 24), (23, 24), (23, 25),
                              (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
                              (29, 31), (30, 32), (27, 31), (28, 32)])

def initial_svm_model():
  ## Preprocess data
  df = preprocessing.preprocess_data([os.path.normpath(i) for i in 
          glob.glob('D:\Documents\CS 198\Data Collection\Dataset\Extracted Data/**/*.csv', 
          recursive = True)])
  # print(df.describe().transpose())
  print(df.shape)
  print(df['target'].value_counts())

  ## Create SVM classifier
  y = df['target']
  X = df.drop(['target'], axis=1)

  # Split the data for testing and training
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=0)

  # Scale data
  scaler = StandardScaler()
  scaler.fit(X_train)
  X_train = scaler.transform(X_train)
  # scaled_frame = pd.DataFrame(X_train,columns=X.columns)
  # print(scaled_frame.describe().transpose())
  X_test = scaler.transform(X_test)

  # scaler = make_pipeline(StandardScaler(), PCA(n_components=40))
  # X_train = scaler.fit_transform(X_train)
  # X_test = scaler.transform(X_test)

  # Fitting a support vector machine
  model = SVC(random_state=0)
  print("SVC parameters:", model.get_params())
  model.fit(X_train, y_train)

  # Compute model predictions
  model_predictions = model.predict(X_test)

  # Compute accuracy score
  orig_acc_score_SVM = accuracy_score(y_test, model_predictions)
  print("Accuracy score:", orig_acc_score_SVM)

  # Plot confusion matrix
  cm = confusion_matrix(y_test, model_predictions, labels=model.classes_)
  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
  disp.plot()  
  print("Figure 3. Confusion Matrix of SVM Model")
  plt.show() 

  # Print classification report
  print("Table 6. Classification Report of SVM Model")
  print(classification_report(y_test, model_predictions))

  return (model, scaler)

def main():
  initial_svm_model()
  return

if __name__ == "__main__":
  main()