from cv2 import split
import pandas as pd
import glob
import os
from IPython.display import display
import joblib

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
import matplotlib.pyplot as plt
import preprocessing

CUSTOM_POSE_CONNECTIONS = frozenset([(11, 12), (11, 23), (12, 24), (23, 24), (23, 25),
                              (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
                              (29, 31), (30, 32), (27, 31), (28, 32)])

def split_data():
  ## Preprocess data
  df_train = preprocessing.preprocess_data([os.path.normpath(i) for i in 
                        glob.glob('D:\Documents\CS 198\Data Collection\Dataset/New Extracted Data/**/*-train.csv', 
                        recursive = True)])
  print("(rows, columns) =", df_train.shape)
  print(df_train['target'].value_counts())

  df_test = preprocessing.preprocess_data([os.path.normpath(i) for i in 
                        glob.glob('D:\Documents\CS 198\Data Collection\Dataset/New Extracted Data/**/*-test.csv', 
                        recursive = True)])
  print("(rows, columns) =", df_test.shape)
  print(df_test['target'].value_counts())

  df = pd.concat([df_train, df_test], ignore_index=True)

  # Pop target column from dataframe
  y = df['target']
  X = df.drop(['target'], axis=1)

  # Split the data for testing and training
  X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=df_test.shape[0], random_state=0, shuffle=False)

  # display(X_test)

  return X_train, X_test, y_train, y_test

def show_performance(clf, X_test, y_test):
  # Predict using classifier
  clf_predictions = clf.predict(X_test)

  # Compute accuracy score
  orig_acc_score_SVM = accuracy_score(y_test, clf_predictions)
  print("Accuracy score:", orig_acc_score_SVM)

  # Show confusion matrix
  cm = ConfusionMatrixDisplay.from_predictions(y_test, clf_predictions, display_labels=clf.classes_, 
          cmap=plt.cm.Blues, normalize='true')
  print("Confusion Matrix of SVM Model")
  print(cm.confusion_matrix)
  plt.show() 

  # Show classification report
  print("Classification Report of SVM Model")
  print(classification_report(y_test, clf_predictions))

  return

def svm_clf(X_train, y_train):
  # Fit a support vector machine
  clf = make_pipeline(StandardScaler(), LogisticRegression(random_state=0, max_iter=1000, solver='liblinear'))
  print(clf['logisticregression'].get_params())
  clf.fit(X_train, y_train)

  return clf

def main():
  # Split data and save to file
  joblib.dump(split_data(), '../assets/preprocessed_data.sav')

  # Load data
  X_train, X_test, y_train, y_test = joblib.load('../assets/preprocessed_data.sav')

  # Save classifier to file
  joblib.dump(svm_clf(X_train, y_train), '../assets/clf.sav')

  # Load classifier and display its performance
  clf = joblib.load('../assets/clf.sav')
  show_performance(clf, X_test, y_test)

  return

if __name__ == "__main__":
  main()