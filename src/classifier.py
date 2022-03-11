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
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import preprocessing

CUSTOM_POSE_CONNECTIONS = frozenset([(11, 12), (11, 23), (12, 24), (23, 24), (23, 25),
                              (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
                              (29, 31), (30, 32), (27, 31), (28, 32)])

def split_data():
  ## Preprocess data
  df = preprocessing.preprocess_data([os.path.normpath(i) for i in 
          glob.glob('D:\Documents\CS 198\Data Collection\Dataset\Extracted Data/**/*.csv', 
          recursive = True)])
  # print(df.describe().transpose())
  print("(rows, columns) =", df.shape)
  print(df['target'].value_counts())

  y = df['target']
  X = df.drop(['target'], axis=1)

  # Split the data for testing and training
  X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.40, random_state=0)

  return X_train, X_test, y_train, y_test

def show_performance(clf, X_test, y_test):
  # Compute classifier predictions
  clf_predictions = clf.predict(X_test)

  # Compute accuracy score
  orig_acc_score_SVM = accuracy_score(y_test, clf_predictions)
  print("Accuracy score:", orig_acc_score_SVM)

  # Plot confusion matrix
  cm = confusion_matrix(y_test, clf_predictions, labels=clf.classes_)
  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
  disp.plot()  
  print("Figure 3. Confusion Matrix of SVM Model")
  plt.show() 

  # Print classification report
  print("Table 6. Classification Report of SVM Model")
  print(classification_report(y_test, clf_predictions))
  return

def svm_clf(X_train, y_train):
  # Fitting a support vector machine
  clf = make_pipeline(StandardScaler(), SVC(random_state=0))
  # print(clf.get_params())
  clf.fit(X_train, y_train)

  return clf

def main():
  # Split data and train classifiers
  # joblib.dump(split_data(), 'preprocessed_data.sav')
  X_train, X_test, y_train, y_test = joblib.load('preprocessed_data.sav')

  # Save classifier to file
  # joblib.dump(svm_clf(X_train, y_train), 'svm_clf.sav')

  # Load classifier and display its performance
  clf = joblib.load('svm_clf.sav')
  show_performance(clf, X_test, y_test)
  return

if __name__ == "__main__":
  main()