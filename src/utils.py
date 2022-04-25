import cv2
import numpy as np

def rescale_frame(frame, percent=75):
  width = int(frame.shape[1] * percent/ 100)
  height = int(frame.shape[0] * percent/ 100)
  dim = (width, height)
  return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

def calculate_angle(a,b,c):
  a = np.array(a) # First
  b = np.array(b) # Mid
  c = np.array(c) # End
  
  radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
  angle = np.abs(radians*180.0/np.pi)
  
  if angle >180.0:
    angle = 360-angle
      
  return angle 

def calculate_distance(a, b):
  a = np.array(a) # Start
  b = np.array(b) # End

  dist = np.linalg.norm(a - b)
  
  return dist

def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)