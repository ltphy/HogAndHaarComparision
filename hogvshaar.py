from PIL import Image
import face_recognition
import os
import numpy
import time
import cv2

count = 1
path = "test"
#size = 1200,1200
sum = 0
result = "result"
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
for file in os.listdir(path):
    full_file_path = os.path.join(path, file)
    #image = face_recognition.load_image_file(full_file_path)
    image = cv2.imread(full_file_path)

    face_locations_hog = face_recognition.face_locations(image)
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    rects = detector.detectMultiScale(image, scaleFactor=1.3, minNeighbors=5)
    face_locations_haar =  [(y, x + w, y + h, x) for (x, y, w, h) in rects]
    
    newpath = os.path.join("result_crop_gray",file)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image, "hog", (5,25), font, 1.0, (255, 0, 0), 2) 
    cv2.putText(image, "haar", (5,50), font, 1.0, (0, 255, 0), 2) 
    for top,right,bottom,left in face_locations_hog:

      # Print the location of each face in this image
      cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2)
    for top,right,bottom,left in face_locations_haar:
      # Print the location of each face in this image
      cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

    name = str(count)+".jpg"
    image_path = os.path.join(result,name)
    cv2.imwrite(image_path,image)
    cv2.imshow('image', image)
    count = count + 1
