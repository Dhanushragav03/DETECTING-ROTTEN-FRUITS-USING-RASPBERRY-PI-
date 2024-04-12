import RPi.GPIO as GPIO
import time
from time import sleep
import os

# Set up GPIO pins
motor_pin1 = 21
motor_pin2 = 20
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(motor_pin1, GPIO.OUT)
GPIO.setup(motor_pin2, GPIO.OUT)

# Load the trained model and the VGG16 model
from tensorflow.keras.utils import load_img, img_to_array 
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model

model = load_model('/home/bit/Downloads/image_classification_model')
model_vgg = VGG16(weights='imagenet', include_top=False)

# Capture video from camera
import cv2
cap = cv2.VideoCapture(0)

# Start capturing frames
start_time = time.time()
while True:
    ret, frame = cap.read()
    if ret:
        # Display the frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Press Esc to exit
            break
        # Capture a photo every 2 seconds
        current_time = time.time()
        if current_time - start_time >= 2:
            cv2.imwrite('/home/bit/last/photo.jpg', frame)
            start_time = current_time
            
            # Preprocess the image and make a prediction
            ip_img = "/home/bit/last/photo.jpg"
            image_array = load_img(ip_img,target_size=(224,224))
            image_array = img_to_array(image_array)
            test = preprocess_input(image_array)
            test = np.expand_dims(test,axis=0)
            test_predict = model_vgg.predict(test)
            test_predict = test_predict.reshape(test_predict.shape[0],25088)
            pred = model.predict(test_predict)
            pred = np.argmax(pred)
            
            # Control the motor based on the prediction
            if pred == 0:
                print("The Fruit is rotten")
               
                GPIO.output(motor_pin1, GPIO.HIGH)
                GPIO.output(motor_pin2, GPIO.LOW)
            elif pred == 1:
                print("The Fruit is fresh")
                GPIO.output(motor_pin1, GPIO.LOW)
            else:
                print("The Fruit not found")
                GPIO.output(motor_pin1, GPIO.LOW)

            # Delete the image file
            os.remove(ip_img)

    else:
        break

cap.release()
cv2.destroyAllWindows()
