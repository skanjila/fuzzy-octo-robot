## From: https://automaticaddison.com/how-to-set-up-real-time-video-using-opencv-on-raspberry-pi-4/

# from picamera import PiCamera
# from time import sleep
# 
# camera = PiCamera()
# camera.start_preview()
# sleep(10)
# camera.stop_preview()

# Credit: Adrian Rosebrock
# https://www.pyimagesearch.com/2015/03/30/accessing-the-raspberry-pi-camera-with-opencv-and-python/
 
# import the necessary packages
from picamera.array import PiRGBArray # Generates a 3D RGB array
from picamera import PiCamera # Provides a Python interface for the RPi Camera Module
import time # Provides time-related functions
import cv2 # OpenCV library
import numpy as np

# Initialize the camera
camera = PiCamera()

backSub = cv2.createBackgroundSubtractorMOG2()
 
# Set the camera resolution
camera.resolution = (640, 480)
 
# Set the number of frames per second
camera.framerate = 32
 
# Generates a 3D RGB array and stores it in rawCapture
bg_capture = PiRGBArray(camera, size=(640, 480))
# Wait a certain number of seconds to allow the camera time to warmup
time.sleep(0.1)


# cap = cv2.VideoCapture(raw_capture)
# # take first frame of the video
# ret,frame = cap.read()

# camera.capture(bg_capture, format="bgr", use_video_port=True)
camera.capture(background.png, use_video_port=True)

# # setup initial location of window
# x, y, w, h = 300, 200, 100, 50 # simply hardcoded the values
# track_window = (x, y, w, h)
# # set up the ROI for tracking
# roi = raw_capture.array[y:y+h, x:x+w]
# hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
# mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
# roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
# cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
 


cv2.imshow("bg", bg_capture.array)

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
 
# aha: For some reason, after we do the original capture, I can't use the raw_capture array in the loop.
## Something about the wrong buffer length for a given width/height of the image
raw_capture = PiRGBArray(camera, size=(640, 480))

max_lowThreshold = 100
window_name = 'Edge Map'
title_trackbar = 'Min Threshold:'
ratio = 3
kernel_size = 3
low_threshold = 2




# Capture frames continuously from the camera
for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
#     fgMask = backSub.apply(frame.array)
    fgMask = cv2.subtract(frame.array, bg_capture.array)
    
    
#     src_gray = cv2.cvtColor(frame.array, cv2.COLOR_BGR2GRAY)
#     img_blur = cv2.blur(src_gray, (3,3))
#     detected_edges = cv2.Canny(img_blur, low_threshold, low_threshold*ratio, kernel_size)
#     mask = detected_edges != 0
#     dst = frame.array * (mask[:,:,None])
#     cv2.imshow("edges", dst)
    
    
#     hsv = cv2.cvtColor(frame.array, cv2.COLOR_BGR2HSV)
#     dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
#     # apply camshift to get the new location
#     ret, track_window = cv2.CamShift(dst, track_window, term_crit)
    
    # Grab the raw NumPy array representing the image
    image = frame.array
    
#     # Draw it on image
#     pts = cv2.boxPoints(ret)
#     pts = np.int0(pts)
#     img2 = cv2.polylines(image,[pts],True, 255,2)
    # cv2.imshow('img2',img2)
    
    # Display the frame using OpenCV
    cv2.imshow("Frame", image)
    cv2.imshow("Mask", fgMask)
     
    # Wait for keyPress for 1 millisecond
    key = cv2.waitKey(1) & 0xFF
     
    # Clear the stream in preparation for the next frame
    raw_capture.truncate(0)
     
    # If the `q` key was pressed, break from the loop
    if key == ord("q"):
        break