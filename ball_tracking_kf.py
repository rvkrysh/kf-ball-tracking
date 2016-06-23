# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 09:34:11 2016

@author: rkunni
"""

import cv2
import numpy as np
import pylab

noOfFrames = 60

im = cv2.imread('ball/1.jpg', 1)
m,n,p = im.shape

# Kalman filter parameters
dt = 1.0

# Process noise std deviation
sd_w = 0.5
# Measurement noise std deviation
sd_vx = 5.0   # measurement noise in the horizontal direction
sd_vy = 5.0   # measurement noise in the vertical direction

# Prediction matrix
F = np.array([[1.0, 0.0,  dt, 0.0],
              [0.0, 1.0, 0.0,  dt],
              [0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 1.0]])
# Control matrix
B = np.array([dt**2/2, dt**2/2, dt, dt])
# Process noise covariance
Q = np.eye(4)
# Measurement noise covariance
R = np.eye(2)
# Transformation matrix
H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])  #only position is measured
# Initial prediction covariance matrix
P = Q
# Acceleration magnitude
ux = 0.01    # acceleration in the horizontal direction
uy = 0.5    # acceleration in the vertical direction
u = np.array([[ux, 0, 0, 0], [0, uy, 0, 0], [0, 0, ux, 0], [0, 0, 0, uy]])

xhat = np.array([132.0, 1.0, 0.0, 0.0])
predStates = xhat

# Construct the background image from first few frames
bgFrame = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
for i in range(2, 5):
    bgFrame = bgFrame/2 + \
    cv2.cvtColor(cv2.imread('ball/'+str(i)+'.jpg'), cv2.COLOR_BGR2GRAY)/2

# Array to save the object locations
objLocs = np.array([None, None])

# Kernel for morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Display the frames like a video
for i in range(6, noOfFrames+1):
    # Read each frame
    frame = cv2.imread('ball/'+str(i)+'.jpg')

    # Perform background subtraction after median filter
    diffFrame = cv2.absdiff(cv2.cvtColor(cv2.medianBlur(frame, 7), \
                cv2.COLOR_BGR2GRAY), cv2.medianBlur(bgFrame, 7))

    # Otsu thresholding to create the binary image
    [th, bwFrame] = cv2.threshold(diffFrame, 0, 255, cv2.THRESH_OTSU)

    # Morphological opening operation to remove small blobs
    bwFrame = cv2.morphologyEx(bwFrame, cv2.MORPH_OPEN, kernel)

    # Find the contours in the binary image
    _, contours, hierarchy = cv2.findContours(bwFrame.copy(), cv2.RETR_TREE,\
                            cv2.CHAIN_APPROX_SIMPLE)
    
    # Calculate and log the contour areas
    contArea = np.array([cv2.contourArea(cont) for cont in contours])
    
    # Find the center and radius of the largest contour
    if contours != []:
        center, radius = cv2.minEnclosingCircle(contours[np.nonzero(contArea \
                            == max(contArea))[0][0]])
    else:
        center = np.array([None, None])

    # Save the current location of the object
    objLocs = np.c_[objLocs, np.array(center)]

    # Draw contours
    cv2.drawContours(frame, contours, -1, (255,20,20), 2)

    # State estimation using Kalman filter
    # Predict
    # Predicted (a priori) state estimate
    xhat = np.dot(F, xhat) + np.dot(B, u)
    # Predicted (a priori) estimate covariance matrix
    P = np.dot(F.dot(P), F.transpose()) + Q
    
    # Update
    z = np.array(center)
    # Innovation or measurement residual
    ytilde = z - np.dot(H, xhat)
    # Innovation (or residual) covariance
    S = np.dot(H.dot(P), H.transpose()) + R
    # Optimal Kalman gain
    if (S.ndim == 0):
        K = np.dot(P.dot(H.transpose()), 1.0/S)
    else:
        K = np.dot(P.dot(H.transpose()), np.linalg.inv(S))
    # Updated (a posteriori) state estimate
    xhat = xhat + np.dot(K, ytilde)
    # Updated (a posteriori) estimate covariance
    P = (np.eye(4) - np.dot(K, H)).dot(P)

    # Save the predicted states
    predStates = np.c_[predStates, xhat]

	# Draw circle around the object
    cv2.circle(frame, (int(round(xhat[0])), int(round(xhat[1]))), \
                int(round(radius)), (50,255,50), 2)

    # Display frames
    cv2.namedWindow('Tracking')
    cv2.imshow('Tracking', frame)
    cv2.waitKey(10) # wait for 16ms

cv2.waitKey(0)
cv2.destroyWindow('Tracking')

# Plotting
# Detected object locations
pylab.plot(objLocs[0, :], objLocs[1, :], 'o-', \
            label = 'Detected location')
# Predicted object locations
pylab.plot(predStates[0, :], predStates[1, :], 'o-', \
            label = 'Predicted location')
pylab.ylabel('Vertical position')
pylab.xlabel('Horizontal position')
pylab.title('Tracking using Kalman filter')
pylab.legend(loc = 'best')
pylab.show()
