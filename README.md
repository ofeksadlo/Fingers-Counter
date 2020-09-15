# Fingers-Counter
Simple fingers counter based on opencv


Fist weights: https://gofile.io/d/KeDSLg

# What it does
You place your fist in the camera. And then stretch your fingers</br> and it counts them down.

# How it works
The program detects your fist using Convolutional Neural Network (yolov3). And then create an ROI (Range Of Interest)</br>
in this ROI we detect angles between contours that under 100 degrees.</br>
For every angle we found we add up 1 finger (Starting from 2). </br>

# Disclaimer
This method is not very good because we cant count 1 finger because there no angles to detect.
A better method would be creating a pollygon in the center of the ROI.</br>
And counting contours that outside of the pollygon.
