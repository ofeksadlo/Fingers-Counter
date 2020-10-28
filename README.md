# Fingers-Counter
Simple fingers counter based on opencv

![showcase](showcase.gif)
</br>
</br>
</br>


# What it does
You place your fist in the camera. And then stretch your fingers</br> and it counts them down.

# How it works
The program detects your fist using Convolutional Neural Network (yolov3). And then create an ROI (Range Of Interest)</br>
in this ROI we detect angles between contours that under 100 degrees.</br>
For every angle we found we add up 1 finger (Starting from 2). </br>

# Disclaimer
This method is not very good because we cant count 1 finger because there's no angles to detect.
A better method would be creating a polygon in the center of the ROI.</br>
And counting contours that outside of the polygon.
