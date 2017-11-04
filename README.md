# Light Me Up

Modules needed:
* OpenCV 2.4.9
* Python 2.7
* Numpy

Dataset used: Extended Cohn-Kanade+ Dataset (available at: http://www.consortium.ri.cmu.edu/ckagree/)

## TL;DR Version

A program that consensually accesses your webcam, takes a picture, applies a Haar Cascade and resizes the cropped face after converting it to grayscale to a 200x200 px image.

Use fisher faces to train the model on the CK+ Dataset, and then run classifier.py after giving it rights to the webcam.

Wait for the verdict, and voila!

The arduino section is currently still being implemented.
