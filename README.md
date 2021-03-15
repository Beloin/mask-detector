# Simple Mask Detection
A simple mask Detector using Python:
 - OpenCV
 - Keras/Tensorflow
 - Sklearn

My main helper was [this article](https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/) and [Prajnasd's github](https://github.com/prajnasb/observations/tree/master/mask_classifier).

Another helper was the Face Recognition preprocessed [model](https://github.com/prajnasb/face_detector).

## To use with main.py:

- To Train the ANN:
```
python3 main.py --train --dataset dataset 
```

- To predict if using mask, or not, with image file:
```
python3 main.py --predict --image <path to image.jpg>
```