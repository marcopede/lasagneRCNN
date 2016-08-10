# lasagneRCNN

Partial implementation of Fast R-CNN [1] in lasagne based on https://github.com/ddtm/theano-roi-pooling

You need to have installed recent versions of:
- Theano
- Lasagne
- Caffe with python
- numpy

You need:
vgg16_fast_rcnn_iter_40000.caffemodel >> a vgg16 Fast R-CNN pretrained on VOC 07 caffe model 
test.prototxt                         >> the corresponding prototxt file

To run the demo type:

>> pytohn RCNN.py

[1] Fast R-CNN: Fast Region-based Convolutional Networks for object detection
Created by Ross Girshick at Microsoft Research, Redmond
