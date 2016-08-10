
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import ReshapeLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from SPP import SpatialPyramidPoolingDNNLayer as SPPolling
from lasagne.nonlinearities import softmax
import lasagne

if 1:
    try:
        import subprocess
        gpu_id = subprocess.check_output('gpu_getIDs.sh', shell=True)
        os.environ["THEANO_FLAGS"]='device=gpu%s'%gpu_id
        print(os.environ["THEANO_FLAGS"])
    except:
        pass

import theano
import theano.tensor as T

# build RCNN model in lasagne
def build_vgg():
    net = {}
    net['input'] = InputLayer((None, 3, 224, 224))
    net['conv1_1'] = ConvLayer(
        net['input'], 64, 3, pad=1, flip_filters=False)
    net['conv1_2'] = ConvLayer(
        net['conv1_1'], 64, 3, pad=1, flip_filters=False)
    net['pool1'] = PoolLayer(net['conv1_2'], 2)
    net['conv2_1'] = ConvLayer(
        net['pool1'], 128, 3, pad=1, flip_filters=False)
    net['conv2_2'] = ConvLayer(
        net['conv2_1'], 128, 3, pad=1, flip_filters=False)
    net['pool2'] = PoolLayer(net['conv2_2'], 2)
    net['conv3_1'] = ConvLayer(
        net['pool2'], 256, 3, pad=1, flip_filters=False)
    net['conv3_2'] = ConvLayer(
        net['conv3_1'], 256, 3, pad=1, flip_filters=False)
    net['conv3_3'] = ConvLayer(
        net['conv3_2'], 256, 3, pad=1, flip_filters=False)
    net['pool3'] = PoolLayer(net['conv3_3'], 2)
    net['conv4_1'] = ConvLayer(
        net['pool3'], 512, 3, pad=1, flip_filters=False)
    net['conv4_2'] = ConvLayer(
        net['conv4_1'], 512, 3, pad=1, flip_filters=False)
    net['conv4_3'] = ConvLayer(
        net['conv4_2'], 512, 3, pad=1, flip_filters=False)
    net['pool4'] = PoolLayer(net['conv4_3'], 2)
    net['conv5_1'] = ConvLayer(
        net['pool4'], 512, 3, pad=1, flip_filters=False)
    net['conv5_2'] = ConvLayer(
        net['conv5_1'], 512, 3, pad=1, flip_filters=False)
    net['conv5_3'] = ConvLayer(
        net['conv5_2'], 512, 3, pad=1, flip_filters=False)
#classification branch
#    net['pool5'] = PoolLayer(net['conv5_3'], 2)
#    net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
#    net['fc6_dropout'] = DropoutLayer(net['fc6'], p=0.5)
#    net['fc7'] = DenseLayer(net['fc6_dropout'], num_units=4096)
#    net['fc7_dropout'] = DropoutLayer(net['fc7'], p=0.5)
#    net['fc8'] = DenseLayer(
#        net['fc7_dropout'], num_units=1000, nonlinearity=None)
#    net['prob'] = NonlinearityLayer(net['fc8'], softmax)
#detection branch
    net['boxes'] = InputLayer((None, 5))
    net['crop'] = SPPolling([net['conv5_3'],net['boxes']],pool_dims=7,sp_scale = 1./float(16))
    net['reshape'] = ReshapeLayer(net['crop'], ([0], -1))
    net['fc6'] = DenseLayer(net['reshape'], num_units=4096)
    net['fc6_dropout'] = DropoutLayer(net['fc6'], p=0.5)
    net['fc7'] = DenseLayer(net['fc6_dropout'], num_units=4096)
    net['fc7_dropout'] = DropoutLayer(net['fc7'], p=0.5)
    net['cls_score'] = DenseLayer(net['fc7_dropout'], num_units=21, nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['cls_score'], softmax)

    return net

def add_names(net_dict):
    for lname,l in net_dict.items():
        l.name=lname

def generate_param_names(net):
    sorted_layers=lasagne.layers.get_all_layers(net)
    param_names = []
    for l in sorted_layers:
        params = l.get_params()
        #print params
        for p in params:
            param_names.append(l.name+"_"+p.name)
    return param_names

def get_param_dict(net):
    #each layer should have a name
    names = generate_param_names(net)
    params = lasagne.layers.get_all_param_values(net)
    param_dict = {names[l]:params[l] for l in range(len(names))}
    return param_dict

# copy parameters from caffe trained model
import pickle
import numpy as np
model_name = 'VGG.lasagnemodel.pkl'
try:
    #force_recompute
    vgg = pickle.load(open(model_name))
except:

    #model_param_values = pickle.load(open('data/vgg16.pkl'))['param values']

    import sys
    #sys.path.append('/home/lear/mpederso/links/wsd/caffe-fast-rcnn/')
    sys.path.append('/home/lear/mpederso/links/wsd/caffe-fast-rcnn/python/')

    import caffe

    param_dir = '/home/lear/mpederso/links/wsd/data/fast_rcnn_models'
    param_fn = '%s/vgg16_fast_rcnn_iter_40000.caffemodel' % param_dir
    model_dir = '/home/lear/mpederso/links/wsd/models/VGG16'
    model_fn = '%s/test.prototxt' % model_dir
    #param_dir = '/home/lear/mpederso/links/wsd/data/imagenet_models'
    #param_fn = '%s/VGG16.v2.caffemodel'% param_dir
    #model_dir = '/home/lear/mpederso/links/wsd/data/imagenet_models'
    #model_fn = '%s/deploy.prototxt' % model_dir

    vgg = build_vgg()
    net = caffe.Net(model_fn, param_fn, caffe.TEST)
    for name, param in net.params.iteritems():
        if not(vgg.has_key(name)):
            print "SKIP:", name
        else:
            layer = vgg[name]#getattr(vgg, name)

            print name, param[0].data.shape, param[1].data.shape,
            print np.array(layer.W.eval()).shape , np.array(layer.b.eval()).shape
            #raw_input()
            if  type(layer)==lasagne.layers.dense.DenseLayer:
                layer.W.set_value(param[0].data.T)
            else:
                layer.W.set_value(param[0].data)#[:,:,::-1,::-1])
            layer.b.set_value(param[1].data)
            #setattr(vgg, name, layer)

    pickle.dump(vgg, open(model_name, 'wb'), -1)


#vgg = build_vgg() #only for test, later it should be removed
cnn_input_var = vgg['input']#.input_var
cnn_input_boxes = vgg['boxes']#.input_var
cnn_output_layer = vgg['prob']

x_boxes = T.matrix()
x_im = T.tensor4()

l_detect = lasagne.layers.get_output(cnn_output_layer,{cnn_input_var:x_im,cnn_input_boxes:x_boxes},deterministic=True)

detect = theano.function([x_im,x_boxes],l_detect)#, mode=theano.compile.mode.Mode(optimizer=None), on_unused_input='warn')#,exception_verbosity='high')#'fast_compile')))

if 1:

    l_conv = lasagne.layers.get_output(vgg['conv5_3'],{cnn_input_var:x_im},deterministic=True)
    conv = theano.function([x_im],l_conv)

    l_spp = lasagne.layers.get_output(vgg['crop'],{cnn_input_var:x_im,cnn_input_boxes:x_boxes},deterministic=True)
    spp = theano.function([x_im,x_boxes],l_spp)

    l_rscr = lasagne.layers.get_output(vgg['cls_score'],{cnn_input_var:x_im,cnn_input_boxes:x_boxes},deterministic=True)
    rscr = theano.function([x_im,x_boxes],l_rscr)

# run on an image
import pylab
import os
import scipy.io as sio
import cv2 as cv

#box_file = os.path.join('/home/lear/mpederso/links/wsd', 'data', 'demo','000004_boxes.mat')
box_file = '000004_boxes.mat'
obj_proposals = sio.loadmat(box_file)['boxes']

#im = cv.imread('/home/lear/mpederso/links/wsd/data/demo/000004.jpg')
im = cv.imread('000004.jpg')

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')
PIXEL_MEANS = np.array([102.9801, 115.9465, 122.7717], dtype=np.float32)

def img_preprocessing(orig_img, pixel_means, max_size=1000, scale=600):
    img = orig_img.astype(np.float32, copy=True)
    img -= pixel_means
    im_size_min = np.min(img.shape[0:2])
    im_size_max = np.max(img.shape[0:2])
    im_scale = float(scale) / float(im_size_min)
    if np.rint(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    img = cv.resize(img, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv.INTER_LINEAR)

    return img.transpose([2, 0, 1]).astype(np.float32), im_scale

_im,_scale = img_preprocessing(im, PIXEL_MEANS, max_size=1000, scale=600)

#x_im = im
#x_boxes = obj_proposals

obj_proposals = (obj_proposals * _scale).astype(np.float32)

boxes = np.concatenate((np.zeros((obj_proposals.shape[0],1),dtype=np.float32),obj_proposals),1)

DEDUP_BOXES = 0.0625
if DEDUP_BOXES > 0:
    v = np.array([1, 1e3, 1e6, 1e9, 1e12])
    hashes = np.round(boxes * DEDUP_BOXES).dot(v)
    _, index, inv_index = np.unique(hashes, return_index=True,
                                    return_inverse=True)
    #blobs['rois'] = blobs['rois'][index, :]
    boxes = boxes[index, :]

#boxes = boxes[::-1,:]

#boxes = boxes[:35]
import time
t = time.time()
l = 0
output = detect(_im[np.newaxis],boxes)

print "Detection Time,",time.time()-t

pylab.imshow(im)

cls = np.argmax(output,1)
scr = np.max(output,1)

thr = 0.9
for idb in range(output.shape[0]):
    if scr[idb]>thr and cls[idb]!=0:
        x0 = boxes[idb,1]/_scale
        y0 = boxes[idb,2]/_scale
        x1 = boxes[idb,3]/_scale
        y1 = boxes[idb,4]/_scale
        pylab.plot([x0,x0,x1,x1,x0],[y0,y1,y1,y0,y0],lw=3)
        pylab.text(x0,y0,'%s:%.3f'%(CLASSES[cls[idb]],scr[idb]))

pylab.show()

#evaluate on VOC07





