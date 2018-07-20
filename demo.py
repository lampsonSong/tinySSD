# created by lampson.song @ 2018-07-18 
# a demo script of tinySSD

import numpy as np  
import sys,os  
import cv2
import time
caffe_root = '/home/lampson/1T_disk/workspace/objectDetection/ssd-caffe/'
sys.path.insert(0, caffe_root + 'python')  
import caffe  
from skimage import img_as_float

net_file= './python_script/deploy.prototxt'  
if not os.path.exists(net_file):
    print("prototxt does not exist, use generate.py.")
    exit()

caffe_model='./model/_iter_290000.caffemodel'  
if not os.path.exists(caffe_model):
    print("model does not exist.")
    exit()

CLASSES = ('background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')


def postprocess(img, out):
    h = img.shape[0]
    w = img.shape[1]
    box = out['detection_out'][0,0,:,3:7] * np.array([w, h, w, h])

    cls = out['detection_out'][0,0,:,1]
    conf = out['detection_out'][0,0,:,2]
    return (box.astype(np.int32), conf, cls)

def detect(imgfile):
    orig_img = cv2.imread(imgfile)

    img = orig_img[:,:,::-1]
    img = img_as_float(img).astype(np.float32)
    img = transformer.preprocess('data', img)

    net.blobs['data'].data[...] = img
    out = net.forward()

    box, conf, cls = postprocess(orig_img, out)

    for i in range(len(box)):
       p1 = (box[i][0], box[i][1])
       p2 = (box[i][2], box[i][3])
       p3 = (max(p1[0], 15), max(p1[1], 15))

       title = "%s:%.2f" % (CLASSES[int(cls[i])], conf[i])
       cv2.putText(orig_img, title, p3, cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)
       cv2.rectangle(orig_img, p1, p2, (0,255,0))

    cv2.imshow("tiny-SSD", orig_img)
    cv2.waitKey(0)


net = caffe.Net(net_file,caffe_model,caffe.TEST)  
net.blobs['data'].reshape(1,3,300,300)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1)) # H * W * C -> C * H * W
transformer.set_mean('data', np.array([104, 117, 123]))
transformer.set_raw_scale('data', 255) # [0, 1] -> [0, 255]
transformer.set_channel_swap('data', (2, 1, 0)) # RGB -> BGR

detect("image/slam.jpeg")
