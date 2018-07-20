# created by lampson.song @ 2018-07-14
# generate tiny-ssd prototxt

import sys, os
sys.path.insert(0,'/home/lampson/1T_disk/workspace/objectDetection/ssd-caffe/python') # your caffe path
import caffe

def fire(net, net_bottom, prefix, out1, out2, out3):
    if(prefix[0:6] == "fire10" or prefix[0:6] == "fire11"):
        net.tops[prefix+'/squeeze1x1'] = caffe.layers.Convolution(net_bottom, num_output=out1, 
                kernel_size=1, weight_filler={"type":"xavier"})
        net.tops[prefix+'/squeeze1x1/bn'] = caffe.layers.BatchNorm(net.tops[prefix+'/squeeze1x1'], param=[dict(lr_mult=0, decay_mult=0), 
            dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)], in_place=True)
        net.tops[prefix+'/squeeze1x1/scale'] = caffe.layers.Scale(net.tops[prefix+'/squeeze1x1/bn'], param=[dict(lr_mult=1.0, decay_mult=0.0),
            dict(lr_mult=2.0, decay_mult=0.0)], scale_param={'filler':{'value':1}, 'bias_term':True, 'bias_filler':{'value':0}}, in_place=True)
    else:
        net.tops[prefix+'/squeeze1x1'] = caffe.layers.Convolution(net_bottom, num_output=out1, 
                kernel_size=1, weight_filler={"type":"xavier"}, param=[dict(lr_mult=1.0, decay_mult=0.0), dict(lr_mult=1.0, decay_mult=0.0)])
   
    if(prefix[0:6] == "fire10" or prefix[0:6] == "fire11"):
        net.tops[prefix+'/expand1x1'] = caffe.layers.Convolution(net.tops[prefix+'/squeeze1x1'], num_output=out2, 
                kernel_size=1, weight_filler={"type":"xavier"})
    else:
        net.tops[prefix+'/expand1x1'] = caffe.layers.Convolution(net.tops[prefix+'/squeeze1x1'], num_output=out2, 
                kernel_size=1, weight_filler={"type":"xavier"}, param=[dict(lr_mult=1.0, decay_mult=0.0), dict(lr_mult=1.0, decay_mult=0.0)])
   
    next_top = net.tops[prefix+'/expand1x1']
    if(prefix[0:5] == "fire9" or prefix[0:6] == "fire10" or prefix[0:6] == "fire11"):
        net.tops[prefix+'/expand1x1/bn'] = caffe.layers.BatchNorm(next_top, param=[dict(lr_mult=0, decay_mult=0), 
            dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)], in_place=True)
        net.tops[prefix+'/expand1x1/scale'] = caffe.layers.Scale(net.tops[prefix+'/expand1x1/bn'], param=[dict(lr_mult=1.0, decay_mult=0.0),
            dict(lr_mult=2.0, decay_mult=0.0)], scale_param={'filler':{'value':1}, 'bias_term':True, 'bias_filler':{'value':0}}, in_place=True)
        
        next_top = net.tops[prefix+'/expand1x1/scale']
    
    net.tops[prefix+'/relu_expand1x1'] = caffe.layers.ReLU(next_top, in_place=True)

    if(prefix[0:6] == "fire10" or prefix[0:6] == "fire11"):
        net.tops[prefix+'/expand3x3'] = caffe.layers.Convolution(net.tops[prefix+'/squeeze1x1'], num_output=out3, pad=1,
                kernel_size=3, weight_filler={'type':'xavier'})
    else:
        net.tops[prefix+'/expand3x3'] = caffe.layers.Convolution(net.tops[prefix+'/squeeze1x1'], num_output=out3, pad=1,
                kernel_size=3, weight_filler={'type':'xavier'}, param=[dict(lr_mult=1.0, decay_mult=0.0), dict(lr_mult=1.0, decay_mult=0.0)])

    next_top = net.tops[prefix+'/expand3x3']
    if(prefix[0:5] == "fire9" or prefix[0:6] == "fire10" or prefix[0:6] == "fire11"):
        net.tops[prefix+'/expand3x3/bn'] = caffe.layers.BatchNorm(next_top, param=[dict(lr_mult=0, decay_mult=0), 
            dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)], in_place=True)
        net.tops[prefix+'/expand3x3/scale'] = caffe.layers.Scale(net.tops[prefix+'/expand3x3/bn'], param=[dict(lr_mult=1.0, decay_mult=0.0),
            dict(lr_mult=2.0, decay_mult=0.0)], scale_param={'filler':{'value':1}, 'bias_term':True, 'bias_filler':{'value':0}}, in_place=True)
        
        next_top = net.tops[prefix+'/expand3x3/scale']
    
    net.tops[prefix+'/relu_expand3x3'] = caffe.layers.ReLU(next_top, in_place=True)
    net.tops[prefix+'/concat'] = caffe.layers.Concat(net.tops[prefix+'/expand1x1'], net.tops[prefix+'/expand3x3'])

    return net.tops[prefix+'/concat'] 


def prior_box(net, net_bottom, prefix, out1, out2, min_size, max_size, aspect_ratio, step):
    # prior box
    if(prefix[0:5] == "fire5"):
        net.tops[prefix+'/normal'] = caffe.layers.BatchNorm(net_bottom, name=prefix+'/bn', param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
        net.tops[prefix+'/scale'] = caffe.layers.Scale(net.tops[prefix+'/normal'], name=prefix+'/scale', param=[dict(lr_mult=1.0, decay_mult=0.0), dict(lr_mult=2.0, decay_mult=0.0)],
                scale_param={'filler':{'value':1}, 'bias_term':True, 'bias_filler':{'value':0}}, in_place=True)
        net_bottom = net.tops[prefix+'/scale']
    net.tops[prefix+'_mbox_loc'] = caffe.layers.Convolution(net_bottom, param=[dict(lr_mult=1.0, decay_mult=1.0), dict(lr_mult=2.0, decay_mult=0.0)],
            convolution_param={'num_output':out1, 'pad':1, 'kernel_size':3, 'stride':1, 'weight_filler':{'type':'msra'},'bias_filler':{'type':'constant', 'value':0.0}})
    net.tops[prefix+'_mbox_loc_perm'] = caffe.layers.Permute(net.tops[prefix+'_mbox_loc'], permute_param={'order':[0,2,3,1]})
    net.tops[prefix+'_mbox_loc_flat'] = caffe.layers.Flatten(net.tops[prefix+'_mbox_loc_perm'], flatten_param={'axis':1})
    
    net.tops[prefix+'_mbox_conf'] = caffe.layers.Convolution(net_bottom, param=[dict(lr_mult=1.0, decay_mult=1.0), dict(lr_mult=2.0, decay_mult=0.0)],
            convolution_param={'num_output':out2, 'pad':1, 'kernel_size':3, 'stride':1, 'weight_filler':{'type':'msra'}, 'bias_filler':{'type':'constant', 'value':0.0}})
    net.tops[prefix+'_mbox_conf_perm'] = caffe.layers.Permute(net.tops[prefix+'_mbox_conf'], permute_param={'order':[0,2,3,1]})
    net.tops[prefix+'_mbox_conf_flat'] = caffe.layers.Flatten(net.tops[prefix+'_mbox_conf_perm'], flatten_param={'axis':1})

    net.tops[prefix+'_mbox_priorbox'] = caffe.layers.PriorBox(net_bottom, net.data, prior_box_param={'min_size':min_size, 'max_size':max_size, 'aspect_ratio':aspect_ratio, 'flip':True,
        'clip':False, 'variance':[0.1, 0.1, 0.2, 0.2], 'step':step})

def generate_net(lmdb, label_file, PHASE, batch_size):
    net = caffe.NetSpec()

    if(PHASE=="TRAIN"):
        # data layer
        net.data, net.label = caffe.layers.AnnotatedData(ntop=2, include={'phase':caffe.TRAIN}, 
                transform_param=dict(mirror=True, mean_value=[104, 117, 123],
                    resize_param=dict(prob=1.0, resize_mode=caffe.params.Resize.WARP, height=300, width=300, 
                    interp_mode=[caffe.params.Resize.LINEAR,caffe.params.Resize.AREA,caffe.params.Resize.NEAREST,caffe.params.Resize.CUBIC,caffe.params.Resize.LANCZOS4]),
                    emit_constraint=dict(emit_type=0), 
                    distort_param=dict(brightness_prob=0.5, brightness_delta=32.0,
                            contrast_prob=0.5, contrast_lower=0.5, contrast_upper=1.5, hue_prob=0.5, hue_delta=18.0,
                            saturation_prob=0.5, saturation_lower=0.5, saturation_upper=1.5, random_order_prob=0.0),
                    expand_param=dict(prob=0.5, max_expand_ratio=4.0)),

                data_param=dict(source=lmdb, batch_size=batch_size, backend=caffe.params.Data.LMDB), 
               
                annotated_data_param=dict(
                batch_sampler=[dict(max_sample=1, max_trials=1),
                dict(sampler=dict(min_scale=0.3, max_scale=1.0,min_aspect_ratio=0.5, max_aspect_ratio=2.0), 
                    sample_constraint=dict(min_jaccard_overlap=0.1), max_sample=1, max_trials=50),
                dict(sampler=dict(min_scale=0.3, max_scale=1.0,min_aspect_ratio=0.5, max_aspect_ratio=2.0), 
                    sample_constraint=dict(min_jaccard_overlap=0.3), max_sample=1, max_trials=50),
                dict(sampler=dict(min_scale=0.3, max_scale=1.0,min_aspect_ratio=0.5, max_aspect_ratio=2.0), 
                    sample_constraint=dict(min_jaccard_overlap=0.5), max_sample=1, max_trials=50),
                dict(sampler=dict(min_scale=0.3, max_scale=1.0,min_aspect_ratio=0.5, max_aspect_ratio=2.0), 
                    sample_constraint=dict(min_jaccard_overlap=0.7), max_sample=1, max_trials=50),
                dict(sampler=dict(min_scale=0.3, max_scale=1.0,min_aspect_ratio=0.5, max_aspect_ratio=2.0), 
                    sample_constraint=dict(min_jaccard_overlap=0.9), max_sample=1, max_trials=50),
                dict(sampler=dict(min_scale=0.3, max_scale=1.0,min_aspect_ratio=0.5, max_aspect_ratio=2.0), 
                    sample_constraint=dict(min_jaccard_overlap=1.0), max_sample=1, max_trials=50)],
                label_map_file=label_file))
    elif(PHASE=="DEPLOY"):
        net.data = caffe.layers.Input(shape={'dim':[1,3,300,300]})
   
    # bone
    net.conv1 = caffe.layers.Convolution(net.data, num_output=57, kernel_size=3, stride=2, weight_filler={"type":"xavier"},
            param=[dict(lr_mult=1.0,decay_mult=0.0),dict(lr_mult=1.0,decay_mult=0.0)])
    net.relu_conv1 = caffe.layers.ReLU(net.conv1,in_place=True)
    net.pool1 = caffe.layers.Pooling(net.relu_conv1, pool=caffe.params.Pooling.MAX, kernel_size=3, stride=2)
   
    # fire1
    net.tops['fire2/concat'] = fire(net, net.pool1, 'fire2', 15,49, 53)
    # fire2
    net.tops['fire3/concat'] = fire(net, net.tops['fire2/concat'], 'fire3', 15, 54, 52)
    net.pool3 = caffe.layers.Pooling(net.tops['fire3/concat'], pool=caffe.params.Pooling.MAX, kernel_size=3, stride=2)
    # fire3
    net.tops['fire4/concat'] = fire(net, net.pool3, 'fire4', 29, 92, 94)
    # fire4
    net.tops['fire5/concat'] = fire(net, net.tops['fire4/concat'], 'fire5', 29, 90, 83)
    net.pool5 = caffe.layers.Pooling(net.tops['fire5/concat'], pool=caffe.params.Pooling.MAX, kernel_size=3, stride=2)
    # fire5
    net.tops['fire6/concat'] = fire(net, net.pool5, 'fire6', 44, 166, 161)
    # fire6
    net.tops['fire7/concat'] = fire(net, net.tops['fire6/concat'], 'fire7', 45, 155, 146)
    # fire7
    net.tops['fire8/concat'] = fire(net, net.tops['fire7/concat'], 'fire8', 49, 163, 171)
    # fire8
    net.tops['fire9/concat'] = fire(net, net.tops['fire8/concat'], 'fire9', 25, 29, 54)
    net.pool9 = caffe.layers.Pooling(net.tops['fire9/concat'], pool=caffe.params.Pooling.MAX, kernel_size=3, stride=2)
    # fire9
    net.tops['fire10/concat'] = fire(net, net.pool9, 'fire10', 37, 45, 56)
    net.pool10 = caffe.layers.Pooling(net.tops['fire10/concat'], pool=caffe.params.Pooling.MAX, kernel_size=3, stride=2)
    # fire10
    net.tops['fire11/concat'] = fire(net, net.pool10, 'fire11', 38, 41, 44)

    # conv12
    net.conv12_1 = caffe.layers.Convolution(net.tops['fire11/concat'], param=[dict(lr_mult=1.0, decay_mult=1.0)],
            convolution_param={'num_output':51, 'bias_term':False, 'kernel_size':1, 'weight_filler':{'type':'msra'}})
    net.tops['conv12_1/bn'] = caffe.layers.BatchNorm(net.conv12_1, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)], in_place=True)
    net.tops['conv12_1/scale'] = caffe.layers.Scale(net.tops['conv12_1/bn'], param=[dict(lr_mult=1.0, decay_mult=0.0), dict(lr_mult=2.0, decay_mult=0.0)], 
            scale_param={'filler':{'value':1}, 'bias_term':True, 'bias_filler':{'value':0}}, in_place=True)
    net.tops['conv12_1/relu'] = caffe.layers.ReLU(net.tops['conv12_1/scale'], in_place=True)
    net.conv12_2 = caffe.layers.Convolution(net.tops['conv12_1/relu'], param=[dict(lr_mult=1.0, decay_mult=1.0)],
            convolution_param={'num_output':46, 'bias_term':False, 'pad':1, 'kernel_size':3, 'stride':2, 'weight_filler':{'type':'msra'}})
    net.tops['conv12_2/bn'] = caffe.layers.BatchNorm(net.conv12_2, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)], in_place=True)
    net.tops['conv12_2/scale'] = caffe.layers.Scale(net.tops['conv12_2/bn'], param=[dict(lr_mult=1.0, decay_mult=0.0), dict(lr_mult=2.0, decay_mult=0.0)], 
            scale_param={'filler':{'value':1}, 'bias_term':True, 'bias_filler':{'value':0}}, in_place=True)
    net.tops['conv12_2/relu'] = caffe.layers.ReLU(net.tops['conv12_2/scale'], in_place=True)

    # conv13
    net.conv13_1 = caffe.layers.Convolution(net.tops['conv12_2/relu'], param=[dict(lr_mult=1.0, decay_mult=1.0)],
            convolution_param={'num_output':55, 'bias_term':False, 'kernel_size':1, 'weight_filler':{'type':'msra'}})
    net.tops['conv13_1/bn'] = caffe.layers.BatchNorm(net.conv13_1, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)], in_place=True)
    net.tops['conv13_1/scale'] = caffe.layers.Scale(net.tops['conv13_1/bn'], param=[dict(lr_mult=1.0, decay_mult=0.0), dict(lr_mult=2.0, decay_mult=0.0)], 
            scale_param={'filler':{'value':1}, 'bias_term':True, 'bias_filler':{'value':0}}, in_place=True)
    net.tops['conv13_1/relu'] = caffe.layers.ReLU(net.conv13_1, in_place=True)
    net.conv13_2 = caffe.layers.Convolution(net.tops['conv13_1/relu'], param=[dict(lr_mult=1.0, decay_mult=1.0)],
            convolution_param={'num_output':85, 'bias_term':False, 'pad':1, 'kernel_size':3, 'stride':2, 'weight_filler':{'type':'msra'}})
    net.tops['conv13_2/bn'] = caffe.layers.BatchNorm(net.conv13_2, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)], in_place=True)
    net.tops['conv13_2/scale'] = caffe.layers.Scale(net.tops['conv13_2/bn'], param=[dict(lr_mult=1.0, decay_mult=0.0), dict(lr_mult=2.0, decay_mult=0.0)], 
            scale_param={'filler':{'value':1}, 'bias_term':True, 'bias_filler':{'value':0}}, in_place=True)
    net.tops['conv13_2/relu'] = caffe.layers.ReLU(net.tops['conv13_2/scale'], in_place=True)


    # fire 5 prior box
    prior_box(net, net.tops['fire5/concat'], 'fire5', 16, 84, 21.0, 45.0, [2.0], 8)
    # fire 9 prior box
    prior_box(net, net.tops['fire9/concat'], 'fire9', 24, 126, 45.0, 99.0, [2.0, 3.0], 16)
    # fire 10 prior box
    prior_box(net, net.tops['fire10/concat'], 'fire10', 24, 126, 99.0, 153.0, [2.0, 3.0], 32)
    # fire 11 prior box
    prior_box(net, net.tops['fire11/concat'], 'fire11', 24, 126, 153.0, 207.0, [2.0, 3.0], 64)
    # conv12_2 prior box
    prior_box(net, net.tops['conv12_2'], 'conv12_2', 24, 126, 207.0, 261.0, [2.0, 3.0], 100)
    # conv13_2 prior box
    prior_box(net, net.tops['conv13_2'], 'conv13_2', 16, 84, 261.0, 315.0, [2.0], 300)

    # last process
    net.tops['mbox_loc'] = caffe.layers.Concat(net.tops['fire5_mbox_loc_flat'], net.tops['fire9_mbox_loc_flat'], net.tops['fire10_mbox_loc_flat'], net.tops['fire11_mbox_loc_flat'],
            net.tops['conv12_2_mbox_loc_flat'], net.tops['conv13_2_mbox_loc_flat'], concat_param={'axis':1})
    net.tops['mbox_conf'] = caffe.layers.Concat(net.tops['fire5_mbox_conf_flat'], net.tops['fire9_mbox_conf_flat'], net.tops['fire10_mbox_conf_flat'], net.tops['fire11_mbox_conf_flat'],
            net.tops['conv12_2_mbox_conf_flat'], net.tops['conv13_2_mbox_conf_flat'], concat_param={'axis':1})
    net.tops['mbox_priorbox'] = caffe.layers.Concat(net.tops['fire5_mbox_priorbox'], net.tops['fire9_mbox_priorbox'], net.tops['fire10_mbox_priorbox'], net.tops['fire11_mbox_priorbox'],
            net.tops['conv12_2_mbox_priorbox'], net.tops['conv13_2_mbox_priorbox'], concat_param={'axis':2})
    if(PHASE=='TRAIN'):
        net.tops['mbox_loss'] = caffe.layers.MultiBoxLoss(net.tops['mbox_loc'], net.tops['mbox_conf'], net.tops['mbox_priorbox'], net.label, include={'phase':caffe.TRAIN},
                propagate_down=[True, True, False, False], loss_param={'normalization':caffe.params.Loss.VALID}, multibox_loss_param={'loc_loss_type':caffe.params.MultiBoxLoss.SMOOTH_L1, 
                    'conf_loss_type':caffe.params.MultiBoxLoss.SOFTMAX, 'loc_weight':1.0, 
                    'num_classes':21, 'share_location':True, 'match_type':caffe.params.MultiBoxLoss.PER_PREDICTION, 'overlap_threshold':0.5, 'use_prior_for_matching':True, 
                    'background_label_id':0, 'use_difficult_gt':True, 'neg_pos_ratio':3.0, 'neg_overlap':0.5, 
                    'code_type':caffe.params.PriorBox.CENTER_SIZE, 'ignore_cross_boundary_bbox':False, 'mining_type':caffe.params.MultiBoxLoss.MAX_NEGATIVE})
    elif(PHASE=='DEPLOY'):
        net.tops['mbox_conf_reshape'] = caffe.layers.Reshape(net.tops['mbox_conf'], reshape_param={'shape':{'dim':[0,-1,21]}})
        net.tops['mbox_conf_softmax'] = caffe.layers.Softmax(net.tops['mbox_conf_reshape'], softmax_param={'axis':2})
        net.tops['mbox_conf_flatten'] = caffe.layers.Flatten(net.tops['mbox_conf_softmax'], flatten_param={'axis':1})
        net.tops['detection_out'] = caffe.layers.DetectionOutput(net.tops['mbox_loc'], net.tops['mbox_conf_flatten'], net.tops['mbox_priorbox'], include={
            'phase':caffe.TEST}, detection_output_param={'num_classes':21, 'share_location':True, 'background_label_id':0, 
            'nms_param':{'nms_threshold':0.45, 'top_k':100}, 'code_type':caffe.params.PriorBox.CENTER_SIZE, 'keep_top_k':100, 'confidence_threshold':0.25})

    return str(net.to_proto())

def write_net(train_proto, train_lmdb, deploy_proto, label_file):
    with open(train_proto, 'w') as f:
        f.write(str(generate_net(train_lmdb, label_file, "TRAIN", batch_size=64)))
        f.close()
    with open(deploy_proto, 'w') as f:
        f.write(str(generate_net(train_lmdb, label_file, "DEPLOY", batch_size=64)))
        f.close()


if __name__ == '__main__':
    project_dir = "./" # add your directory absolute path here

    label_file = project_dir + "data/VOC0712/labelmap_voc.prototxt"
    train_lmdb = project_dir + "-- your lmdb path -- /trian.lmdb" # your lmdb path
    train_proto = project_dir + "python_script/train.prototxt"
    deploy_proto = project_dir + "python_script/deploy.prototxt"

    write_net(train_proto, train_lmdb, deploy_proto, label_file)
