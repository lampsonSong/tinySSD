# created by lampson.song @ 2018-07-14
# generate tiny-ssd prototxt

import sys, os
sys.path.insert(0,'/home/lampson/1T_disk/workspace/objectDetection/ssd-caffe/python')
import caffe

def fire(net, net_bottom, prefix, out1, out2, out3):
    net.tops[prefix+'/squeeze1x1'] = caffe.layers.Convolution(net_bottom, num_output=out1, 
            kernel_size=1, weight_filler={"type":"xavier"})
    net.tops[prefix+'/relu_squeeze1x1'] = caffe.layers.ReLU(net.tops[prefix+'/squeeze1x1'], in_place=True)
    
    net.tops[prefix+'/expand1x1'] = caffe.layers.Convolution(net.tops[prefix+'/relu_squeeze1x1'], num_output=out2, 
            kernel_size=1, weight_filler={"type":"xavier"})
    net.tops[prefix+'/relu_expand1x1'] = caffe.layers.ReLU(net.tops[prefix+'/expand1x1'], in_place=True)

    net.tops[prefix+'/expand3x3'] = caffe.layers.Convolution(net.tops[prefix+'/squeeze1x1'], num_output=out3, pad=1,
            kernel_size=3, weight_filler={'type':'xavier'})
    
    net.tops[prefix+'/relu_expand3x3'] = caffe.layers.ReLU(net.tops[prefix+'/expand3x3'], in_place=True)
    net.tops[prefix+'/concat'] = caffe.layers.Concat(net.tops[prefix+'/expand1x1'], net.tops[prefix+'/expand3x3'])

    return net.tops[prefix+'/concat'] 


def prior_box(net, net_bottom, prefix, out1, out2, min_size, max_size, aspect_ratio, step):
    # prior box
    if(prefix[0:4]=="fire"):
        net.tops[prefix+'_norm'] = caffe.layers.Normalize(net_bottom, norm_param={'across_spatial':False, 'scale_filler':{'type':'constant', 'value':20.0}, 
            'channel_shared':False})
        net_bottom = net.tops[prefix+'_norm']
        
    net.tops[prefix+'_mbox_loc'] = caffe.layers.Convolution(net_bottom, param=[dict(lr_mult=1.0, decay_mult=1.0), dict(lr_mult=2.0, decay_mult=0.0)],
            convolution_param={'num_output':out1, 'pad':1, 'kernel_size':3, 'stride':1, 'weight_filler':{'type':'xavier'},'bias_filler':{'type':'constant', 'value':0.0}})
    net.tops[prefix+'_mbox_loc_perm'] = caffe.layers.Permute(net.tops[prefix+'_mbox_loc'], permute_param={'order':[0,2,3,1]})
    net.tops[prefix+'_mbox_loc_flat'] = caffe.layers.Flatten(net.tops[prefix+'_mbox_loc_perm'], flatten_param={'axis':1})
    
    net.tops[prefix+'_mbox_conf'] = caffe.layers.Convolution(net_bottom, param=[dict(lr_mult=1.0, decay_mult=1.0), dict(lr_mult=2.0, decay_mult=0.0)],
            convolution_param={'num_output':out2, 'pad':1, 'kernel_size':3, 'stride':1, 'weight_filler':{'type':'xavier'}, 'bias_filler':{'type':'constant', 'value':0.0}})
    net.tops[prefix+'_mbox_conf_perm'] = caffe.layers.Permute(net.tops[prefix+'_mbox_conf'], permute_param={'order':[0,2,3,1]})
    net.tops[prefix+'_mbox_conf_flat'] = caffe.layers.Flatten(net.tops[prefix+'_mbox_conf_perm'], flatten_param={'axis':1})

    net.tops[prefix+'_mbox_priorbox'] = caffe.layers.PriorBox(net.data, net_bottom, prior_box_param={'min_size':min_size, 'max_size':max_size, 'aspect_ratio':aspect_ratio, 'flip':True,
        'clip':False, 'variance':[0.1, 0.1, 0.2, 0.2], 'step':step, 'offset':0.5})

def generate_net(lmdb, mean_file, label_file, PHASE, batch_size):
    net = caffe.NetSpec()

    # data layer
    net.data, net.label = caffe.layers.AnnotatedData(ntop=2, include={'phase':caffe.TRAIN}, 
            transform_param=dict(mirror=True, mean_value=[104.0, 117.0, 123.0],
                resize_param=dict(prob=1.0, resize_mode=1, height=300, width=300, interp_mode=[1,2,3,4,5]),
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
   
    # bone
    net.conv1 = caffe.layers.Convolution(net.data, num_output=57, kernel_size=3, stride=2, pad=0, weight_filler={"type":"xavier"},
            bias_filler={"type":"constant", "value":0.0}, param=[dict(lr_mult=1.0,decay_mult=1.0),dict(lr_mult=2.0,decay_mult=0.0)])
    net.relu1_1 = caffe.layers.ReLU(net.conv1,in_place=True)
    net.pool1 = caffe.layers.Pooling(net.relu1_1, pool=caffe.params.Pooling.MAX, kernel_size=3, stride=2)
   
    # fire1
    net.tops['fire1/concat'] = fire(net, net.pool1, 'fire1', 15,49, 53)
    # fire2
    net.tops['fire2/concat'] = fire(net, net.tops['fire1/concat'], 'fire2', 15, 54, 52)
    net.pool3 = caffe.layers.Pooling(net.tops['fire2/concat'], pool=caffe.params.Pooling.MAX, kernel_size=3, stride=2)
    # fire3
    net.tops['fire3/concat'] = fire(net, net.pool3, 'fire3', 29, 92, 94)
    # fire4
    net.tops['fire4/concat'] = fire(net, net.tops['fire3/concat'], 'fire4', 29, 90, 83)
    net.pool5 = caffe.layers.Pooling(net.tops['fire4/concat'], pool=caffe.params.Pooling.MAX, kernel_size=3, stride=2)
    # fire5
    net.tops['fire5/concat'] = fire(net, net.pool5, 'fire5', 44, 166, 83)
    # fire6
    net.tops['fire6/concat'] = fire(net, net.tops['fire5/concat'], 'fire6', 45, 155, 146)
    # fire7
    net.tops['fire7/concat'] = fire(net, net.tops['fire6/concat'], 'fire7', 49, 163, 171)
    # fire8
    net.tops['fire8/concat'] = fire(net, net.tops['fire7/concat'], 'fire8', 25, 29, 54)
    net.pool9 = caffe.layers.Pooling(net.tops['fire8/concat'], pool=caffe.params.Pooling.MAX, kernel_size=3, stride=2)
    # fire9
    net.tops['fire9/concat'] = fire(net, net.pool9, 'fire9', 37, 45, 56)
    net.pool10 = caffe.layers.Pooling(net.tops['fire9/concat'], pool=caffe.params.Pooling.MAX, kernel_size=3, stride=2)
    # fire10
    net.tops['fire10/concat'] = fire(net, net.pool10, 'fire10', 38, 41, 44)

    # conv12
#    net.conv12_1 = caffe.layers.Convolution(net.tops['fire10/concat'], param=[dict(lr_mult=1.0, decay_mult=1.0), dict(lr_mult=2.0, decay_mult=0.0)],
#            num_output=51, pad=3, stride=2, kernel_size=3, weight_filler={'type':'xavier'}, bias_filler={'type':'constant', 'value':0.0})
    net.conv12_1 = caffe.layers.Convolution(net.tops['fire10/concat'], param=[dict(lr_mult=1.0, decay_mult=1.0), dict(lr_mult=2.0, decay_mult=0.0)],
            convolution_param={'num_output':51, 'pad':3, 'stride':2, 'kernel_size':3, 'weight_filler':{'type':'xavier'}, 'bias_filler':{'type':'constant', 'value':0.0}})
    net.relu122_1 = caffe.layers.ReLU(net.conv12_1, in_place=True)
    net.conv12_2 = caffe.layers.Convolution(net.relu122_1, param=[dict(lr_mult=1.0, decay_mult=1.0), dict(lr_mult=2.0, decay_mult=0.0)],
            convolution_param={'num_output':46, 'pad':0, 'kernel_size':3, 'weight_filler':{'type':'xavier'}, 'bias_filler':{'type':'constant', 'value':0.0}})
    net.relu12_2 = caffe.layers.ReLU(net.conv12_2, in_place=True)

    # conv13
    net.conv13_1 = caffe.layers.Convolution(net.relu12_2, param=[dict(lr_mult=1.0, decay_mult=1.0), dict(lr_mult=2.0, decay_mult=0.0)],
            convolution_param={'num_output':55, 'pad':1, 'kernel_size':3, 'weight_filler':{'type':'xavier'}, 'bias_filler':{'type':'constant', 'value':0.0}})
    net.relu13_1 = caffe.layers.ReLU(net.conv13_1, in_place=True)
    net.conv13_2 = caffe.layers.Convolution(net.relu13_1, param=[dict(lr_mult=1.0, decay_mult=1.0), dict(lr_mult=2.0, decay_mult=0.0)],
            convolution_param={'num_output':85, 'pad':1, 'kernel_size':3, 'stride':2, 'bias_filler':{'type':'constant', 'value':0.0}})
    

    # fire 4 prior box
    prior_box(net, net.tops['fire4/concat'], 'fire4', 16, 84, 30.0, 60.0, [2.0], 8.0)
    # fire 8 prior box
    prior_box(net, net.tops['fire8/concat'], 'fire8', 24, 126, 60.0, 111.0, [2.0, 3.0], 16.0)
    # fire 9 prior box
    prior_box(net, net.tops['fire9/concat'], 'fire9', 24, 126, 111.0, 162.0, [2.0, 3.0], 34.0)
    # fire 10 prior box
    prior_box(net, net.tops['fire10/concat'], 'fire10', 24, 126, 162.0, 213.0, [2.0, 3.0], 75.0)
    # conv12_2 prior box
    prior_box(net, net.tops['conv12_2'], 'conv12_2', 24, 126, 213.0, 264.0, [2.0, 3.0], 150.0)
    # conv13_2 prior box
    prior_box(net, net.tops['conv13_2'], 'conv13_2', 16, 84, 264.0, 315.0, [2.0], 300.0)


    # last process
    net.tops['mbox_loc'] = caffe.layers.Concat(net.tops['fire4_mbox_loc_flat'], net.tops['fire8_mbox_loc_flat'], net.tops['fire9_mbox_loc_flat'], net.tops['fire10_mbox_loc_flat'],
            net.tops['conv12_2_mbox_loc_flat'], net.tops['conv13_2_mbox_loc_flat'], concat_param={'axis':1})
    net.tops['mbox_conf'] = caffe.layers.Concat(net.tops['fire4_mbox_conf_flat'], net.tops['fire8_mbox_conf_flat'], net.tops['fire9_mbox_conf_flat'], net.tops['fire10_mbox_conf_flat'],
            net.tops['conv12_2_mbox_conf_flat'], net.tops['conv13_2_mbox_conf_flat'], concat_param={'axis':1})
    net.tops['mbox_priorbox'] = caffe.layers.Concat(net.tops['fire4_mbox_priorbox'], net.tops['fire8_mbox_priorbox'], net.tops['fire9_mbox_priorbox'], net.tops['fire10_mbox_priorbox'],
            net.tops['conv12_2_mbox_priorbox'], net.tops['conv13_2_mbox_priorbox'], concat_param={'axis':2})
    net.tops['mbox_loss'] = caffe.layers.MultiBoxLoss(net.tops['mbox_loc'], net.tops['mbox_conf'], net.tops['mbox_priorbox'], net.label, include={'phase':caffe.TRAIN},
            propagate_down=[True, True, False, False], loss_param={'normalization':1}, multibox_loss_param={'loc_loss_type':1, 'conf_loss_type':0, 'loc_weight':1.0, 
                'num_classes':21, 'share_location':True, 'match_type':1, 'overlap_threshold':0.5, 'use_prior_for_matching':True, 'background_label_id':0, 'use_difficult_gt':True,
                'neg_pos_ratio':3.0, 'neg_overlap':0.5, 'code_type':2, 'ignore_cross_boundary_bbox':False, 'mining_type':1})

    return str(net.to_proto())

def write_net(mean_file, train_proto, train_lmdb, test_proto, val_lmdb, label_file):
    with open(train_proto, 'w') as f:
        f.write(str(generate_net(train_lmdb, mean_file, label_file, "TRAIN", batch_size=64)))

if __name__ == '__main__':
    project_dir = "/home/lampson/1T_disk/workspace/objectDetection/tiny-ssd/"

    label_file = "data/VOC0712/labelmap_voc.prototxt"
    train_lmdb = project_dir + "models/tiny_ssd/python_script/trian.lmdb"
    val_lmdb = project_dir + "models/tiny_ssd/python_script/test_lmdb"
    train_proto = project_dir + "models/tiny_ssd/python_script/train.prototxt"
    test_proto = project_dir + "models/tiny_ssd/python_script/test.prototxt"
    solver_proto = project_dir + "models/tiny_ssd/python_script/solver.prototxt"
    mean_file = project_dir + "mean.binaryproto"

    write_net(mean_file, train_proto, train_lmdb, test_proto, val_lmdb, label_file)
