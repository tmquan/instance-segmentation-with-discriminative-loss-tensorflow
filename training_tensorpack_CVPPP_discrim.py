# Hidden 2 domains no constrained
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys, argparse, glob, time, cv2

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

# Misc. libraries
from six.moves import map, zip, range
from natsort import natsorted 
from enet import *
from loss import *
#from Utility import *
# Array and image processing toolboxes
import numpy as np 
import skimage
import skimage.io
import skimage.transform
import skimage.segmentation
import malis

# For Augmentor
import PIL
from PIL import Image

# Augmentor
import Augmentor

# Tensorflow 
import tensorflow as tf

# Tensorpack toolbox
import tensorpack.tfutils.symbolic_functions as symbf
from tensorpack import *
from tensorpack.dataflow import dataset
from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.utils.utils import get_rng
from tensorpack.tfutils import optimizer, gradproc
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary, add_tensor_summary
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
from tensorpack.utils import logger
from tensorpack.models.common import layer_register, VariableHolder
from tensorpack.tfutils.common import get_tf_version_number
from tensorpack.utils.argtools import shape2d, shape4d, get_data_format
from tensorpack.models.tflayer import rename_get_variable, convert_to_tflayer_args


# Tensorlayer
from tensorlayer.cost import binary_cross_entropy, absolute_difference_error, dice_coe, cross_entropy

DIMZ=1
DIMY=256
DIMX=256
###############################################################################

def calc_dic(n_objects_gt, n_objects_pred):
    return np.abs(n_objects_gt - n_objects_pred)


def calc_dice(gt_seg, pred_seg):

    nom = 2 * np.sum(gt_seg * pred_seg)
    denom = np.sum(gt_seg) + np.sum(pred_seg)

    dice = float(nom) / float(denom)
    return dice


def calc_bd(ins_seg_gt, ins_seg_pred):

    gt_object_idxes = list(set(np.unique(ins_seg_gt)).difference([0]))
    pred_object_idxes = list(set(np.unique(ins_seg_pred)).difference([0]))

    best_dices = []
    for gt_idx in gt_object_idxes:
        _gt_seg = (ins_seg_gt == gt_idx).astype('bool')
        dices = []
        for pred_idx in pred_object_idxes:
            _pred_seg = (ins_seg_pred == pred_idx).astype('bool')

            dice = calc_dice(_gt_seg, _pred_seg)
            dices.append(dice)
        best_dice = np.max(dices)
        best_dices.append(best_dice)

    best_dice = np.mean(best_dices)

    return best_dice


def calc_sbd(ins_seg_gt, ins_seg_pred):

    _dice1 = calc_bd(ins_seg_gt, ins_seg_pred)
    _dice2 = calc_bd(ins_seg_pred, ins_seg_gt)
    return min(_dice1, _dice2)


import matplotlib.pyplot as plt
def get_colors(inp, colormap, vmin=None, vmax=None):
    norm = plt.Normalize(vmin, vmax)
    return colormap(norm(inp))

###############################################################################
def time_seed ():
    seed = None
    while seed == None:
        cur_time = time.time ()
        seed = int ((cur_time - int (cur_time)) * 1000000)
    return seed

class ImageDataFlow(RNGDataFlow):
    def __init__(self, 
        imageDir, 
        labelDir, 
        size, 
        dtype='float32', 
        isTrain=False, 
        isValid=False, 
        isTest=False, 
        pruneLabel=False, 
        shape=[1, 512, 512]):

        self.dtype      = dtype
        self.imageDir   = imageDir
        self.labelDir   = labelDir
        self._size      = size
        self.isTrain    = isTrain
        self.isValid    = isValid

        imageFiles = natsorted (glob.glob(self.imageDir + '/*.*'))
        labelFiles = natsorted (glob.glob(self.labelDir + '/*.*'))
        print(imageFiles)
        print(labelFiles)
        self.images = []
        self.labels = []
        self.data_seed = time_seed ()
        self.data_rand = np.random.RandomState(self.data_seed)
        self.rng = np.random.RandomState(999)
        for imageFile in imageFiles: #[:100]:
            #image = skimage.io.imread (imageFile)
            image = cv2.imread(imageFile)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #image = (image.astype(np.float32)-mean)/std

            self.images.append(image)
        for labelFile in labelFiles: #[:100]:
            label = cv2.imread(labelFile, cv2.IMREAD_COLOR)
            label = skimage.io.imread (labelFile)
            self.labels.append(label)
            
        self.DIMZ = shape[0]
        self.DIMY = shape[1]
        self.DIMX = shape[2]
        self.pruneLabel = pruneLabel

    def size(self):
        return self._size
###############################################################################

class TUImageDataFlow(ImageDataFlow):
    ###############################################################################
    def AugmentPair(self, src_image, src_label, pipeline, seed=None, verbose=False):
        np.random.seed(seed) if seed else np.random.seed(2015)
        # print(src_image.shape, src_label.shape, aug_image.shape, aug_label.shape) if verbose else ''
        if src_image.ndim==2:
            src_image = np.expand_dims(src_image, 2)
        if src_label.ndim==2:
            src_label = np.expand_dims(src_label, 2)
        
        # Create the result
        aug_images = [] #np.zeros_like(src_image)
        aug_labels = [] #np.zeros_like(src_label)
        #print(src_image.shape)
        #print(src_label.shape)
        # print(src_image.shape, src_label.shape)
        for z in range(src_image.shape[2]):
            #Image and numpy has different matrix order
            pipeline.set_seed(seed)
            aug_image = pipeline._execute_with_array(src_image[...,z]) 
            aug_images.append(aug_image)
        for z in range(src_label.shape[2]):
            pipeline.set_seed(seed)
            aug_label = pipeline._execute_with_array(src_label[...,z])        
            aug_labels.append(aug_label)

        aug_images = np.array(aug_images).astype(np.float32)
        aug_labels = np.array(aug_labels).astype(np.float32)

        aug_images = np.transpose(aug_images, [1, 2, 0])
        aug_labels = np.transpose(aug_labels, [1, 2, 0])
        # print(aug_images.shape, aug_labels.shape)

        return aug_images, aug_labels

  
    ###############################################################################
    def get_data(self):
        for k in range(self._size):
            #
            # Pick randomly a tuple of training instance
            #
            rand_index = self.data_rand.randint(0, len(self.images))
            image_p = self.images[rand_index].copy ()
            label_p = self.labels[rand_index].copy ()

            seed = time_seed () #self.rng.randint(0, 20152015)
            
           
            # p_total = Augmentor.Pipeline()
            # p_total.resize(probability=1, width=self.DIMY, height=self.DIMX, resample_filter='NEAREST')
            # image_p, label_p = self.AugmentPair(image_p.copy(), label_p.copy(), p_total, seed=seed)
            image_shape = (DIMY, DIMX)
            image_p = cv2.resize(image_p.copy(), image_shape, interpolation=cv2.INTER_LINEAR)
            label_p = cv2.resize(label_p.copy(), image_shape, interpolation=cv2.INTER_NEAREST)

            if self.isTrain:
                # Augment the pair image for same seed
                p_train = Augmentor.Pipeline()
                p_train.rotate_random_90(probability=0.75, resample_filter=Image.NEAREST)
                p_train.rotate(probability=1, max_left_rotation=10, max_right_rotation=10, resample_filter=Image.NEAREST)
                p_train.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=5)
                # p_train.zoom_random(probability=0.5, percentage_area=0.8)
                p_train.flip_random(probability=0.75)

                image_p, label_p = self.AugmentPair(image_p.copy(), label_p.copy(), p_train, seed=seed)
            else:
                p_valid = Augmentor.Pipeline()
                # p_valid.zoom_random(probability=0.5, percentage_area=0.8)
                p_valid.flip_random(probability=0.75)

                image_p, label_p = self.AugmentPair(image_p.copy(), label_p.copy(), p_valid, seed=seed)
            # # Calculate linear label
            if self.pruneLabel:
                label_p, nb_labels_p = skimage.measure.label(label_p.copy(), return_num=True)        

            # Calculate membrane
            def membrane(label):
                membr = np.zeros_like(np.squeeze(label))
                #for z in range(membr.shape[0]):
                #    membr[z,...] = 1-skimage.segmentation.find_boundaries(np.squeeze(label[z,...]), mode='thick') #, mode='inner'
                membr = 1-skimage.segmentation.find_boundaries(np.squeeze(label), mode='thick') #, mode='inner'
                membr[np.squeeze(label)==0] = 0 
                membr = np.reshape(membr, label.shape)
                return membr
            #print(label_p.shape)
            membr_p = membrane(label_p.copy())

            #Expand dim to make single channel
            image_p = np.expand_dims(image_p, axis=0)
            label_p = np.expand_dims(label_p, axis=0)
            membr_p = np.expand_dims(membr_p, axis=0)

            #label_p = np.expand_dims(label_p, axis=-1)
            #membr_p = np.expand_dims(membr_p, axis=-1)
            yield [image_p.astype(np.float32), 
                   membr_p.astype(np.float32),
                   label_p.astype(np.float32), 
                   ] 



###############################################################################
def get_data(dataDir, isTrain=False, isValid=False, isTest=False):
    # Process the directories 
    if isTrain:
        num=3000 #100 #3000
        names = ['trainA', 'trainB']
    if isValid:
        num=10
        names = ['validA', 'validB']
    if isTest:
        num=10
        names = ['testA', 'testB']

    
    dset  = TUImageDataFlow(os.path.join(dataDir, names[0]),
                               os.path.join(dataDir, names[1]),
                               num, 
                               isTrain=isTrain, 
                               isValid=isValid, 
                               isTest =isTest)
    dset.reset_state()
    return dset

###############################################################################
class Model(ModelDesc):
   

    @auto_reuse_variable_scope
    def load_enet(self, input_image, feature_dim=3, last_dim=1):
        assert input_image is not None
        num_initial_blocks = 1
        skip_connections = False
        stage_two_repeat = 2
        batch_size=1
        with slim.arg_scope(ENet_arg_scope()):
            last_prelu, _ = ENet(input_image,
                         num_classes=last_dim,
                         batch_size=batch_size,
                         is_training=True,
                         reuse=None,
                         num_initial_blocks=num_initial_blocks,
                         stage_two_repeat=stage_two_repeat,
                         skip_connections=skip_connections)

        # variables_to_restore = slim.get_variables_to_restore()
        # saver = tf.train.Saver(variables_to_restore)
        # saver.restore(sess, checkpoint)
        semantic   = slim.conv2d_transpose(last_prelu, last_dim, [2,2], stride=2, 
                                        biases_initializer=tf.constant_initializer(10.0), 
                                        weights_initializer=tf.contrib.layers.xavier_initializer(), 
                                        scope='Semantic/transfer_layer/conv2d_transpose')
        logits = slim.conv2d_transpose(last_prelu, feature_dim, [2,2], stride=2, 
                                        biases_initializer=tf.constant_initializer(10.0), 
                                        weights_initializer=tf.contrib.layers.xavier_initializer(), 
                                        scope='Instance/transfer_layer/conv2d_transpose')

        # with tf.variable_scope('', reuse=True):
        #     weight = tf.get_variable('Instance/transfer_layer/conv2d_transpose/weights')
        #     bias = tf.get_variable('Instance/transfer_layer/conv2d_transpose/biases')
        #     sess.run([weight.initializer, bias.initializer])

        return semantic, logits

    def inputs(self):
        return [
            tf.placeholder(tf.float32, (None, DIMY, DIMX, 3), 'image'),
            tf.placeholder(tf.float32, (None, DIMY, DIMX, 1), 'membr'),
            tf.placeholder(tf.float32, (None, DIMY, DIMX, 1), 'label'),
            ]

    def build_graph(self, image, membr, label):
        # G = tf.get_default_graph()
        # with G.gradient_override_map({"Round": "Identity", "ArgMax": "Identity"}):
            # pi, pl = image, label
        input_image, membrane, correct_label = image, membr, label


        # Calculate the loss
        feature_dim = 16
        param_var   = 1.0
        param_dist  = 1.0
        param_reg   = 0.001
        delta_v = 0.5
        delta_d = 1.5
        image_shape=(DIMY, DIMX)
        # Run the model
        semantic, prediction = self.load_enet(input_image, feature_dim)
        semantic = tf.identity(semantic, 'semantic')

        prediction = tf.identity(prediction, 'prediction')

        
        # Summary the loss
        losses = []
        with tf.name_scope('loss_aff'):
            aff_im = tf.identity(1.0 - dice_coe(semantic, membrane, axis=[0,1,2,3], loss_type='jaccard'), 
                                 name='aff_im')  
            losses.append(1e1*aff_im)
            add_moving_summary(aff_im)
        with tf.name_scope('loss_discrim'):
            disc_loss, l_var, l_dist, l_reg = discriminative_loss_single(prediction, correct_label, feature_dim, image_shape, 
                                                    delta_v, delta_d, param_var, param_dist, param_reg)
            losses.append(1e0*disc_loss)
            add_moving_summary(disc_loss)

        self.cost = tf.reduce_mean(losses)
        #add_moving_summary(disc_loss)
        #add_moving_summary(l_var)
        #add_moving_summary(l_dist)
        #add_moving_summary(l_reg)
        
        # Summary the image
        tf.summary.image('image_', input_image,   max_outputs=50)
        tf.summary.image('label_', correct_label, max_outputs=50)
        tf.summary.image('membr_', 255*membrane,      max_outputs=50)

        tf.summary.image('preds0', prediction[...,0:1], max_outputs=50)
        tf.summary.image('preds1', prediction[...,1:2], max_outputs=50)
        tf.summary.image('preds2', prediction[...,2:3], max_outputs=50)

        viz = tf.concat([tf.concat([input_image[...,0:1], 255*membrane[...,0:1],
                                    correct_label[...,0:1], 255*semantic[...,0:1]], axis=2),                
                     ], axis=1)
        viz = tf.cast(tf.clip_by_value(viz, 0, 255), tf.uint8, name='viz')
        tf.summary.image('labelized', viz, max_outputs=50)

        return self.cost

    def optimizer(self):
        lr = symbolic_functions.get_scalar_var('learning_rate', 2e-4, summary=True)
        return tf.train.AdamOptimizer(lr, beta1=0.5, epsilon=1e-3)
###############################################################################
from sklearn.cluster import MeanShift, estimate_bandwidth
def cluster(prediction, bandwidth):
    ms = MeanShift(bandwidth, bin_seeding=True)
    print ('Mean shift clustering, might take some times ...')
    tic = time.time()
    ms.fit(prediction)
    print ('time for clustering', time.time() - tic)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    
    num_clusters = cluster_centers.shape[0]

    return num_clusters, labels, cluster_centers

def get_instance_masks(prediction, bandwidth):
    batch_size, h, w, feature_dim = prediction.shape

    num_clusters, labels, cluster_centers = cluster(prediction.reshape([h*w, feature_dim]), bandwidth)
    print ('Number of predicted clusters', num_clusters)
    labels = np.array(labels, dtype=np.uint8).reshape([h,w])
    return labels



class VisualizeRunner(Callback):
    def __init__(self, input, tower_name='InferenceTower', device=0):
        self.dset = input 
        self._tower_name = tower_name
        self._device = device

    def _setup_graph(self):
        self.pred = self.trainer.get_predictor(
            ['image', 'membr', 'label'], ['prediction', 'viz'])

    def _before_train(self):
        pass

    def _trigger(self):
        for lst in self.dset.get_data():
            prediction, viz = self.pred(lst)
            inst_masks = get_instance_masks(prediction, 0.7) ### Bandwith is here


            self.trainer.monitors.put_image('pil_test', inst_masks)
            viz = np.squeeze(np.array(viz))
            self.trainer.monitors.put_image('viz_test', viz)
###############################################################################
def sample(dataDir, model_path, prefix='.'):
    print("Starting...")
    # print(dataDir)
    imageFiles = glob.glob(os.path.join(dataDir, 'testA/*.png'))
    labelFiles = glob.glob(os.path.join(dataDir, 'testB/*.png'))
    
    imageFiles = natsorted(imageFiles)
    labelFiles = natsorted(labelFiles)

    def AugmentPair(src_image, src_label, pipeline, seed=None, verbose=False):
        np.random.seed(seed) if seed else np.random.seed(2015)
        # print(src_image.shape, src_label.shape, aug_image.shape, aug_label.shape) if verbose else ''
        if src_image.ndim==2:
            src_image = np.expand_dims(src_image, 2)
        if src_label.ndim==2:
            src_label = np.expand_dims(src_label, 2)
        
        # Create the result
        aug_images = [] #np.zeros_like(src_image)
        aug_labels = [] #np.zeros_like(src_label)
        #print(src_image.shape)
        #print(src_label.shape)
        # print(src_image.shape, src_label.shape)
        for z in range(src_image.shape[2]):
            #Image and numpy has different matrix order
            pipeline.set_seed(seed)
            aug_image = pipeline._execute_with_array(src_image[...,z]) 
            aug_images.append(aug_image)
        for z in range(src_label.shape[2]):
            pipeline.set_seed(seed)
            aug_label = pipeline._execute_with_array(src_label[...,z])        
            aug_labels.append(aug_label)

        aug_images = np.array(aug_images).astype(np.float32)
        aug_labels = np.array(aug_labels).astype(np.float32)

        aug_images = np.transpose(aug_images, [1, 2, 0])
        aug_labels = np.transpose(aug_labels, [1, 2, 0])
        # print(aug_images.shape, aug_labels.shape)

        return aug_images, aug_labels

       

        aug_images = np.array(aug_images).astype(np.float32)
        aug_labels = np.array(aug_labels).astype(np.float32)
        # print(aug_images.shape, aug_labels.shape)
        aug_images = np.expand_dims(aug_images, axis=-1)
        aug_labels = np.expand_dims(aug_labels, axis=-1)
        return aug_images, aug_labels



    
    p_total = Augmentor.Pipeline()
    p_total.resize(probability=1, width=DIMY, height=DIMX, resample_filter='NEAREST')



    predict_func = OfflinePredictor(PredictConfig(
        model=Model(),
        session_init=get_model_loader(model_path),
        input_names=['image'],
        output_names=['semantic', 'prediction']))

    sbds = []
    for k in range(len(imageFiles)): #range(1): #(len(imageFiles)):
        image = skimage.io.imread(imageFiles[k])
        label = skimage.io.imread(labelFiles[k])
        image, label = AugmentPair(image.copy(), label.copy(), p_total, seed=None)
        print(image.shape)

        # image = np.expand_dims(image, axis=3)
        label, nb_labels = skimage.measure.label(label.copy(), return_num=True)        
        ### Start deployment
        image = np.expand_dims(image, axis=0)
        # label = np.expand_dims(label, axis=0)
        # membr = np.zeros_like(label)

        print image.shape
        print label.shape

        pred_pim, pred_pif = predict_func(image)
        pred_pim = np.array(pred_pim)
        pred_pif = np.array(pred_pif)

        print pred_pim.shape
        print pred_pif.shape



        # def np_func(X, algorithm, feature_dim=None, label_shape=None, n_clusters=12, n_jobs=4):
        #     # Perform clustering on high dimensional channel image
        #     feats_shape = X.shape
        #     if feature_dim==None:
        #         feature_dim = feats_shape[-1]
        #         print 'Feature_dim', feature_dim
            
        #     # Flatten and normalize X
        #     X_flatten = np.reshape(X, newshape=(-1, feature_dim))
        #     avg = X_flatten.mean()
        #     std = X_flatten.std()
        #     X_flatten -= avg
        #     X_flatten /= std
        #     print(X.shape)
        #     print(X_flatten.shape)
           

        #     print ('Perform clustering, might take some time ...')
        #     tic = time.time()
        #     algorithm.fit(X_flatten)
        #     # y_pred_flatten = algorithm.fit_predict(X_flatten)
        #     print ('Time for clustering', time.time() - tic)


        #     # Get the result in float32
        #     y_pred_flatten = algorithm.labels_.astype(np.float32)
        #     if label_shape:
        #         y_pred = np.reshape(y_pred_flatten, label_shape)
        #     else:
        #         y_pred = y_pred_flatten

        #     print(label_shape)
        #     print(y_pred_flatten.shape)
        #     print(X_flatten.max())
        #     print(X_flatten.min())
        #     print(X_flatten.mean())
        #     print(X_flatten.std())
        #     print(y_pred_flatten.max())
        #     print(y_pred_flatten.min())


        #     return y_pred

        # # Having mask and high dimensional space, so what's next?
        
        # feature_dim=16
        # # First squeeze everything
        # pim = np.squeeze(np.array(pred_pim)) #2d image
        # pif = np.squeeze(np.array(pred_pif)) #2d image

        # pim_flatten = np.reshape(pim, [-1])
        # pif_flatten = np.reshape(pif, [-1, feature_dim])

        # loc_1d = pim_flatten>0.5      # Find the location of semantic segmentation
        # idx_1d = np.where(loc_1d)   # Find the indices of semantic segmentation

        # #Filter the high dim space
        # # isFiltered=False
        # # if isFiltered==True:
        # #     pif_masked_flatten = pif_flatten[loc_1d]
        # # else:
        # #     pif_masked_flatten = pif_flatten

        # pif_masked_flatten = pif_flatten
        # # Cluster them
        # from sklearn.cluster import MeanShift, estimate_bandwidth
        # from sklearn.cluster import DBSCAN, SpectralClustering
        # from sklearn import cluster

        # bandwidth = 0.5 #0.7 #
        # # bandwidth = cluster.estimate_bandwidth(pif_masked_flatten, quantile=0.3)
        # algorithm = cluster.MeanShift(bandwidth= bandwidth, bin_seeding=True, n_jobs=4)
        # # algorithm = cluster.SpectralClustering(n_clusters=label.max(), eigen_solver='arpack', affinity="nearest_neighbors")

        # pil_masked_flatten = np_func(pif_masked_flatten, algorithm, 
        #                              feature_dim=feature_dim)

        # # Reshape the label
        # pil_flatten = np.zeros_like(pim_flatten)
        # pil_flatten[idx_1d] = pil_masked_flatten
        # pil = np.reshape(pil_flatten, pim.shape)

        pil = get_instance_masks(pred_pif, 1.0) ### Bandwith is here


        sbd = calc_sbd(label, pil)
        print 'Sbd ', sbd
        sbds.append(sbd)

        label = np.squeeze(label)
        pil = np.squeeze(pil)
        skimage.io.imsave('result_discrim/groundtruth/{}.png'.format(k+1), get_colors(label, plt.cm.tab20c)) #plt.cm.PiYG))
        skimage.io.imsave('result_discrim/predict/{}.png'.format(k+1), get_colors(pil, plt.cm.tab20c)) #plt.cm.PiYG))
    


    mean_sbd = np.mean(sbds)
    print 'Mean sbds ', mean_sbd
    print("Ending...")
    return None
###############################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',    default='0', help='comma seperated list of GPU(s) to use.')
    parser.add_argument('--data',   default='data_tu/', required=True, 
                                    help='Data directory, contain trainA/trainB/validA/validB')
    parser.add_argument('--load',   help='Load the model path')
    parser.add_argument('--sample', help='Run the deployment on an instance',
                                    action='store_true')

    args = parser.parse_args()
    
    train_ds = get_data(args.data, isTrain=True, isValid=False, isTest=False)
    valid_ds = get_data(args.data, isTrain=False, isValid=True, isTest=False)

    train_ds  = PrefetchDataZMQ(train_ds, 4)
    train_ds  = PrintData(train_ds)

    model     = Model()

    os.environ['PYTHONWARNINGS'] = 'ignore'

    # Set the GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Running train or deploy
    if args.sample:
        # TODO
        print("Deploy the data")
        sample(args.data, args.load, prefix='deploy_')
    else:
        # Set up configuration
        logger.auto_set_dir()

        # Set up configuration
        config = TrainConfig(
            model           =   model, 
            dataflow        =   train_ds,
            callbacks       =   [
                PeriodicTrigger(ModelSaver(), every_k_epochs=50),
                PeriodicTrigger(VisualizeRunner(valid_ds), every_k_epochs=5),
                ScheduledHyperParamSetter('learning_rate', [(0, 2e-4), (100, 1e-4), (200, 1e-5), (300, 1e-6)], interp='linear')
                ],
            max_epoch       =   500, 
            session_init    =    SaverRestore(args.load) if args.load else None,
            )
    
        # Train the model
        launch_train_with_config(config, QueueInputTrainer())
