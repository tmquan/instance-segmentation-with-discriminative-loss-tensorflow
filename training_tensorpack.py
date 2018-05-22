# Hidden 2 domains no constrained
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys, argparse, glob, time

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

# Misc. libraries
from six.moves import map, zip, range
from natsort import natsorted 
from enet import *
from loss import *
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



###############################################################################
class EMDataFlow(ImageDataFlow):
    ###############################################################################
    def AugmentPair(self, src_image, src_label, pipeline, seed=None, verbose=False):
        np.random.seed(seed) if seed else np.random.seed(2015)
        # print(src_image.shape, src_label.shape, aug_image.shape, aug_label.shape) if verbose else ''
        if src_image.ndim==2:
            src_image = np.expand_dims(src_image, 0)
            src_label = np.expand_dims(src_label, 0)
        
        # Create the result
        aug_images = [] #np.zeros_like(src_image)
        aug_labels = [] #np.zeros_like(src_label)
        
        # print(src_image.shape, src_label.shape)
        for z in range(src_image.shape[0]):
            #Image and numpy has different matrix order
            pipeline.set_seed(seed)
            aug_image = pipeline._execute_with_array(src_image[z,...]) 
            pipeline.set_seed(seed)
            aug_label = pipeline._execute_with_array(src_label[z,...])        
            aug_images.append(aug_image)
            aug_labels.append(aug_label)
        aug_images = np.array(aug_images).astype(np.float32)
        aug_labels = np.array(aug_labels).astype(np.float32)
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
            label_p = np.dstack((label_p, label_p, label_p))

            seed = time_seed () #self.rng.randint(0, 20152015)
            
           
            p_total = Augmentor.Pipeline()
            p_total.resize(probability=1, width=self.DIMY, height=self.DIMX, resample_filter='NEAREST')
            image_p, label_p = self.AugmentPair(image_p.copy(), label_p.copy(), p_total, seed=seed)

            if self.isTrain:
            	# Augment the pair image for same seed
                p_train = Augmentor.Pipeline()
                p.flip_left_right(probability=0.5)

                image_p, label_p = self.AugmentPair(image_p.copy(), label_p.copy(), p_train, seed=seed)
                
            	
            # # Calculate linear label
            if self.pruneLabel:
                label_p, nb_labels_p = skimage.measure.label(label_p.copy(), return_num=True)        

            #Expand dim to make single channel
            # image_p = np.expand_dims(image_p, axis=-1)

            label_p = np.expand_dims(label_p, axis=-1)
            yield [image_p.astype(np.float32), 
                   label_p.astype(np.float32), 
                   ] 

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
        for imageFile in imageFiles:
            image = skimage.io.imread (imageFile)
            self.images.append(image)
        for labelFile in labelFiles:
            label = skimage.io.imread (labelFile)
            self.labels.append(label)
            
        self.DIMZ = shape[0]
        self.DIMY = shape[1]
        self.DIMX = shape[2]
        self.pruneLabel = pruneLabel

    def size(self):
        return self._size

###############################################################################
def get_data(dataDir, isTrain=False, isValid=False, isTest=False):
	# Process the directories 
	if isTrain:
		num=3000
		names = ['trainA', 'trainB']
	if isValid:
		num=500
		names = ['validA', 'validB']
	if isTest:
		num=10
		names = ['testA', 'testB']

	
	dset  = ImageDataFlow(os.path.join(dataDir, names[0]),
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
    def load_enet(self, input_image, feature_dim=3):
        assert input_image is not None
        num_initial_blocks = 1
	    skip_connections = False
	    stage_two_repeat = 2
	    batch_size=1
	    with slim.arg_scope(ENet_arg_scope()):
	        last_prelu, _ = ENet(input_image,
	                     num_classes=12,
	                     batch_size=batch_size,
	                     is_training=True,
	                     reuse=None,
	                     num_initial_blocks=num_initial_blocks,
	                     stage_two_repeat=stage_two_repeat,
	                     skip_connections=skip_connections)

	    # variables_to_restore = slim.get_variables_to_restore()
	    # saver = tf.train.Saver(variables_to_restore)
	    # saver.restore(sess, checkpoint)

	    logits = slim.conv2d_transpose(last_prelu, feature_dim, [2,2], stride=2, 
	                                    biases_initializer=tf.constant_initializer(10.0), 
	                                    weights_initializer=tf.contrib.layers.xavier_initializer(), 
	                                    scope='Instance/transfer_layer/conv2d_transpose')

	    # with tf.variable_scope('', reuse=True):
	    #     weight = tf.get_variable('Instance/transfer_layer/conv2d_transpose/weights')
	    #     bias = tf.get_variable('Instance/transfer_layer/conv2d_transpose/biases')
	    #     sess.run([weight.initializer, bias.initializer])

	    return last_prelu, logits

    def inputs(self):
        return [
            tf.placeholder(tf.float32, (None, DIMY, DIMX, 3), 'image'),
            tf.placeholder(tf.float32, (None, DIMY, DIMX, 1), 'label'),
            ]

    def build_graph(self, image, label):
        G = tf.get_default_graph()
        with G.gradient_override_map({"Round": "Identity", "ArgMax": "Identity"}):
            # pi, pl = image, label
            input_image, correct_label = image, label

            ### Optimization operations
            feature_dim = 3
		    param_var 	= 1.0
		    param_dist 	= 1.0
		    param_reg 	= 0.001
		    delta_v = 0.5
		    delta_d = 1.5

	        disc_loss, l_var, l_dist, l_reg = discriminative_loss(prediction, correct_label, feature_dim, image_shape, 
	                                                    delta_v, delta_d, param_var, param_dist, param_reg)
	        self.cost = disc_loss
	        add_moving_summary(disc_loss)
	        add_moving_summary(l_var)
	        add_moving_summary(l_dist)
	        add_moving_summary(l_reg)
            # with tf.variable_scope('gen'):
            #     with tf.device('/device:GPU:0'):
            #         with tf.variable_scope('image2level'):
            #             pil = self.generator(tf_2tanh(pi), 
            #                                      # last_dim=1, 
            #                                      # nl=tf.nn.tanh, 
            #                                      # nb_filters=32
            #                                      )
            #             pil = tf_2imag(pil, maxVal=24)
                      
            # losses = []         
            
            # with tf.name_scope('loss_mae'):
            #     mae_il = tf.reduce_mean(tf.abs(pl -  pil),
            #                             name='mae_il')
            #     losses.append(1e0*mae_il)
            #     add_moving_summary(mae_il)

            #     mae_if = tf.reduce_mean(tf.abs( pl - tf.cast(tf.cast(pl, tf.int32), tf.float32) -
            #                                     pil - tf.cast(tf.cast(pl, tf.int32), tf.float32)), 
                                             
            #                             name='mae_if')
            #     losses.append(1e2*mae_if)
            #     add_moving_summary(mae_if)

         
            # # Collect the result
            # pil = tf.identity(pil, name='pil')
            # self.cost = tf.reduce_sum(losses, name='self.cost')
            # add_moving_summary(self.cost)
            # # Visualization

            # # Segmentation
            # pz = tf.zeros_like(pi)
            # viz = tf.concat([tf.concat([pi, 20*pl,  20*pil], axis=2),
            #                  tf.concat([pz, 255*2*(pl-tf.cast(tf.cast(pl, tf.int32), tf.float32)),  
            #                                 255*2*(pil-tf.cast(tf.cast(pil, tf.int32), tf.float32)),  
            #                                 ], axis=2),
            #                  ], axis=1)
            # # viz = tf_2imag(viz)
            # viz = tf.cast(tf.clip_by_value(viz, 0, 255), tf.uint8, name='viz')
            # tf.summary.image('labelized', viz, max_outputs=50)

    def optimizer(self):
        lr = symbolic_functions.get_scalar_var('learning_rate', 2e-4, summary=True)
        return tf.train.AdamOptimizer(lr, beta1=0.5, epsilon=1e-3)
###############################################################################
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--gpu',    default='0', help='comma seperated list of GPU(s) to use.')
	parser.add_argument('--data',  	default='data/', required=True, 
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
		# sample
		pass
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
				PeriodicTrigger(InferenceRunner(valid_ds, [ScalarStats('loss_mae/mae_il')]), every_k_epochs=1),
				ScheduledHyperParamSetter('learning_rate', [(0, 2e-4), (100, 1e-4), (200, 1e-5), (300, 1e-6)], interp='linear')
				],
			max_epoch       =   500, 
			session_init    =    SaverRestore(args.load) if args.load else None,
			)
	
		# Train the model
		launch_train_with_config(config, QueueInputTrainer())
