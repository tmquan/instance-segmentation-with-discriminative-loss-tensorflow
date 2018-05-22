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
