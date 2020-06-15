import time

from data.mateinone import MateInOne
from data.cifar import CIFAR10
from data.cityskape import CityScape
from data.graph import TriangleFinding, Transitivity, ComponentLabeling
from data.matrices import BitwiseXOR
from data.matrix import Squaring, Transpose, Identity, Rotate90
from data.sudoku import Sudoku
from models.switcblade4segment import Switchblade4Segment
from models.switchblade import Switchblade
from models.switchblade4cifar import Switchblade4CIFAR

"""Data and checkpoint placement"""

train_dir = '/tmp/switchblade_train'
data_dir = '/tmp/data'
generate_in_files = True
force_file_generation = False

"""Training configuration"""

visible_gpus = '0'
use_xla = True
mixed_precision_training = True  # If problems with memory leaks, disable this

tpu_train_batch_size = 1024
train_batch_size = 32
eval_batch_size = 32
test_batch_size = 1

save_checkpoint_steps = 1000
train_steps = 500000
eval_steps = 2000
shuffle_size = train_batch_size

adjust_class_imbalance = False
use_jaccard_loss = False
reduce_void_weight = False  # works only when adjust_class_imbalance is on

"""Hyperparameters"""

learning_rate = 0.0001
label_smoothing = 0.1
L2_decay = 0.001

"""TensorBoard logging"""

enable_summary_hooks = True
log_input_image = False
log_segmentation_output = False
log_gradients = False
log_layer_outputs = False
log_2d_outputs = True

"""Task specific configurations: """

chess = False

""" Recommended settings for CIFAR-10 """
# model = Switchblade4CIFAR(feature_maps=192, block_count=1)
# dataset = CIFAR10()
# log_input_image = True
# L2_decay = 0.01

""" Recommended settings for CityScapes pixel-level segmentation """
# model = Switchblade4Segment(feature_maps=192, block_count=1)
# dataset = CityScape()
# train_batch_size = 4
# eval_batch_size = 2
# log_input_image = True
# log_segmentation_output = True
# use_jaccard_loss = True
# adjust_class_imbalance = True
# reduce_void_weight = True
# mixed_precision_training = False

""" Recommended settings for Matrix Squaring task """
# model = Switchblade(feature_maps=96, block_count=2)
# dataset = Squaring()

""" Recommended settings for Matrix Transpose task """
model = Switchblade(feature_maps=96, block_count=2)
dataset = Transpose()

""" Recommended settings for Matrix Identity task """
# model = Switchblade(feature_maps=96, block_count=2)
# dataset = Identity()

""" Recommended settings for Matrix Rotate by 90 degrees task """
# model = Switchblade(feature_maps=96, block_count=2)
# dataset = Rotate90()

""" Recommended settings for bitwise XOR of matrices """
# model = Switchblade(feature_maps=192, block_count=2)
# dataset = BitwiseXOR()

""" Recommended settings for Triangle Finding in graph """
# model = Switchblade(feature_maps=192, block_count=2)
# dataset = TriangleFinding()
# adjust_class_imbalance = True

""" Recommended settings for Transitive Path in the graph """
# model = Switchblade(feature_maps=192, block_count=2)
# dataset = Transitivity()
# adjust_class_imbalance = True

""" Recommended settings for Component Labeling in the graph """
# model = Switchblade(feature_maps=192, block_count=2)
# dataset = ComponentLabeling()

""" Recommended settings for Sudoku puzzle solving """
# model = Switchblade(feature_maps=192, block_count=2)
# dataset = Sudoku()
# eval_batch_size = 1
# L2_decay = 0.01
# train_steps = 1000000

""" Recommended settings for Mate in One (chess) task"""
# model = Switchblade(feature_maps=192, block_count=2)
# dataset = MateInOne()
# chess = True

""" Model directory"""
start_date = time.strftime("%y_%m_%d_%T", time.gmtime(time.time()))
model_dir = f"{train_dir}/{start_date}_{dataset.__class__.__name__.lower()}"
gen_test_file = "/gen_test.txt"
