import time

from data.graph import TriangleFinding, Transitivity, ComponentLabeling
from data.matrices import BitwiseXOR
from data.matrix import Transpose, Squaring, Rotate90
from data.sudoku import SudokuHard, Sudoku
from models.matrixse import MatrixSE
from models.matrixse_multisteps import MatrixSEMultistep
from models.resnet import ResNet

"""Data and checkpoint placement"""

train_dir = '/tmp/matrix_se'
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
shuffle_size = 10000

adjust_class_imbalance = False
reduce_void_weight = False  # works only when adjust_class_imbalance is on

""" Fully-connected CNN receptive field calculation """
calculate_fcn_receptive_field = False  # Remove normalization and use ReLU for this to work

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

# Generic version
middle_scale = 2

"""Task specific configurations for Matrix Shuffle-Exchange: """

""" Recommended settings for Matrix Squaring task """
model = MatrixSE(feature_maps=96, block_count=2)
dataset = Squaring()

""" Recommended settings for Matrix Transpose task """
# model = MatrixSE(feature_maps=96, block_count=2)
# dataset = Transpose()

""" Recommended settings for Matrix Rotate by 90 degrees task """
# model = MatrixSE(feature_maps=96, block_count=2)
# dataset = Rotate90()

""" Recommended settings for bitwise XOR of matrices """
# model = MatrixSE(feature_maps=96, block_count=2)
# dataset = BitwiseXOR()

""" Recommended settings for Triangle Finding in graph """
# model = MatrixSE(feature_maps=96, block_count=2)
# dataset = TriangleFinding()
# adjust_class_imbalance = True

""" Recommended settings for Transitive Path in the graph """
# model = MatrixSE(feature_maps=192, block_count=2)
# dataset = Transitivity()
# adjust_class_imbalance = True

""" Recommended settings for Component Labeling in the graph """
# model = MatrixSE(feature_maps=192, block_count=2)
# dataset = ComponentLabeling()

""" Recommended settings for Sudoku puzzle solving """
# model = MatrixSEMultistep(feature_maps=96, block_count=2, train_steps=10, eval_steps=30)
# dataset = Sudoku()
# train_steps = 1000000

""" Recommended settings for Sudoku-hard puzzle solving """
# model = MatrixSEMultistep(feature_maps=96, block_count=2, train_steps=10, eval_steps=30)
# dataset = SudokuHard()
# train_steps = 1000000

""" Task specific configurations for ResNet: """

""" Recommended settings for Matrix Squaring task """
# model = ResNet(feature_maps=128, kernel_size=3, residual_blocks=14)
# dataset = Squaring()

""" Recommended settings for Matrix Transpose task """
# model = ResNet(feature_maps=128, kernel_size=3, residual_blocks=14)
# dataset = Transpose()

""" Recommended settings for Matrix Rotate by 90 degrees task """
# model = ResNet(feature_maps=128, kernel_size=3, residual_blocks=14)
# dataset = Rotate90()

""" Recommended settings for bitwise XOR of matrices """
# model = ResNet(feature_maps=128, kernel_size=3, residual_blocks=14)
# dataset = BitwiseXOR()

""" Model directory"""
start_date = time.strftime("%y_%m_%d_%T", time.gmtime(time.time()))
model_dir = f"{train_dir}/{start_date}_{dataset.__class__.__name__.lower()}"
gen_test_file = "gen_test.txt"
