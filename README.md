# Switchblade - a Neural Network for Hard 2D Tasks

## Requirements:
* Python 3.6 or higher;
* Nvidia T4 GPU or equivalent;
* TensorFlow 1.15 (see `requirements.txt`).

## Running experiments:
* set training directory `train_dir` in `config.py` file;
* set data directory `data_dir` in `config.py` file;
* select task by uncommenting task section in `config.py` file.
Matrix transpose task is selected by default. Matrix transpose tasks in `config.py`
is represented as: 
```python3
""" Recommended settings for Matrix Transpose task """
model = Switchblade(feature_maps=96, block_count=2)
dataset = Transpose()
```
* CityScapes datas should be manually downloaded from [CityScapes site](https://www.cityscapes-dataset.com)
and placed in `data_dir/cityscapes` (optional);
* Sudoku data should be manually downloaded from [GitHub](https://github.com/Kyubyong/sudoku)
and extracted in `data_dir/sudoku` (optional);
* Mate-In-One data is available [here]()
and should be extracted in `data_dir/mateinone` (optional);

* then run `python3 trainer.py --train --eval`. Checkpoints will be saved in `train_dir`;
* for algorithmic tasks, datasets will be generated in `data_dir` on the first run, sequential runs
will reuse this dataset;
* to evaluated generalization on larger inputs add `--gen_test` when starting training. Generalization results will be available in `train_dir/gen_test.txt`;
* you can restart training or evaluation from previous checkpoint using `--continue <checkpoint_dir>` flag.

## Task description:

### Algorithmic tasks on matrices:

#### Matrix squaring (Squaring):
Given binary matrix model is required to output its square (matrix multiplication with itself) modulo 2.
Data generation for this task is given in `data/matrix.py` file.

To choose this task, in `config.py` uncomment:
```
""" Recommended settings for Matrix Squaring task """
model = Switchblade(feature_maps=96, block_count=2)
dataset = Squaring()
```

#### Matrix transpose (Transpose):
Given matrix with elements of alphabet 1-11, model is required to output its transposed matrix.
Data generation for this task is given in `data/matrix.py` file.

To choose this task, in `config.py` uncomment:
```
""" Recommended settings for Matrix Transpose task """
model = Switchblade(feature_maps=96, block_count=2)
dataset = Transpose()
```
#### Matrix Rotation by 90 degrees (Rotate90):
Given matrix with elements of alphabet 1-11, model is required to output matrix that has been rotated by 90 degrees clockwise.
Data generation for this task is given in `data/matrix.py` file.

To choose this task, in `config.py` uncomment:
```
""" Recommended settings for Matrix Rotate by 90 degrees task """
# model = Switchblade(feature_maps=96, block_count=2)
# dataset = Rotate90()
```

#### Element-wise XOR operation of two matrices (BitwiseXOR):
Given two binary matrices, model is required to output matrix that has been obtained by element-wise XOR operation.
Data generation for this task is given in `data/matrices.py` file.

To chose this task, in `config.py` uncomment:
```
""" Recommended settings for bitwise XOR of matrices """
# model = Switchblade(feature_maps=192, block_count=2)
# dataset = BitwiseXOR()
```

### Algorithmic task on graphs:
Graphs are represented as adjacency matrices.

#### Connected Component Labeling (ComponentLabeling):
We initialize a labeled graph with random edge labels in range 2-100. 
The task is to label all edges of each connected component with the lowest label among all the component’s edges.
Data generation for this task is given in `data/graph.py` file.

To choose this task, in `config.py` uncomment:
```
""" Recommended settings for Component Labeling in the graph """
# model = Switchblade(feature_maps=192, block_count=2)
# dataset = ComponentLabeling()
```

#### Triangle Finding in graph (TriangleFinding):
We generate random complete bipartite graphs and add a few random edges.
The goal is to return all the edges belonging to any triangle.
Data generation for this task is given in `data/graph.py` file.

To choose this task, in `config.py` uncomment:
```
""" Recommended settings for Triangle Finding in graph """
# model = Switchblade(feature_maps=192, block_count=2)
# dataset = TriangleFinding()
# adjust_class_imbalance = True
```

#### Transitive path finding in graph (Transitivity):
Given a directed graph, the goal is to add edges to graph for every two vertices
that have a transitive path of length 2 between them.
Data generation for this task is given in `data/graph.py` file.

To choose this task, in `config.py` uncomment:
```
""" Recommended settings for Transitive Path in the graph """
model = Switchblade(feature_maps=192, block_count=2)
dataset = Transitivity()
adjust_class_imbalance = True
```

### Image tasks:

#### CIFAR-10 image classification:
CIFAR-10 is a conventional image classification task that consists of 50000
train and 10000 test 32×32 images in 10 mutually exclusive classes.
Data preparation for this task is given in `data/cifar.py` file.

To choose this task, in `config.py` uncomment:
```
""" Recommended settings for CIFAR-10 """
model = Switchblade4CIFAR(feature_maps=192, block_count=1)
dataset = CIFAR10()
log_input_image = True
L2_decay = 0.01
```

#### CityScapes pixel-level semantic segmentation (CityScape)
Given image, model is required to classify each pixel in one of 20 classes.
Data preparation for this task is given in `data/cityscape.py` file.

To choose this task, in `config.py` uncomment:
```
""" Recommended settings for CityScapes pixel-level segmentation """
model = Switchblade4Segment(feature_maps=192, block_count=1)
dataset = CityScape()
train_batch_size = 4
eval_batch_size = 2
log_input_image = True
log_segmentation_output = True
use_jaccard_loss = True
adjust_class_imbalance = True
reduce_void_weight = True
mixed_precision_training = False
```

### Puzzles & Games

#### Sudoku puzzle solving (Sudoku)
Given empty Sudoku puzzle, model is required to give solution in one step.
Data preparation for this task is given in `data/sudoku.py` file.

To choose this task, in `config.py` uncomment:
```
""" Recommended settings for Sudoku puzzle solving """
model = Switchblade(feature_maps=192, block_count=2)
dataset = Sudoku()
eval_batch_size = 1
L2_decay = 0.01
train_steps = 1000000
```

#### Mate-in-one chess dataset (MateInOne)
Given chess board where mate move is possible, model is required to return mating move.
Data preparation for this task is given in `data/mateinone.py` file.

To choose this task, in `config.py` uncomment:
```
""" Recommended settings for Mate in One (chess) task"""
model = Switchblade(feature_maps=192, block_count=2)
dataset = MateInOne()
chess = True
```

