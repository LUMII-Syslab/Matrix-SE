# Official _TensorFlow_ Implementation Of Switchblade model

This repository contains the official _TensorFlow_ implementation of the following paper:

>**Switchblade - a Neural Network for Hard 2D Tasks**
>
> by Emīls Ozoliņš, Kārlis Freivalds, Agris Šostaks
>
> [[arXiv]()]
>
>Abstract: _Convolutional neural networks have become the main tools for
> processing two-dimensional data. They work well for images, yet convolutions
> have a limited receptive field that prevents its applications to more
> complex 2D tasks. We propose a new neural network model, named Switchblade,
> that can efficiently exploit long-range dependencies in 2D data and
> solve much more challenging tasks. It has close-to-optimal  O(n² log n) 
>complexity for processing n×n data matrix. Besides the common image
> classification and segmentation,  we consider a diverse set of algorithmic 
>tasks on matrices and graphs. Switchblade can infer highly complex matrix 
>squaring and graph triangle finding algorithms purely from input-output
> examples. We show that our model is likewise suitable for logical 
>reasoning tasks -- it attains perfect accuracy on Sudoku puzzle solving.
> Additionally, we introduce a new dataset for predicting the checkmating
> move in chess on which our model achieves 72.5% accuracy._

## What is _Switchblade_?

Switchblade is generalization of [Neural Shuffle-Exchange](https://github.com/LUMII-Syslab/shuffle-exchange) networks to two dimensions. 
It is suitable for broad range of problems that can be represented as matrix. Switchblade model
can induce O(n^2 log n) time complexity, where n is length of matrix side, algorithms and model long-range dependencies.

Switchblade model consists of cascaded Quaternary Switch and Quaternary Shuffle layers, that forms
Beneš blocks. Here model structure is represented:
![](assets/switchblade_model.jpg)

Quaternary Switch layer divides inpute elements into tuples of 4 and the applies
Quaternary Switch Unit (QSU) to each tuple. QSU is a learnable
4-to-4 function, derived from [Residual Switch Unit](https://github.com/LUMII-Syslab/RSE), and is given as:

![](assets/qsu.png)

The Quaternary Shuffle layer rearranges elements according to cyclic base-4 digit permutation.
Quaternary Shuffle permutation can be interpreted as splitting matrix rows into two halves 
(white and green) and interleaving the halves, then applying the same
transformation to columns (white and red). 
Example permuation on 4×4 matrix:
![](assets/quaternary_shuffle.jpg)

## Preview Of Results
Evaluated on several problems.

Algorithmic tasks on matrices:
![](assets/algorithmic.jpg) 

Sudoku puzzle:
![](assets/sudoku.jpg)
Chess dataset Mate-In-One:
![](assets/chess.jpg)

## Running experiments
### Requirements
* Python 3.6 or higher;
* Nvidia T4 GPU or equivalent;
* TensorFlow 1.15 (see `requirements.txt`).

### Training & Evaluation
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
* Chess - Mate-In-One dataset and description is available in [wiki](https://github.com/LUMII-Syslab/Switchblade/wiki/Chess-Dataset---Mate-in-one-problems)
and should be extracted in `data_dir/mateinone` (optional);

* then run `python3 trainer.py --train --eval`. Checkpoints will be saved in `train_dir`;
* for algorithmic tasks, datasets will be generated in `data_dir` on the first run, sequential runs
will reuse this dataset;
* to evaluated generalization on larger inputs add `--gen_test` when starting training. Generalization results will be available in `train_dir/gen_test.txt`;
* you can restart training or evaluation from previous checkpoint using `--continue <checkpoint_dir>` flag.

## Task description

### Algorithmic tasks on matrices

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

### Algorithmic task on graphs
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

### Image tasks

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
Dataset and its description is availabel in our [wiki page](https://github.com/LUMII-Syslab/Switchblade/wiki/Chess-Dataset---Mate-in-one-problems).
Data preparation for this task is given in `data/mateinone.py` file.

To choose this task, in `config.py` uncomment:
```
""" Recommended settings for Mate in One (chess) task"""
model = Switchblade(feature_maps=192, block_count=2)
dataset = MateInOne()
chess = True
```

