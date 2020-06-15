# Official _TensorFlow_ implementation of Switchblade model

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


## What is Switchblade?

Switchblade is the generalization of [Neural Shuffle-Exchange](https://github.com/LUMII-Syslab/shuffle-exchange) networks to two dimensions. 
It is suitable for a broad range of problems that can be represented as a matrix and can induce O(n² log n) time complexity algorithms, where n is the length of the input matrix side.

The Switchblade model consists of cascaded Quaternary Switch and Quaternary Shuffle layers that form Beneš blocks. Additionally, Beneš blocks are enclosed by Quaternary Shuffle and Quaternary Reshape layers. Switchblade model of 2 Beneš blocks:

![](https://github.com/LUMII-Syslab/Switchblade/blob/master/assets/switchblade_model.jpg)


Quaternary Switch layer divides elements into tuples of 4 and then applies Quaternary Switch Unit (QSU) to each tuple. QSU is a learnable
4-to-4 function derived from [Residual Switch Unit](https://github.com/LUMII-Syslab/RSE), and is given as:

![](https://github.com/LUMII-Syslab/Switchblade/blob/master/assets/qsu.png)

The Quaternary Shuffle layer rearranges elements according to a cyclic digit rotation permutation. Permuatiton implemented by Quaternary Shuffle can be interpreted as splitting matrix rows into two halves 
(white and green) and interleaving the halves, then applying the same
transformation to columns (white and red). 
Example of permutation implemented by Quaternary Shuffle layer on 4×4 matrix:

![](https://github.com/LUMII-Syslab/Switchblade/blob/master/assets/quaternary_shuffle.jpg)
## Preview of results

We evaluate the Switchblade model on several 2D tasks. For image tasks, we chose two widely used benchmarks: CIFAR-10 image classification and CityScapes semantic segmentation. But our main emphasis is on hard, previously unsolved tasks: algorithmic tasks on matrices and graphs and logical reasoning tasks -- solving Sudoku puzzle and predicting the mating move in chess. All tasks were trained and evaluated on one Nvidia T4 GPU card using softmax cross-entropy loss.

### Algorithmic tasks on matrices:
We propose new datasets for 2D algorithmic tasks, where input/output data can be represented as a matrix. For matrix tasks, we chose transpose of a matrix (_Transpose_), matrix rotation by 90 degrees (_Rotate90_), elementwise XOR of two matrices (_Bitwise XOR_) and matrix multiplication by itself (_Matrix Squaring_). For graph tasks, we chose transitive path finding (_Transitivity_), connected component labeling (_Component Labeling_), and triangle labeling in the graph (_Triangle Finding_), and represent graphs as an adjacency matrix. Dataset generators for algorithmic tasks are available in [data folder](https://github.com/LUMII-Syslab/Switchblade/tree/master/data).

We train the Switchblade model on up to inputs of size 32×32 and evaluated generalization on up to size 1024×1024. Generalization results of Switchblade model on algorithmic tasks are as follows:


![](https://github.com/LUMII-Syslab/Switchblade/blob/master/assets/algorithmic.jpg)

###  Sudoku puzzle:
The Switchblade model is evaluated on the [Sudoku](https://en.wikipedia.org/wiki/Sudoku) puzzle-solving task. Our model trained on the [1M Sudoku puzzle dataset](https://github.com/Kyubyong/sudoku) achieves 100\% accuracy on the test dataset. In contrast with other Soduko solvers, that regularly use backtracking algorithms, we ask Switchblade to predict puzzle solution in one step. The following image represents an empty puzzle from the test set and its solution:


![](https://github.com/LUMII-Syslab/Switchblade/blob/master/assets/sudoku.jpg)


### Chess dataset - Mate-In-One:
We have created a chess dataset - Mate-In-One, where we consider only a single chess move  -- the last one where the white has to move a piece to checkmate. We feed a chess position where the mate is possible and train Switchblade to output the mating move. A detailed description of the dataset and download links is available in the [Mate-In-One wiki page](https://github.com/LUMII-Syslab/Switchblade/wiki/Chess-Dataset---Mate-in-one-problems). The Switchblade model obtains __72.5\%__ test accuracy on this dataset.

![](https://github.com/LUMII-Syslab/Switchblade/blob/master/assets/chess.jpg)

## Running experiments
See guide on running experiments in the [wiki](https://github.com/LUMII-Syslab/Switchblade/wiki/Running-experiments).

## Citing _Switcblade_

If you use _Switchblade_ or any of our datasets, please use the following _BibTeX_ entry:
```
@missing
```

## Contact information

For help or issues using _Switchblade_, please submit a _GitHub_ issue.

For personal communication related to _Switchblade_, please contact Emīls Ozoliņš ([emils.ozolins@lumii.lv](mailto:emils.ozolins@lumii.lv)).

