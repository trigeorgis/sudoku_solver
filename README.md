# Sudoku Solver

This project contains a simple sudoku parser and solver.


## Installation

This project depends on the [Menpo Project](http://www.menpo.org/),
which is multi-platform (Linux, Windows, OS X). As explained in [Menpo's
installation isntructions](http://www.menpo.org/installation/), it is
highly recommended to use
[conda](http://conda.pydata.org/miniconda.html) as your Python
distribution.

Once downloading and installing
[conda](http://conda.pydata.org/miniconda.html), this project can be
installed by:

**Step 1:** Create a new conda environment and activate it:
```console
$ conda create -n sudoku_solver python=2.7
$ source activate sudoku_solver
``` 

**Step 2:** Install [menpo](http://www.menpo.org/menpo/) and
[menpowidgets](http://www.menpo.org/menpowidgets/) and opencv from the menpo
channel: 
```console
(sudoku_solver)$ conda install -c menpo menpo menpowidgets opencv
```

**Step 3:** Install the [TensorFlow](https://www.tensorflow.org/)
dependencies following the [installation
instructions](https://www.tensorflow.org/versions/r0.10/get_started/index.html).

For Ubuntu/Linux the wheel file is
```console
(sudoku_solver) export TF_SERVER=https://storage.googleapis.com/tensorflow/linux/cpu/
(sudoku_solver) export TF_BINARY_NAME=tensorflow-0.10.0-cp27-none-linux_x86_64.whl
(sudoku_solver) pip install --ignore-installed --upgrade $TF_SERVER/$TF_BINARY_URL
```

## Training the convnet classifier (optional)

To train our convnet we use the traditional MNIST dataset for
handwritten digits [1] and a very small dataset [2] of cells from sudoku
puzzles.

``[1] http://yann.lecun.com/exdb/mnist/``
``[2]
https://github.com/eatonk/sudoku-image-solver/tree/master/ocr_data``

We use a small network which is comprised of 3 hidden layers -- 2
convolutionals and a fully-connected one. Each of this is followed by a
batchnorm and a ReLU nonlinearity. Finally the task loss is to maximize
the crossentropy for our 11 classes (10 for the digits and one for the
background).

```console
(sudoku_solver) python digit_train.py
```

We train the above network for about 200k gradient steps, at which point we reach a plataeu
on the accuracy on our validation set.

Having training the model we freeze the final weights to a single
TensorFlow graph using the
[freeze_graph.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py)
script contained in the TensorFlow framework.

## Running the solver

The main code for this project is in the [``Sudoku
Solver.ipynb``](https://github.com/trigeorgis/sudoku_solver/blob/master/Sudoku%20Solver.ipynb) file
which is a documented notebook which goes through the whole process of
creating the sudoku solver.
