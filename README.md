# mnist_playground
Implementation of various models on the mnist dataset.

Models currently implemented : -nn
                               -dnn
                               -cnn 

                              

usage: launch.py [-h] [--arch ARCH] [--batch_size BATCH_SIZE]
                 [--learning_rate LEARNING_RATE] [--num_steps NUM_STEPS]
                 [--shuffle SHUFFLE]

optional arguments:
  -h, --help            show this help message and exit
  --arch ARCH           the ML model to be used.
  --batch_size BATCH_SIZE
                        size of the mini batch
  --learning_rate LEARNING_RATE
                        learning rate
  --num_steps NUM_STEPS
                        number of steps
  --shuffle SHUFFLE     flag to shuffle the dataset before training

