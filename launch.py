from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import argparse
import estimator

def get_parser():
    """ CLI Argument Parser."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="nn", type=str,
        help="the ML model to be used.")
    parser.add_argument("--batch_size", default=100, type=int,
        help="size of the mini batch")
    parser.add_argument("--learning_rate", default=0.1, type=float,
        help="learning rate")
    parser.add_argument("--num_steps", default=1000, type=int,
        help="number of steps")
    parser.add_argument("--shuffle", default=True, type=bool,
        help="flag to shuffle the dataset before training")

    return parser

def main():

    # Parse the CLI Arguments.
    parser = get_parser()
    args = parser.parse_args()
    # Obtain the mnist dataset.
    mnist = input_data.read_data_sets("/tmp/data/",one_hot=False)
    print(mnist.train.images.shape)
    print(mnist.train.labels.shape)
    print(mnist.test.images.shape)
    print(mnist.test.labels.shape)
    # Get the model function.
    model = tf.estimator.Estimator(model_fn=estimator.model_fn, params=args)
    # Model Training.
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"images" : mnist.train.images}, y=mnist.train.labels, batch_size=args.batch_size,
        num_epochs=None, shuffle=args.shuffle)
    model.train(input_fn, steps=args.num_steps)
    # Model Evaluation.
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"images" : mnist.test.images}, y=mnist.test.labels, batch_size=args.batch_size,
        shuffle=args.shuffle)
    e = model.evaluate(input_fn)
    
    print("Testing Accuracy: ", e['accuracy'])

if __name__ == '__main__':
    main()
