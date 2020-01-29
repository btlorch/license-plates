import argparse
import tensorflow as tf
from model import LicensePlatesCNN


parser = argparse.ArgumentParser()
parser.add_argument("training_data", type=str, help="Path to training data")
parser.add_argument("validation_data", type=str, help="Path to validation data")
parser.add_argument("--input_channels", type=int, help="Number of input channels. 1 for grayscale, 3 for color",
                    default=3)
parser.add_argument("--checkpoint_dir", type=str, help="Directory where to store checkpoint", default="checkpoint")
parser.add_argument("--summary_dir", type=str, help="Directory where to store summary", default="summary")
args = vars(parser.parse_args())

with tf.compat.v1.Session() as sess:
    cnn = LicensePlatesCNN(sess,
                           args["checkpoint_dir"],
                           args["summary_dir"],
                           input_channels=args["input_channels"])
    cnn.train(args["training_data"], args["validation_data"])
