import argparse
import os
import tensorflow as tf
from model import LicensePlatesCNN
from utils import setup_custom_logger


log = setup_custom_logger(os.path.basename(__file__))


def eval(test_data, store_results_path, input_channels=3, checkpoint_dir="checkpoint", summary_dir="summary"):
    with tf.Session() as sess:
        cnn = LicensePlatesCNN(sess=sess,
                               checkpoint_dir=checkpoint_dir,
                               summary_dir=summary_dir,
                               input_channels=input_channels)

        # Load the trained weights
        if not cnn.load():
            log.error("Unable to restore model from checkpoint")
            return

        # Feed test data through network
        cnn.evaluate(test_data, store_results_path=store_results_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("test_data", type=str, help="Path to test data")
    parser.add_argument("store_results_path", type=str, help="Where to store predictions", default=None)
    parser.add_argument("--input_channels", type=int, help="Number of input channels. 1 for grayscale, 3 for color", default=3)
    parser.add_argument("--checkpoint_dir", type=str, help="Directory where to read checkpoint from", default="checkpoint")
    parser.add_argument("--summary_dir", type=str, help="Directory where to store evaluation summary", default="summary")
    args = vars(parser.parse_args())

    eval(**args)
