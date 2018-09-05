from matplotlib.ticker import FuncFormatter
from itertools import chain
import numpy as np
import logging
import string
import sys


def setup_custom_logger(name):
    # Source: https://stackoverflow.com/questions/7621897/python-logging-module-globally
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    return logger


def visualize_predictions(ax, char_probabilities, char_labels=None):
    """
    Draws the top-5 predictions as bar plot for each character
    :param ax: axis to draw on
    :param char_probabilities: prediction probabilities as matrix of shape 7 x 37
    :param char_labels: one-hot encoded labels of shape 7 x 37
    """
    # Some constants
    # Maximum number of characters is 7
    max_length = 7
    bar_width = 0.7
    bar_default_color = "#2C3E50"
    bar_highlight_correct_color = "#18CDCA"
    bar_highlight_incorrect_color = "#FC4349"
    chars = np.array(list(string.ascii_uppercase + string.digits + "_"))
    y_axis_formatter = FuncFormatter(lambda y, _: "{:.0%}".format(y))

    # Collect x-axis ticks and labels
    probabilities_ax_xticks = []
    probabilities_ax_xticklabels = []

    # Clear prediction axis
    # Traverse patches in reverse order to clear them immediately
    for p in reversed(ax.patches):
        p.remove()

    # Iterate over character positions
    for pos in range(max_length):
        # Find the 5 most likely characters at the current position
        most_likely_char_indices = np.argsort(-char_probabilities[pos])[:5]
        # Map most likely prediction to corresponding character
        most_likely_chars = list(
            map(lambda ind: chars[ind] if ind < 36 else r"$\diamond$", most_likely_char_indices))

        axis_char_offset = pos * 7
        indices = np.arange(5)
        rects = ax.bar(axis_char_offset + indices, char_probabilities[pos][most_likely_char_indices], bar_width,
                       color=bar_default_color, align="center", zorder=1)

        if char_labels is not None:
            # Given label annotations, colorize bars
            # Undo one-hot encoding
            correct_char_index = np.argmax(char_labels[pos])
            correct_classification_idx = np.where(most_likely_char_indices == correct_char_index)[0]

            # If the correct label is among the top-5 most likely characters
            if len(correct_classification_idx) > 0:
                correct_classification_idx = correct_classification_idx[0]
                if correct_classification_idx == 0:
                    # If the most likely character is the correct one, set "correct color"
                    rects[correct_classification_idx].set_color(bar_highlight_correct_color)
                else:
                    # If the most likely character is not the correct one, set "incorrect color"
                    rects[correct_classification_idx].set_color(bar_highlight_incorrect_color)

                # Bold tick label for correct character
                if correct_char_index == 36:
                    most_likely_chars[correct_classification_idx] = r"$\bm{\diamond}$"
                else:
                    most_likely_chars[correct_classification_idx] = r"\textbf{{{}}}".format(
                        most_likely_chars[correct_classification_idx])

        probabilities_ax_xticks.append(axis_char_offset + indices)
        probabilities_ax_xticklabels.append(most_likely_chars)

    # Unravel and set x-axis ticks and labels
    ax.set_xticks(list(chain(*probabilities_ax_xticks)))
    ax.set_xticklabels(list(chain(*probabilities_ax_xticklabels)))
    ax.tick_params(axis='x', labelsize=8)
    ax.yaxis.set_major_formatter(y_axis_formatter)
    ax.set_ylabel("Probability (\%)")
