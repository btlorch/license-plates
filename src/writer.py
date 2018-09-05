from utils import setup_custom_logger
import numpy as np
import itertools
import h5py
import os


# Set up constants for data sets
DATA_IMAGES = "images"
DATA_LENGTH_LABELS = "length_labels"
DATA_CHAR_LABELS = "char_labels"
DATA_LICENSE_NUMBERS = "license_numbers"
DATA_SCALING_TARGET_WIDTHS = "scaling_target_widths"
DATA_NOISE_SNRS_DB = "noise_SNRs_db"
DATA_LENGTH_PROBABILITIES = "length_probabilities"
DATA_CHAR_PROBABILITIES = "char_probabilities"
DATA_ROTATION_ANGLES = "rotation_angles"
DATA_FILE_PATHS = "file_path"
DATA_CLASS_LABELS = "class_labels"
DATA_CLASS_PROBABILITIES = "class_probabilities"


REQUIRED_DATA_SETS = [DATA_IMAGES, DATA_LENGTH_LABELS, DATA_CHAR_LABELS, DATA_LICENSE_NUMBERS, DATA_SCALING_TARGET_WIDTHS, DATA_NOISE_SNRS_DB]
ALL_DATA_SETS = [DATA_IMAGES, DATA_LENGTH_LABELS, DATA_CHAR_LABELS, DATA_LICENSE_NUMBERS, DATA_SCALING_TARGET_WIDTHS, DATA_NOISE_SNRS_DB, DATA_LENGTH_PROBABILITIES, DATA_CHAR_PROBABILITIES, DATA_ROTATION_ANGLES, DATA_CLASS_LABELS, DATA_CLASS_PROBABILITIES]


log = setup_custom_logger(os.path.basename(__file__))


class BufferedWriter(object):
    def __init__(self, output_filename, chunk_size=256, disable_keys_check=False):
        self._output_filename = output_filename
        self._disabled_keys_check = disable_keys_check
        self._chunk_size = chunk_size

        # Buffering data structures for incremental writes
        self._written_items_count = 0
        # Set up empty list for each buffer
        self._wait_for_write_buffer = dict((key, []) for key in ALL_DATA_SETS)
        self._num_samples_in_write_buffer = 0
        self._chunk_remainder_buffer = dict()
        self._num_samples_in_chunk_remainder_buffer = 0

    @staticmethod
    def _count_samples(buffers_dict):
        """
        Counts the number of samples in the given dictionary.
        Assumes that the number of samples is the same for each data set.
        Empty data sets are ignored.
        :param buffers_dict: dictionary with different data sets
        :return: 0 if dictionary is empty or the number of samples
        """
        if not buffers_dict:
            return 0

        for key in list(buffers_dict.keys()):
            num_items = len(buffers_dict[key])
            # There might be empty data sets in the buffer which we want to ignore
            if 0 != num_items:
                return num_items

        return 0

    def _concat_buffers(self):
        """
        Concatenates the items in the wait_for_write_buffer itself.
        Concatenates the result with the remaining items in the _chunk_remainder.buffer.
        :return: Dictionary of data sets, where each value is either a flat list or an ndarray with the number of samples in the first dimension
        """
        if self._num_samples_in_write_buffer == 0:
            return self._chunk_remainder_buffer

        # Concatenate samples from list
        concat_buffer = dict()
        for key, data_buffer in self._wait_for_write_buffer.items():
            # Ignore empty data buffers
            if len(data_buffer) == 0:
                continue

            if isinstance(data_buffer[0], list):
                concat_buffer[key] = list(itertools.chain.from_iterable(data_buffer))
            elif isinstance(data_buffer[0], np.ndarray):
                concat_buffer[key] = np.concatenate(data_buffer)
            else:
                log.error("Unexpected buffer type")
                raise ValueError("Unexpected buffer type")

        # Concatenate with remaining data from previous write
        if self._num_samples_in_chunk_remainder_buffer > 0:
            for key, data_buffer in concat_buffer.items():
                if isinstance(data_buffer, list):
                    concat_buffer[key] = self._chunk_remainder_buffer[key] + data_buffer
                elif isinstance(data_buffer, np.ndarray):
                    concat_buffer[key] = np.concatenate((self._chunk_remainder_buffer[key], data_buffer))
                else:
                    log.error("Unexpected buffer type")
                    raise ValueError("Unexpected buffer type")

        return concat_buffer

    def write(self, batch):
        """
        Writes the given batch to a local buffer.
        Once the local buffer reaches the predefined chunk size, multiples of the chunk size are written to disk.
        The remaining items are kep in the chunk_remainder_buffer.
        :param batch: dictionary with data sets as list
        :return:
        """

        num_batch_samples = self._count_samples(batch)

        if not self._disabled_keys_check:
            # Make sure that the batch contains at least all required keys
            for key in REQUIRED_DATA_SETS:
                assert key in batch, "Given batch does not contain key {}".format(key)

        # Copy batch data sets into local buffers
        for key, data_buffer in batch.items():
            if not self._disabled_keys_check:
                # Make sure that the batch does not contain unknown keys
                assert key in ALL_DATA_SETS, "Batch contains unknown key {}".format(key)

            # Append to list
            self._wait_for_write_buffer[key].append(data_buffer)

        self._num_samples_in_write_buffer += num_batch_samples

        total_num_samples = self._num_samples_in_write_buffer + self._num_samples_in_chunk_remainder_buffer
        # Once we have reached to chunk size, write as many multiples of the chunk size as possible to disk.
        if total_num_samples >= self._chunk_size:
            concat_buffer = self._concat_buffers()

            while total_num_samples >= self._chunk_size:
                # Extract a chunk of data
                chunk = dict()
                for key, data_buffer in concat_buffer.items():
                    # Move chunk into new buffer
                    chunk[key] = data_buffer[:self._chunk_size]
                    # Remove chunk from buffer
                    concat_buffer[key] = data_buffer[self._chunk_size:]

                self._write(chunk)
                total_num_samples -= self._chunk_size

            self._chunk_remainder_buffer = concat_buffer
            # Reset write buffer
            for key in ALL_DATA_SETS:
                self._wait_for_write_buffer[key].clear()

            self._num_samples_in_write_buffer = 0
            self._num_samples_in_chunk_remainder_buffer = self._count_samples(self._chunk_remainder_buffer)

    def _write(self, chunk):
        """

        :param chunk: data already concatenated and ready to be written
        :return:
        """
        chunk_size = self._count_samples(chunk)
        if 0 == chunk_size:
            return

        if 0 == self._written_items_count:
            # Create new output file
            with h5py.File(self._output_filename, "w") as f:
                # Initialize a resizable data set to hold the output
                for key, data_buffer in chunk.items():
                    kwargs = {}
                    if isinstance(data_buffer, list):
                        shape = (chunk_size,)
                        maxshape = (None,)
                        # Select dtype based on data's type
                        if isinstance(data_buffer[0], (np.integer, int)):
                            dtype = int
                        elif isinstance(data_buffer[0], (np.float, float)):
                            dtype = float
                        else:
                            dtype = h5py.special_dtype(vlen=bytes)

                    elif isinstance(data_buffer, np.ndarray):
                        shape = data_buffer.shape
                        maxshape = (None,) + data_buffer.shape[1:]
                        dtype = data_buffer.dtype
                        kwargs["compression"] = "gzip"
                        kwargs["chunks"] = (self._chunk_size,) + data_buffer.shape[1:]
                    else:
                        log.error("Unknown item type")

                    # Set up data set
                    dataset = f.create_dataset(key, shape=shape, maxshape=maxshape, dtype=dtype, **kwargs)
                    dataset[:] = data_buffer

        else:
            # Append to existing output file
            with h5py.File(self._output_filename, "a") as f:
                for key, data_buffer in chunk.items():
                    dataset = f[key]
                    # Resize the data set to accommodate the next chunk of rows
                    dataset.resize(self._written_items_count + chunk_size, axis=0)
                    # Write the next chunk
                    dataset[self._written_items_count:] = data_buffer

        # Increment the row counter
        self._written_items_count += chunk_size
        log.debug("Written {:d} items in total".format(self._written_items_count))

    def flush(self):
        if 0 == self._num_samples_in_write_buffer + self._num_samples_in_chunk_remainder_buffer:
            return

        chunk = self._concat_buffers()
        self._write(chunk)
        self._num_samples_in_write_buffer = 0
        self._num_samples_in_chunk_remainder_buffer = 0
