# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Position Encoding Layer as used in:
https://www.tensorflow.org/text/tutorials/transformer#positional_encoding
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer


# The official RelativePositionEmbedding concatenates the sine and cosine instead
# of interleaving.  Also doesn't add to inputs if passed in, just used for shape
# information.  So, we use this implementation.  This is numerically equivalent
# to the PyTorch implementation
# https://github.com/pytorch/examples/blob/13acec6d7c78dacd5e1fe9b0b4a325e1d39abc15/word_language_model/model.py#L65-L106
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(
        np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model
    )

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionEncoding(Layer):
    def __init__(self, max_len: int, dropout_rate: float, **kwargs):
        self.max_len = max_len
        self.dropout_rate = dropout_rate

        super().__init__(**kwargs)

    def get_config(self):
        config = {
            "max_len": self.max_len,
            "dropout_rate": self.dropout_rate,
        }

        config.update(super().get_config())

        return config

    def build(self, input_shape):
        emb_size = input_shape[-1]
        pe = positional_encoding(self.max_len, emb_size)
        self.pe = tf.Variable(pe, trainable=False)

        self.built = True

    def call(self, inputs, training=False):
        # Expect inputs to have shape (B, T, H) - may have a sequence length
        # that is less than or up to the max length.
        tf.debugging.assert_shapes(
            [
                (inputs, ("B", "T1", "H")),
                (self.pe, ("T2", "H")),
            ]
        )

        _, seq_len, _ = tf.shape(inputs)
        tf.assert_less(seq_len, self.max_len, "Have an OOB sequence")

        out = inputs + self.pe[tf.newaxis, :seq_len, :]

        if training:
            out = tf.nn.dropout(out, self.dropout_rate)

        return out
