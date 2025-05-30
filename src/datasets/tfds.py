# This code is based on materials from the Big Vision [https://github.com/google-research/big_vision].
# Thanks to Big Vision  for their contributions to the field of computer vision and for their open-source contributions to this project.
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

"""TensorFlow Datasets as data source for big_vision."""
import functools

import jax
import overrides
import tensorflow_datasets as tfds

from src.datasets import core as ds_core


class DataSource(ds_core.DataSource):
  """Use TFDS as a data source."""

  def __init__(self, name, split, data_dir=None, skip_decode=("image",)):
    if 'imagenet2012' in name:
        self.builder = _get_builder(name, data_dir)
    elif 'cifar' in name:
        self.builder = tfds.builder(name)
        self.builder.download_and_prepare()
    elif 'coco' in name:
        self.builder = tfds.builder(name)
        self.builder.download_and_prepare()
    else:
        self.builder = tfds.builder_from_directory(data_dir)
    self.split = split
    # Each host is responsible for a fixed subset of data
    process_splits = tfds.even_splits(split, jax.process_count())
    self.process_split = process_splits[jax.process_index()]
    self.skip_decoders = {
        f: tfds.decode.SkipDecoding()
        for f in skip_decode
        if f in self.builder.info.features
    }

  @overrides.overrides
  def get_tfdata(self, ordered=False):
    return self.builder.as_dataset(
        split=self.process_split,
        shuffle_files=not ordered,
        read_config=tfds.ReadConfig(
            skip_prefetch=True,  # We prefetch after pipeline.
            try_autocache=False,  # We control this, esp. for few-shot.
            add_tfds_id=True,
        ),
        decoders=self.skip_decoders)

  @property
  @overrides.overrides
  def total_examples(self):
    return self.builder.info.splits[self.split].num_examples

  @overrides.overrides
  def num_examples_per_process(self, nprocess=None):
    splits = tfds.even_splits(self.split, nprocess or jax.process_count())
    return [self.builder.info.splits[s].num_examples for s in splits]


@functools.lru_cache(maxsize=None)
def _get_builder(dataset, data_dir):
  return tfds.builder(dataset, data_dir=data_dir, try_gcs=True)
