# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.4
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# + [markdown] id="Jxv6goXm7oGF"
# ##### Copyright 2018 The TensorFlow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

# + cellView="form" id="llMNufAK7nfK"
#@title Licensed under the Apache License, Version 2.0 (the "License"); { display-mode: "form" }
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# + [markdown] id="8Byow2J6LaPl"
# # tf.data: Build TensorFlow input pipelines

# + [markdown] id="kGXS3UWBBNoc"
# <table class="tfo-notebook-buttons" align="left">
#   <td>
#     <a target="_blank" href="https://www.tensorflow.org/guide/data"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/guide/data.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://github.com/tensorflow/docs/blob/master/site/en/guide/data.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
#   </td>
#   <td>
#     <a href="https://storage.googleapis.com/tensorflow_docs/docs/site/en/guide/data.ipynb"><img src="https://www.tensorflow.org/images/download_logo_32px.png" />Download notebook</a>
#   </td>
# </table>

# + [markdown] id="9Qo3HgDjbDcI"
# The `tf.data` API enables you to build complex input pipelines from simple,
# reusable pieces. For example, the pipeline for an image model might aggregate
# data from files in a distributed file system, apply random perturbations to each
# image, and merge randomly selected images into a batch for training. The
# pipeline for a text model might involve extracting symbols from raw text data,
# converting them to embedding identifiers with a lookup table, and batching
# together sequences of different lengths. The `tf.data` API makes it possible to
# handle large amounts of data, read from different data formats, and perform
# complex transformations.
#
# The `tf.data` API introduces a `tf.data.Dataset` abstraction that represents a
# sequence of elements, in which each element consists of one or more components.
# For example, in an image pipeline, an element might be a single training
# example, with a pair of tensor components representing the image and its label.
#
# There are two distinct ways to create a dataset:
#
# *   A data **source** constructs a `Dataset` from data stored in memory or in
#     one or more files.
#
# *   A data **transformation** constructs a dataset from one or more
#     `tf.data.Dataset` objects.
#

# + id="UJIEjEIBdf-h"
import tensorflow as tf

# + id="7Y0JtWBNR9E5"
import pathlib
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

np.set_printoptions(precision=4)

# + [markdown] id="0l4a0ALxdaWF"
# ## Basic mechanics
# <a id="basic-mechanics"/>
#
# To create an input pipeline, you must start with a data *source*. For example,
# to construct a `Dataset` from data in memory, you can use
# `tf.data.Dataset.from_tensors()` or `tf.data.Dataset.from_tensor_slices()`.
# Alternatively, if your input data is stored in a file in the recommended
# TFRecord format, you can use `tf.data.TFRecordDataset()`.
#
# Once you have a `Dataset` object, you can *transform* it into a new `Dataset` by
# chaining method calls on the `tf.data.Dataset` object. For example, you can
# apply per-element transformations such as `Dataset.map()`, and multi-element
# transformations such as `Dataset.batch()`. See the documentation for
# `tf.data.Dataset` for a complete list of transformations.
#
# The `Dataset` object is a Python iterable. This makes it possible to consume its
# elements using a for loop:

# + id="0F-FDnjB6t6J"
dataset = tf.data.Dataset.from_tensor_slices([8, 3, 0, 8, 2, 1])
dataset

# + id="pwJsRJ-FbDcJ"
for elem in dataset:
  print(elem.numpy())

# + [markdown] id="m0yy80MobDcM"
# Or by explicitly creating a Python iterator using `iter` and consuming its
# elements using `next`:

# + id="03w9oxFfbDcM"
it = iter(dataset)

print(next(it).numpy())

# + [markdown] id="Q4CgCL8qbDcO"
# Alternatively, dataset elements can be consumed using the `reduce`
# transformation, which reduces all elements to produce a single result. The
# following example illustrates how to use the `reduce` transformation to compute
# the sum of a dataset of integers.

# + id="C2bHAeNxbDcO"
print(dataset.reduce(0, lambda state, value: state + value).numpy())

# + [markdown] id="B2Fzwt2nbDcR"
# <!-- TODO(jsimsa): Talk about `tf.function` support. -->
#
# <a id="dataset_structure"></a>
# ### Dataset structure
#
# A dataset produces a sequence of *elements*, where each element is
# the same (nested) structure of *components*. Individual components
# of the structure can be of any type representable by
# `tf.TypeSpec`, including `tf.Tensor`, `tf.sparse.SparseTensor`,
# `tf.RaggedTensor`, `tf.TensorArray`, or `tf.data.Dataset`.
#
# The Python constructs that can be used to express the (nested)
# structure of elements include `tuple`, `dict`, `NamedTuple`, and
# `OrderedDict`. In particular, `list` is not a valid construct for
# expressing the structure of dataset elements. This is because
# early tf.data users felt strongly about `list` inputs (e.g. passed
# to `tf.data.Dataset.from_tensors`) being automatically packed as
# tensors and `list` outputs (e.g. return values of user-defined
# functions) being coerced into a `tuple`. As a consequence, if you
# would like a `list` input to be treated as a structure, you need
# to convert it into `tuple` and if you would like a `list` output
# to be a single component, then you need to explicitly pack it
# using `tf.stack`.
#
# The `Dataset.element_spec` property allows you to inspect the type
# of each element component. The property returns a *nested structure*
# of `tf.TypeSpec` objects, matching the structure of the element,
# which may be a single component, a tuple of components, or a nested
# tuple of components. For example:

# + id="Mg0m1beIhXGn"
dataset1 = tf.data.Dataset.from_tensor_slices(tf.random.uniform([4, 10]))

dataset1.element_spec

# + id="cwyemaghhXaG"
dataset2 = tf.data.Dataset.from_tensor_slices(
   (tf.random.uniform([4]),
    tf.random.uniform([4, 100], maxval=100, dtype=tf.int32)))

dataset2.element_spec

# + id="1CL7aB0ahXn_"
dataset3 = tf.data.Dataset.zip((dataset1, dataset2))

dataset3.element_spec

# + id="m5bz7R1xhX1f"
# Dataset containing a sparse tensor.
dataset4 = tf.data.Dataset.from_tensors(tf.SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4]))

dataset4.element_spec

# + id="lVOPHur_hYQv"
# Use value_type to see the type of value represented by the element spec
dataset4.element_spec.value_type

# + [markdown] id="r5xNsFFvhUnr"
# The `Dataset` transformations support datasets of any structure. When using the
# `Dataset.map()`, and `Dataset.filter()` transformations,
# which apply a function to each element, the element structure determines the
# arguments of the function:

# + id="2myAr3Pxd-zF"
dataset1 = tf.data.Dataset.from_tensor_slices(
    tf.random.uniform([4, 10], minval=1, maxval=10, dtype=tf.int32))

dataset1

# + id="woPXMP14gUTg"
for z in dataset1:
  print(z.numpy())

# + id="53PA4x6XgLar"
dataset2 = tf.data.Dataset.from_tensor_slices(
   (tf.random.uniform([4]),
    tf.random.uniform([4, 100], maxval=100, dtype=tf.int32)))

dataset2

# + id="2ju4sNSebDcR"
dataset3 = tf.data.Dataset.zip((dataset1, dataset2))

dataset3

# + id="BgxsfAS2g6gk"
for a, (b,c) in dataset3:
  print('shapes: {a.shape}, {b.shape}, {c.shape}'.format(a=a, b=b, c=c))

# + [markdown] id="M1s2K0g-bDcT"
# ## Reading input data
#

# + [markdown] id="F3JG2f0h2683"
# ### Consuming NumPy arrays
#
# See [Loading NumPy arrays](../tutorials/load_data/numpy.ipynb) for more examples.
#
# If all of your input data fits in memory, the simplest way to create a `Dataset`
# from them is to convert them to `tf.Tensor` objects and use
# `Dataset.from_tensor_slices()`.

# + id="NmaE6PjjhQ47"
train, test = tf.keras.datasets.fashion_mnist.load_data()

# + id="J6cNiuDBbDcU"
images, labels = train
images = images/255

dataset = tf.data.Dataset.from_tensor_slices((images, labels))
dataset


# + [markdown] id="XkwrDHN5bDcW"
# Note: The above code snippet will embed the `features` and `labels` arrays
# in your TensorFlow graph as `tf.constant()` operations. This works well for a
# small dataset, but wastes memory---because the contents of the array will be
# copied multiple times---and can run into the 2GB limit for the `tf.GraphDef`
# protocol buffer.

# + [markdown] id="pO4ua2gEmIhR"
# ### Consuming Python generators
#
# Another common data source that can easily be ingested as a `tf.data.Dataset` is the python generator.
#
# Caution: While this is a convienient approach it has limited portability and scalibility. It must run in the same python process that created the generator, and is still subject to the Python [GIL](https://en.wikipedia.org/wiki/Global_interpreter_lock).

# + id="9njpME-jmDza"
def count(stop):
  i = 0
  while i<stop:
    yield i
    i += 1


# + id="xwqLrjnTpD8Y"
for n in count(5):
  print(n)

# + [markdown] id="D_BB_PhxnVVx"
# The `Dataset.from_generator` constructor converts the python generator to a fully functional `tf.data.Dataset`.
#
# The constructor takes a callable as input, not an iterator. This allows it to restart the generator when it reaches the end. It takes an optional `args` argument, which is passed as the callable's arguments.
#
# The `output_types` argument is required because `tf.data` builds a `tf.Graph` internally, and graph edges require a `tf.dtype`.

# + id="GFga_OTwm0Je"
ds_counter = tf.data.Dataset.from_generator(count, args=[25], output_types=tf.int32, output_shapes = (), )

# + id="fel1SUuBnDUE"
for count_batch in ds_counter.repeat().batch(10).take(10):
  print(count_batch.numpy())


# + [markdown] id="wxy9hDMTq1zD"
# The `output_shapes` argument is not *required* but is highly recomended as many tensorflow operations do not support tensors with unknown rank. If the length of a particular axis is unknown or variable, set it as `None` in the `output_shapes`.
#
# It's also important to note that the `output_shapes` and `output_types` follow the same nesting rules as other dataset methods.
#
# Here is an example generator that demonstrates both aspects, it returns tuples of arrays, where the second array is a vector with unknown length.

# + id="allFX1g8rGKe"
def gen_series():
  i = 0
  while True:
    size = np.random.randint(0, 10)
    yield i, np.random.normal(size=(size,))
    i += 1


# + id="6Ku26Yb9rcJX"
for i, series in gen_series():
  print(i, ":", str(series))
  if i > 5:
    break

# + [markdown] id="LmkynGilx0qf"
# The first output is an `int32` the second is a `float32`.
#
# The first item is a scalar, shape `()`, and the second is a vector of unknown length, shape `(None,)` 

# + id="zDTfhEzhsliM"
ds_series = tf.data.Dataset.from_generator(
    gen_series, 
    output_types=(tf.int32, tf.float32), 
    output_shapes=((), (None,)))

ds_series

# + [markdown] id="WWxvSyQiyN0o"
# Now it can be used like a regular `tf.data.Dataset`. Note that when batching a dataset with a variable shape, you need to use `Dataset.padded_batch`.

# + id="A7jEpj3As1lO"
ds_series_batch = ds_series.shuffle(20).padded_batch(10)

ids, sequence_batch = next(iter(ds_series_batch))
print(ids.numpy())
print()
print(sequence_batch.numpy())

# + [markdown] id="_hcqOccJ1CxG"
# For a more realistic example, try wrapping `preprocessing.image.ImageDataGenerator` as a `tf.data.Dataset`.
#
# First download the data:

# + id="g-_JCFRQ1CXM"
flowers = tf.keras.utils.get_file(
    'flower_photos',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    untar=True)

# + [markdown] id="UIjPhvQ87jUT"
# Create the `image.ImageDataGenerator`

# + id="vPCZeBQE5DfH"
img_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, rotation_range=20)

# + id="my4PxqfH26p6"
images, labels = next(img_gen.flow_from_directory(flowers))

# + id="Hd96nH1w3eKH"
print(images.dtype, images.shape)
print(labels.dtype, labels.shape)

# + id="KvRwvt5E2rTH"
ds = tf.data.Dataset.from_generator(
    lambda: img_gen.flow_from_directory(flowers), 
    output_types=(tf.float32, tf.float32), 
    output_shapes=([32,256,256,3], [32,5])
)

ds.element_spec

# + id="LcaULBCXj_2_"
for images, label in ds.take(1):
  print('images.shape: ', images.shape)
  print('labels.shape: ', labels.shape)


# + [markdown] id="ma4XoYzih2f4"
# ### Consuming TFRecord data
#
# See [Loading TFRecords](../tutorials/load_data/tfrecord.ipynb) for an end-to-end example.
#
# The `tf.data` API supports a variety of file formats so that you can process
# large datasets that do not fit in memory. For example, the TFRecord file format
# is a simple record-oriented binary format that many TensorFlow applications use
# for training data. The `tf.data.TFRecordDataset` class enables you to
# stream over the contents of one or more TFRecord files as part of an input
# pipeline.

# + [markdown] id="LiatWUloRJc4"
# Here is an example using the test file from the French Street Name Signs (FSNS).

# + id="jZo_4fzdbDcW"
# Creates a dataset that reads all of the examples from two files.
fsns_test_file = tf.keras.utils.get_file("fsns.tfrec", "https://storage.googleapis.com/download.tensorflow.org/data/fsns-20160927/testdata/fsns-00000-of-00001")

# + [markdown] id="seD5bOH3RhBP"
# The `filenames` argument to the `TFRecordDataset` initializer can either be a
# string, a list of strings, or a `tf.Tensor` of strings. Therefore if you have
# two sets of files for training and validation purposes, you can create a factory
# method that produces the dataset, taking filenames as an input argument:
#

# + id="e2WV5d7DRUA-"
dataset = tf.data.TFRecordDataset(filenames = [fsns_test_file])
dataset

# + [markdown] id="62NC3vz9U8ww"
# Many TensorFlow projects use serialized `tf.train.Example` records in their TFRecord files. These need to be decoded before they can be inspected:

# + id="3tk29nlMl5P3"
raw_example = next(iter(dataset))
parsed = tf.train.Example.FromString(raw_example.numpy())

parsed.features.feature['image/text']

# + [markdown] id="qJAUib10bDcb"
# ### Consuming text data
#
# See [Loading Text](../tutorials/load_data/text.ipynb) for an end to end example.
#
# Many datasets are distributed as one or more text files. The
# `tf.data.TextLineDataset` provides an easy way to extract lines from one or more
# text files. Given one or more filenames, a `TextLineDataset` will produce one
# string-valued element per line of those files.

# + id="hQMoFu2TbDcc"
directory_url = 'https://storage.googleapis.com/download.tensorflow.org/data/illiad/'
file_names = ['cowper.txt', 'derby.txt', 'butler.txt']

file_paths = [
    tf.keras.utils.get_file(file_name, directory_url + file_name)
    for file_name in file_names
]

# + id="il4cOjiVwj95"
dataset = tf.data.TextLineDataset(file_paths)

# + [markdown] id="MevIbDiwy4MC"
# Here are the first few lines of the first file:

# + id="vpEHKyvHxu8A"
for line in dataset.take(5):
  print(line.numpy())

# + [markdown] id="lJyVw8ro7fey"
# To alternate lines between files use `Dataset.interleave`. This makes it easier to shuffle files together. Here are the first, second and third lines from each translation:

# + id="1UCveWOt7fDE"
files_ds = tf.data.Dataset.from_tensor_slices(file_paths)
lines_ds = files_ds.interleave(tf.data.TextLineDataset, cycle_length=3)

for i, line in enumerate(lines_ds.take(9)):
  if i % 3 == 0:
    print()
  print(line.numpy())

# + [markdown] id="2F_pOIDubDce"
# By default, a `TextLineDataset` yields *every* line of each file, which may
# not be desirable, for example, if the file starts with a header line, or contains comments. These lines can be removed using the `Dataset.skip()` or
# `Dataset.filter()` transformations. Here, you skip the first line, then filter to
# find only survivors.

# + id="X6b20Gua2jPO"
titanic_file = tf.keras.utils.get_file("train.csv", "https://storage.googleapis.com/tf-datasets/titanic/train.csv")
titanic_lines = tf.data.TextLineDataset(titanic_file)

# + id="5M1pauNT68B2"
for line in titanic_lines.take(10):
  print(line.numpy())


# + id="dEIP95cibDcf"
def survived(line):
  return tf.not_equal(tf.strings.substr(line, 0, 1), "0")

survivors = titanic_lines.skip(1).filter(survived)

# + id="odQ4618h1XqD"
for line in survivors.take(10):
  print(line.numpy())

# + [markdown] id="x5z5B11UjDTd"
# ### Consuming CSV data

# + [markdown] id="ChDHNi3qbDch"
# See [Loading CSV Files](../tutorials/load_data/csv.ipynb), and [Loading Pandas DataFrames](../tutorials/load_data/pandas_dataframe.ipynb) for more examples. 
#
# The CSV file format is a popular format for storing tabular data in plain text.
#
# For example:

# + id="kj28j5u49Bjm"
titanic_file = tf.keras.utils.get_file("train.csv", "https://storage.googleapis.com/tf-datasets/titanic/train.csv")

# + id="ghvtmW40LM0B"
df = pd.read_csv(titanic_file)
df.head()

# + [markdown] id="J9uBqt5oGsR-"
# If your data fits in memory the same `Dataset.from_tensor_slices` method works on dictionaries, allowing this data to be easily imported:

# + id="JmAMCiPJA0qO"
titanic_slices = tf.data.Dataset.from_tensor_slices(dict(df))

for feature_batch in titanic_slices.take(1):
  for key, value in feature_batch.items():
    print("  {!r:20s}: {}".format(key, value))

# + [markdown] id="47yippqaHFk6"
# A more scalable approach is to load from disk as necessary. 
#
# The `tf.data` module provides methods to extract records from one or more CSV files that comply with [RFC 4180](https://tools.ietf.org/html/rfc4180).
#
# The `experimental.make_csv_dataset` function is the high level interface for reading sets of csv files. It supports column type inference and many other features, like batching and shuffling, to make usage simple.

# + id="zHUDrM_s_brq"
titanic_batches = tf.data.experimental.make_csv_dataset(
    titanic_file, batch_size=4,
    label_name="survived")

# + id="TsZfhz79_Wlg"
for feature_batch, label_batch in titanic_batches.take(1):
  print("'survived': {}".format(label_batch))
  print("features:")
  for key, value in feature_batch.items():
    print("  {!r:20s}: {}".format(key, value))

# + [markdown] id="k_5N7CdNGYAa"
# You can use the `select_columns` argument if you only need a subset of columns.

# + id="H9KNHyDwF2Sc"
titanic_batches = tf.data.experimental.make_csv_dataset(
    titanic_file, batch_size=4,
    label_name="survived", select_columns=['class', 'fare', 'survived'])

# + id="7C2uosFnGIT8"
for feature_batch, label_batch in titanic_batches.take(1):
  print("'survived': {}".format(label_batch))
  for key, value in feature_batch.items():
    print("  {!r:20s}: {}".format(key, value))

# + [markdown] id="TSVgJJ1HJD6M"
# There is also a lower-level `experimental.CsvDataset` class which provides finer grained control. It does not support column type inference. Instead you must specify the type of each column. 

# + id="wP1Y_NXA8bYl"
titanic_types  = [tf.int32, tf.string, tf.float32, tf.int32, tf.int32, tf.float32, tf.string, tf.string, tf.string, tf.string] 
dataset = tf.data.experimental.CsvDataset(titanic_file, titanic_types , header=True)

for line in dataset.take(10):
  print([item.numpy() for item in line])

# + [markdown] id="oZSuLVsTbDcj"
# If some columns are empty, this low-level interface allows you to provide default values instead of column types.


# + id="d5_hbiE9bDck"
# Creates a dataset that reads all of the records from two CSV files, each with
# four float columns which may have missing values.

record_defaults = [999,999,999,999]
dataset = tf.data.experimental.CsvDataset("missing.csv", record_defaults)
dataset = dataset.map(lambda *items: tf.stack(items))
dataset

# + id="__jc7iD9M9FC"
for line in dataset:
  print(line.numpy())

# + [markdown] id="z_4g0cIvbDcl"
# By default, a `CsvDataset` yields *every* column of *every* line of the file,
# which may not be desirable, for example if the file starts with a header line
# that should be ignored, or if some columns are not required in the input.
# These lines and fields can be removed with the `header` and `select_cols`
# arguments respectively.

# + id="p2IF_K0obDcm"
# Creates a dataset that reads all of the records from two CSV files with
# headers, extracting float data from columns 2 and 4.
record_defaults = [999, 999] # Only provide defaults for the selected columns
dataset = tf.data.experimental.CsvDataset("missing.csv", record_defaults, select_cols=[1, 3])
dataset = dataset.map(lambda *items: tf.stack(items))
dataset

# + id="-5aLprDeRNb0"
for line in dataset:
  print(line.numpy())

# + [markdown] id="-CJfhb03koVN"
# ### Consuming sets of files

# + [markdown] id="yAO7SZDSk57_"
# There are many datasets distributed as a set of files, where each file is an example.

# + id="1dZwN3CS-jV2"
flowers_root = tf.keras.utils.get_file(
    'flower_photos',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    untar=True)
flowers_root = pathlib.Path(flowers_root)


# + [markdown] id="4099UU8n-jHP"
# Note: these images are licensed CC-BY, see LICENSE.txt for details.

# + [markdown] id="FCyTYpmDs_jE"
# The root directory contains a directory for each class:

# + id="_2iCXsHu6jJH"
for item in flowers_root.glob("*"):
  print(item.name)

# + [markdown] id="Ylj9fgkamgWZ"
# The files in each class directory are examples:

# + id="lAkQp5uxoINu"
list_ds = tf.data.Dataset.list_files(str(flowers_root/'*/*'))

for f in list_ds.take(5):
  print(f.numpy())


# + [markdown] id="91CPfUUJ_8SZ"
# Read the data using the `tf.io.read_file` function and extract the label from the path, returning `(image, label)` pairs:

# + id="-xhBRgvNqRRe"
def process_path(file_path):
  label = tf.strings.split(file_path, os.sep)[-2]
  return tf.io.read_file(file_path), label

labeled_ds = list_ds.map(process_path)

# + id="kxrl0lGdnpRz"
for image_raw, label_text in labeled_ds.take(1):
  print(repr(image_raw.numpy()[:100]))
  print()
  print(label_text.numpy())

# + [markdown] id="yEh46Ee0oSH5"
# <!--
# TODO(mrry): Add this section.
#
# ### Handling text data with unusual sizes
# -->
#
# ## Batching dataset elements
#

# + [markdown] id="gR-2xY-8oSH4"
# ### Simple batching
#
# The simplest form of batching stacks `n` consecutive elements of a dataset into
# a single element. The `Dataset.batch()` transformation does exactly this, with
# the same constraints as the `tf.stack()` operator, applied to each component
# of the elements: i.e. for each component *i*, all elements must have a tensor
# of the exact same shape.

# + id="xB7KeceLoSH0"
inc_dataset = tf.data.Dataset.range(100)
dec_dataset = tf.data.Dataset.range(0, -100, -1)
dataset = tf.data.Dataset.zip((inc_dataset, dec_dataset))
batched_dataset = dataset.batch(4)

for batch in batched_dataset.take(4):
  print([arr.numpy() for arr in batch])

# + [markdown] id="LlV1tpFdoSH0"
# While `tf.data` tries to propagate shape information, the default settings of `Dataset.batch` result in an unknown batch size because the last batch may not be full. Note the `None`s in the shape:

# + id="yN7hn7OBoSHx"
batched_dataset

# + [markdown] id="It1fPA3NoSHw"
# Use the `drop_remainder` argument to ignore that last batch, and get full shape propagation:

# + id="BycWC7WCoSHt"
batched_dataset = dataset.batch(7, drop_remainder=True)
batched_dataset

# + [markdown] id="mj9nRxFZoSHs"
# ### Batching tensors with padding
#
# The above recipe works for tensors that all have the same size. However, many
# models (e.g. sequence models) work with input data that can have varying size
# (e.g. sequences of different lengths). To handle this case, the
# `Dataset.padded_batch` transformation enables you to batch tensors of
# different shape by specifying one or more dimensions in which they may be
# padded.

# + id="kycwO0JooSHn"
dataset = tf.data.Dataset.range(100)
dataset = dataset.map(lambda x: tf.fill([tf.cast(x, tf.int32)], x))
dataset = dataset.padded_batch(4, padded_shapes=(None,))

for batch in dataset.take(2):
  print(batch.numpy())
  print()


# + [markdown] id="wl3yhth1oSHm"
# The `Dataset.padded_batch` transformation allows you to set different padding
# for each dimension of each component, and it may be variable-length (signified
# by `None` in the example above) or constant-length. It is also possible to
# override the padding value, which defaults to 0.
#
# <!--
# TODO(mrry): Add this section.
#
# ### Dense ragged -> tf.SparseTensor
# -->
#

# + [markdown] id="G8zbAxMwoSHl"
# ## Training workflows
#

# + [markdown] id="UnlhzF_AoSHk"
# ### Processing multiple epochs
#
# The `tf.data` API offers two main ways to process multiple epochs of the same
# data.
#
# The simplest way to iterate over a dataset in multiple epochs is to use the
# `Dataset.repeat()` transformation. First, create a dataset of titanic data:

# + id="0tODHZzRoSHg"
titanic_file = tf.keras.utils.get_file("train.csv", "https://storage.googleapis.com/tf-datasets/titanic/train.csv")
titanic_lines = tf.data.TextLineDataset(titanic_file)


# + id="LMO6mlXxoSHc"
def plot_batch_sizes(ds):
  batch_sizes = [batch.shape[0] for batch in ds]
  plt.bar(range(len(batch_sizes)), batch_sizes)
  plt.xlabel('Batch number')
  plt.ylabel('Batch size')


# + [markdown] id="WfVzmqL7oSHa"
# Applying the `Dataset.repeat()` transformation with no arguments will repeat
# the input indefinitely.
#
# The `Dataset.repeat` transformation concatenates its
# arguments without signaling the end of one epoch and the beginning of the next
# epoch. Because of this a `Dataset.batch` applied after `Dataset.repeat` will yield batches that straddle epoch boundaries:

# + id="nZ0G1cztoSHX"
titanic_batches = titanic_lines.repeat(3).batch(128)
plot_batch_sizes(titanic_batches)

# + [markdown] id="moH-4gBEoSHW"
# If you need clear epoch separation, put `Dataset.batch` before the repeat:

# + id="wmbmdK1qoSHS"
titanic_batches = titanic_lines.batch(128).repeat(3)

plot_batch_sizes(titanic_batches)

# + [markdown] id="DlEM5f9loSHR"
# If you would like to perform a custom computation (e.g. to collect statistics) at the end of each epoch then it's simplest to restart the dataset iteration on each epoch:

# + id="YyekyeY7oSHO"
epochs = 3
dataset = titanic_lines.batch(128)

for epoch in range(epochs):
  for batch in dataset:
    print(batch.shape)
  print("End of epoch: ", epoch)

# + [markdown] id="_Bci79WCoSHN"
# ### Randomly shuffling input data
#
# The `Dataset.shuffle()` transformation maintains a fixed-size
# buffer and chooses the next element uniformly at random from that buffer.
#
# Note: While large buffer_sizes shuffle more thoroughly, they can take a lot of memory, and significant time to fill. Consider using `Dataset.interleave` across files if this becomes a problem.

# + [markdown] id="6YvXr-qeoSHL"
# Add an index to the dataset so you can see the effect:

# + id="Io4iJH1toSHI"
lines = tf.data.TextLineDataset(titanic_file)
counter = tf.data.experimental.Counter()

dataset = tf.data.Dataset.zip((counter, lines))
dataset = dataset.shuffle(buffer_size=100)
dataset = dataset.batch(20)
dataset

# + [markdown] id="T6tNYRcsoSHH"
# Since the `buffer_size` is 100, and the batch size is 20, the first batch contains no elements with an index over 120.

# + id="ayM3FFFAoSHC"
n,line_batch = next(iter(dataset))
print(n.numpy())

# + [markdown] id="PLrfIjTHoSHB"
# As with `Dataset.batch` the order relative to `Dataset.repeat` matters.
#
# `Dataset.shuffle` doesn't signal the end of an epoch until the shuffle buffer is empty. So a shuffle placed before a repeat will show every element of one epoch before moving to the next: 

# + id="YX3pe7zZoSG6"
dataset = tf.data.Dataset.zip((counter, lines))
shuffled = dataset.shuffle(buffer_size=100).batch(10).repeat(2)

print("Here are the item ID's near the epoch boundary:\n")
for n, line_batch in shuffled.skip(60).take(5):
  print(n.numpy())

# + id="H9hlE-lGoSGz"
shuffle_repeat = [n.numpy().mean() for n, line_batch in shuffled]
plt.plot(shuffle_repeat, label="shuffle().repeat()")
plt.ylabel("Mean item ID")
plt.legend()

# + [markdown] id="UucIgCxWoSGx"
# But a repeat before a shuffle mixes the epoch boundaries together:

# + id="Bhxb5YGZoSGm"
dataset = tf.data.Dataset.zip((counter, lines))
shuffled = dataset.repeat(2).shuffle(buffer_size=100).batch(10)

print("Here are the item ID's near the epoch boundary:\n")
for n, line_batch in shuffled.skip(55).take(15):
  print(n.numpy())

# + id="VAM4cbpZoSGL"
repeat_shuffle = [n.numpy().mean() for n, line_batch in shuffled]

plt.plot(shuffle_repeat, label="shuffle().repeat()")
plt.plot(repeat_shuffle, label="repeat().shuffle()")
plt.ylabel("Mean item ID")
plt.legend()

# + [markdown] id="ianlfbrxbDco"
# ## Preprocessing data
#
# The `Dataset.map(f)` transformation produces a new dataset by applying a given
# function `f` to each element of the input dataset. It is based on the
# [`map()`](https://en.wikipedia.org/wiki/Map_\(higher-order_function\)) function
# that is commonly applied to lists (and other structures) in functional
# programming languages. The function `f` takes the `tf.Tensor` objects that
# represent a single element in the input, and returns the `tf.Tensor` objects
# that will represent a single element in the new dataset. Its implementation uses
# standard TensorFlow operations to transform one element into another.
#
# This section covers common examples of how to use `Dataset.map()`.
#

# + [markdown] id="UXw1IZVdbDcq"
# ### Decoding image data and resizing it
#
# <!-- TODO(markdaoust): link to image augmentation when it exists -->
# When training a neural network on real-world image data, it is often necessary
# to convert images of different sizes to a common size, so that they may be
# batched into a fixed size.
#
# Rebuild the flower filenames dataset:

# + id="rMGlj8V-u-NH"
list_ds = tf.data.Dataset.list_files(str(flowers_root/'*/*'))


# + [markdown] id="GyhZLB8N5jBm"
# Write a function that manipulates the dataset elements.

# + id="fZObC0debDcr"
# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
def parse_image(filename):
  parts = tf.strings.split(filename, os.sep)
  label = parts[-2]

  image = tf.io.read_file(filename)
  image = tf.io.decode_jpeg(image)
  image = tf.image.convert_image_dtype(image, tf.float32)
  image = tf.image.resize(image, [128, 128])
  return image, label


# + [markdown] id="e0dVJlCA5qHA"
# Test that it works.

# + id="y8xuN_HBzGup"
file_path = next(iter(list_ds))
image, label = parse_image(file_path)

def show(image, label):
  plt.figure()
  plt.imshow(image)
  plt.title(label.numpy().decode('utf-8'))
  plt.axis('off')

show(image, label)

# + [markdown] id="d3P8N-S55vDu"
# Map it over the dataset.

# + id="SzO8LI_H5Sk_"
images_ds = list_ds.map(parse_image)

for image, label in images_ds.take(2):
  show(image, label)

# + [markdown] id="3Ff7IqB9bDcs"
# ### Applying arbitrary Python logic
#
# For performance reasons, use TensorFlow operations for
# preprocessing your data whenever possible. However, it is sometimes useful to
# call external Python libraries when parsing your input data. You can use the `tf.py_function()` operation in a `Dataset.map()` transformation.

# + [markdown] id="R2u7CeA67DU8"
# For example, if you want to apply a random rotation, the `tf.image` module only has `tf.image.rot90`, which is not very useful for image augmentation. 
#
# Note: `tensorflow_addons` has a TensorFlow compatible `rotate` in `tensorflow_addons.image.rotate`.
#
# To demonstrate `tf.py_function`, try using the `scipy.ndimage.rotate` function instead:

# + id="tBUmbERt7Czz"
import scipy.ndimage as ndimage

def random_rotate_image(image):
  image = ndimage.rotate(image, np.random.uniform(-30, 30), reshape=False)
  return image


# + id="_wEyL7bS9S6t"
image, label = next(iter(images_ds))
image = random_rotate_image(image)
show(image, label)


# + [markdown] id="KxVx7z-ABNyq"
# To use this function with `Dataset.map` the same caveats apply as with `Dataset.from_generator`, you need to describe the return shapes and types when you apply the function:

# + id="Cn2nIu92BMp0"
def tf_random_rotate_image(image, label):
  im_shape = image.shape
  [image,] = tf.py_function(random_rotate_image, [image], [tf.float32])
  image.set_shape(im_shape)
  return image, label


# + id="bWPqKbTnbDct"
rot_ds = images_ds.map(tf_random_rotate_image)

for image, label in rot_ds.take(2):
  show(image, label)

# + [markdown] id="ykx59-cMBwOT"
# ### Parsing `tf.Example` protocol buffer messages
#
# Many input pipelines extract `tf.train.Example` protocol buffer messages from a
# TFRecord format. Each `tf.train.Example` record contains one or more "features",
# and the input pipeline typically converts these features into tensors.

# + id="6wnE134b32KY"
fsns_test_file = tf.keras.utils.get_file("fsns.tfrec", "https://storage.googleapis.com/download.tensorflow.org/data/fsns-20160927/testdata/fsns-00000-of-00001")
dataset = tf.data.TFRecordDataset(filenames = [fsns_test_file])
dataset

# + [markdown] id="HGypdgYOlXZz"
# You can work with `tf.train.Example` protos outside of a `tf.data.Dataset` to understand the data:

# + id="4znsVNqnF73C"
raw_example = next(iter(dataset))
parsed = tf.train.Example.FromString(raw_example.numpy())

feature = parsed.features.feature
raw_img = feature['image/encoded'].bytes_list.value[0]
img = tf.image.decode_png(raw_img)
plt.imshow(img)
plt.axis('off')
_ = plt.title(feature["image/text"].bytes_list.value[0])

# + id="cwzqp8IGC_vQ"
raw_example = next(iter(dataset))


# + id="y2X1dQNfC8Lu"
def tf_parse(eg):
  example = tf.io.parse_example(
      eg[tf.newaxis], {
          'image/encoded': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
          'image/text': tf.io.FixedLenFeature(shape=(), dtype=tf.string)
      })
  return example['image/encoded'][0], example['image/text'][0]


# + id="lGJhKDp_61A_"
img, txt = tf_parse(raw_example)
print(txt.numpy())
print(repr(img.numpy()[:20]), "...")

# + id="8vFIUFzD5qIC"
decoded = dataset.map(tf_parse)
decoded

# + id="vRYNYkEej7Ix"
image_batch, text_batch = next(iter(decoded.batch(10)))
image_batch.shape

# + [markdown] id="ry1n0UBeczit"
# <a id="time_series_windowing"></a>
#
# ### Time series windowing

# + [markdown] id="t0JMgvXEz9y1"
# For an end to end time series example see: [Time series forecasting](../../tutorials/structured_data/time_series.ipynb).

# + [markdown] id="hzBABBkAkkVJ"
# Time series data is often organized with the time axis intact.
#
# Use a simple `Dataset.range` to demonstrate:

# + id="kTQgo49skjuY"
range_ds = tf.data.Dataset.range(100000)

# + [markdown] id="o6GLGhxgpazJ"
# Typically, models based on this sort of data will want a contiguous time slice. 
#
# The simplest approach would be to batch the data:

# + [markdown] id="ETqB7QvTCNty"
# #### Using `batch`

# + id="pSs9XqwQpvIN"
batches = range_ds.batch(10, drop_remainder=True)

for batch in batches.take(5):
  print(batch.numpy())


# + [markdown] id="mgb2qikEtk5W"
# Or to make dense predictions one step into the future, you might shift the features and labels by one step relative to each other:

# + id="47XfwPhetkIN"
def dense_1_step(batch):
  # Shift features and labels one step relative to each other.
  return batch[:-1], batch[1:]

predict_dense_1_step = batches.map(dense_1_step)

for features, label in predict_dense_1_step.take(3):
  print(features.numpy(), " => ", label.numpy())

# + [markdown] id="DjsXuINKqsS_"
# To predict a whole window instead of a fixed offset you can split the batches into two parts:

# + id="FMmkQB1Gqo6x"
batches = range_ds.batch(15, drop_remainder=True)

def label_next_5_steps(batch):
  return (batch[:-5],   # Inputs: All except the last 5 steps
          batch[-5:])   # Labels: The last 5 steps

predict_5_steps = batches.map(label_next_5_steps)

for features, label in predict_5_steps.take(3):
  print(features.numpy(), " => ", label.numpy())

# + [markdown] id="5a611Qr3jlhl"
# To allow some overlap between the features of one batch and the labels of another, use `Dataset.zip`:

# + id="11dF3wyFjk2J"
feature_length = 10
label_length = 3

features = range_ds.batch(feature_length, drop_remainder=True)
labels = range_ds.batch(feature_length).skip(1).map(lambda labels: labels[:label_length])

predicted_steps = tf.data.Dataset.zip((features, labels))

for features, label in predicted_steps.take(5):
  print(features.numpy(), " => ", label.numpy())

# + [markdown] id="adew3o2mCURC"
# #### Using `window`

# + [markdown] id="fF6pEdlduq8E"
# While using `Dataset.batch` works, there are situations where you may need finer control. The `Dataset.window` method gives you complete control, but requires some care: it returns a `Dataset` of `Datasets`. See [Dataset structure](#dataset_structure) for details.

# + id="ZEI2W_EBw2OX"
window_size = 5

windows = range_ds.window(window_size, shift=1)
for sub_ds in windows.take(5):
  print(sub_ds)

# + [markdown] id="r82hWdk4x-46"
# The `Dataset.flat_map` method can take a dataset of datasets and flatten it into a single dataset:

# + id="SB8AI03mnF8u"
 for x in windows.flat_map(lambda x: x).take(30):
   print(x.numpy(), end=' ')


# + [markdown] id="sgLIwq9Anc34"
# In nearly all cases, you will want to `.batch` the dataset first:

# + id="5j_y84rmyVQa"
def sub_to_batch(sub):
  return sub.batch(window_size, drop_remainder=True)

for example in windows.flat_map(sub_to_batch).take(5):
  print(example.numpy())


# + [markdown] id="hVugrmND3Grp"
# Now, you can see that the `shift` argument controls how much each window moves over.
#
# Putting this together you might write this function:

# + id="LdFRv_0D4FqW"
def make_window_dataset(ds, window_size=5, shift=1, stride=1):
  windows = ds.window(window_size, shift=shift, stride=stride)

  def sub_to_batch(sub):
    return sub.batch(window_size, drop_remainder=True)

  windows = windows.flat_map(sub_to_batch)
  return windows



# + id="-iVxcVfEdf5b"
ds = make_window_dataset(range_ds, window_size=10, shift = 5, stride=3)

for example in ds.take(10):
  print(example.numpy())

# + [markdown] id="fMGMTPQ4w8pr"
# Then it's easy to extract labels, as before:

# + id="F0fPfZkZw6j_"
dense_labels_ds = ds.map(dense_1_step)

for inputs,labels in dense_labels_ds.take(3):
  print(inputs.numpy(), "=>", labels.numpy())

# + [markdown] id="vyi_-ft0kvy4"
# ### Resampling
#
# When working with a dataset that is very class-imbalanced, you may want to resample the dataset. `tf.data` provides two methods to do this. The credit card fraud dataset is a good example of this sort of problem.
#
# Note: See [Imbalanced Data](../tutorials/keras/imbalanced_data.ipynb) for a full tutorial.
#

# + id="U2e8dxVUlFHO"
zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/download.tensorflow.org/data/creditcard.zip',
    fname='creditcard.zip',
    extract=True)

csv_path = zip_path.replace('.zip', '.csv')

# + id="EhkkM4Wx75S_"
creditcard_ds = tf.data.experimental.make_csv_dataset(
    csv_path, batch_size=1024, label_name="Class",
    # Set the column types: 30 floats and an int.
    column_defaults=[float()]*30+[int()])


# + [markdown] id="A8O47EmHlxYX"
# Now, check the distribution of classes, it is highly skewed:

# + id="a8-Ss69XlzXD"
def count(counts, batch):
  features, labels = batch
  class_1 = labels == 1
  class_1 = tf.cast(class_1, tf.int32)

  class_0 = labels == 0
  class_0 = tf.cast(class_0, tf.int32)

  counts['class_0'] += tf.reduce_sum(class_0)
  counts['class_1'] += tf.reduce_sum(class_1)

  return counts


# + id="O1a3t_B4l_f6"
counts = creditcard_ds.take(10).reduce(
    initial_state={'class_0': 0, 'class_1': 0},
    reduce_func = count)

counts = np.array([counts['class_0'].numpy(),
                   counts['class_1'].numpy()]).astype(np.float32)

fractions = counts/counts.sum()
print(fractions)

# + [markdown] id="z1b8lFhSnDdv"
# A common approach to training with an imbalanced dataset is to balance it. `tf.data` includes a few methods which enable this workflow:

# + [markdown] id="y8jQWsgMnjQG"
# #### Datasets sampling

# + [markdown] id="ov14SRrQyQE3"
# One approach to resampling a dataset is to use `sample_from_datasets`. This is more applicable when you have a separate `data.Dataset` for each class.
#
# Here, just use filter to generate them from the credit card fraud data:

# + id="6YKfCPa-nioA"
negative_ds = (
  creditcard_ds
    .unbatch()
    .filter(lambda features, label: label==0)
    .repeat())
positive_ds = (
  creditcard_ds
    .unbatch()
    .filter(lambda features, label: label==1)
    .repeat())

# + id="8FNd3sQjzl9-"
for features, label in positive_ds.batch(10).take(1):
  print(label.numpy())

# + [markdown] id="GxLAr-7p0ATX"
# To use `tf.data.experimental.sample_from_datasets` pass the datasets, and the weight for each:

# + id="vjdPVIFCngOb"
balanced_ds = tf.data.experimental.sample_from_datasets(
    [negative_ds, positive_ds], [0.5, 0.5]).batch(10)

# + [markdown] id="2K4ObOms082B"
# Now the dataset produces examples of each class with 50/50 probability:

# + id="Myvkw21Rz-fH"
for features, labels in balanced_ds.take(10):
  print(labels.numpy())


# + [markdown] id="OUTE3eb9nckY"
# #### Rejection resampling

# + [markdown] id="kZ9ezkK6irMD"
# One problem with the above `experimental.sample_from_datasets` approach is that
# it needs a separate `tf.data.Dataset` per class. Using `Dataset.filter`
# works, but results in all the data being loaded twice.
#
# The `data.experimental.rejection_resample` function can be applied to a dataset to rebalance it, while only loading it once. Elements will be dropped from the dataset to achieve balance.
#
# `data.experimental.rejection_resample` takes a `class_func` argument. This `class_func` is applied to each dataset element, and is used to determine which class an example belongs to for the purposes of balancing.
#
# The elements of `creditcard_ds` are already `(features, label)` pairs. So the `class_func` just needs to return those labels:

# + id="zC_Cuzw8lhI5"
def class_func(features, label):
  return label


# + [markdown] id="DdKmE8Jumlp0"
# The resampler also needs a target distribution, and optionally an initial distribution estimate:

# + id="9tv0tWNxmkzM"
resampler = tf.data.experimental.rejection_resample(
    class_func, target_dist=[0.5, 0.5], initial_dist=fractions)

# + [markdown] id="YxJrOZVToGuE"
# The resampler deals with individual examples, so you must `unbatch` the dataset before applying the resampler:

# + id="fY6VIhr3oGHG"
resample_ds = creditcard_ds.unbatch().apply(resampler).batch(10)

# + [markdown] id="L-HnC1s8idqV"
# The resampler returns creates `(class, example)` pairs from the output of the `class_func`. In this case, the `example` was already a `(feature, label)` pair, so use `map` to drop the extra copy of the labels:

# + id="KpfCGU6BiaZq"
balanced_ds = resample_ds.map(lambda extra_label, features_and_label: features_and_label)

# + [markdown] id="j3d2jyEhx9kD"
# Now the dataset produces examples of each class with 50/50 probability:

# + id="XGLYChBQwkDV"
for features, labels in balanced_ds.take(10):
  print(labels.numpy())

# + [markdown] id="vYFKQx3bUBeU"
# ## Iterator Checkpointing

# + [markdown] id="SOGg1UFhUE4z"
# Tensorflow supports [taking checkpoints](https://www.tensorflow.org/guide/checkpoint) so that when your training process restarts it can restore the latest checkpoint to recover most of its progress. In addition to checkpointing the model variables, you can also checkpoint the progress of the dataset iterator. This could be useful if you have a large dataset and don't want to start the dataset from the beginning on each restart. Note however that iterator checkpoints may be large, since transformations such as `shuffle` and `prefetch` require buffering elements within the iterator. 
#
# To include your iterator in a checkpoint, pass the iterator to the `tf.train.Checkpoint` constructor.

# + id="3Fsm9wvKUsNC"
range_ds = tf.data.Dataset.range(20)

iterator = iter(range_ds)
ckpt = tf.train.Checkpoint(step=tf.Variable(0), iterator=iterator)
manager = tf.train.CheckpointManager(ckpt, '/tmp/my_ckpt', max_to_keep=3)

print([next(iterator).numpy() for _ in range(5)])

save_path = manager.save()

print([next(iterator).numpy() for _ in range(5)])

ckpt.restore(manager.latest_checkpoint)

print([next(iterator).numpy() for _ in range(5)])

# + [markdown] id="gxWglTwX9Fex"
# Note: It is not possible to checkpoint an iterator which relies on external state such as a `tf.py_function`. Attempting to do so will raise an exception complaining about the external state.

# + [markdown] id="uLRdedPpbDdD"
# ## Using tf.data with tf.keras

# + [markdown] id="JTQe8daMcgFz"
# The `tf.keras` API simplifies many aspects of creating and executing machine
# learning models. Its `.fit()` and `.evaluate()` and `.predict()` APIs support datasets as inputs. Here is a quick dataset and model setup:

# + id="-bfjqm0hOfES"
train, test = tf.keras.datasets.fashion_mnist.load_data()

images, labels = train
images = images/255.0
labels = labels.astype(np.int32)

# + id="wDhF3rGnbDdD"
fmnist_train_ds = tf.data.Dataset.from_tensor_slices((images, labels))
fmnist_train_ds = fmnist_train_ds.shuffle(5000).batch(32)

model = tf.keras.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              metrics=['accuracy'])

# + [markdown] id="Rdogg8CfHs-G"
#  Passing a dataset of `(feature, label)` pairs is all that's needed for `Model.fit` and `Model.evaluate`:

# + id="9cu4kPzOHnlt"
model.fit(fmnist_train_ds, epochs=2)

# + [markdown] id="FzpAQfJMJF41"
# If you pass an infinite dataset, for example by calling `Dataset.repeat()`, you just need to also pass the `steps_per_epoch` argument:

# + id="Bp1BpzlyJinb"
model.fit(fmnist_train_ds.repeat(), epochs=2, steps_per_epoch=20)

# + [markdown] id="iTLsw_nqJpTw"
# For evaluation you can pass the number of evaluation steps:

# + id="TnlRHlaL-XUI"
loss, accuracy = model.evaluate(fmnist_train_ds)
print("Loss :", loss)
print("Accuracy :", accuracy)

# + [markdown] id="C8UBU3CJKEA4"
# For long datasets, set the number of steps to evaluate:

# + id="uVgamf9HKDon"
loss, accuracy = model.evaluate(fmnist_train_ds.repeat(), steps=10)
print("Loss :", loss)
print("Accuracy :", accuracy)

# + [markdown] id="aZYhJ_YSIl6w"
# The labels are not required in when calling `Model.predict`. 

# + id="343lXJ-pIqWD"
predict_ds = tf.data.Dataset.from_tensor_slices(images).batch(32)
result = model.predict(predict_ds, steps = 10)
print(result.shape)

# + [markdown] id="YfzZORwLI202"
# But the labels are ignored if you do pass a dataset containing them:

# + id="mgQJTPrT-2WF"
result = model.predict(fmnist_train_ds, steps = 10)
print(result.shape)
