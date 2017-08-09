import tensorflow as tf
import os

'''
python src/train.py --train_dir=/tmp/tfmodel  \
    --dataset_name=decorations  \
    --dataset_dir=/home/paul/Data/decorations/TFRecords/2017  \
    --num_classes=4  \
    --max_number_of_steps=1000  \
    --batch_size=2  \
'''

slim = tf.contrib.slim

classes = ["glasses", "hat", "package", "tie"]
FILE_PATTERN = '*.tfrecords'
ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'shape': 'Shape of the image',
    'object/bbox': 'A list of bounding boxes, one per each object.',
    'object/label': 'A list of labels, one per each object.',
}

SPLITS_TO_SIZES = {
    'train': 762,
}

MAX_BOX_NUM_PER_IMAGE = {
    'train': 9,
}

NUM_CLASSES = 4


def get_split(split_name, dataset_dir, file_pattern=None, reader=None,
              split_to_sizes=SPLITS_TO_SIZES, items_to_descriptions=ITEMS_TO_DESCRIPTIONS, num_classes=NUM_CLASSES):
    """Gets a dataset tuple with instructions for reading Pascal VOC type dataset.

    Args:
      split_name: A train/test split name.
      dataset_dir: The base directory of the dataset sources.
      file_pattern: The file pattern to use when matching the dataset sources.
        It is assumed that the pattern contains a '%s' string so that the split
        name can be inserted.
      reader: The TensorFlow reader type.

    Returns:
      A `Dataset` namedtuple.

    Raises:
        ValueError: if `split_name` is not a valid train/test split.
    """
    if split_name not in split_to_sizes:
        raise ValueError('split name %s was not recognized.' % split_name)
    if file_pattern is None:
        file_pattern = FILE_PATTERN

    file_pattern = os.path.join(dataset_dir, split_name, file_pattern)

    # Allowing None in the signature so that dataset_factory can use the default.
    if reader is None:
        reader = tf.TFRecordReader
    # Features in Pascal VOC type TFRecords.
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/height': tf.FixedLenFeature([1], tf.int64),
        'image/width': tf.FixedLenFeature([1], tf.int64),
        'image/channels': tf.FixedLenFeature([1], tf.int64),
        'image/shape': tf.FixedLenFeature([3], tf.int64),
        'image/box_num': tf.FixedLenFeature([1], tf.int64),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64),
        'image/object/bbox/difficult': tf.VarLenFeature(dtype=tf.int64),
        'image/object/bbox/truncated': tf.VarLenFeature(dtype=tf.int64),
    }
    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
        'shape': slim.tfexample_decoder.Tensor('image/shape'),
        'box_num': slim.tfexample_decoder.Tensor('image/box_num', shape=[1]),
        'object/bbox': slim.tfexample_decoder.BoundingBox(
            ['xmin', 'ymin', 'xmax', 'ymax'], 'image/object/bbox/'),
        'object/label': slim.tfexample_decoder.Tensor('image/object/bbox/label'),
        'object/difficult': slim.tfexample_decoder.Tensor('image/object/bbox/difficult'),
        'object/truncated': slim.tfexample_decoder.Tensor('image/object/bbox/truncated'),
    }
    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    labels_to_names = {}
    for label in classes:
        labels_to_names[classes.index(label)] = label

    return slim.dataset.Dataset(
        data_sources=file_pattern,
        reader=reader,
        decoder=decoder,
        num_samples=split_to_sizes[split_name],
        items_to_descriptions=items_to_descriptions,
        num_classes=num_classes,
        labels_to_names=labels_to_names)
