from dataset_utils import int64_feature, float_feature, bytes_feature
import decorations
import math
import os
import xml.etree.ElementTree as ElementTree
import shutil
import tensorflow as tf

tf.app.flags.DEFINE_string(
    'path_to_decorations', '/home/paul/Data/decorations', 'path to decorations dataset')
tf.app.flags.DEFINE_string(
    'image_list', '/home/paul/Data/decorations/2017.txt', 'image_list')
tf.app.flags.DEFINE_string(
    'year', '2017', '2017')
tf.app.flags.DEFINE_string(
    'type', 'train', 'all or train or val or test')

FLAGS = tf.app.flags.FLAGS

# Small graph for image decoding
decoder_sess = tf.Session()
image_placeholder = tf.placeholder(dtype=tf.string)
decoded_jpeg = tf.image.decode_jpeg(image_placeholder, channels=3)


def process_image(image_path, anno_path):
    image_data = tf.gfile.FastGFile(image_path, 'r').read()

    with open(anno_path) as f:
        xml_tree = ElementTree.parse(f)
    root = xml_tree.getroot()

    # Image shape.
    size = root.find('size')
    shape = [int(size.find('height').text),
             int(size.find('width').text),
             int(size.find('depth').text)]
    # Find annotations.
    bboxes = []
    labels = []
    labels_text = []
    difficult = []
    truncated = []
    for obj in root.iter('object'):

        label = obj.find('name').text
        if label not in decorations.classes:  # exclude difficult or unlisted classes
            continue

        labels.append(decorations.classes.index(label))
        labels_text.append(label)

        if obj.find('difficult'):
            difficult.append(int(obj.find('difficult').text))
        else:
            difficult.append(0)
        if obj.find('truncated'):
            truncated.append(int(obj.find('truncated').text))
        else:
            truncated.append(0)

        bbox = obj.find('bndbox')
        bboxes.append((float(bbox.find('ymin').text) / shape[0],
                       float(bbox.find('xmin').text) / shape[1],
                       float(bbox.find('ymax').text) / shape[0],
                       float(bbox.find('xmax').text) / shape[1]
                       ))

    return image_data, shape, bboxes, labels, labels_text, difficult, truncated


def convert_to_example(image_data, labels, labels_text, bboxes, shape,
                       difficult, truncated):
    """Build an Example proto for an image example.

    Args:
      image_data: string, JPEG encoding of RGB image;
      labels: list of integers, identifier for the ground truth;
      labels_text: list of strings, human-readable labels;
      bboxes: list of bounding boxes; each box is a list of integers;
          specifying [xmin, ymin, xmax, ymax]. All boxes are assumed to belong
          to the same label as the image label.
      shape: 3 integers, image shapes in pixels.
    Returns:
      Example proto
    """
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    for b in bboxes:
        assert len(b) == 4
        # pylint: disable=expression-not-assigned
        [l.append(point) for l, point in zip([ymin, xmin, ymax, xmax], b)]
        # pylint: enable=expression-not-assigned

    image_format = b'JPEG'
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(shape[0]),
        'image/width': int64_feature(shape[1]),
        'image/channels': int64_feature(shape[2]),
        'image/shape': int64_feature(shape),
        'image/box_num': int64_feature(len(labels)),
        'image/object/bbox/xmin': float_feature(xmin),
        'image/object/bbox/xmax': float_feature(xmax),
        'image/object/bbox/ymin': float_feature(ymin),
        'image/object/bbox/ymax': float_feature(ymax),
        'image/object/bbox/label': int64_feature(labels),
        'image/object/bbox/label_text': bytes_feature(labels_text),
        'image/object/bbox/difficult': int64_feature(difficult),
        'image/object/bbox/truncated': int64_feature(truncated),
        'image/format': bytes_feature(image_format),
        'image/encoded': bytes_feature(image_data)}))
    return example


def process_dataset(name, image_paths, anno_paths, result_path, example_num_per_file):
    """Process selected Pascal VOC type dataset to generate TFRecords files.

    Parameters
    ----------
    name : string
        Name of resulting dataset 'train' or 'test'.
    image_paths : list
        List of paths to images to include in dataset.
    anno_paths : list
        List of paths to corresponding image annotations.
    result_path : string
        Path to put resulting TFRecord files.
    example_num_per_file : int
        how many examples one TFRecord file has.
    """

    total_example_num = len(image_paths)
    total_files_num = int(math.ceil(total_example_num / example_num_per_file))
    tfrecords_path_list_f = open(result_path + '/tfrecords_list.txt', 'w')
    writer = None
    max_box_num = 0
    for i in range(0, total_example_num):
        if i % example_num_per_file == 0:
            if i != 0:
                writer.close()
            tfrecords_name = '{}-{:05d}-of-{:05d}.tfrecords'.format(name, int(math.ceil(i / example_num_per_file)),
                                                                    total_files_num)
            tfrecords_name = os.path.join(result_path, tfrecords_name)
            print(tfrecords_name + '\n')
            tfrecords_path_list_f.write(tfrecords_name + '\n')
            writer = tf.python_io.TFRecordWriter(tfrecords_name)

        image_file = image_paths[i]
        anno_file = anno_paths[i]

        image_data, shape, bboxes, labels, labels_text, difficult, truncated = process_image(image_file, anno_file)
        if (len(labels) > max_box_num):
            max_box_num = len(labels)
            print(max_box_num)
        example = convert_to_example(image_data, labels, labels_text, bboxes, shape, difficult, truncated)

        # write to writer
        writer.write(example.SerializeToString())
    writer.close()
    tfrecords_path_list_f.close()


if __name__ == '__main__':

    """Locate files for data sets and then generate TFRecords."""
    path = FLAGS.path_to_decorations
    path = os.path.expanduser(path)

    image_paths = []
    anno_paths = []

    f = open(FLAGS.image_list)
    for line in f:
        temp = line.rstrip()
        image_paths.append(temp)
        temp = temp.replace('images', 'annotations')
        temp = temp.replace('jpg', 'xml')
        anno_paths.append(temp)
    f.close()
    save_path = os.path.join(path, 'TFRecords', FLAGS.year, FLAGS.type)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        shutil.rmtree(save_path)
        os.makedirs(save_path)

    process_dataset(FLAGS.type, image_paths, anno_paths, save_path, 100)
