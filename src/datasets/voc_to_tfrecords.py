from dataset_utils import int64_feature, float_feature, bytes_feature
import voc_common
import math
import os
import xml.etree.ElementTree as ElementTree
import shutil
import tensorflow as tf

tf.app.flags.DEFINE_string(
    'path_to_voc', '~/Data/VOC/VOCdevkit/', 'path to Pascal VOC dataset')
tf.app.flags.DEFINE_string(
    'year', 'all', '2007 or 2012 or all')
tf.app.flags.DEFINE_string(
    'type', 'all', 'train or val or test')

FLAGS = tf.app.flags.FLAGS

# Small graph for image decoding
decoder_sess = tf.Session()
image_placeholder = tf.placeholder(dtype=tf.string)
decoded_jpeg = tf.image.decode_jpeg(image_placeholder, channels=3)


def get_data_set():
    data_set = []
    if (FLAGS.year == 'all'):
        if (FLAGS.type == 'all'):
            data_set = [('2007', 'train'), ('2007', 'val'), ('2007', 'test'), ('2012', 'train'), ('2012', 'val')]
        elif ((FLAGS.type == 'train') | (FLAGS.type == 'val')):
            data_set.append(('2007', FLAGS.type))
            data_set.append(('2012', FLAGS.type))
        elif (FLAGS.type == 'test'):
            data_set.append(('2007', 'test'))
            print("only 2007 has test dataset !\n")
        else:
            print("unknow dataset type !\n")
    elif (FLAGS.year == '2007'):
        if (FLAGS.type == 'all'):
            data_set = [(FLAGS.year, 'train'), (FLAGS.year, 'val'), (FLAGS.year, 'test')]
        elif ((FLAGS.type == 'train') | (FLAGS.type == 'val') | (FLAGS.type == 'test')):
            data_set.append((FLAGS.year, FLAGS.type))
        else:
            print("unknow dataset type !\n")
    elif (FLAGS.year == '2012'):
        if (FLAGS.type == 'all'):
            data_set = [(FLAGS.year, 'train'), (FLAGS.year, 'val')]
        elif ((FLAGS.type == 'train') | (FLAGS.type == 'val')):
            data_set.append((FLAGS.year, FLAGS.type))
        elif (FLAGS.type == 'test'):
            print("only 2007 has test dataset !\n")
        else:
            print("unknow dataset type !\n")
    else:
        print("unknow dataset year !\n")

    print('data_set: ' + str(data_set) + '\n')
    return data_set


def get_ids(voc_path, year, type):
    ids = []
    id_file = os.path.join(voc_path, 'VOC{}/ImageSets/Main/{}.txt'.format(
        year, type))
    with open(id_file, 'r') as image_ids:
        ids.extend(map(str.strip, image_ids.readlines()))
    return ids


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
        if label not in voc_common.classes:  # exclude difficult or unlisted classes
            continue

        labels.append(voc_common.classes.index(label))
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
        'image/box_num':int64_feature(len(labels)),
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


def get_image_path(voc_path, year, image_id):
    """Get path to image for given year and image id."""
    return os.path.join(voc_path, 'VOC{}/JPEGImages/{}.jpg'.format(year,
                                                                   image_id))


def get_anno_path(voc_path, year, image_id):
    """Get path to image annotation for given year and image id."""
    return os.path.join(voc_path, 'VOC{}/Annotations/{}.xml'.format(year,
                                                                    image_id))


def get_save_path(voc_path, year, type):
    save_path = os.path.join(voc_path, 'TFRecords', year, type)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        shutil.rmtree(save_path)
        os.makedirs(save_path)
    return save_path


def get_process_dataset_params(voc_path, year, type):
    ids = get_ids(voc_path, year, type)
    image_paths = [get_image_path(voc_path, year, i) for i in ids]
    anno_paths = [get_anno_path(voc_path, year, i) for i in ids]
    save_path = get_save_path(voc_path, year, type)
    return image_paths, anno_paths, save_path


def process_dataset(name, image_paths, anno_paths, result_path, example_num_per_file):
    """Process selected Pascal VOC dataset to generate TFRecords files.

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
        if(len(labels)> max_box_num):
            max_box_num = len(labels)
            print(max_box_num)
        example = convert_to_example(image_data, labels, labels_text, bboxes, shape, difficult, truncated)

        # write to writer
        writer.write(example.SerializeToString())
    writer.close()
    tfrecords_path_list_f.close()


if __name__ == '__main__':

    """Locate files for data sets and then generate TFRecords."""
    voc_path = FLAGS.path_to_voc
    voc_path = os.path.expanduser(voc_path)

    data_set = get_data_set()
    for year, type in data_set:
        image_paths, anno_paths, save_path = get_process_dataset_params(voc_path, year, type)
        process_dataset(type, image_paths, anno_paths, save_path, 100)
