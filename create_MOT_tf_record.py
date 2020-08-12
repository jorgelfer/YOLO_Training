"""
Convert raw MOT dataset to TFRecord for object_detection.
By jorge
"""

import tensorflow as tf
import os
from absl import app, logging
import numpy as np
import io
import PIL.Image
from yolov3_tf2 import dataset_util
from tqdm import tqdm

flags = tf.compat.v1.flags
flags.DEFINE_string('data_dir','/home/jorge/Desktop/yolov3-tf2-master/data/MOT20/train/', 'Root directory to raw MOT dataset.')
flags.DEFINE_string('classes_file','./data/MOT2020.names', 'path to classes file')
flags.DEFINE_string('output_path','/home/jorge/Desktop/yolov3-tf2-master/data/MOT20_tfrecords/', 'Path to output TFRecord')
FLAGS = flags.FLAGS

dirs = ['MOT20-01','MOT20-02','MOT20-03','MOT20-05']

def create_tf_example(filename, objs, class_names):
    with tf.io.gfile.GFile(filename, 'rb') as fid:
        encoded_jpg = fid.read()
    # check if image is JPEG, otherwise raise an error.
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    image_shape = tf.image.decode_jpeg(encoded_jpg).shape
    height = image_shape[0] # Image height
    width = image_shape[1] # Image width

    xmin = [] # List of normalized left x coordinates in bounding box (1 per box)
    ymin = [] # List of normalized top y coordinates in bounding box (1 per box)
    xmax = [] # List of normalized right x coordinates in bounding box
    # (1 per box)
    ymax = [] # List of normalized bottom y coordinates in bounding box
    # (1 per box)
    classes_text = [] # List of classes names of bounding box (1 per box)
    evaluation = [] # List of [0 or 1] evaluation flag of bounding box (1 per box)
    classes = [] # List of integer class id of bounding box (1 per box)
    visi = [] # List of visibility of bounding box (1 per box)
    for obj in objs:
      if int(obj[7].numpy()) != 1:
          continue
      xmin.append(float(obj[2].numpy()) / width)
      ymin.append(float(obj[3].numpy()) / height)
      xmax.append(float(obj[4].numpy()+obj[2].numpy()) / width)
      ymax.append(float(obj[5].numpy()+obj[3].numpy()) / height)
      classes_text.append(class_names[int(obj[7].numpy())-1].encode('utf8'))
      evaluation.append(int(obj[6].numpy()))
      classes.append(int(obj[7].numpy())-1)
      visi.append(float(obj[8].numpy()))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/evaluation': dataset_util.int64_list_feature(evaluation),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/class/vis': dataset_util.float_list_feature(visi),
    }))
    return tf_example


def main(_argv):
    # Check if output dir exists
    if not os.path.isdir(FLAGS.output_path):
        os.mkdir(FLAGS.output_path)
    # Get classes names
    class_names = [c.strip() for c in open(FLAGS.classes_file).readlines()]
    logging.info('classes loaded')
    n_images_shard = 200
    for i, dirin in enumerate(dirs):
        logging.info('Reading from %s', dirin)
        sequence_dir = os.path.join(FLAGS.data_dir,dirin)
        image_dir = os.path.join(sequence_dir, "img1")
        image_filenames = {
            int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
            for f in sorted(os.listdir(image_dir))}
        groundtruth_file = os.path.join(sequence_dir, "gt/gt.txt")
        groundtruth = None
        if os.path.exists(groundtruth_file):
            groundtruth = np.loadtxt(groundtruth_file, delimiter=',')
            groundtruth = tf.convert_to_tensor(groundtruth)
        n_shards = int(len(image_filenames) / n_images_shard) + (1 if len(image_filenames) % n_images_shard != 0 else 0)
        index = 1
        for shard in tqdm(range(n_shards)):
            tfrecords_shard_path = "{}_{}_{}_{}.records".format('MOT20', 'train',
                                                             '%.5d-of-%.5d' % (i, len(dirs) - 1),
                                                             '%.5d-of-%.5d' % (shard, n_shards - 1))
            end = index + n_images_shard if len(image_filenames) > (index + n_images_shard) else len(image_filenames) + 1
            if end == len(image_filenames) + 1:
                tfrecords_shard_path = "{}_{}_{}_{}.records".format('MOT20', 'val', '%.5d-of-%.5d' % (i, len(dirs) - 1), '%.5d-of-%.5d' % (shard, n_shards - 1))
            image_filenames_2 = {x:image_filenames[x] for x in list(range(index, end))}
            with tf.io.TFRecordWriter(os.path.join(FLAGS.output_path,
                                                   tfrecords_shard_path)) as writer:
                for frame, filename in image_filenames_2.items():
                    frame_objs = tf.equal(groundtruth[:,0],frame)
                    tf_example = create_tf_example(filename,
                                                   tf.boolean_mask(groundtruth,frame_objs),
                                                   class_names)
                    writer.write(tf_example.SerializeToString())
            writer.close()
            index = end

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
