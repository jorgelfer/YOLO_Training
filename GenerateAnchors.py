from absl import app, flags, logging
from absl.flags import FLAGS
import numpy as np
from kmeans import kmeans, avg_iou
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny,
)
import yolov3_tf2.dataset as dataset
from tqdm import tqdm

flags.DEFINE_string('dataset','./data/AOLP_tfrecords/AOLP_train_*-of-*_*-of-*.records','path to dataset')
flags.DEFINE_string('classes','./data/MOT2020.names', 'path to classes file')
flags.DEFINE_integer('size', 416, 'image size')
flags.DEFINE_integer('clusters', 9, 'Number of clusters')

def main(_argv):
    train_dataset = dataset.load_tfrecord_dataset(FLAGS.dataset, FLAGS.classes, FLAGS.size)
    data = []
    for imgs, labels in tqdm(train_dataset):
        for x1, y1, x2, y2, label in labels:
            if x1 == 0 and x2 == 0:
                continue
            data.append([x2-x1, y2-y1])
    out = kmeans(np.asarray(data), k=FLAGS.clusters)
    print("Accuracy: {:.2f}%".format(avg_iou(np.asarray(data), out) * 100))
    print("Boxes:\n {}".format(out))
    #
    ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
    print("Ratios:\n {}".format(sorted(ratios)))
    #
    # sort BB according to their area.
    areas = np.multiply(out[:, 0], out[:, 1])
    out_sorted = out[sorted(range(len(areas)), key=lambda k: areas[k]), :]
    print("Sorted boxes according to area:\n {}".format(out_sorted))

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
