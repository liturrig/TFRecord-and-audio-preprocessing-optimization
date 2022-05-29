import argparse
import csv
import tensorflow as tf
from datetime import datetime
import time
import os

# initialize the parameter collector and add required arguments
parser = argparse.ArgumentParser()
parser.add_argument('--input', type = str, required = True)
parser.add_argument('--output', type = str, required = True)
parser.add_argument('--normalize', action = 'store_true')
parametri = vars(parser.parse_args())

# define normalization boundaries (from DHT data sheet)

temp_boundaries = [0,50]
hum_boundaries = [20,90]

# initialize the array to store csv readings
data = []

# read the csv and transfer data as a new element in the array above
with open(parametri['input']) as f:
    reader = csv.reader(f)
    for row in reader:
        data.append([row[0],row[1],int(row[2]),int(row[3])])

# begin the writing part (for normalized option)
if parametri['normalize']:
    # initialize the Writer object, by passing the output path
    with tf.io.TFRecordWriter(parametri['output']) as writer:
        for row in data:
            # generate POSIX by splitting the 'date' field
            day, month, year = row[0].split('/')
            hour, min, sec = row[1].split(':')
            d = datetime(int(year), int(month), int(day), int(hour), int(min), int(sec))
            posix = int(time.mktime(d.timetuple()))

            # istantiate the 3 features and write them in the TFRecord file
            date_feature = tf.train.Feature(int64_list = tf.train.Int64List(value = [posix]))

            temp_feature = tf.train.Feature(float_list = tf.train.FloatList(value = [(row[2]-temp_boundaries[0])/(temp_boundaries[1] - temp_boundaries[0])]))

            hum_feature = tf.train.Feature(float_list = tf.train.FloatList(value = [(row[3] - hum_boundaries[0])/(hum_boundaries[1] - hum_boundaries[0])]))

            mapping = {'Date': date_feature, 'Temperature': temp_feature, 'Humidity': hum_feature}

            example = tf.train.Example(features=tf.train.Features(feature=mapping))

            writer.write(example.SerializeToString())

# begin the writing part (for the non-normalized option)
else:
    # initialize the Writer object, by passing the output path
    with tf.io.TFRecordWriter(parametri['output']) as writer:
        for row in data:
            # generate POSIX by splitting the 'date' field
            day, month, year = row[0].split('/')
            hour, min, sec = row[1].split(':')
            d = datetime(int(year), int(month), int(day), int(hour), int(min), int(sec))
            posix = int(time.mktime(d.timetuple()))
            # istantiate the 3 features and write them in the TFRecord file
            date_feature = tf.train.Feature(int64_list = tf.train.Int64List(value = [posix]))

            temp_feature = tf.train.Feature(int64_list = tf.train.Int64List(value = [row[2]]))

            hum_feature = tf.train.Feature(int64_list = tf.train.Int64List(value = [row[3]]))

            mapping = {'Date': date_feature, 'Temperature': temp_feature, 'Humidity': hum_feature}

            example = tf.train.Example(features=tf.train.Features(feature=mapping))

            writer.write(example.SerializeToString())


# print the final TFRecord file size
print(f'Generated file size: {os.path.getsize(parametri["output"])} B.')

