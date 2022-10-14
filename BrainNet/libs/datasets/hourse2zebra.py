import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
orig_image_size  = (286, 286)
input_image_size = (256, 256, 3)
class Horse2Zebra:
    def __init__(self,i_buffer_size=256,i_batch_size=1,i_shuffle=True):
        self.buffer_size = i_buffer_size
        self.batch_size  = i_batch_size
        self.shuffle     = i_shuffle
        """Load dataset as it stored in tensorflow datasets"""
        loaded_dataset = tfds.load("cycle_gan/horse2zebra",with_info=True,as_supervised=True)[0]
        self.train_horses, self.train_zebras = loaded_dataset["trainA"], loaded_dataset["trainB"]
        self.test_horses, self.test_zebras   = loaded_dataset["testA"], loaded_dataset["testB"]
        """These above datasets are tensorflow dataset with format (image,label) in types((tf.uint8,tf.int64))"""
        if self.shuffle:
            self.train_horses = self.train_horses.map(map_func=self.preprocess_train_image).cache().shuffle(self.buffer_size).batch(self.batch_size)
            self.train_zebras = self.train_zebras.map(map_func=self.preprocess_train_image).cache().shuffle(self.buffer_size).batch(self.batch_size)
            self.test_horses  = self.test_horses.map(map_func=self.preprocess_test_image).cache().shuffle(self.buffer_size).batch(self.batch_size)
            self.test_zebras  = self.test_zebras.map(map_func=self.preprocess_test_image).cache().shuffle(self.buffer_size).batch(self.batch_size)
        else:
            self.train_horses = self.train_horses.map(map_func=self.preprocess_train_image).cache().batch(self.batch_size)
            self.train_zebras = self.train_zebras.map(map_func=self.preprocess_train_image).cache().batch(self.batch_size)
            self.test_horses = self.test_horses.map(map_func=self.preprocess_test_image).cache().batch(self.batch_size)
            self.test_zebras = self.test_zebras.map(map_func=self.preprocess_test_image).cache().batch(self.batch_size)
    @staticmethod
    def normalize_image(i_image=None):
        image = tf.cast(i_image,dtype=tf.float32)
        # Map values in the range [-1, 1]
        return (image/127.5)-1.0
    @staticmethod
    def preprocess_train_image(i_image=None,i_label=None):
        image,label = i_image,i_label
        image = tf.image.random_flip_left_right(image)
        image = tf.image.resize(image,size=orig_image_size)
        image = tf.image.random_crop(image,size=input_image_size)
        image = Horse2Zebra.normalize_image(i_image=image)
        return image
    @staticmethod
    def preprocess_test_image(i_image=None,i_label=None):
        # Only resizing and normalization for the test images.
        image, label = i_image,i_label
        img = tf.image.resize(image, [input_image_size[0], input_image_size[1]])
        img = Horse2Zebra.normalize_image(img)
        return img
    def imshow(self):
        _, ax = plt.subplots(4, 2, figsize=(10, 15))
        for i, samples in enumerate(zip(self.train_horses.take(4), self.train_zebras.take(4))):
            horse = (((samples[0][0] * 127.5) + 127.5).numpy()).astype(np.uint8)
            zebra = (((samples[1][0] * 127.5) + 127.5).numpy()).astype(np.uint8)
            ax[i, 0].imshow(horse)
            ax[i, 1].imshow(zebra)
        plt.show()
    @staticmethod
    def convert(i_db = None):
        assert isinstance(i_db,tf.data.Dataset)
        np_db = []
        for item in i_db:
            image = ((item[0]*127.5)+127.5).numpy().astype(np.uint8)
            np_db.append(image)
        print('Dataset size: {}'.format(len(np_db)))
        return np_db
    def get_numpy_db(self):
        trainX = self.convert(self.train_horses)
        trainY = self.convert(self.train_zebras)
        valX   = self.convert(self.test_horses)
        valY   = self.convert(self.test_zebras)
        return trainX,trainY,valX,valY
if __name__ == '__main__':
    print('This module is to load and prepare Horse2Zebra dataset for training CycleGAN network')
    db = Horse2Zebra()
    tx,ty,vx,vy = db.get_numpy_db()
    for images in tx:
        print(type(images))
        plt.imshow(images)
        plt.show()
"""=================================================================================================================="""