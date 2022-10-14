import numpy as np
from PIL import Image
import tensorflow as tf
from classification_models.keras import Classifiers
class DeepFeatures:
    def __init__(self,
                 i_models      = None,
                 i_input_shape = (256,256,3),
                 i_flip_up     = True,
                 i_flip_lr     = True,
                 i_crop_ratio  = 0.90):
        assert isinstance(i_models,(list,tuple))
        assert len(i_models)>0
        assert isinstance(i_input_shape,(list,tuple))
        assert len(i_input_shape)==3 #(Height, Width, Depth)
        assert i_input_shape[-1]==3 #Color image
        """Suppose that all model use same input image (image shape must be same)"""
        self.input_shape = i_input_shape
        self.flip_ud     = i_flip_up
        self.flip_lr     = i_flip_lr
        self.crop_size   = (int(self.input_shape[0]*i_crop_ratio),int(self.input_shape[1]*i_crop_ratio),self.input_shape[-1])
        """i_models is a list of (model,output_layer_name) that is used for image feature extraction"""
        """Init the models that will be used for feature extraction"""
        self.models = []
        for (model,output_layer) in i_models:
            if isinstance(model,str):
                model = self.init_model(i_model_name=model,i_pooling=output_layer)
                self.models.append(model)
            else:
                assert isinstance(model,tf.keras.models.Model)
                model = tf.keras.models.Model(inputs=model.input,outputs=model.get_layer(output_layer).output)
                self.models.append(model)
    def init_model(self,i_model_name=None,i_pooling='avg'):
        """Init pretrained model based on imagenet dataset"""
        assert isinstance(i_pooling,str)
        assert isinstance(i_model_name,str)
        if i_model_name=='VGG16':
            return tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=self.input_shape,pooling=i_pooling)
        elif i_model_name=='VGG19':
            return tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_shape=self.input_shape,pooling=i_pooling)
        elif i_model_name == 'ResNet50':
            return tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=self.input_shape,pooling=i_pooling)
        elif i_model_name == 'ResNet101':
            return tf.keras.applications.ResNet101(include_top=False, weights='imagenet', input_shape=self.input_shape,pooling=i_pooling)
        elif i_model_name == 'ResNet152':
            return tf.keras.applications.ResNet152(include_top=False, weights='imagenet', input_shape=self.input_shape,pooling=i_pooling)
        elif i_model_name == 'ResNet50V2':
            return tf.keras.applications.ResNet50V2(include_top=False, weights='imagenet', input_shape=self.input_shape,pooling=i_pooling)
        elif i_model_name == 'ResNet101V2':
            return tf.keras.applications.ResNet101V2(include_top=False, weights='imagenet', input_shape=self.input_shape,pooling=i_pooling)
        elif i_model_name == 'ResNet152V2':
            return tf.keras.applications.ResNet152V2(include_top=False, weights='imagenet', input_shape=self.input_shape,pooling=i_pooling)
        elif i_model_name == 'InceptionV3':
            return tf.keras.applications.InceptionV3(include_top=False, weights='imagenet', input_shape=self.input_shape,pooling=i_pooling)
        elif i_model_name == 'InceptionResNetV2':
            return tf.keras.applications.InceptionResNetV2(include_top=False, weights='imagenet', input_shape=self.input_shape,pooling=i_pooling)
        elif i_model_name == 'MobileNetV2':
            return tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=self.input_shape,pooling=i_pooling)
        elif i_model_name == 'DenseNet121':
            return tf.keras.applications.DenseNet121(include_top=False, weights='imagenet', input_shape=self.input_shape,pooling=i_pooling)
        elif i_model_name == 'DenseNet169':
            return tf.keras.applications.DenseNet169(include_top=False, weights='imagenet', input_shape=self.input_shape,pooling=i_pooling)
        elif i_model_name == 'DenseNet201':
            return tf.keras.applications.DenseNet201(include_top=False, weights='imagenet', input_shape=self.input_shape,pooling=i_pooling)
        elif i_model_name == 'NASNetLarge':
            return tf.keras.applications.NASNetLarge(include_top=False, weights='imagenet', input_shape=self.input_shape,pooling=i_pooling)
        elif i_model_name == 'Xception':
            return tf.keras.applications.Xception(include_top=False, weights='imagenet', input_shape=self.input_shape,pooling=i_pooling)
        elif i_model_name == 'ResNeXt50':
            """Reference: https://github.com/qubvel/classification_models"""
            ResNeXt50, preprocess_input = Classifiers.get('resnext50')
            base_model = ResNeXt50(include_top=False,input_shape=self.input_shape)
            output     = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
            model      = tf.keras.models.Model(inputs=[base_model.input], outputs=[output])
            return model
        else:
            return tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=self.input_shape,pooling=i_pooling)

    @classmethod
    def imresize(cls, i_image=None, i_tsize=(512, 512), i_min=None, i_max=None):
        """i_tsize in format (height, width). But, PIL.Image.fromarray.resize(width, height). So, we must invert i_tsize before scaling"""
        assert isinstance(i_image, np.ndarray)
        assert isinstance(i_tsize, (list, tuple))
        assert len(i_tsize) == 2
        assert i_tsize[0] > 0
        assert i_tsize[1] > 0
        tsize = (i_tsize[1], i_tsize[0])
        if i_image.dtype != np.uint8:  # Using min-max scaling to make unit8 image
            if i_min is None:
                min_val = np.min(i_image)
            else:
                assert isinstance(i_min, (int, float))
                min_val = i_min
            if i_max is None:
                max_val = np.max(i_image)
            else:
                assert isinstance(i_max, (int, float))
                max_val = i_max
            assert min_val <= max_val
            image = (i_image - min_val) / (max_val - min_val + 1e-10)
            image = (image * 255.0).astype(np.uint8)
        else:
            image = i_image.copy()
        assert image.dtype == np.uint8
        assert isinstance(image, np.ndarray)
        image = Image.fromarray(np.squeeze(image)).resize(tsize)
        image = np.asarray(image)
        if len(image.shape) == 2:
            image = np.expand_dims(image, -1)
        else:
            assert len(image.shape) == 3
        return image
    def extract_image_features(self,i_image=None):
        """Extract deep features for a single image"""
        assert isinstance(i_image, np.ndarray)
        """Making uniform size of input image"""
        """Image size normalization"""
        image = self.imresize(i_image=i_image, i_tsize=self.input_shape[0:2])
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
            image = np.concatenate((image, image, image), axis=-1)
        else:
            if image.shape[-1] == 1:
                image = np.concatenate((image, image, image), axis=-1)
            else:
                pass
        assert image.shape[-1] == 3
        """Image value normalization"""
        #image = (image - np.min(image)) / (np.max(image) - np.min(image))
        image = image/255.0
        """Making batch of image"""
        image = np.expand_dims(image, axis=0)
        """Feature extraction"""
        features = None
        for index, model in enumerate(self.models):
            assert isinstance(model,tf.keras.models.Model)
            feature = model.predict(image)[0]
            if index==0:
                features = feature
            else:
                features = np.concatenate((features,feature),axis=-1)
        return features
    def extract_db_features(self,i_db=None):
        """Extract deep features for a dataset (multiple images)"""
        assert isinstance(i_db, (list, tuple))  # i_db is a list of images or list of tuple of (image,mask,label) or something like that
        features = []
        print('Extracting image features using pretrained CNN models...')
        print('Input dataset is a list of images or list of tuple of (image,mask,label) or something like that')
        for item in i_db:
            if isinstance(item, (list, tuple)):
                image = item[0]
            else:
                assert isinstance(item, np.ndarray)
                image = item.copy()
            assert len(image.shape) in (2, 3)
            feature = self.extract_image_features(i_image=image)
            features.append(feature.flatten())
        features = np.array(features)
        print('Finished extracting image features using pretrained CNN models...')
        print('Final feature shape = {}'.format(features.shape))
        return features
    """Build a feature extractor generator in form of tf.data.Dataset object"""
    def aug_image(self,i_image=None):
        assert isinstance(i_image,tf.Tensor)
        if self.flip_ud:
            image = tf.image.random_flip_up_down(i_image)
        else:
            image = tf.identity(i_image)
        if self.flip_lr:
            image = tf.image.random_flip_left_right(image)
        else:
            pass
        if image.shape[-1]==1:
            image = tf.concat((image,image,image),axis=-1)
        else:
            pass
        image = tf.image.random_crop(value=image, size=self.crop_size)
        image = tf.image.resize(image, size=self.input_shape[0:2])
        """Normalization"""
        image = tf.cast(image, tf.dtypes.float32) / 255.0
        image = tf.expand_dims(image,axis=0)
        """Feature extraction"""
        features = None
        for index, model in enumerate(self.models):
            assert isinstance(model, tf.keras.models.Model)
            feature = model(image)[0]
            if index == 0:
                features = feature
            else:
                features = tf.concat((features, feature), axis=-1)
        return features
    def extract_db_features_with_data_augumentation(self,i_db=None,i_repeat=10,i_batch_size=4,i_is_train_db=True):
        assert isinstance(i_db,(list,tuple))#In format (image,label)
        assert isinstance(i_repeat,int)
        assert isinstance(i_batch_size,int)
        assert i_repeat>0
        assert i_batch_size>0
        for item in i_db:
            assert isinstance(item,(list,tuple))
        images,labels = list(zip(*i_db))
        images  = np.array(images)
        labels  = np.array(labels).reshape((-1,1))
        images  = tf.data.Dataset.from_tensor_slices(images)
        labels  = tf.data.Dataset.from_tensor_slices(labels)
        dataset = tf.data.Dataset.zip((images,labels))
        if i_is_train_db:
            dataset = dataset.map(map_func=lambda image,label:(self.aug_image(i_image=image),label)).shuffle(buffer_size=len(i_db)).repeat(i_repeat).batch(i_batch_size)
        else:
            dataset = dataset.map(map_func=lambda image, label: (self.aug_image(i_image=image), label)).batch(i_batch_size)
        return dataset
    def convert_pair(self,i_image_a=None,i_image_b=None,i_label=None):
        return (self.aug_image(i_image_a),self.aug_image(i_image_b)),i_label
if __name__ == '__main__':
    print('Extract image features using imagenet-based pretrained models ')
    print('Reference: https://keras.io/api/applications/')
    print('Reference: https://github.com/qubvel/classification_models')
    import matplotlib.pyplot as plt
    extractor = DeepFeatures(i_models= (('VGG16','avg'),('ResNeXt50','avg')), i_input_shape=(256, 256, 3))
    example_image = np.random.randint(low=0,high=255,size=(256,256,3))
    train_features = extractor.extract_db_features(i_db=((example_image,1),))
    plt.plot(train_features[0])
    plt.show()
"""=================================================================================================================="""