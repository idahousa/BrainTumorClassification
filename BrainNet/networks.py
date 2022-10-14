import os
import random
import imageio
import numpy as np
import scipy.special
import tensorflow as tf
from BrainNet.libs.logs import Logs
from BrainNet.libs.deep_features import DeepFeatures
from BrainNet.libs.ml.svms import SupportVectorMachines
class FewShotLearning:
    debug = False
    def __init__(self,
                 i_models       = (('DenseNet169','avg'),('InceptionV3','avg'),('ResNeXt50','avg')),
                 i_input_shape  = (256,256,3),
                 i_num_classes  = 2,
                 i_train_db     = None,
                 i_val_db       = None,
                 i_ckpts_path   = None,
                 i_clsnet_epoch = 10,
                 i_fsnet_epoch  = 10,
                 i_suffix       = 'images'):
        assert isinstance(i_models,(list,tuple))
        for model in i_models:
            assert isinstance(model,(list,tuple))
        assert isinstance(i_input_shape,(list,tuple)) #In format (height, width, depth)
        if i_train_db is not None:
            assert isinstance(i_train_db,(list,tuple))    #In format (image,label)
        else:
            pass
        assert isinstance(i_num_classes,int)
        assert i_num_classes>0
        assert isinstance(i_suffix,str)
        self.suffix = i_suffix
        self.models      = i_models
        self.input_shape = i_input_shape
        self.num_classes = i_num_classes
        if i_ckpts_path is None:
            self.ckpts_path = os.path.join(os.getcwd(),'ckpts')
        else:
            self.ckpts_path = i_ckpts_path
        if not os.path.exists(self.ckpts_path):
            os.makedirs(self.ckpts_path)
        else:
            pass
        self.model_path = os.path.join(self.ckpts_path,'Models')
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        else:
            pass
        """1. Extract deep features using pretrained CNN network on imagenet dataset"""
        self.feature_extractor = DeepFeatures(i_models=self.models,i_input_shape=self.input_shape)
        self.train_db_full     = i_train_db
        self.val_db_full       = i_val_db
        if i_train_db is not  None:
            self.train_labels      = np.array(list(zip(*i_train_db))[-1])
            self.train_db_features = self.feature_extractor.extract_db_features(i_db=i_train_db)
            """At the begining, the support set is set to be same as the train set"""
            self.support_labels    = np.array(list(zip(*i_train_db))[-1])
            self.support_features  = self.feature_extractor.extract_db_features(i_db=i_train_db)
        else:
            self.train_labels      = None
            self.train_db_features = None
            self.support_labels    = None
            self.support_features  = None
        if i_val_db is not None:
            self.val_db_features  = self.feature_extractor.extract_db_features(i_db=i_val_db)
            self.val_labels       = np.array(list(zip(*i_val_db))[-1])
        else:
            self.val_db_features  = None
            self.val_labels = None
        if self.train_db_features is not None:
            self.feature_shape  = (self.train_db_features.shape[-1],)
        else:
            self.feature_shape = None
        if self.train_db_full is not None:
            self.visualize_data(i_db=self.train_db_full,i_num_rows=4,i_prefix='train')
        else:
            pass
        if self.val_db_full is not None:
            self.visualize_data(i_db=self.val_db_full,i_num_rows=10,i_prefix='val')
        else:
            pass
        """SVM models"""
        self.svm_linear_model       = None
        self.svm_linear_final_model = None
        self.svm_linear_final_model_v3 = None
        self.svm_linear_model_path       = os.path.join(self.model_path,'svm_linear.obj')
        self.svm_linear_final_model_path = os.path.join(self.model_path,'svm_linear_final.obj')
        self.svm_linear_final_model_path_v3 = os.path.join(self.model_path, 'svm_linear_final_v3x.obj')
        self.svm_rbf_model          = None
        self.svm_rbf_final_model    = None
        self.svm_rbf_final_model_v3 = None
        self.svm_rbf_model_path       = os.path.join(self.model_path, 'svm_rbf.obj')
        self.svm_rbf_final_model_path = os.path.join(self.model_path, 'svm_rbf_final.obj')
        self.svm_rbf_final_model_path_v3 = os.path.join(self.model_path, 'svm_rbf_final_v3x.obj')
        self.svm_poly_model         = None
        self.svm_poly_final_model   = None
        self.svm_poly_final_model_v3 = None
        self.svm_poly_model_path       = os.path.join(self.model_path, 'svm_poly.obj')
        self.svm_poly_final_model_path = os.path.join(self.model_path, 'svm_poly_final.obj')
        self.svm_poly_final_model_path_v3 = os.path.join(self.model_path, 'svm_poly_final_v3x.obj')
        self.svm_sigmoid_model       = None
        self.svm_sigmoid_final_model = None
        self.svm_sigmoid_final_model_v3 = None
        self.svm_sigmoid_model_path       = os.path.join(self.model_path, 'svm_sigmoid.obj')
        self.svm_sigmoid_final_model_path = os.path.join(self.model_path, 'svm_sigmoid_final.obj')
        self.svm_sigmoid_final_model_path_v3 = os.path.join(self.model_path, 'svm_sigmoid_final_v3x.obj')
        self.init_SVMs()
        """CNN model"""
        self.cls_net_path = os.path.join(self.model_path,'ClsNet.h5')
        if os.path.exists(self.cls_net_path):
            self.ClsNet = tf.keras.models.load_model(self.cls_net_path,custom_objects={})
        else:
            if self.train_db_full is not None:
                self.ClsNet = self.build_clsnet(i_input_shape=self.feature_shape,i_num_classes=self.num_classes,i_name='ClsNet')
                self.train_ClsNet(i_cls_lr=0.0001, i_db_repeat=2, i_batch_size=4, i_num_epochs=i_clsnet_epoch)
            else:
                raise Exception('Invalid training data')
        """FS model"""
        self.fs_net_path = os.path.join(self.model_path,'FsNet.h5')
        if os.path.exists(self.fs_net_path):
            self.FsNet =  tf.keras.models.load_model(self.fs_net_path,custom_objects={})
        else:
            if self.train_db_full is not None:
                self.FsNet  = self.build_fsnet(i_input_shape=self.feature_shape,i_name='FsNet')
                self.train_FsNet(i_fs_lr=0.0001, i_db_repeat=2, i_batch_size=4, i_num_epochs=i_fsnet_epoch)
            else:
                raise Exception('Invalid training data')
    def visualize_data(self,i_db=None,i_num_rows=4,i_prefix='train'):
        """i_db is list or tuple in format (image,label)"""
        save_data, save_labels = list(zip(*i_db))  # As my design. See the  BTSmall2C_DB() for more details
        """Making images for training and testing sets"""
        num_rows = i_num_rows
        composite_images = [[] for _ in range(self.num_classes)]
        for label in range(self.num_classes):
            for index, image in enumerate(save_data):
                if save_labels[index] == label:
                    composite_images[label].append(image)
            if len(composite_images[label]) % num_rows == 0:
                pass
            else:
                while True:
                    composite_images[label].append(np.zeros(shape=self.input_shape, dtype=np.uint8))
                    if len(composite_images[label]) % num_rows == 0:
                        break
                    else:
                        continue
            num_cols = len(composite_images[label]) // num_rows
            final_image = np.zeros(shape=(num_rows * self.input_shape[0], num_cols * self.input_shape[1], self.input_shape[2]))
            for i in range(num_rows):
                for j in range(num_cols):
                    final_image[i * self.input_shape[0]:(i + 1) * self.input_shape[1],j * self.input_shape[0]:(j + 1) * self.input_shape[1], :] = composite_images[label][i * num_cols + j]
            imageio.imwrite(os.path.join(self.ckpts_path,'{}_images_{}.jpg'.format(i_prefix,label)), final_image)
    def update_support_set(self,i_support_set=None):
        assert isinstance(i_support_set,(list,tuple)) #In format ((image,label),...,(image,label)).
        """Update the support sets that to be different from the training set. Used for update new well-known cases"""
        self.support_labels  = np.array(list(zip(*i_support_set))[-1])
        self.support_features = self.feature_extractor.extract_db_features(i_db=i_support_set)
        Logs.log('Finished update the support sets')
    """1. Train the SVM model for classification"""
    """2. Load or Train SVM-based model for image classificaiton using extracted deep features"""
    def init_SVMs(self):
        """Init svm models"""
        """Init Linear model"""
        if os.path.exists(self.svm_linear_model_path):
            self.svm_linear_model = SupportVectorMachines().load(self.svm_linear_model_path)
        else:
            if self.train_db_features is not None and self.train_labels is not None:
                self.svm_linear_model = SupportVectorMachines(i_kernel_name = 'linear',
                                                              i_znorm_flag  = False,
                                                              i_pca_flag    = False,
                                                              i_pca_comps   = 10,
                                                              i_num_classes = self.num_classes).train(i_features=self.train_db_features,i_labels=self.train_labels)
                self.svm_linear_model.save(i_file_path=self.svm_linear_model_path)
                self.svm_linear_model.evaluation(i_db=list(zip(self.val_db_features,self.val_labels)))
            else:
                raise Exception('Invalid training data')
        if os.path.exists(self.svm_linear_final_model_path):
            self.svm_linear_final_model = SupportVectorMachines().load(self.svm_linear_final_model_path)
        else:
            pass
        if os.path.exists(self.svm_linear_final_model_path_v3):
            self.svm_linear_final_model_v3 = SupportVectorMachines().load(self.svm_linear_final_model_path_v3)
        else:
            pass
        """Poly model"""
        if os.path.exists(self.svm_poly_model_path):
            self.svm_poly_model = SupportVectorMachines().load(self.svm_poly_model_path)
        else:
            if self.train_db_features is not None and self.train_labels is not None:
                self.svm_poly_model = SupportVectorMachines(i_kernel_name   = 'poly',
                                                            i_znorm_flag  = False,
                                                            i_pca_flag    = False,
                                                            i_pca_comps   = 10,
                                                            i_num_classes = self.num_classes).train(i_features=self.train_db_features,i_labels=self.train_labels)
                self.svm_poly_model.save(i_file_path=self.svm_poly_model_path)
                self.svm_poly_model.evaluation(i_db=list(zip(self.val_db_features, self.val_labels)))
            else:
                raise Exception('Invalid training data')
        if os.path.exists(self.svm_poly_final_model_path):
            self.svm_poly_final_model = SupportVectorMachines().load(self.svm_poly_final_model_path)
        else:
            pass
        if os.path.exists(self.svm_poly_final_model_path_v3):
            self.svm_poly_final_model_v3 = SupportVectorMachines().load(self.svm_poly_final_model_path_v3)
        else:
            pass
        """RBF model"""
        if os.path.exists(self.svm_rbf_model_path):
            self.svm_rbf_model = SupportVectorMachines().load(self.svm_rbf_model_path)
        else:
            if self.train_db_features is not None and self.train_labels is not None:
                self.svm_rbf_model = SupportVectorMachines(i_kernel_name    = 'rbf',
                                                              i_znorm_flag  = False,
                                                              i_pca_flag    = False,
                                                              i_pca_comps   = 10,
                                                              i_num_classes = self.num_classes).train(i_features=self.train_db_features,i_labels=self.train_labels)
                self.svm_rbf_model.save(i_file_path=self.svm_rbf_model_path)
                self.svm_rbf_model.evaluation(i_db=list(zip(self.val_db_features, self.val_labels)))
            else:
                raise Exception('Invalid training data')
        if os.path.exists(self.svm_rbf_final_model_path):
            self.svm_rbf_final_model = SupportVectorMachines().load(self.svm_rbf_final_model_path)
        else:
            pass
        if os.path.exists(self.svm_rbf_final_model_path_v3):
            self.svm_rbf_final_model_v3 = SupportVectorMachines().load(self.svm_rbf_final_model_path_v3)
        else:
            pass
        """Sigmoid model"""
        if os.path.exists(self.svm_sigmoid_model_path):
            self.svm_sigmoid_model = SupportVectorMachines().load(self.svm_sigmoid_model_path)
        else:
            if self.train_db_features is not None and self.train_labels is not None:
                self.svm_sigmoid_model = SupportVectorMachines(i_kernel_name = 'sigmoid',
                                                              i_znorm_flag   = False,
                                                              i_pca_flag     = False,
                                                              i_pca_comps    = 10,
                                                              i_num_classes  = self.num_classes).train(i_features=self.train_db_features,i_labels=self.train_labels)
                self.svm_sigmoid_model.save(i_file_path=self.svm_sigmoid_model_path)
                self.svm_sigmoid_model.evaluation(i_db=list(zip(self.val_db_features, self.val_labels)))
            else:
                raise Exception('Invalid training data')
        if os.path.exists(self.svm_sigmoid_final_model_path):
            self.svm_sigmoid_final_model = SupportVectorMachines().load(self.svm_sigmoid_final_model_path)
        else:
            pass
        if os.path.exists(self.svm_sigmoid_final_model_path_v3):
            self.svm_sigmoid_final_model_v3 = SupportVectorMachines().load(self.svm_sigmoid_final_model_path_v3)
        else:
            pass
    """3. Build a CNN-based classification network based on MLBP with extraced image features (fine-tuning) (ClsNet)"""
    @staticmethod
    def build_clsnet(i_input_shape=(1024,),i_num_classes = 2, i_name='ClsNet'):
        assert isinstance(i_input_shape,(list,tuple))
        assert len(i_input_shape)==1 #We use the input is the extracted image features
        assert isinstance(i_num_classes,int)
        assert i_num_classes>0
        inputs = tf.keras.layers.Input(shape=i_input_shape, name=i_name + "_input")
        outputs = tf.keras.layers.Dense(units=512,activation='relu')(inputs)
        outputs = tf.keras.layers.Dense(units=1024, activation='relu')(outputs)
        outputs = tf.keras.layers.Dropout(rate=0.1)(outputs)
        outputs = tf.keras.layers.Dense(units=i_num_classes)(outputs)
        model   = tf.keras.models.Model(inputs=inputs, outputs=outputs, name=i_name)
        model.summary()
        return model
    """4. Build MLBP based few-shot learning network (FSNet)"""
    @staticmethod
    def get_transformer(i_input_shape=(1024,)):
        inputs  = tf.keras.layers.Input(shape=i_input_shape, name="transformer_input")
        outputs = tf.keras.layers.Dense(units=512, activation='relu')(inputs)
        outputs = tf.keras.layers.Dense(units=1024, activation='relu')(outputs)
        model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name='transformer')
        return model
    @staticmethod
    def build_fsnet(i_input_shape=(1024,),i_name='FsNet'):
        assert isinstance(i_input_shape, (list, tuple))
        assert len(i_input_shape) == 1  # We use the input is the extracted image features
        input_a = tf.keras.layers.Input(shape=i_input_shape, name=i_name + "_input_a")
        input_b = tf.keras.layers.Input(shape=i_input_shape, name=i_name + "_input_b")
        transformer = FewShotLearning.get_transformer(i_input_shape=i_input_shape)
        output_a = transformer(input_a)
        output_b = transformer(input_b)
        """Cosine distance"""
        output_sim = tf.reduce_sum(tf.keras.layers.Multiply()([output_a, output_b]), axis=-1)
        output_sim = tf.math.divide(output_sim,tf.math.sqrt(tf.math.reduce_sum(tf.math.multiply(output_a, output_a), axis=-1)))
        output_sim = tf.math.divide(output_sim,tf.math.sqrt(tf.math.reduce_sum(tf.math.multiply(output_b, output_b), axis=-1)))
        output_sim = tf.abs(output_sim)
        model = tf.keras.models.Model(inputs=[input_a, input_b], outputs=[output_sim],name=i_name)
        model.summary()
        return model
    """5. Train the ClsNet and FSNet"""
    def train_ClsNet(self,i_cls_lr=0.0001,i_db_repeat=10,i_batch_size=4,i_num_epochs=10):
        assert isinstance(i_cls_lr,float)
        assert 0<i_cls_lr<0.001
        assert isinstance(i_db_repeat,int)
        assert i_db_repeat>0
        assert isinstance(i_batch_size,int)
        assert i_batch_size>0
        assert isinstance(i_num_epochs,int)
        assert i_num_epochs>0
        """Train ClsNet and FsNet using training and validation data"""
        assert isinstance(self.ClsNet, tf.keras.Model)
        self.ClsNet.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=i_cls_lr),
                            loss      = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                            metrics   = [tf.keras.metrics.SparseCategoricalAccuracy()])
        cls_callback = Monitor(i_save_path=self.model_path,i_model_name='ClsNet.h5')
        train_db = self.feature_extractor.extract_db_features_with_data_augumentation(i_db=self.train_db_full,i_repeat=i_db_repeat,i_batch_size=i_batch_size,i_is_train_db=True)
        val_db   = self.feature_extractor.extract_db_features_with_data_augumentation(i_db=self.val_db_full,i_batch_size=i_batch_size,i_is_train_db=False)
        self.ClsNet.fit(x               = train_db,
                        epochs          = i_num_epochs,
                        verbose         = 1,
                        shuffle         = True,
                        validation_data = val_db,
                        callbacks       = [cls_callback])
    def eval_ClsNet(self,i_db=None):
        assert isinstance(i_db, (list, tuple))  # In format: (image,label)
        confusion_matrix = np.zeros(shape=(self.num_classes, self.num_classes))
        for index, (image, label) in enumerate(i_db):
            print('Index = {}'.format(index))
            preds = self.predict(i_image=image, i_rule='max', i_all_predictions=True)
            pred = preds[1] # As my design. See the 'predict' for more details
            confusion_matrix[label][pred] += 1
            """Save images"""
            if self.debug:
                save_path = os.path.join(self.ckpts_path, 'results_sp_size_of_{}_{}'.format(len(self.support_labels),self.suffix), 'CNNs')
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                else:
                    pass
                save_image_path = os.path.join(save_path, 'image_index_{}_pred_{}_to_{}.jpg'.format(index, label, pred))
                imageio.imwrite(save_image_path, image)
            else:
                pass
        Logs.log_matrix('Confusion Matrix (ClsNet)', confusion_matrix)
        Logs.log('Accuracy: {:0.3f} (%)'.format(np.trace(confusion_matrix) * 100 / np.sum(confusion_matrix)))
        return confusion_matrix
    def eval_SVMs(self,i_db=None,i_kernel='linear'):
        assert isinstance(i_db, (list, tuple))  # In format: (image,label)
        assert isinstance(i_kernel,str)
        assert i_kernel in ('linear','rbf','poly','sigmoid')
        confusion_matrix = np.zeros(shape=(self.num_classes, self.num_classes))
        for index, (image, label) in enumerate(i_db):
            print('Index = {}'.format(index))
            preds = self.predict(i_image=image, i_rule='max', i_all_predictions=True,i_svms=True)
            if i_kernel == 'linear':
                pred = preds[0]   # As my design. See the 'predict' for more details
            elif i_kernel == 'rbf':
                pred = preds[1]   # As my design. See the 'predict' for more details
            elif i_kernel == 'poly':
                pred = preds[2]   # As my design. See the 'predict' for more details
            else:
                assert i_kernel == 'sigmoid'
                pred = preds[3]   # As my design. See the 'predict' for more details
            confusion_matrix[label][pred] += 1
            """Save images"""
            if self.debug:
                save_path = os.path.join(self.ckpts_path,'results_sp_size_of_{}_{}'.format(len(self.support_labels),self.suffix),'svms',i_kernel)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                else:
                    pass
                save_image_path = os.path.join(save_path,'image_index_{}_pred_{}_to_{}.jpg'.format(index,label,pred))
                imageio.imwrite(save_image_path,image)
            else:
                pass
        Logs.log_matrix('Confusion Matrix (SVM - {})'.format(i_kernel), confusion_matrix)
        Logs.log('Accuracy: {:0.3f} (%)'.format(np.trace(confusion_matrix) * 100 / np.sum(confusion_matrix)))
        return confusion_matrix
    def train_FsNet(self,i_fs_lr=0.0001,i_db_repeat=10,i_batch_size=4,i_num_epochs=10):
        """Init all possible pair in training and validation dataset"""
        assert isinstance(i_num_epochs,int)
        assert i_num_epochs>0
        train_db,val_db=[],[]
        for (image_a,label_a) in self.train_db_full:
            for (image_b,label_b) in self.train_db_full:
                if label_a==label_b:
                    train_db.append((image_a,image_b,1))
                else:
                    train_db.append((image_a,image_b,0))
        for (image_a,label_a) in self.val_db_full:
            for (image_b,label_b) in self.val_db_full:
                if label_a==label_b:
                    val_db.append((image_a,image_b,1))
                else:
                    val_db.append((image_a,image_b,0))
        print('Summary: train_db length: {} vs val_db length: {}'.format(len(train_db),len(val_db)))
        """Create tf.data.Dataset objects for train_db and val_db"""
        train_db_x,train_db_y,train_db_l = list(zip(*train_db))
        val_db_x,val_db_y,val_db_l       = list(zip(*val_db))
        """Train dataset"""
        train_db_x = np.array(train_db_x)
        train_db_y = np.array(train_db_y)
        train_db_l = np.array(train_db_l).reshape((-1,1))
        train_db = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(train_db_x),
                                        tf.data.Dataset.from_tensor_slices(train_db_y),
                                        tf.data.Dataset.from_tensor_slices(train_db_l)))
        train_db = train_db.map(lambda img_a, img_b, label: self.feature_extractor.convert_pair(img_a, img_b, label)).repeat(i_db_repeat).batch(i_batch_size)
        """Validation dataset"""
        val_db_x = np.array(val_db_x)
        val_db_y = np.array(val_db_y)
        val_db_l = np.array(val_db_l).reshape((-1, 1))
        val_db   = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(val_db_x),
                                        tf.data.Dataset.from_tensor_slices(val_db_y),
                                        tf.data.Dataset.from_tensor_slices(val_db_l)))
        val_db   = val_db.map(lambda img_a, img_b, label: self.feature_extractor.convert_pair(img_a, img_b, label)).batch(i_batch_size)
        """Train the model"""
        assert isinstance(self.FsNet, tf.keras.Model)
        self.FsNet.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=i_fs_lr),
                           loss      = tf.keras.losses.BinaryCrossentropy(),
                           metrics   = [tf.keras.metrics.BinaryAccuracy()])
        fs_callback = Monitor(i_save_path=self.model_path, i_model_name='FsNet.h5')
        self.FsNet.fit(x               = train_db,
                       epochs          = i_num_epochs,
                       verbose         = 1,
                       shuffle         = True,
                       validation_data = val_db,#None (for fast) or val_db for measuring the validation loss
                       callbacks       = [fs_callback])
    def eval_FsNet(self,i_db=None,i_rule='max'):
        assert isinstance(i_db, (list, tuple))  # In format: (image,label)
        assert isinstance(i_rule,str)
        assert i_rule in ('max','avg')
        confusion_matrix = np.zeros(shape=(self.num_classes, self.num_classes))
        for index, (image, label) in enumerate(i_db):
            print('Index = {}'.format(index))
            preds = self.predict(i_image=image,i_rule=i_rule,i_all_predictions=True)
            if i_rule=='max':
                pred = preds[-1][0][0] #As my design. See the 'predict' for more details
            else:
                assert i_rule =='avg'
                pred = preds[-1][1][0] #As my design. See the 'predict' for more details
            confusion_matrix[label][pred] += 1
            """Save images"""
            if self.debug:
                save_path = os.path.join(self.ckpts_path, 'results_sp_size_of_{}_{}'.format(len(self.support_labels),self.suffix), 'FS', i_rule)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                else:
                    pass
                save_image_path = os.path.join(save_path, 'image_index_{}_pred_{}_to_{}.jpg'.format(index, label, pred))
                imageio.imwrite(save_image_path, image)
            else:
                pass
        Logs.log_matrix('Confusion Matrix (FsNet_{})'.format(i_rule), confusion_matrix)
        Logs.log('Accuracy: {:0.3f} (%)'.format(np.trace(confusion_matrix) * 100 / np.sum(confusion_matrix)))
        return confusion_matrix
    """6. Evaluation classificaiton accuracy of a dataset"""
    def eval(self,i_db=None):
        assert isinstance(i_db,(list,tuple)) #In format: (image,label)
        confusion_matrix = np.zeros(shape=(self.num_classes,self.num_classes))
        for index, (image,label) in enumerate(i_db):
            print('Index = {}'.format(index))
            pred = self.predict(i_image=image)
            confusion_matrix[label][pred]+=1
        Logs.log_matrix('Confusion Matrix  - Proposed@@',confusion_matrix)
        Logs.log('Accuracy: {:0.3f} (%)'.format(np.trace(confusion_matrix)*100/np.sum(confusion_matrix)))
        return confusion_matrix
    """(Main function) Making prediction"""
    def predict(self,i_image=None,i_rule='max',i_all_predictions=False,i_svms=False):
        assert isinstance(i_image,np.ndarray)
        assert isinstance(i_rule,str)
        assert isinstance(i_all_predictions,bool)
        assert i_rule in ('max','avg')
        assert isinstance(i_svms,bool)
        assert isinstance(self.svm_linear_model,SupportVectorMachines)
        assert isinstance(self.svm_poly_model, SupportVectorMachines)
        assert isinstance(self.svm_sigmoid_model, SupportVectorMachines)
        assert isinstance(self.svm_rbf_model, SupportVectorMachines)
        assert isinstance(self.ClsNet,tf.keras.Model)
        assert isinstance(self.FsNet,tf.keras.Model)
        """0. Extract image features"""
        features    = self.feature_extractor.extract_image_features(i_image=i_image).reshape((1,-1))
        """2. SVM model predction"""
        linear_svm  = self.svm_linear_model.predict(i_features=features)
        rbf_svm     = self.svm_rbf_model.predict(i_features=features)
        poly_svm    = self.svm_poly_model.predict(i_features=features)
        sigmoid_svm = self.svm_sigmoid_model.predict(i_features=features)
        if i_svms:#Return svm-based prediction
            return linear_svm[0],rbf_svm[0],poly_svm[0],sigmoid_svm[0]
        else:
            pass
        """3. Deep model prediction"""
        cls_pred = self.ClsNet.predict(features)
        cls_pred = np.argmax(cls_pred[0])
        """FS model prediction"""
        fs_preds = []
        for index, f in enumerate(self.support_features):
            fs_pred = self.FsNet.predict((features,f.reshape((1,-1))))[0]
            fs_preds.append(fs_pred)
        """Prediction based on max rule"""
        fs_pred_max       = self.support_labels[np.argmax(fs_preds)]
        fs_pred_max_score = fs_preds[np.argmax(fs_preds)]
        fs_pred_max_index = np.argmax(fs_preds)
        """Prediction based on avg rule"""
        fs_pred_avg       = 0
        fs_pred_avg_score = 0.0
        fs_pred_avg_index = None
        match_score       = 0.0
        for label in range(self.num_classes):
            segment  = [fs_preds[index] for index in range(len(fs_preds)) if self.support_labels[index]==label]
            mask     = [fs_preds[index] if self.support_labels[index]==label else -1.0 for index in range(len(fs_preds))]
            avg_score = np.average(segment)
            if fs_pred_avg_score<avg_score:
                fs_pred_avg       = label
                fs_pred_avg_score = avg_score
                match_score       = np.max(segment)
                fs_pred_avg_index = np.argmax(mask)
        fs_pred_max_score = np.round(fs_pred_max_score,3)
        fs_pred_avg_score = np.round(fs_pred_avg_score,3)
        match_score       = np.round(match_score,3)
        svm_preds = (linear_svm[0],rbf_svm[0],poly_svm[0],sigmoid_svm[0])
        cnn_pred  = cls_pred
        sim_preds = (fs_pred_max,fs_pred_max_score,fs_pred_max_index),(fs_pred_avg,fs_pred_avg_score,match_score,fs_pred_avg_index)
        print('Prediction raw:')
        print(svm_preds)
        print(cnn_pred)
        print(sim_preds)
        """Combination rules"""
        if i_all_predictions:
            return svm_preds,cnn_pred,sim_preds
        else:
            if svm_preds[0]==svm_preds[1]==svm_preds[2]==svm_preds[3]==cnn_pred:
                return cnn_pred
            else:
                if i_rule=='max':
                    return sim_preds[0][0]
                else:
                    return sim_preds[1][0]
    """Score level fusion approach"""
    def extract_score(self,i_image=None):
        """Extract prediction score for the input image"""
        assert isinstance(i_image, np.ndarray)
        assert isinstance(self.svm_linear_model, SupportVectorMachines)
        assert isinstance(self.svm_poly_model, SupportVectorMachines)
        assert isinstance(self.svm_sigmoid_model, SupportVectorMachines)
        assert isinstance(self.svm_rbf_model, SupportVectorMachines)
        assert isinstance(self.ClsNet, tf.keras.Model)
        assert isinstance(self.FsNet, tf.keras.Model)
        """0. Extract image features"""
        features = self.feature_extractor.extract_image_features(i_image=i_image).reshape((1, -1))
        """2. SVM model predction"""
        linear_svm  = self.svm_linear_model.scoring(i_features=features)
        rbf_svm     = self.svm_rbf_model.scoring(i_features=features)
        poly_svm    = self.svm_poly_model.scoring(i_features=features)
        sigmoid_svm = self.svm_sigmoid_model.scoring(i_features=features)
        """3. Deep model prediction"""
        cls_pred = self.ClsNet.predict(features)[0]
        cls_pred = scipy.special.softmax(cls_pred)
        """FS model prediction"""
        fs_preds = []
        for index, f in enumerate(self.support_features):
            fs_pred = self.FsNet.predict((features, f.reshape((1, -1))))[0]
            fs_preds.append(fs_pred)
        """Prediction score by averaging distance to samples in classes"""
        avg_scores = []
        for label in range(self.num_classes):
            segment = [fs_preds[index] for index in range(len(fs_preds)) if self.support_labels[index] == label]
            avg_scores.append(np.average(segment))
        """Summary results"""
        if self.num_classes==2:
            """In this case, the problem come to one-class classification. Therfore, there is only one decision function"""
            """As a result, the score is a single number and the decision is based on the sign of the evaluation of decision function"""
            """Therefore, if the score is smaller than 0, then prediction is class 0, else prediction is class 1"""
            linear_svm  = [linear_svm[0]*-1,  linear_svm[0]]
            rbf_svm     = [rbf_svm[0]*-1,     rbf_svm[0]]
            poly_svm    = [poly_svm[0]*-1,    poly_svm[0]]
            sigmoid_svm = [sigmoid_svm[0]*-1, sigmoid_svm[0]]
        else:
            assert self.num_classes>2
            linear_svm  = linear_svm[0]
            rbf_svm     = rbf_svm[0]
            poly_svm    = poly_svm[0]
            sigmoid_svm = sigmoid_svm[0]
        linear_svm   = list(scipy.special.softmax(linear_svm))
        rbf_svm      = list(scipy.special.softmax(rbf_svm))
        poly_svm     = list(scipy.special.softmax(poly_svm))
        sigmoid_svm  = list(scipy.special.softmax(sigmoid_svm))
        cls_score    = list(cls_pred)
        fs_score_avg = list(scipy.special.softmax(avg_scores))
        return  np.array(linear_svm),np.array(rbf_svm),np.array(poly_svm),np.array(sigmoid_svm),np.array(cls_score),np.array(fs_score_avg)
    """Weighted SUM and Weighted PRODUCT Prediction"""
    def predict_fusion(self,i_image=None,i_rule='wsum',i_weights=(1.0, 1.0, 0.9, 1.0, 0.2, 0.7)):
        assert isinstance(i_image,np.ndarray)
        assert isinstance(i_rule,str)
        assert isinstance(i_weights,(list,tuple))
        assert len(i_weights)==6
        for weight in i_weights:
            assert isinstance(weight,float)
            assert 0.0<=weight<=1.0
        assert i_rule in ('wsum','wSUM','wPROD','wPRODUCT')
        scores = self.extract_score(i_image=i_image)
        fscore = None
        if i_rule in ('wsum','wSUM'):
            for index,score in enumerate(scores):
                if index==0:
                    fscore = score * i_weights[index]
                else:
                    fscore += score *i_weights[index]
            return np.argmax(fscore)
        else:
            for index, score in enumerate(scores):
                if index == 0:
                    fscore  = pow(score,i_weights[index])
                else:
                    fscore *= pow(score,i_weights[index])
            return np.argmax(fscore)
""""Training monitors"""
class Monitor(tf.keras.callbacks.Callback):
    """A callback to generate and save images after each epoch"""
    def __init__(self,i_save_path=None,i_model_name=None):
        assert isinstance(i_save_path,str)
        assert isinstance(i_model_name,str)
        assert os.path.exists(i_save_path)
        self.ckpts_path  = i_save_path
        self.model_name  = i_model_name
        self.losses      = 0.0
        self.batch_index = 0
        Logs.log('Start Training')
    def on_epoch_end(self, epoch, logs=None):
        Logs.log('Epoch: {} => {}'.format(epoch, logs))
        """Save the models"""
        self.model.save(os.path.join(self.ckpts_path, self.model_name))
    @staticmethod
    def on_train_end(logs=None):
        print(logs)
        Logs.log('Training Ended')
if __name__ == '__main__':
    print('This module is to implement my custom image classification network')
    print('This version perform classification based on three sub-network:')
    print('1. SVM-based classification model using deep features')
    print('2. CNN-based classification model using fine-tuning')
    print('3. Few-shot learning-based classification for multiple classes based on MLP')
    from BrainNet.libs.datasets.hourse2zebra import Horse2Zebra
    db_size = 10
    dataset = Horse2Zebra(i_shuffle=False)
    tX, tY, vX, vY = dataset.get_numpy_db()
    db_train,db_val,db_test=[],[],[]
    for item in tX:
        if len(db_train)>db_size:
            db_test.append((item, 0))
        else:
            db_train.append((item, 0))
    for item in tY:
        if len(db_train)>=db_size*2:
            db_test.append((item, 1))
        else:
            db_train.append((item, 1))
    for item in vX:
        if len(db_val)<db_size:
            db_val.append((item, 0))
        else:
            pass
        db_test.append((item, 0))
    for item in vY:
        if len(db_val)<2*db_size:
            db_val.append((item, 1))
        else:
            pass
        db_test.append((item, 1))
    random.shuffle(db_train)
    print('Train length: {}'.format(len(db_train)))
    print('Val length: {}'.format(len(db_val)))
    print('Test length: {}'.format(len(db_test)))
    """Init model"""
    cls_model = FewShotLearning(i_models       = (('DenseNet169','avg'),),#('InceptionV3','avg'),('ResNeXt50','avg')
                                i_input_shape  = (256,256,3),
                                i_num_classes  = 2,
                                i_train_db     = db_train,
                                i_val_db       = db_val,
                                i_clsnet_epoch = 5,
                                i_fsnet_epoch  = 5,
                                i_ckpts_path   = None)
    cls_model.eval_ClsNet(i_db=db_test)
    cls_model.eval_FsNet(i_db=db_test)
    cls_model.eval_SVMs(i_db=db_test,i_kernel='linear')
    cls_model.eval_SVMs(i_db=db_test, i_kernel='rbf')
    cls_model.eval_SVMs(i_db=db_test, i_kernel='poly')
    cls_model.eval_SVMs(i_db=db_test, i_kernel='sigmoid')
"""=================================================================================================================="""