""""====================================================================================================================
- This module is to train or infer the result for brain tumor image classification using our proposed method
====================================================================================================================="""
import os
import imageio
import matplotlib.pyplot as plt
from BrainNet.networks import FewShotLearning
def load_support_set(i_support_path=None):
    assert isinstance(i_support_path,str)
    assert os.path.exists(i_support_path)
    sp_sets = []
    dirs = os.listdir(i_support_path)
    for label in dirs:
        class_label = int(label)
        images = os.listdir(os.path.join(i_support_path,label))
        for image in images:
            image_path = os.path.join(i_support_path,label,image)
            sp_sets.append((imageio.imread(image_path),class_label))
    return sp_sets
"""0. Prepare Data. If inference, then put the db_train and db_val to None"""
"""db_train and db_val are the training and validation dataset in format: ((image,label),...,(image,label))"""
db_train = None
db_val   = None
support_set = load_support_set(i_support_path=os.path.join(os.getcwd(),'support_sets'))
test_image_name = 'image_0.jpg'
"""1. Init model and train"""
pretrained_nets = (('DenseNet169','avg'),('InceptionV3','avg'),('ResNeXt50','avg'))
pnets = None
for index, pnet in enumerate(pretrained_nets):
    if index==0:
        pnets = pnet[0]
    else:
        pnets += '_{}'.format(pnet[0])
ckpts_path = os.path.join(os.getcwd(),'ckpts','ckpts_BTSmall',pnets)
print('Looking for/Saving to: {}'.format(ckpts_path))
cls_model = FewShotLearning(i_models       = pretrained_nets,
                            i_input_shape  = (256,256,3),
                            i_num_classes  = 2,
                            i_train_db     = db_train,
                            i_val_db       = db_val,
                            i_clsnet_epoch = 10,
                            i_fsnet_epoch  = 10,
                            i_ckpts_path   = ckpts_path)
"""Update the support set"""
cls_model.update_support_set(i_support_set=support_set)
"""3. Load image and prediction"""
tumor_image_path = os.path.join(os.getcwd(),'images','tumor',test_image_name)
nontumor_image_path = os.path.join(os.getcwd(),'images','notumor',test_image_name)
tumor_image = imageio.imread(tumor_image_path)
nontumor_image = imageio.imread(nontumor_image_path)
tumor_image_pred = cls_model.predict_fusion(i_image=tumor_image,i_rule='wsum',i_weights=(1.0,0.0,0.0,0.0,0.0,0.0))
nontumor_image_pred = cls_model.predict_fusion(i_image=nontumor_image,i_rule='wsum',i_weights=(1.0,0.0,0.0,0.0,0.0,0.0))
plt.subplot(1,2,1)
plt.imshow(tumor_image)
plt.title('GroundTruth = {} vs Prediction = {}'.format('Tumor','Tumor' if tumor_image_pred==1 else 'NonTumor'))
plt.subplot(1,2,2)
plt.imshow(nontumor_image)
plt.title('GroundTruth = {} vs Prediction = {}'.format('NonTumor','Tumor' if nontumor_image_pred==1 else 'NonTumor'))
plt.show()
"""========================================================Finish===================================================="""