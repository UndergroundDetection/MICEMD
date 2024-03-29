import numpy as np
import MicEMD.tdem as tdem
from MicEMD.preprocessor import data_prepare
from MicEMD.handler import TDEMHandler
import math

# the attribute of the steel, Ni, and Al including permeability, permeability of vacuum and conductivity
attribute = np.array([[696.3028547, 4*math.pi*1e-7, 50000000], [99.47183638, 4*math.pi*1e-7, 14619883.04],
                      [1.000022202, 4*math.pi*1e-7, 37667620.91]])
# create and initial the target, detector, collection class of Tdem
target = tdem.Target(material=['Steel', 'Ni', 'Al'], shape=['Oblate spheroid', 'Prolate spheroid'],
                     attribute=attribute, ta_min=0.01, ta_max=1.5, tb_min=0.01, tb_max=1.5, a_r_step=0.08,
                     b_r_step=0.08)
detector = tdem.Detector(0.4, 20, 0, 0)
collection = tdem.Collection(t_split=20, snr=30)

# call the simulate interface, the forward_result is a tuple which conclude the Sample Set and a random
# sample of Sample Set, the random sample of Sample Set is used to visualize
fwd_res = tdem.simulate(target, detector, collection, model='dipole')

# split data sets and normalization for the Sample Set, Here we classify materials
ori_dataset_material = data_prepare(fwd_res[0], task='material')

# dimensionality reduction, return a tuple conclude train_set and test_set
dim_dataset_material = tdem.preprocess(ori_dataset_material, dim_red_method='PCA', n_components=20)

# parameters setting of the classification model by dict
para = {'solver': 'lbfgs', 'hidden_layer_sizes': (50,), 'activation': 'tanh'}

# call the classify interface
# the res of the classification which is a dict that conclude accuracy, predicted value and true value
cls_material_res = tdem.classify(dim_dataset_material, 'ANN', para)


# create the TDEMHandler and call the methods to show and save the results
# set the TDEMHandler without parameters to save the results
# the file path of the results is generated by your settings
handler = TDEMHandler()

# save the forward results and one sample data
handler.save_fwd_data(fwd_res[0], file_name='fwd_res.csv')
handler.save_sample_data(fwd_res[1], file_name='sample.csv', show=True)

# save the original dataset that distinguishes material
handler.save_fwd_data(ori_dataset_material[0], file_name='ori_material_train.csv')
handler.save_fwd_data(ori_dataset_material[1], file_name='ori_material_test.csv')

# save the final dataset after dimensionality reduction
handler.save_fwd_data(dim_dataset_material[0], file_name='dim_material_train.csv')
handler.save_fwd_data(dim_dataset_material[1], file_name='dim_material_test.csv')

# save the classification results
handler.show_cls_res(cls_material_res, ['Steel', 'Ni', 'Al'], show=True, save=True, file_name='cls_result_material.pdf')
handler.save_cls_res(cls_material_res, 'cls_material_res.csv')


# classify the shape of the targets
ori_dataset_shape = data_prepare(fwd_res[0], task='shape')
dim_dataset_shape = tdem.preprocess(ori_dataset_shape, dim_red_method='PCA', n_components=20)
cls_shape_res = tdem.classify(dim_dataset_shape, 'ANN', para)
# save the original dataset that distinguishes material
handler.save_fwd_data(ori_dataset_shape[0], file_name='ori_shape_train.csv')
handler.save_fwd_data(ori_dataset_shape[1], file_name='ori_shape_test.csv')
# save the final dataset after dimensionality reduction
handler.save_fwd_data(dim_dataset_shape[0], file_name='dim_shape_train.csv')
handler.save_fwd_data(dim_dataset_shape[1], file_name='dim_shape_test.csv')
handler.show_cls_res(cls_shape_res, ['Oblate spheroid', 'Prolate spheroid'], show=True, save=True, file_name='cls_result_shape.pdf')
handler.save_cls_res(cls_shape_res, 'cls_shape_res.csv')
