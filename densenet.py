import sys

sys.path.insert(0, "/usr/local/lib/python3.7/site-packages")

from keras.applications.mobilenet import MobileNet
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from keras.applications.densenet import DenseNet201,DenseNet169,DenseNet121
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np # linear algebra
from keras.layers import Activation, Dense
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from itertools import chain
import os
from glob import glob
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau

#Read Labels from csv file
xray_df = pd.read_csv("./sample/sample_labels.csv")
#Make dictionary of files and path
image_files = {os.path.basename(x): x for x in glob(os.path.join(os.path.join('.','sample','sample','images', '*.png')))}


print('Scans found:', len(image_files), ', Total Headers', xray_df.shape[0])
#Create Column in Labels File storing image path
xray_df['path'] = xray_df['Image Index'].map(image_files.get)
xray_df.sample(3)

#Remove "No Finding Labels with ""
xray_df['Finding Labels'] = xray_df['Finding Labels'].map(lambda x: x.replace('No Finding', ''))

#Find Unique Labels in Multiple Labels
labels = np.unique(list(chain(*xray_df['Finding Labels'].map(lambda x: x.split('|')).tolist())))
labels = [x for x in labels if len(x)>0]

print('All Labels ({}): {}'.format(len(labels), labels))

for c_label in labels:
    if len(c_label)>1: # leave out empty labels
        xray_df[c_label] = xray_df['Finding Labels'].map(lambda finding: 1.0 if c_label in finding else 0)


MIN_CASES = 1000
labels = [c_label for c_label in labels if xray_df[c_label].sum()>MIN_CASES]
xray_df['disease_vec'] = xray_df.apply(lambda x: [x[labels].values], 1).map(lambda x: x[0])

#Split train and test data set
train_df, valid_df = train_test_split(xray_df,
                                   test_size = 0.25,
                                   random_state = 42)
                                   #stratify = xray_df['Finding Labels'].map(lambda x: x[:4]))

#Standard Image Size for DensetNet(224 * 224 * 3)
IMG_SIZE = (224, 224)
#Image Generator
core_idg = ImageDataGenerator(samplewise_center=True,
                              samplewise_std_normalization=True,
                              horizontal_flip = True,
                              vertical_flip = False,
                              height_shift_range= 0.05,
                              width_shift_range=0.1,
                              rotation_range=5,
                              shear_range = 0.1,
                              fill_mode = 'reflect',
                              zoom_range=0.15)

def flow_from_dataframe(img_data_gen, in_df, path_col, y_col, **dflow_args):
    base_dir = os.path.dirname(in_df[path_col].values[0])
    print('## Ignore next message from keras, values are replaced anyways')
    df_gen = img_data_gen.flow_from_directory(base_dir,
                                     class_mode = 'input',
                                    **dflow_args)

    df_gen.filenames = in_df[path_col].values
    df_gen.classes = np.stack(in_df[y_col].values)
    df_gen.samples = in_df.shape[0]
    df_gen.n = in_df.shape[0]
    df_gen._set_index_array()
    df_gen.directory = '' # since we have the full path
    print('Reinserting dataframe: {} images'.format(in_df.shape[0]))
    return df_gen

#Training Data
train_gen = flow_from_dataframe(core_idg, train_df,
                             path_col = 'path',
                            y_col = 'disease_vec',
                            target_size = IMG_SIZE,
                             color_mode = 'rgb',
                            batch_size = 32)
#Validation Data
valid_gen = flow_from_dataframe(core_idg, valid_df,
                             path_col = 'path',
                            y_col = 'disease_vec',
                            target_size = IMG_SIZE,
                             color_mode = 'rgb',
                            batch_size = 256)

#Get  Test Data from next Batch
test_X, test_Y = next(flow_from_dataframe(core_idg,
                               valid_df,
                             path_col = 'path',
                            y_col = 'disease_vec',
                            target_size = IMG_SIZE,
                             color_mode = 'rgb',
                            batch_size = 1024))

#Get  Test Data from next Batch
t_x, t_y = next(train_gen)

#Prepare Model
base_mobilenet_model =  DenseNet201(input_shape =  t_x.shape[1:],
                                 include_top = False, weights = 'imagenet')
multi_disease_model = Sequential()
multi_disease_model.add(base_mobilenet_model)
multi_disease_model.add(GlobalAveragePooling2D())
multi_disease_model.add(Dropout(0.5))
multi_disease_model.add(Dense(512))
multi_disease_model.add(Activation('relu'))
multi_disease_model.add(Dropout(0.5))
multi_disease_model.add(Dense(len(labels), activation = 'sigmoid'))
multi_disease_model.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                           metrics = ['binary_accuracy', 'mae'])
#Print Model Summary
multi_disease_model.summary()


weight_path="{}_weights.best.hdf5".format('xray_class')

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1,
                             save_best_only=True, mode='min', save_weights_only = True)

early = EarlyStopping(monitor="val_loss",
                      mode="min",
                      patience=3)
callbacks_list = [checkpoint, early]

#Train Model
multi_disease_model.fit_generator(train_gen,
                                  steps_per_epoch=100,
                                  validation_data = (test_X, test_Y),
                                  epochs = 1,
                                  callbacks = callbacks_list)

#Print Accuracy
for c_label, s_count in zip(labels, 100*np.mean(test_Y,0)):
    print('%s: %2.2f%%' % (c_label, s_count))

#Make Predictions
pred_Y = multi_disease_model.predict(test_X, batch_size = 32, verbose = True)

#Plot ROC Curves
from sklearn.metrics import roc_curve, auc
fig, c_ax = plt.subplots(1,1, figsize = (9, 9))
for (idx, c_label) in enumerate(labels):
    fpr, tpr, thresholds = roc_curve(test_Y[:,idx].astype(int), pred_Y[:,idx])
    c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (c_label, auc(fpr, tpr)))
c_ax.legend()
c_ax.set_xlabel('False Positive Rate')
c_ax.set_ylabel('True Positive Rate')
fig.savefig('barely_trained_net.png')
