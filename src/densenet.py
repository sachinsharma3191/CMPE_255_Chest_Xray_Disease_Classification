import sys

sys.path.insert(0, "/usr/local/lib/python3.7/site-packages")

from keras.applications.mobilenet import MobileNet
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from keras.applications.densenet import DenseNet201, DenseNet169, DenseNet121
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np  # linear algebra
from sklearn.metrics import roc_curve, auc
from keras.layers import Activation, Dense
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from itertools import chain
import os
from cv2 import cv2 as cv
from glob import glob
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau

# Read Labels from csv file
xray_df = pd.read_csv("kaggle/input/sample/sample_labels.csv")

PATH = os.path.abspath(os.path.join('kaggle', 'input/sample/'))
SOURCE_IMAGES = os.path.join(PATH, "sample", "images")
images = glob(os.path.join(SOURCE_IMAGES, "*.png"))
# Make dictionary of files and path
image_files = {os.path.basename(x): x for x in images}

print(image_files)
print('Scans found:', len(image_files), ', Total Headers', xray_df.shape[0])
# Create Column in Labels File storing image path
xray_df['path'] = xray_df['Image Index'].map(image_files.get)
xray_df['Patient Age'] = xray_df['Patient Age'].map(lambda x: int(x[:-1]))

print(xray_df.head())

xray_df = xray_df.drop(['Patient Age', 'Patient Gender', 'Follow-up #', 'Patient ID', 'View Position',
                        'OriginalImageWidth', 'OriginalImageHeight', 'OriginalImagePixelSpacing_x',
                        'OriginalImagePixelSpacing_y'], axis=1)

xray_df['Finding Labels'] = xray_df['Finding Labels'].map(lambda x: x.replace('No Finding', ''))

print(xray_df.head())

labels = np.unique(list(chain(*xray_df['Finding Labels'].map(lambda x: x.split('|')).tolist())))
labels = [x for x in labels if len(x) > 0]
print('All Labels ({}): {}'.format(len(labels), labels))

for c_label in labels:
    if len(c_label) > 1:  # leave out empty labels
        xray_df[c_label] = xray_df['Finding Labels'].map(lambda finding: 1.0 if c_label in finding else 0)

# MIN_CASES = 1000
# labels = [c_label for c_label in labels if xray_df[c_label].sum()>MIN_CASES]
xray_df['disease_vec'] = xray_df.apply(lambda x: [x[labels].values], 1).map(lambda x: x[0])

print(xray_df.head())

train_df, test_df = train_test_split(xray_df,
                                     test_size=0.15,
                                     random_state=2018,
                                     stratify=xray_df['Finding Labels'].map(lambda x: x[:4]))

X_train = train_df['path'].values.tolist()
y_train = np.asarray(train_df['disease_vec'].values.tolist())
X_test = test_df['path'].values.tolist()
y_test = np.asarray(test_df['disease_vec'].values.tolist())

# Standard Image Size for DenseNet(224 * 224 * 3)
IMG_SIZE = [224, 224, 3]
DENSE_NET_IMAGE_SIZE = (224, 224)

print(cv.imread(X_train[0]).shape)

# Creating Train and Test Data for Image
train_images = np.zeros([len(X_train), IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]])
test_images = np.zeros([len(X_test), IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]])

# Reading Images for Training Data
for index, row in enumerate(X_train):
    image = cv.imread(row)
    resize_image = cv.resize(image, (224, 224), interpolation=cv.INTER_AREA)
    train_images[index] = resize_image

print(train_images.shape)

# Reading Images for Test Data
for index, row in enumerate(X_test):
    image = cv.imread(row)
    resize_image = cv.resize(image, DENSE_NET_IMAGE_SIZE, interpolation=cv.INTER_AREA)
    test_images[index] = resize_image

print(test_images.shape)

X_train = train_images.reshape(len(X_train), 224, 224, 3)
X_test = test_images.reshape(len(X_test), 224, 224, 3)

# Prepare Model
base_mobilenet_model = DenseNet201(input_shape=(224, 224, 3),
                                   include_top=False, weights=None)
multi_disease_model = Sequential()
multi_disease_model.add(base_mobilenet_model)
multi_disease_model.add(GlobalAveragePooling2D())
multi_disease_model.add(Dropout(0.5))
multi_disease_model.add(Dense(512))
multi_disease_model.add(Activation('relu'))
multi_disease_model.add(Dropout(0.5))
multi_disease_model.add(Dense(len(labels), activation='sigmoid'))
multi_disease_model.compile(optimizer='adam', loss='binary_crossentropy',
                            metrics=['binary_accuracy', 'mae'])
# Print Model Summary
multi_disease_model.summary()

# Callbacks
weight_path = "{}_weights.best.hdf5".format('xray_class')

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1,
                             save_best_only=True, mode='min', save_weights_only=True)

early = EarlyStopping(monitor="val_loss",
                      mode="min",
                      patience=3)
callbacks_list = [checkpoint, early]

# Fitting Model on Test Data
multi_disease_model.fit(X_train, y_train, epochs=300, verbose=1,
                        validation_data=(X_test, y_test),
                        callbacks=callbacks_list)

# Printing Accuracy
for c_label, s_count in zip(labels, 100 * np.mean(y_test, 0)):
    print('%s: %2.2f%%' % (c_label, s_count))

# Model Predictions
pred_Y = multi_disease_model.predict(X_test, batch_size=32, verbose=True)

# Plotting ROC Curve
fig, c_ax = plt.subplots(1, 1, figsize=(9, 9))
for (idx, c_label) in enumerate(labels):
    fpr, tpr, thresholds = roc_curve(y_test[:, idx].astype(int), pred_Y[:, idx])
    c_ax.plot(fpr, tpr, label='%s (AUC:%0.2f)' % (c_label, auc(fpr, tpr)))
c_ax.legend()
c_ax.set_xlabel('False Positive Rate')
c_ax.set_ylabel('True Positive Rate')
fig.savefig('barely_trained_net.png')
