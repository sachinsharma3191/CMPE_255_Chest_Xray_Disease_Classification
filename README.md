# Chest X-Ray Disease Classification
This research group project was completed in partial fulfilment of the requirements for the CMPE 255 Data Mining course at the San Jose State University and under the careful supervision of Dr.	Gheorghi Guzun.<br />
  
Chest X-Rays are extensively used in field of medical to examine chest,heart,lungs blood veseels and other surrounding structure in human body for possible medical conditions and identifications of diseased ailments such as Pneumonia,Cardiomegaly,Effusion,Pleural_Thickening and other chest issues.Recently some of COVID-19 cases has been detected using Chest X-Rays.All classification of diseases is done manually.Presently,there is no automated framework exist to accurately classification diseases using Chest X-Ray.Standford ML Group has developed [CheXNet](https://stanfordmlgroup.github.io/projects/chexnet/) to detect pneumonia from chest X-rays at a level exceeding practicing radiologists.This has caused huge interest in developing automated framework to correctly predict diseases.<br/>

Our project aims to apply image classfication techniques using Deep Learning to classify X-Rays into 14 categories 
* Atelectasis
* Cardiomegaly
* Consolidation 
* Edema 
* Effusion  
* Emphysema 
* Fibrosis 
* Infiltration
* Mass 
* Nodule
* Pleural_Thickening
* Pneumonia
* Pneumothorax

## Technologies and Tools
* Tensorflow 
* Keras
* OpenCV-Python (Python API for Python)

## Tensorflow
Tensorflow is an open source machine learning developed by Google as part of its Google Brain Project.

## Keras
Keras is an open source neural network libray written in Python.It is built on top on Tensorflow and can be easily integrated with other neural network libraries

## OpenCV-Python
OpenCV is computer vision library release by Intel in 1999 and first release in 2000.OpenCV-Python is the Python API of OpenCV. It combines the best qualities of OpenCV C++ API and Python language.

## Dataset
National Institutes of Health Chest X-Ray Dataset. It consists of 112,120 X-ray images with disease labels from 30,805 unique patients 
[Dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/37178474737) 

Sample Test Data of Chest X-Ray Dataset.It consists 5,606 X-ray images images with size 1024 x 1024
[Dataset](https://drive.google.com/open?id=1VIqUu4_OhmG1AR9duWYdNvIKYZIGA9k2)

## Alogrthims 
* ResNet50
* DenseNet169
* MobileNet

## ResNet50
* ResNet makes it possible to train up to hundreds or even thousands of layers and still achieves compelling performance.

## DenseNet  
* Improved version of ResNet with with deeper convolutional layers
* Solves Vanishing Gradient Descent of CNN
* Variations
   * DenseNet121
   * DenseNet169
   * DenseNet201

## MobileNet
* CNN architecture for Image Classification and Mobile Vision
* Requires less computation power for model 
* Good for low power devices such as mobiles, embedded devices ,devices without GPU


## Data Preprocessing
### Image Augmentation: 
It is required to artificially increase the dataet.  For instance, if we have an image, we can flip it horizontally or vertically, rotate it by a certain degree, shift it by height or width. Although, our intent is increasing the performance of the model, it may end up in over-fitting. Data Augmentation of each of the image is done using in-place data Augmentation. For this study, we use ImageGenerator class provided by Keras



## Implemenation

### Model Building
### ResNet50
In the first layer, Conv2D is used. The input_shape is (128,128,1) and its output shape is (None, 224, 224, 3). Conv2D is usually used to create a convolution matrix which can help in edge detection.This  is followed by  adding resnet50 model which is pretrained on Image-Net. The input shape will be (224,224,3) and output shape will be multiple. This is followed by applying GlobalAveragePooling2D(). To this a Dense layer (Dense(no of labels)) is added with activation equals to sigmoid with output size equivalent to the number of labels, 15 in the case of this study.

### DenseNet
For this study, we use a Sequential class constructor to create the model and layers are added using the add() method. Three different convolution neural network under DenseNet with different number of hidden layers were tried ( DenseNet121, DenseNet 169, DenseNet 201 ). DenseNet 169 seemed to be the best of the lot. 
In the first layer, DenseNet169 is used where the input_shape is given, (224,224,3) and output shape is (10,10,1920) which is pre-trained using ImageNet. This is followed by dimensionality reduction using GlobalAveragePooling2D(). This layer transforms the dimensions from (4*4*1920) to (1*1*1920). This is followed by applying regularization technique (Dropout(0.5)). Probability of 0.5 is the common value that is chosen. To this a Dense layer (Dense (512)) is added which has an output size of 512 followed by regularization. Also, we have added activation functions of relu to improve losses. An additional Dense layer (Dense(no of labels)) is added with output size equivalent to the number of labels, 13 in the case of this study.

### MobileNet
Layers are stacked up using Keras Sequential model and the list of layers were passed to the constructor. The base layer for this model is Mobilenet (input shape( 128,128,1)  and output shape(4,4,1024)).This is followed by GlobalAveragePooling2D() which performs average pooling for spatial data which means it applies average pooling on the spatial dimensions until each of the spatial dimension is one leaving the other unchanged. In this model this layer transforms the dimensions from (4*4*1024) to (1*1*1024). This is followed by applying regularization technique (Dropout(0.5)). Probability of 0.5 is common value that is chosen [9]. The goal of applying this is to avoid over-fitting. To this a Dense layer (Dense(512)) is added which has an output size of 512 followed by regularization. This combination is added thrice. The aim was to see if addition of layers increases performance. An additional Dense layer (Dense(no of labels)) is added with output size equivalent to the number of labels, 13 in the case of this study.

### Model Compilation
Compiling the model requires three key factors. 
* Optimizer we use Adam gradient-based optimizer
* Loss function, we use binary_crossentropy
* Metrics we use accuracy.

### Model Training
The model is trained for 1 and 5 epochs respectively. Each epoch runs in steps is (train_gen.n // train_gen.batch_size) which is equal to 1226.

## Results

### Model Accuracy

| Metrics        | MobileNet(Epoch =1) |MobileNet(Epoch =5)  | ResNet (Epoch =1)  | ResNet (Epoch =5) | ResNet (Epoch =10) | DensetNet169 (Epoch= 1) | Denset169 (Epoch = 5)
| ------: |:--------:| -----:|  ------: |:-----:| -----:| ---------: | ------ | 
| Binary_Accuracy      | 0.8786 |	0.8791 |	0.9116 |	0.9150 | 0.9180 |	0.8371 |0.8602
| Val_loss      | 0.3688 |	0.3443 |	0.2457 |	0.2018 |	0.2387 |	1.3201 |	2.7672
| Val_binary_accuracy | 0.8789 | 0.8793 |	0.9124 |0.9137	| 0.9157 | 0.8401 |	0. 8210
