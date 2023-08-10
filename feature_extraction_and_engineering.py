# -*- coding: utf-8 -*-
"""
Created on Fri May 19 17:11:43 2023

@authors:
    Taofik Ahmed Suleiman
    Daniel Tweneboah Anyimadu
    Andrew Dwi Permana
    Hsham Abdalgny Abdalwhab Ngim

@supervisor: 
    Professor Alessandra Scotto di Freca
"""

from data_preparation import *
from data_splitting import *

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.layers import GlobalAveragePooling2D

from keras.applications.vgg16 import VGG16
from keras.applications import ResNet50
from keras.applications import DenseNet121

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


''' #### Build and train the first model - Benign vs Others and then the second model - 
    melanoma vs seborrheic_keratosis

Steps:
1. Define the number of classes, this is set to 1 for binary classification
2. Create a sequential model.
3. Add convolutional layers - we used Conv2D layers to extract features from the input 
   images.The specified parameters include the number of filters (256, 128, 64), the 
   filter/kernel size (3x3), and the activation function ("relu").
4. Add max pooling layers - we used MaxPool2D layers to reduce the spatial dimensions of the
   feature maps and help the model to focus on the most relevant features while reducing the
   number of parameters.
5. Add dropout layers - we used Dropout layers to mitigate overfitting by preventing the 
   model from relying too heavily on specific features and improves generalization.
6. Flatten the output - we used Flatten layer to convert the 2D feature maps into a 1D 
   vector before applying the machine learning'''


#=================================================
# FIRST STEP FEATURE EXTRACTION USING DEEP LEARNING
#=================================================
feature_extractor = Sequential()
feature_extractor.add(Conv2D(256, (3, 3), activation="relu", input_shape=(SIZE, SIZE, 3)))
feature_extractor.add(MaxPool2D(pool_size=(2, 2)))  
feature_extractor.add(Dropout(0.3))

feature_extractor.add(Conv2D(128, (3, 3), activation='relu'))
feature_extractor.add(MaxPool2D(pool_size=(2, 2)))  
feature_extractor.add(Dropout(0.3))

feature_extractor.add(Conv2D(64, (3, 3), activation='relu'))
feature_extractor.add(MaxPool2D(pool_size=(2, 2)))  
feature_extractor.add(Dropout(0.3))

feature_extractor.add(Flatten())
x_train_ex = feature_extractor.predict(x_train)


#========================================================
# FIRST STEP FEATURE ENGINEERING: STANDARDIZATION AND PCA
#========================================================
# Normalize the training data
scaler = StandardScaler()
x_train_normalized = scaler.fit_transform(x_train_ex)

# Normalize the validation data using the same scaler
x_test_normalized = scaler.transform(feature_extractor.predict(x_test))

# Perform PCA on the normalized training data
pca = PCA(n_components=50)
x_train_pca = pca.fit_transform(x_train_normalized)
x_test_pca = pca.transform(x_test_normalized)

#=================================================
# FIRST STEP FEATURE EXTRACTION USING VGG16
#=================================================
vgg16_feature_extractor = VGG16(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))
vgg16_feature_extractor.trainable = False

model_vgg16 = Sequential()
model_vgg16.add(vgg16_feature_extractor)
model_vgg16.add(GlobalAveragePooling2D())

x_train_vgg16_ex = model_vgg16.predict(x_train)

#========================================================
# FIRST STEP FEATURE ENGINEERING: STANDARDIZATION AND PCA
#========================================================
# Normalize the training data
scaler_vgg16 = StandardScaler()
x_train_vgg16_normalized = scaler_vgg16.fit_transform(x_train_vgg16_ex)

# Normalize the validation data using the same scaler
x_test_vgg16_normalized = scaler_vgg16.transform(model_vgg16.predict(x_test))

# Perform PCA on the normalized training data
pca_vgg16 = PCA(n_components=50)
x_train_vgg16_pca = pca_vgg16.fit_transform(x_train_vgg16_normalized)
x_test_vgg16_pca = pca_vgg16.transform(x_test_vgg16_normalized)


#=================================================
# FIRST STEP FEATURE EXTRACTION USING RESNET
#=================================================
resnet_feature_extractor = ResNet50(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))
resnet_feature_extractor.trainable = False

model_resnet = Sequential()
model_resnet.add(resnet_feature_extractor)
model_resnet.add(GlobalAveragePooling2D())

x_train_resnet_ex = model_resnet.predict(x_train)

#========================================================
# FIRST STEP FEATURE ENGINEERING: STANDARDIZATION AND PCA
#========================================================
# Normalize the training data
scaler_resnet = StandardScaler()
x_train_resnet_normalized = scaler_resnet.fit_transform(x_train_resnet_ex)

# Normalize the validation data using the same scaler
x_test_resnet_normalized = scaler_resnet.transform(model_resnet.predict(x_test))

# Perform PCA on the normalized training data
pca_resnet = PCA(n_components=50)
x_train_resnet_pca = pca_resnet.fit_transform(x_train_resnet_normalized)
x_test_resnet_pca = pca_resnet.transform(x_test_resnet_normalized)


#=================================================
# FIRST STEP FEATURE EXTRACTION USING DENSENET
#=================================================
densenet_feature_extractor = DenseNet121(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))
densenet_feature_extractor.trainable = False

model_densenet = Sequential()
model_densenet.add(densenet_feature_extractor)
model_densenet.add(GlobalAveragePooling2D())

x_train_densenet_ex = model_densenet.predict(x_train)

#========================================================
# FIRST STEP FEATURE ENGINEERING: STANDARDIZATION AND PCA
#========================================================
# Normalize the training data
scaler_densenet = StandardScaler()
x_train_densenet_normalized = scaler_densenet.fit_transform(x_train_densenet_ex)

# Normalize the validation data using the same scaler
x_test_densenet_normalized = scaler_densenet.transform(model_densenet.predict(x_test))

# Perform PCA on the normalized training data
pca_densenet = PCA(n_components=50)
x_train_densenet_pca = pca_densenet.fit_transform(x_train_densenet_normalized)
x_test_densenet_pca = pca_densenet.transform(x_test_densenet_normalized)

#=================================================
# SECOND STEP FEATURE EXTRACTION USING DEEP LEARNING
#=================================================
feature_extractor2 = Sequential()
feature_extractor2.add(Conv2D(256, (3, 3), activation="relu", input_shape=(SIZE, SIZE, 3)))
feature_extractor2.add(MaxPool2D(pool_size=(2, 2)))  
feature_extractor2.add(Dropout(0.3))

feature_extractor2.add(Conv2D(128, (3, 3), activation='relu'))
feature_extractor2.add(MaxPool2D(pool_size=(2, 2)))  
feature_extractor2.add(Dropout(0.3))

feature_extractor2.add(Conv2D(64, (3, 3), activation='relu'))
feature_extractor2.add(MaxPool2D(pool_size=(2, 2)))  
feature_extractor2.add(Dropout(0.3))

feature_extractor2.add(Flatten())
x_train_step2_ex = feature_extractor2.predict(x_train_step2)


#========================================================
# SECOND STEP FEATURE ENGINEERING: STANDARDIZATION AND PCA
#========================================================
# Normalize the training data
scaler2 = StandardScaler()
x_train2_normalized = scaler2.fit_transform(x_train_step2_ex)

# Normalize the validation data using the same scaler
x_test2_normalized = scaler2.transform(feature_extractor2.predict(x_test_step2))

# Perform PCA on the normalized training data
pca2 = PCA(n_components=70)
x_train2_pca = pca2.fit_transform(x_train2_normalized)
x_test2_pca = pca2.transform(x_test2_normalized)

#=================================================
# SECOND STEP FEATURE EXTRACTION USING VGG16
#=================================================
vgg16_feature_extractor2 = VGG16(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))
vgg16_feature_extractor2.trainable = False

model_vgg16_2 = Sequential()
model_vgg16_2.add(vgg16_feature_extractor2)
model_vgg16_2.add(GlobalAveragePooling2D())

x_train_vgg16_ex2 = model_vgg16_2.predict(x_train_step2)

#========================================================
# SECOND STEP FEATURE ENGINEERING: STANDARDIZATION AND PCA
#========================================================
# Normalize the training data
scaler_vgg16_2 = StandardScaler()
x_train_vgg16_normalized2 = scaler_vgg16_2.fit_transform(x_train_vgg16_ex2)

# Normalize the validation data using the same scaler
x_test_vgg16_normalized2 = scaler_vgg16_2.transform(model_vgg16_2.predict(x_test_step2))

# Perform PCA on the normalized training data
pca_vgg16_2 = PCA(n_components=70)
x_train_vgg16_step2_pca = pca_vgg16_2.fit_transform(x_train_vgg16_normalized2)
x_test_vgg16_step2_pca = pca_vgg16_2.transform(x_test_vgg16_normalized2)


#=================================================
# SECOND STEP FEATURE EXTRACTION USING RESNET
#=================================================
resnet_feature_extractor2 = ResNet50(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))
resnet_feature_extractor2.trainable = False

model_resnet_2 = Sequential()
model_resnet_2.add(resnet_feature_extractor2)
model_resnet_2.add(GlobalAveragePooling2D())

x_train_resnet_ex2 = model_resnet_2.predict(x_train_step2)

#========================================================
# SECOND STEP FEATURE ENGINEERING: STANDARDIZATION AND PCA
#========================================================
# Normalize the training data
scaler_resnet_2 = StandardScaler()
x_train_resnet_normalized2 = scaler_resnet_2.fit_transform(x_train_resnet_ex2)

# Normalize the validation data using the same scaler
x_test_resnet_normalized2 = scaler_resnet_2.transform(model_resnet_2.predict(x_test_step2))

# Perform PCA on the normalized training data
pca_resnet_2 = PCA(n_components=70)
x_train_resnet_step2_pca = pca_resnet_2.fit_transform(x_train_resnet_normalized2)
x_test_resnet_step2_pca = pca_resnet_2.transform(x_test_resnet_normalized2)


#=================================================
# SECOND STEP FEATURE EXTRACTION USING DENSENET
#=================================================
densenet_feature_extractor2 = DenseNet121(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))
densenet_feature_extractor2.trainable = False

model_densenet_2 = Sequential()
model_densenet_2.add(densenet_feature_extractor2)
model_densenet_2.add(GlobalAveragePooling2D())

x_train_densenet_ex2 = model_densenet_2.predict(x_train_step2)

#========================================================
# SECOND STEP FEATURE ENGINEERING: STANDARDIZATION AND PCA
#========================================================
# Normalize the training data
scaler_densenet_2 = StandardScaler()
x_train_densenet_normalized2 = scaler_densenet_2.fit_transform(x_train_densenet_ex2)

# Normalize the validation data using the same scaler
x_test_densenet_normalized2 = scaler_densenet_2.transform(model_densenet_2.predict(x_test_step2))

# Perform PCA on the normalized training data
pca_densenet_2 = PCA(n_components=70)
x_train_densenet_step2_pca = pca_densenet_2.fit_transform(x_train_densenet_normalized2)
x_test_densenet_step2_pca = pca_densenet_2.transform(x_test_densenet_normalized2)
