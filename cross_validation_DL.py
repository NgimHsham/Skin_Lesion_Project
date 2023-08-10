# -*- coding: utf-8 -*-
"""
Created on Mon May 29 10:01:37 2023

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

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications import ResNet50
from keras.applications import DenseNet121
from keras.optimizers import Adam

''' Build and train the first and second model - Benign vs Others then the second model - 
    melanoma vs seborrheic_keratosis

we have implemented our own CNN model from scratch using autokeras and we have also used
several pretrained models '''

#==============================================================================
# FIRST STEP CLASSIFICATION
#==============================================================================
#=================================================
# CNN build up for the first step classification
#=================================================

def build_cnn_model():
    num_classes = 1
    model = Sequential()
    model.add(Conv2D(256, (3, 3), activation="relu", input_shape=(SIZE, SIZE, 3)))
    model.add(MaxPool2D(pool_size=(2, 2)))  
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))  
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))  
    model.add(Dropout(0.3))

    model.add(Flatten())

    model.add(Dense(32))
    model.add(Dense(num_classes, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['acc'])
    return model

#====================================================================
# Pretrained model - VGG16 build up for the first step classification
#====================================================================

def build_vgg16_model():
    num_classes = 1
    VGG_model = VGG16(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))

    for layer in VGG_model.layers:
        layer.trainable = False

    model_vgg = Sequential()
    model_vgg.add(VGG_model)
    model_vgg.add(Flatten())
    model_vgg.add(Dense(32))
    model_vgg.add(Dense(num_classes, activation='sigmoid'))

    model_vgg.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['acc'])
    return model_vgg

#=====================================================================
# Pretrained model - RESNET build up for the first step classification
#=====================================================================

def build_resnet_model():
    num_classes = 1
    resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))

    for layer in resnet_model.layers:
        layer.trainable = False

    model_resnet = Sequential()
    model_resnet.add(resnet_model)
    model_resnet.add(Flatten())
    model_resnet.add(Dense(32))
    model_resnet.add(Dense(num_classes, activation='sigmoid'))

    model_resnet.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
    return model_resnet

#==========================================================================
# Pretrained model - DenseNet build up for the first step classification
#==========================================================================

def build_densenet_model():
    num_classes = 1
    densenet_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))

    for layer in densenet_model.layers:
        layer.trainable = False

    x = Flatten()(densenet_model.output)
    x = Dense(256, activation='relu')(x)
    x = Dense(num_classes, activation='sigmoid')(x)

    model_densenet = Model(inputs=densenet_model.input, outputs=x)

    model_densenet.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    return model_densenet

#==============================================================================
# SECOND STEP
#==============================================================================

#=================================================
# CNN build up for the second step classification
#=================================================

def build_cnn_model2():
    num_classes = 1
    model2 = Sequential()
    model2.add(Conv2D(256, (3, 3), activation="relu", input_shape=(SIZE, SIZE, 3)))
    model2.add(MaxPool2D(pool_size=(2, 2)))  
    model2.add(Dropout(0.3))

    model2.add(Conv2D(128, (3, 3), activation='relu'))
    model2.add(MaxPool2D(pool_size=(2, 2)))  
    model2.add(Dropout(0.3))

    model2.add(Conv2D(64, (3, 3), activation='relu'))
    model2.add(MaxPool2D(pool_size=(2, 2)))  
    model2.add(Dropout(0.3))

    model2.add(Flatten())

    model2.add(Dense(32))
    model2.add(Dense(num_classes, activation='sigmoid'))

    model2.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['acc'])
    return model2

#====================================================================
# Pretrained model - VGG16 build up for the second step classification
#====================================================================

def build_vgg16_model2():
    num_classes = 1
    VGG_model2 = VGG16(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))

    for layer in VGG_model2.layers:
        layer.trainable = False

    model_vgg2 = Sequential()
    model_vgg2.add(VGG_model2)
    model_vgg2.add(Flatten())
    model_vgg2.add(Dense(32))
    model_vgg2.add(Dense(num_classes, activation='sigmoid'))

    model_vgg2.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['acc'])
    return model_vgg2

#=====================================================================
# Pretrained model - RESNET build up for the second step classification
#=====================================================================

def build_resnet_model2():
    num_classes = 1
    resnet_model2 = ResNet50(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))

    for layer in resnet_model2.layers:
        layer.trainable = False

    model_resnet2 = Sequential()
    model_resnet2.add(resnet_model2)
    model_resnet2.add(Flatten())
    model_resnet2.add(Dense(32))
    model_resnet2.add(Dense(num_classes, activation='sigmoid'))

    model_resnet2.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
    return model_resnet2

#===========================================================================
# Pretrained model - Densenet build up for the second step classification
#===========================================================================

def build_densenet_model2():
    num_classes = 1
    densenet_model2 = DenseNet121(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))

    for layer in densenet_model2.layers:
        layer.trainable = False

    y = Flatten()(densenet_model2.output)
    y = Dense(256, activation='relu')(y)
    y = Dense(num_classes, activation='sigmoid')(y)

    model_densenet2 = Model(inputs=densenet_model2.input, outputs=y)

    model_densenet2.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    return model_densenet2

# =======================================================
# First step cross validation results
# =======================================================

# Lists to store the cross-validation results
acc_scores_cnn = []
acc_scores_vgg16 = []
acc_scores_resnet = []
acc_scores_densenet = []
batch_size = 32 
epochs = 45

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Loop through each fold and perform cross-validation for CNN, VGG16, ResNet,
# and DenseNet models
for fold, (train_index, val_index) in enumerate(kf.split(x_train, y_train)):
    print('\n', f"Fold {fold + 1}", '\n')
    x_train_fold, y_train_fold = x_train[train_index], y_train[train_index]
    x_val_fold, y_val_fold = x_train[val_index], y_train[val_index]

    # Build and train the CNN model for this fold
    model_cnn = build_cnn_model()
    history_cnn = model_cnn.fit(
        x_train_fold, y_train_fold,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val_fold, y_val_fold),
        verbose=2
    )
    score_cnn_cv = model_cnn.evaluate(x_val_fold, y_val_fold)
    print('\n', f"Fold {fold + 1} - CNN Validation Accuracy: {score_cnn_cv[1]}", '\n')
    acc_scores_cnn.append(score_cnn_cv[1])

    # Build and train the VGG16 model for this fold
    model_vgg16 = build_vgg16_model()
    history_vgg16 = model_vgg16.fit(
        x_train_fold, y_train_fold,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val_fold, y_val_fold),
        verbose=2
    )
    score_vgg16_cv = model_vgg16.evaluate(x_val_fold, y_val_fold)
    print('\n', f"Fold {fold + 1} - VGG16 Validation Accuracy: {score_vgg16_cv[1]}", '\n')
    acc_scores_vgg16.append(score_vgg16_cv[1])

    # Build and train the ResNet model for this fold
    model_resnet = build_resnet_model()
    history_resnet = model_resnet.fit(
        x_train_fold, y_train_fold,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val_fold, y_val_fold),
        verbose=2
    )
    score_resnet_cv = model_resnet.evaluate(x_val_fold, y_val_fold)
    print('\n', f"Fold {fold + 1} - ResNet Validation Accuracy: {score_resnet_cv[1]}", '\n')
    acc_scores_resnet.append(score_resnet_cv[1])

    # Build and train the DenseNet model for this fold
    model_densenet = build_densenet_model()
    history_densenet = model_densenet.fit(
        x_train_fold, y_train_fold,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val_fold, y_val_fold),
        verbose=2
    )
    score_densenet_cv = model_densenet.evaluate(x_val_fold, y_val_fold)
    print('\n', f"Fold {fold + 1} - DenseNet Validation Accuracy: {score_densenet_cv[1]}", '\n')
    acc_scores_densenet.append(score_densenet_cv[1])
   
score_cnn = model_cnn.evaluate(x_test, y_test)
score_vgg16 = model_vgg16.evaluate(x_test, y_test)
score_resnet = model_resnet.evaluate(x_test, y_test)
score_densenet = model_densenet.evaluate(x_test, y_test)

# =======================================================
# Second step cross validation results
# =======================================================

# Lists to store the cross-validation results
acc_scores_cnn2 = []
acc_scores_vgg162 = []
acc_scores_resnet2 = []
acc_scores_densenet2 = []
batch_size2 = 32 
epochs2 = 50

n_splits2 = 5
kf2 = KFold(n_splits=n_splits2, shuffle=True, random_state=42)

# Loop through each fold and perform cross-validation for CNN, VGG16, ResNet,
# and DenseNet models
for fold2, (train_index2, val_index2) in enumerate(kf2.split(x_train_step2, y_train_step2)):
    print('\n', f"Fold {fold2 + 1}", '\n')
    x_train_fold2, y_train_fold2 = x_train_step2[train_index2],  y_train_step2[train_index2]
    x_val_fold2, y_val_fold2 = x_train_step2[val_index2], y_train_step2[val_index2]


    model_cnn2 = build_cnn_model2()
    history_cnn2 = model_cnn2.fit(
        x_train_fold2, y_train_fold2,
        epochs=epochs2,
        batch_size=batch_size2,
        validation_data=(x_val_fold2, y_val_fold2),
        verbose=2
    )

    score_cnn_cv2 = model_cnn2.evaluate(x_val_fold2, y_val_fold2)
    print('\n', f"Fold {fold2 + 1} - CNN Validation Accuracy: {score_cnn_cv2[1]}", '\n')
    acc_scores_cnn2.append(score_cnn_cv2[1])
    
    
    model_vgg162 = build_vgg16_model2()
    history_vgg162 = model_vgg162.fit(
        x_train_fold2, y_train_fold2,
        epochs=epochs2,
        batch_size=batch_size2,
        validation_data=(x_val_fold2, y_val_fold2),
        verbose=2
    )
    score_vgg16_cv2 = model_vgg162.evaluate(x_val_fold2, y_val_fold2)
    print('\n', f"Fold {fold2 + 1} - VGG16 Validation Accuracy: {score_vgg16_cv2[1]}", '\n')
    acc_scores_vgg162.append(score_vgg16_cv2[1])


    model_resnet2 = build_resnet_model2()
    history_resnet2 = model_resnet2.fit(
        x_train_fold2, y_train_fold2,
        epochs=epochs2,
        batch_size=batch_size2,
        validation_data=(x_val_fold2, y_val_fold2),
        verbose=2
    )
    score_resnet_cv2 = model_resnet2.evaluate(x_val_fold2, y_val_fold2)
    print('\n', f"Fold {fold2 + 1} - ResNet Validation Accuracy: {score_resnet_cv2[1]}", '\n')
    acc_scores_resnet2.append(score_resnet_cv2[1])

    # Build and train the DenseNet model for this fold
    model_densenet2 = build_densenet_model2()
    history_densenet2 = model_densenet2.fit(
        x_train_fold2, y_train_fold2,
        epochs=epochs2,
        batch_size=batch_size2,
        validation_data=(x_val_fold2, y_val_fold2),
        verbose=2
    )
    score_densenet_cv2 = model_densenet2.evaluate(x_val_fold2, y_val_fold2)
    print('\n', f"Fold {fold2 + 1} - DenseNet Validation Accuracy: {score_densenet_cv2[1]}", '\n')
    acc_scores_densenet2.append(score_densenet_cv2[1])
   
score_cnn2 = model_cnn2.evaluate(x_test_step2, y_test_step2)
score_vgg162 = model_vgg162.evaluate(x_test_step2, y_test_step2)
score_resnet2 = model_resnet2.evaluate(x_test_step2, y_test_step2)
score_densenet2 = model_densenet2.evaluate(x_test_step2, y_test_step2)


print(f"Average CNN Cross Validation Accuracy for first step: {np.mean(acc_scores_cnn)}")
print('CNN Test accuracy for first step:', score_cnn[1], '\n')

print(f"Average VGG16 Cross Validation Accuracy for first step: {np.mean(acc_scores_vgg16)}")
print('VGG16 Test accuracy for first step:', score_vgg16[1], '\n')

print(f"Average ResNet Cross Validation Accuracy for first step: {np.mean(acc_scores_resnet)}")
print('ResNet Test accuracy for first step:', score_resnet[1], '\n')

print(f"Average DenseNet Cross Validation Accuracy for first step: {np.mean(acc_scores_densenet)}")
print('DenseNet Test accuracy for first step:', score_densenet[1], '\n')

print(f"Average CNN Cross Validation Accuracy for second step: {np.mean(acc_scores_cnn2)}")
print('CNN Test accuracy for  second step:', score_cnn2[1], '\n')

print(f"Average VGG16 Cross Validation Accuracy for  second step: {np.mean(acc_scores_vgg162)}")
print('VGG16 Test accuracy for  second  step:', score_vgg162[1], '\n')

print(f"Average ResNet Cross Validation Accuracy for  second step: {np.mean(acc_scores_resnet2)}")
print('ResNet Test accuracy for  second step:', score_resnet2[1], '\n')

print(f"Average DenseNet Cross Validation Accuracy for  second step: {np.mean(acc_scores_densenet2)}")
print('DenseNet Test accuracy for  second step:', score_densenet2[1], '\n')
