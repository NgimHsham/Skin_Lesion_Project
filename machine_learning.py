# -*- coding: utf-8 -*-
"""
Created on Fri May 19 23:41:23 2023

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
from feature_extraction_and_engineering import *


from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

import warnings
from sklearn.exceptions import ConvergenceWarning


''' We build four machine learning models with 5-fold cross validation '''

#========================================================
# MACHINE LEARNING MODELS FOR THE FIRST STEP USING CNN
#========================================================

# Random Forest
rfc_pca = RandomForestClassifier(n_estimators=50, random_state=42)
cross_val_scores_pca = cross_val_score(rfc_pca, x_train_pca, y_train, cv=5)
rfc_pca.fit(x_train_pca, y_train)
# Predict on the PCA-transformed validation data
prediction_1_rf = rfc_pca.predict(x_test_pca)
# Evaluate the model on the validation dataset
test_accuracy1_rf = accuracy_score(y_test, prediction_1_rf)

# SVM
svm = SVC(kernel='linear')
cross_val_scores_svm = cross_val_score(svm, x_train_pca, y_train, cv=5)
svm.fit(x_train_pca, y_train)
prediction_svm = svm.predict(x_test_pca)
test_accuracy_svm = accuracy_score(y_test, prediction_svm)

# KNN
knn = KNeighborsClassifier(n_neighbors=5)
cross_val_scores_knn = cross_val_score(knn, x_train_pca, y_train, cv=5)
knn.fit(x_train_pca, y_train)
prediction_knn = knn.predict(x_test_pca)
test_accuracy_knn = accuracy_score(y_test, prediction_knn)

# Logistic Regression
warnings.filterwarnings("ignore", category=ConvergenceWarning)
logreg = LogisticRegression()
cross_val_scores_logreg = cross_val_score(logreg, x_train_pca, y_train, cv=5)
logreg.fit(x_train_pca, y_train)
prediction_logreg = logreg.predict(x_test_pca)
test_accuracy_logreg = accuracy_score(y_test, prediction_logreg)


# Print the cross-validation scores and validation accuracies for the first model
print('\n','The results of the first step classification model', '\n')
print("First Step Cross-Validation Scores (Random Forest):", cross_val_scores_pca)
print("First Step Test Accuracy (Random Forest):", test_accuracy1_rf, '\n')
print("First Step Cross-Validation Scores (SVM):", cross_val_scores_svm)
print("First Step Test Accuracy (SVM):", test_accuracy_svm, '\n')
print("First Step Cross-Validation Scores (KNN):", cross_val_scores_knn)
print("First Step Test Accuracy (KNN):", test_accuracy_knn, '\n')
print("First Step Cross-Validation Scores (Logistic Regression):", cross_val_scores_logreg)
print("First Step Test Accuracy (Logistic Regression):", test_accuracy_logreg, '\n')


#========================================================
# MACHINE LEARNING MODELS FOR THE FIRST STEP USING VGG16
#========================================================

# Random Forest
rfc_pca_vgg16 = RandomForestClassifier(n_estimators=50, random_state=42)
cross_val_scores_pca_vgg16 = cross_val_score(rfc_pca_vgg16, x_train_vgg16_pca, y_train, cv=5)
rfc_pca_vgg16.fit(x_train_vgg16_pca, y_train)
prediction_1_rf_vgg16 = rfc_pca_vgg16.predict(x_test_vgg16_pca)
test_accuracy1_rf_vgg16 = accuracy_score(y_test, prediction_1_rf_vgg16)

# SVM
svm_vgg16 = SVC(kernel='linear')
cross_val_scores_svm_vgg16 = cross_val_score(svm_vgg16, x_train_vgg16_pca, y_train, cv=5)
svm_vgg16.fit(x_train_vgg16_pca, y_train)
prediction_svm_vgg16 = svm_vgg16.predict(x_test_vgg16_pca)
test_accuracy_svm_vgg16 = accuracy_score(y_test, prediction_svm_vgg16)

# KNN
knn_vgg16 = KNeighborsClassifier(n_neighbors=5)
cross_val_scores_knn_vgg16 = cross_val_score(knn_vgg16, x_train_vgg16_pca, y_train, cv=5)
knn_vgg16.fit(x_train_vgg16_pca, y_train)
prediction_knn_vgg16 = knn_vgg16.predict(x_test_vgg16_pca)
test_accuracy_knn_vgg16 = accuracy_score(y_test, prediction_knn_vgg16)

# Logistic Regression
logreg_vgg16 = LogisticRegression()
cross_val_scores_logreg_vgg16 = cross_val_score(logreg_vgg16, x_train_vgg16_pca, y_train, cv=5)
logreg_vgg16.fit(x_train_vgg16_pca, y_train)
prediction_logreg_vgg16 = logreg_vgg16.predict(x_test_vgg16_pca)
test_accuracy_logreg_vgg16 = accuracy_score(y_test, prediction_logreg_vgg16)

# Print the cross-validation scores and validation accuracies for the first step with VGG16 features
print('\n','The results of the first step classification model using VGG16 features', '\n')
print("First Step Cross-Validation Scores (Random Forest):", cross_val_scores_pca_vgg16)
print("First Step Test Accuracy (Random Forest):", test_accuracy1_rf_vgg16, '\n')
print("First Step Cross-Validation Scores (SVM):", cross_val_scores_svm_vgg16)
print("First Step Test Accuracy (SVM):", test_accuracy_svm_vgg16, '\n')
print("First Step Cross-Validation Scores (KNN):", cross_val_scores_knn_vgg16)
print("First Step Test Accuracy (KNN):", test_accuracy_knn_vgg16, '\n')
print("First Step Cross-Validation Scores (Logistic Regression):", cross_val_scores_logreg_vgg16)
print("First Step Test Accuracy (Logistic Regression):", test_accuracy_logreg_vgg16, '\n')



#========================================================
# MACHINE LEARNING MODELS FOR THE FIRST STEP USING RESNET
#========================================================

# Random Forest
rfc_pca_resnet = RandomForestClassifier(n_estimators=50, random_state=42)
cross_val_scores_pca_resnet = cross_val_score(rfc_pca_resnet, x_train_resnet_pca, y_train, cv=5)
rfc_pca_resnet.fit(x_train_resnet_pca, y_train)
prediction_1_rf_resnet = rfc_pca_resnet.predict(x_test_resnet_pca)
test_accuracy1_rf_resnet = accuracy_score(y_test, prediction_1_rf_resnet)

# SVM
svm_resnet = SVC(kernel='linear')
cross_val_scores_svm_resnet = cross_val_score(svm_resnet, x_train_resnet_pca, y_train, cv=5)
svm_resnet.fit(x_train_resnet_pca, y_train)
prediction_svm_resnet = svm_resnet.predict(x_test_resnet_pca)
test_accuracy_svm_resnet = accuracy_score(y_test, prediction_svm_resnet)

# KNN
knn_resnet = KNeighborsClassifier(n_neighbors=5)
cross_val_scores_knn_resnet = cross_val_score(knn_resnet, x_train_resnet_pca, y_train, cv=5)
knn_resnet.fit(x_train_resnet_pca, y_train)
prediction_knn_resnet = knn_resnet.predict(x_test_resnet_pca)
test_accuracy_knn_resnet = accuracy_score(y_test, prediction_knn_resnet)

# Logistic Regression
logreg_resnet = LogisticRegression()
cross_val_scores_logreg_resnet = cross_val_score(logreg_resnet, x_train_resnet_pca, y_train, cv=5)
logreg_resnet.fit(x_train_resnet_pca, y_train)
prediction_logreg_resnet = logreg_resnet.predict(x_test_resnet_pca)
test_accuracy_logreg_resnet = accuracy_score(y_test, prediction_logreg_resnet)

# Print the cross-validation scores and validation accuracies for the first step with ResNet features
print('\n','The results of the first step classification model using ResNet features', '\n')
print("First Step Cross-Validation Scores (Random Forest):", cross_val_scores_pca_resnet)
print("First Step Test Accuracy (Random Forest):", test_accuracy1_rf_resnet, '\n')
print("First Step Cross-Validation Scores (SVM):", cross_val_scores_svm_resnet)
print("First Step Test Accuracy (SVM):", test_accuracy_svm_resnet, '\n')
print("First Step Cross-Validation Scores (KNN):", cross_val_scores_knn_resnet)
print("First Step Test Accuracy (KNN):", test_accuracy_knn_resnet, '\n')
print("First Step Cross-Validation Scores (Logistic Regression):", cross_val_scores_logreg_resnet)
print("First Step Test Accuracy (Logistic Regression):", test_accuracy_logreg_resnet, '\n')


#========================================================
# MACHINE LEARNING MODELS FOR THE FIRST STEP USING DENSENET
#========================================================

# Random Forest
rfc_pca_densenet = RandomForestClassifier(n_estimators=50, random_state=42)
cross_val_scores_pca_densenet = cross_val_score(rfc_pca_densenet, x_train_densenet_pca, y_train, cv=5)
rfc_pca_densenet.fit(x_train_densenet_pca, y_train)
prediction_1_rf_densenet = rfc_pca_densenet.predict(x_test_densenet_pca)
test_accuracy1_rf_densenet = accuracy_score(y_test, prediction_1_rf_densenet)

# SVM
svm_densenet = SVC(kernel='linear')
cross_val_scores_svm_densenet = cross_val_score(svm_densenet, x_train_densenet_pca, y_train, cv=5)
svm_densenet.fit(x_train_densenet_pca, y_train)
prediction_svm_densenet = svm_densenet.predict(x_test_densenet_pca)
test_accuracy_svm_densenet = accuracy_score(y_test, prediction_svm_densenet)

# KNN
knn_densenet = KNeighborsClassifier(n_neighbors=5)
cross_val_scores_knn_densenet = cross_val_score(knn_densenet, x_train_densenet_pca, y_train, cv=5)
knn_densenet.fit(x_train_densenet_pca, y_train)
prediction_knn_densenet = knn_densenet.predict(x_test_densenet_pca)
test_accuracy_knn_densenet = accuracy_score(y_test, prediction_knn_densenet)

# Logistic Regression
logreg_densenet = LogisticRegression()
cross_val_scores_logreg_densenet = cross_val_score(logreg_densenet, x_train_densenet_pca, y_train, cv=5)
logreg_densenet.fit(x_train_densenet_pca, y_train)
prediction_logreg_densenet = logreg_densenet.predict(x_test_densenet_pca)
test_accuracy_logreg_densenet = accuracy_score(y_test, prediction_logreg_densenet)

# Print the cross-validation scores and validation accuracies for the first model
print('\n','The results of the first step classification model using DenseNet features', '\n')
print("First Step Cross-Validation Scores (Random Forest):", cross_val_scores_pca_densenet)
print("First Step Test Accuracy (Random Forest):", test_accuracy1_rf_densenet, '\n')
print("First Step Cross-Validation Scores (SVM):", cross_val_scores_svm_densenet)
print("First Step Test Accuracy (SVM):", test_accuracy_svm_densenet, '\n')
print("First Step Cross-Validation Scores (KNN):", cross_val_scores_knn_densenet)
print("First Step Test Accuracy (KNN):", test_accuracy_knn_densenet, '\n')
print("First Step Cross-Validation Scores (Logistic Regression):", cross_val_scores_logreg_densenet)
print("First Step Test Accuracy (Logistic Regression):", test_accuracy_logreg_densenet, '\n')


#=======================================================
# MACHINE LEARNING MODELS FOR THE SECOND STEP USING CNN
#=======================================================

# Random Forest
rfc_pca2 = RandomForestClassifier(n_estimators=50, random_state=42)
cross_val_scores_rf2 = cross_val_score(rfc_pca2, x_train2_pca, y_train_step2, cv=5)
rfc_pca2.fit(x_train2_pca, y_train_step2)
# Predict on the PCA-transformed validation data
prediction_2_rf = rfc_pca2.predict(x_test2_pca)
# Evaluate the model on the validation dataset
test_accuracy2_rf = accuracy_score(y_test_step2, prediction_2_rf)

# SVM
svm_model = SVC()
svm_cross_val_scores2 = cross_val_score(svm_model, x_train2_pca, y_train_step2, cv=5)
svm_model.fit(x_train2_pca, y_train_step2)
prediction2_svm = svm_model.predict(x_test2_pca)
accuracy2_svm = accuracy_score(y_test_step2, prediction2_svm)

# KNN
knn_model = KNeighborsClassifier()
knn_cross_val_scores2 = cross_val_score(knn_model, x_train2_pca, y_train_step2, cv=5)
knn_model.fit(x_train2_pca, y_train_step2)
prediction2_knn = knn_model.predict(x_test2_pca)
accuracy2_knn = accuracy_score(y_test_step2, prediction2_knn)

# Logistic Regression
warnings.filterwarnings("ignore", category=ConvergenceWarning)
logreg_model = LogisticRegression()
logreg_cross_val_scores2 = cross_val_score(logreg_model, x_train2_pca, y_train_step2, cv=5)
logreg_model.fit(x_train2_pca, y_train_step2)
prediction2_logreg = logreg_model.predict(x_test2_pca)
accuracy2_logreg = accuracy_score(y_test_step2, prediction2_logreg)

# Print the cross-validation scores and validation accuracies for the second model
print('\n','The results of the second step classification model', '\n')
print("Second Step Cross-Validation Scores for Random Forest:", cross_val_scores_rf2)
print("Second Step Test Accuracy for Random Forest:", test_accuracy2_rf, '\n')
print("Second Step Cross-Validation Scores for SVM:", svm_cross_val_scores2)
print("Second Step Test Accuracy for SVM:", accuracy2_svm, '\n')
print("Second Step Cross-Validation Scores for KNN:", knn_cross_val_scores2)
print("Second Step Test Accuracy for KNN:", accuracy2_knn, '\n')
print("Second Step Cross-Validation Scores for Logistic Regression:", logreg_cross_val_scores2)
print("Second Step Test Accuracy for Logistic Regression:", accuracy2_logreg, '\n')


#=========================================================
# MACHINE LEARNING MODELS FOR THE SECOND STEP USING VGG16
#=========================================================

# Random Forest
rfc_pca_vgg16_step2 = RandomForestClassifier(n_estimators=50, random_state=42)
cross_val_scores_pca_vgg16_step2 = cross_val_score(rfc_pca_vgg16_step2, x_train_vgg16_step2_pca, y_train_step2, cv=5)
rfc_pca_vgg16_step2.fit(x_train_vgg16_step2_pca, y_train_step2)
prediction_2_rf_vgg16 = rfc_pca_vgg16_step2.predict(x_test_vgg16_step2_pca)
test_accuracy2_rf_vgg16 = accuracy_score(y_test_step2, prediction_2_rf_vgg16)

# SVM
svm_vgg16_step2 = SVC()
cross_val_scores_svm_vgg16_step2 = cross_val_score(svm_vgg16_step2, x_train_vgg16_step2_pca, y_train_step2, cv=5)
svm_vgg16_step2.fit(x_train_vgg16_step2_pca, y_train_step2)
prediction2_svm_vgg16 = svm_vgg16_step2.predict(x_test_vgg16_step2_pca)
accuracy2_svm_vgg16 = accuracy_score(y_test_step2, prediction2_svm_vgg16)

# KNN
knn_vgg16_step2 = KNeighborsClassifier()
cross_val_scores_knn_vgg16_step2 = cross_val_score(knn_vgg16_step2, x_train_vgg16_step2_pca, y_train_step2, cv=5)
knn_vgg16_step2.fit(x_train_vgg16_step2_pca, y_train_step2)
prediction2_knn_vgg16 = knn_vgg16_step2.predict(x_test_vgg16_step2_pca)
accuracy2_knn_vgg16 = accuracy_score(y_test_step2, prediction2_knn_vgg16)

# Logistic Regression
logreg_vgg16_step2 = LogisticRegression()
cross_val_scores_logreg_vgg16_step2 = cross_val_score(logreg_vgg16_step2, x_train_vgg16_step2_pca, y_train_step2, cv=5)
logreg_vgg16_step2.fit(x_train_vgg16_step2_pca, y_train_step2)
prediction2_logreg_vgg16 = logreg_vgg16_step2.predict(x_test_vgg16_step2_pca)
accuracy2_logreg_vgg16 = accuracy_score(y_test_step2, prediction2_logreg_vgg16)

# Print the cross-validation scores and validation accuracy for the second step with VGG16 features
print('\n', 'The results of the second step classification model using VGG16 features', '\n')
print("Second Step Cross-Validation Scores for Random Forest (VGG16):", cross_val_scores_pca_vgg16_step2)
print("Second Step Test Accuracy for Random Forest (VGG16):", test_accuracy2_rf_vgg16, '\n')
print("Second Step Cross-Validation Scores for SVM (VGG16):", cross_val_scores_svm_vgg16_step2)
print("Second Step Test Accuracy for SVM (VGG16):", accuracy2_svm_vgg16, '\n')
print("Second Step Cross-Validation Scores for KNN (VGG16):", cross_val_scores_knn_vgg16_step2)
print("Second Step Test Accuracy for KNN (VGG16):", accuracy2_knn_vgg16, '\n')
print("Second Step Cross-Validation Scores for Logistic Regression (VGG16):", cross_val_scores_logreg_vgg16_step2)
print("Second Step Test Accuracy for Logistic Regression (VGG16):", accuracy2_logreg_vgg16, '\n')

#==========================================================
# MACHINE LEARNING MODELS FOR THE SECOND STEP USING RESNET
#==========================================================

# Random Forest
rfc_pca_resnet_step2 = RandomForestClassifier(n_estimators=50, random_state=42)
cross_val_scores_pca_resnet_step2 = cross_val_score(rfc_pca_resnet_step2, x_train_resnet_step2_pca, y_train_step2, cv=5)
rfc_pca_resnet_step2.fit(x_train_resnet_step2_pca, y_train_step2)
prediction_2_rf_resnet = rfc_pca_resnet_step2.predict(x_test_resnet_step2_pca)
test_accuracy2_rf_resnet = accuracy_score(y_test_step2, prediction_2_rf_resnet)

# SVM
svm_resnet_step2 = SVC()
cross_val_scores_svm_resnet_step2 = cross_val_score(svm_resnet_step2, x_train_resnet_step2_pca, y_train_step2, cv=5)
svm_resnet_step2.fit(x_train_resnet_step2_pca, y_train_step2)
prediction2_svm_resnet = svm_resnet_step2.predict(x_test_resnet_step2_pca)
accuracy2_svm_resnet = accuracy_score(y_test_step2, prediction2_svm_resnet)

# KNN
knn_resnet_step2 = KNeighborsClassifier()
cross_val_scores_knn_resnet_step2 = cross_val_score(knn_resnet_step2, x_train_resnet_step2_pca, y_train_step2, cv=5)
knn_resnet_step2.fit(x_train_resnet_step2_pca, y_train_step2)
prediction2_knn_resnet = knn_resnet_step2.predict(x_test_resnet_step2_pca)
accuracy2_knn_resnet = accuracy_score(y_test_step2, prediction2_knn_resnet)

# Logistic Regression
logreg_resnet_step2 = LogisticRegression()
cross_val_scores_logreg_resnet_step2 = cross_val_score(logreg_resnet_step2, x_train_resnet_step2_pca, y_train_step2, cv=5)
logreg_resnet_step2.fit(x_train_resnet_step2_pca, y_train_step2)
prediction2_logreg_resnet = logreg_resnet_step2.predict(x_test_resnet_step2_pca)
accuracy2_logreg_resnet = accuracy_score(y_test_step2, prediction2_logreg_resnet)

# Print the cross-validation scores and validation accuracies for the second step with ResNet features
print('\n','The results of the second step classification model using ResNet features', '\n')
print("Second Step Cross-Validation Scores for Random Forest:", cross_val_scores_pca_resnet_step2)
print("Second Step Test Accuracy for Random Forest:", test_accuracy2_rf_resnet, '\n')
print("Second Step Cross-Validation Scores for SVM:", cross_val_scores_svm_resnet_step2)
print("Second Step Test Accuracy for SVM:", accuracy2_svm_resnet, '\n')
print("Second Step Cross-Validation Scores for KNN:", cross_val_scores_knn_resnet_step2)
print("Second Step Test Accuracy for KNN:", accuracy2_knn_resnet, '\n')
print("Second Step Cross-Validation Scores for Logistic Regression:", cross_val_scores_logreg_resnet_step2)
print("Second Step Test Accuracy for Logistic Regression:", accuracy2_logreg_resnet, '\n')


#============================================================
# MACHINE LEARNING MODELS FOR THE SECOND STEP USING DENSENET
#============================================================

# Random Forest
rfc_pca_densenet_step2 = RandomForestClassifier(n_estimators=50, random_state=42)
cross_val_scores_rf_densenet_step2 = cross_val_score(rfc_pca_densenet_step2, x_train_densenet_step2_pca, y_train_step2, cv=5)
rfc_pca_densenet_step2.fit(x_train_densenet_step2_pca, y_train_step2)
# Predict on the PCA-transformed validation data
prediction_2_rf_densenet = rfc_pca_densenet_step2.predict(x_test_densenet_step2_pca)
# Evaluate the model on the validation dataset
test_accuracy2_rf_densenet = accuracy_score(y_test_step2, prediction_2_rf_densenet)

# SVM
svm_densenet_step2 = SVC()
cross_val_scores_svm_densenet_step2 = cross_val_score(svm_densenet_step2, x_train_densenet_step2_pca, y_train_step2, cv=5)
svm_densenet_step2.fit(x_train_densenet_step2_pca, y_train_step2)
prediction2_svm_densenet = svm_densenet_step2.predict(x_test_densenet_step2_pca)
accuracy2_svm_densenet = accuracy_score(y_test_step2, prediction2_svm_densenet)

# KNN
knn_densenet_step2 = KNeighborsClassifier()
cross_val_scores_knn_densenet_step2 = cross_val_score(knn_densenet_step2, x_train_densenet_step2_pca, y_train_step2, cv=5)
knn_densenet_step2.fit(x_train_densenet_step2_pca, y_train_step2)
prediction2_knn_densenet = knn_densenet_step2.predict(x_test_densenet_step2_pca)
accuracy2_knn_densenet = accuracy_score(y_test_step2, prediction2_knn_densenet)

# Logistic Regression
logreg_densenet_step2 = LogisticRegression()
cross_val_scores_logreg_densenet_step2 = cross_val_score(logreg_densenet_step2, x_train_densenet_step2_pca, y_train_step2, cv=5)
logreg_densenet_step2.fit(x_train_densenet_step2_pca, y_train_step2)
prediction2_logreg_densenet = logreg_densenet_step2.predict(x_test_densenet_step2_pca)
accuracy2_logreg_densenet = accuracy_score(y_test_step2, prediction2_logreg_densenet)
warnings.filterwarnings("default", category=ConvergenceWarning)

# Print the cross-validation scores and validation accuracies for the second model
print('\n','The results of the second step classification model using DenseNet features', '\n')
print("Second Step Cross-Validation Scores for Random Forest:", cross_val_scores_rf_densenet_step2)
print("Second Step Test Accuracy for Random Forest:", test_accuracy2_rf_densenet, '\n')
print("Second Step Cross-Validation Scores for SVM:", cross_val_scores_svm_densenet_step2)
print("Second Step Test Accuracy for SVM:", accuracy2_svm_densenet, '\n')
print("Second Step Cross-Validation Scores for KNN:", cross_val_scores_knn_densenet_step2)
print("Second Step Test Accuracy for KNN:", accuracy2_knn_densenet, '\n')
print("Second Step Cross-Validation Scores for Logistic Regression:", cross_val_scores_logreg_densenet_step2)
print("Second Step Test Accuracy for Logistic Regression:", accuracy2_logreg_densenet, '\n')