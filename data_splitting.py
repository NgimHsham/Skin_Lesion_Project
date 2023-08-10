# -*- coding: utf-8 -*-
"""
Created on Mon May 15 15:11:59 2023

@authors:
    Taofik Ahmed Suleiman
    Daniel Tweneboah Anyimadu
    Andrew Dwi Permana
    Hsham Abdalgny Abdalwhab Ngim

@supervisor: 
    Professor Alessandra Scotto di Freca
"""

from data_preparation import *


''' Split the data for the first classification - Benign vs Others
Steps:
1. Split the data class-wise into a training and a test
2. Set training set = first 70% of the images, class-wise
3. Set test set = second 30% of the images, class-wise
4. No random split (this way different groups will adopt the same fixed split)
5. Shuffle the data to avoid learning order bias '''

# Split benign and others data separately
Y = benign_others_data['others']
benign_data = image_pixel[Y == 0]
others_data = image_pixel[Y == 1]

# Calculate the index for splitting
benign_split_index = int(len(benign_data) * 0.7)
others_split_index = int(len(others_data) * 0.7)

# Split into training and test sets
benign_train = benign_data[:benign_split_index]
benign_test = benign_data[benign_split_index:]

others_train = others_data[:others_split_index]
others_test = others_data[others_split_index:]

# Combine the training and test sets
x_train1 = np.concatenate([benign_train, others_train])
y_train1 = np.concatenate([np.zeros(len(benign_train)), np.ones(len(others_train))])

x_test1 = np.concatenate([benign_test, others_test])
y_test1 = np.concatenate([np.zeros(len(benign_test)), np.ones(len(others_test))])

# Shuffle to avoid learning other bias
train_data_new = list(zip(x_train1, y_train1))
random.shuffle(train_data_new)
x_train_shuffled, y_train_shuffled = zip(*train_data_new)

# Convert the shuffled lists back to numpy arrays
x_train = np.array(x_train_shuffled)
y_train = np.array(y_train_shuffled)

test_data_new = list(zip(x_test1, y_test1))
random.shuffle(test_data_new)
x_test_shuffled, y_test_shuffled = zip(*test_data_new)

# Convert the shuffled lists back to numpy arrays
x_test = np.array(x_test_shuffled)
y_test = np.array(y_test_shuffled)




''' Split the data for the second classification melanoma vs seborrheic_keratosis

Steps:
1. Create a new column for melanoma vs seborrheic, this column returns 0 for benign, 
   1 for melanoma and 2 for seborrheic
2. From the new column, extract the columns that correspond to melanoma into melanoma data 
   and the other into seborrheic
3. Split the data class-wise into a training and a test
4. Set training set = first 70% of the images, class-wise
5. Set test set = second 30% of the images, class-wise
6. No random split (this way different groups will adopt the same fixed split)
7. Shuffle the data to avoid learning order bias

NB: When the model is used in real life, the second step will be extracting the columns 
    predicted as others from the first step.'''

benign_others_data['melanoma_vs_seborrheic'] = np.where(benign_others_data['melanoma'] == 1, 1,
                                                      np.where(benign_others_data['seborrheic_keratosis'] == 1, 2, 0))

# Split benign and others data separately
Y_step2 = benign_others_data['melanoma_vs_seborrheic']
melanoma_data = image_pixel[Y_step2 == 1]
seborrheic_data = image_pixel[Y_step2 == 2]

# Calculate the index for splitting
melanoma_split_index = int(len(melanoma_data) * 0.7)
seborrheic_split_index = int(len(seborrheic_data) * 0.7)

# Split into training and test sets
melanoma_train = melanoma_data[:melanoma_split_index]
melanoma_test = melanoma_data[melanoma_split_index:]

seborrheic_train = seborrheic_data[:seborrheic_split_index]
seborrheic_test = seborrheic_data[seborrheic_split_index:]

# Combine the training and test sets
x_train2 = np.concatenate([melanoma_train, seborrheic_train])
y_train2 = np.concatenate([np.zeros(len(melanoma_train)), np.ones(len(seborrheic_train))])

x_test2 = np.concatenate([melanoma_test, seborrheic_test])
y_test2 = np.concatenate([np.zeros(len(melanoma_test)), np.ones(len(seborrheic_test))])

# Shuffle to avoid learning other bias
train2_data_new = list(zip(x_train2, y_train2))
random.shuffle(train2_data_new)
x_train2_shuffled, y_train2_shuffled = zip(*train2_data_new)

# Convert the shuffled lists back to numpy arrays
x_train_step2 = np.array(x_train2_shuffled)
y_train_step2 = np.array(y_train2_shuffled)

test2_data_new = list(zip(x_test2, y_test2))
random.shuffle(test2_data_new)
x_test2_shuffled, y_test2_shuffled = zip(*test2_data_new)

# Convert the shuffled lists back to numpy arrays
x_test_step2 = np.array(x_test2_shuffled)
y_test_step2 = np.array(y_test2_shuffled)
