""" Main Python """

#-----------------------------------------------------------------------------
#  Person Identification from Face Images
#
#  CSC 481-01 Summer 2018 Haronne Cuffy. <cuffyh1@southernct.edu>
#  CSC 581-01 Summer 2018 Joshua Connor. <connorj4@southernct.edu>
#
#  Distributed under the terms of the GNU GENERAL PUBLIC LICENSE, 
#  Version 3, 29 June 2007
#-----------------------------------------------------------------------------

import os 
import sys
import numpy as np
import pylab as pl
#import pandas as pd

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn import metrics



'''
    Load Data
'''

def read_files(gender, person, image):

    try:
        face_data = []
        with open(os.path.join('points_22/'+gender+'-'+str('{0:03}'.format(person)), gender+'-'+str('{0:03}'.format(person))+'-0'+str(image)+'.pts'), 'r') as file_object:
            data = file_object.read().split()
            #print('data: ', data)
            for element in data:
                try:
                    face_data.append(float(element))
                except ValueError:
                    pass
            face_data = face_data[2:]
            #print('face_data ', face_data)
        file_object.close()
        return face_data

    except IOError as e:
        #print('File: ' + gender+'-'+str('{0:03}'.format(person))+'-01.pts - was not found')
        pass


def import_points():
    #dataset = pd.DataFrame(columns=['data','target'])
    data_training = []
    target_training = []

    data_testing = []
    target_testing = []
    
    img_sample = 0
    while img_sample < 3:
        # Testing data creation
        counter = 0
        if img_sample % 2 == 0:
            while counter < 20:
                try:
                    # Read the files
                    face_array = read_files('m', counter + 1, img_sample + 1)
                    # Convert to nupmy array
                    vector_pts = np.array(face_array)
                    vector_pts = vector_pts.reshape(22, 2)

                    print('last cood train: ', img_sample, ':', counter, vector_pts[-1])
                    # Add Features to the dataset
                    #dataset = dataset.append({'data': feature_extraction(vector_pts)}, {'target': counter}, ignore_index=True)
                    data_training.append(feature_extraction(vector_pts))
                    target_training.append(counter)
                    counter += 1
                #except IOError as err:
                    #print("I/O error: {0}".format(err))
                    #pass
                except:
                    print('Person: ', img_sample, ': ', counter, ' was skipped.')
                    #print("Unexpected error:", sys.exc_info()[0])
                    #pass
                # Count up
            img_sample += 1
        else:
            while counter < 20:
                try:
                    # Read the files
                    face_array = read_files('m', counter + 1, img_sample + 1)
                    # Convert to nupmy array
                    vector_pts = np.array(face_array)
                    vector_pts = vector_pts.reshape(22, 2)

                    print('last coord test: ', img_sample, ':', counter, vector_pts[-1])
                    # Add Features to the dataset
                    #dataset = dataset.append({'data': feature_extraction(vector_pts)}, {'target': counter}, ignore_index=True)
                    data_testing.append(feature_extraction(vector_pts))
                    target_testing.append(counter)
                    counter += 1
                #except IOError as err:
                    #print("I/O error: {0}".format(err))
                    #pass
                except:
                    print('Person: ', img_sample, ': ', counter, ' was skipped.')
                    #print("Unexpected error:", sys.exc_info()[0])
                    pass
                # Count up
            img_sample += 1


    #print('\ndata: ', data, len(data), '\n------------\n')
    #print('\ntarget: ', target, len(target), '\n------------\n')

    data_training = np.array(data_training)
    target_training = np.array(target_training)
    data_testing = np.array(data_testing)
    target_testing = np.array(target_testing)

    dataset = {'data_train': data_training, 'target_train': target_training, 'data_test': data_testing, 'target_test': target_testing}
    #dataset_testing = {'data_test': data_testing, 'target_test': target_testing}
    #print('\ndict: ', d, len(d), '\n------------\n')
    #Convert to panda dataframe
    #dataset = pd.DataFrame(data=d)

    return dataset

'''
    Define Features
'''

# 01. Eye length ratio
def eye_length_ratio(pt_9, pt_10, pt_11, pt_12, pt_8, pt_13 ):
    eye_left = np.linalg.norm(pt_9-pt_10)
    eye_right = np.linalg.norm(pt_11-pt_12)
    if eye_left > eye_right:
        feature_0 = eye_left/np.linalg.norm(pt_8-pt_13)
    else:
        feature_0 = eye_right/np.linalg.norm(pt_8-pt_13)
    return feature_0

# 02. Eye distance ratio
def eye_distance_ratio(pt_0, pt_1, pt_8, pt_13):
    feature_1 = np.linalg.norm(pt_0-pt_1) / np.linalg.norm(pt_8-pt_13)
    return feature_1

# 03. Nose ratio
def nose_ratio(pt_15, pt_16, pt_20, pt_21):
    feature_2 = np.linalg.norm(pt_15-pt_16) / np.linalg.norm(pt_20-pt_21)
    return feature_2

# 04. Lip size ratio
def lip_size_ratio(pt_2, pt_3, pt_17, pt_18):
    feature_3 = np.linalg.norm(pt_2-pt_3) / np.linalg.norm(pt_17-pt_18)
    return feature_3

# 05. Lip length ratio
def lip_length_ratio(pt_2, pt_3, pt_20, pt_21):
    feature_4 = np.linalg.norm(pt_2-pt_3) / np.linalg.norm(pt_20-pt_21)
    return feature_4

# 06. Eye-brow length ratio
def eye_brow_length_ratio(pt_4, pt_5, pt_6, pt_7, pt_8, pt_13):
    eyebr_left = np.linalg.norm(pt_4-pt_5)
    eyebr_right = np.linalg.norm(pt_6-pt_7)
    if eyebr_left > eyebr_right:
        feature_5 = eyebr_left/np.linalg.norm(pt_8-pt_13)
    else:
        feature_5 = eyebr_right/np.linalg.norm(pt_8-pt_13)
    return feature_5

# 07. Aggressive ratio
def aggressive_ratio(pt_10, pt_19, pt_20, pt_21):
    feature_6 = np.linalg.norm(pt_10-pt_19) / np.linalg.norm(pt_20-pt_21)
    return feature_6


'''
    Feature Extraction
'''
def feature_extraction(vector_pts):
    feature_0 = eye_length_ratio(vector_pts[9, ], vector_pts[10, ], vector_pts[11, ], vector_pts[12, ], vector_pts[8, ], vector_pts[13, ])
    feature_1 = eye_distance_ratio(vector_pts[0, ], vector_pts[1, ],vector_pts[8, ], vector_pts[13, ])
    feature_2 = nose_ratio(vector_pts[15, ], vector_pts[16, ], vector_pts[20, ], vector_pts[21, ])
    feature_3 = lip_size_ratio(vector_pts[2, ], vector_pts[3, ], vector_pts[17, ], vector_pts[18, ])
    feature_4 = lip_length_ratio(vector_pts[2, ], vector_pts[3, ], vector_pts[20, ], vector_pts[21, ])
    feature_5 = eye_length_ratio(vector_pts[4, ], vector_pts[5, ], vector_pts[6, ], vector_pts[7, ], vector_pts[8, ], vector_pts[13, ])
    feature_6 = lip_length_ratio(vector_pts[10, ], vector_pts[19, ], vector_pts[20, ], vector_pts[21, ])
    features = [feature_0,feature_1,feature_2,feature_3,feature_4,feature_5,feature_6]
    #features = [feature_0,feature_1,feature_3,feature_4,feature_5]
    #print('features: ', features)
    return features

'''
Training Data and Testing Data Split
'''
#def split_data(dataset):
    
    #print(X_train.shape, y_train.shape)
    #print(X_test.shape, y_test.shape)

    #return X_train, X_test, y_train, y_test

'''
Classifiers
'''

# I.	K-Nearest Neighbors

# II.	Artificial Neural Network

# III.	Naïve Bayes


'''
    Summarize Scores
'''

'''
    Summarize Selected Features
'''

'''
    Main Person Identification
'''
def main():
    try:
        print("\n\n Program Has Begun... \n ------------------------------------------------------------- \n")
        # The dataset
        data_set = import_points()
        #print('Just features: \n', data_set, '\n\n')

        X_train = data_set['data_train']
        y_train = data_set['target_train']

        X_test = data_set['data_test']
        y_test = data_set['target_test']

        # Pre processing
        scaler = StandardScaler()
        # Fit only to the training data
        scaler.fit(X_train)

        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        mlp = MLPClassifier(hidden_layer_sizes=(7,7,7),max_iter=500)
        
        print('\n-------------------------------------\n')

        mlp.fit(X_train, y_train)
        print(mlp, '\n')

        print('\n-------------------------------------\n')
        # Compare y_test
        #print('\nY Test Class: ', y_test, len(y_test), '\n')
        test_predict = mlp.predict(X_test)
        print('Test Predict: ', test_predict, len(test_predict))

        accuracy = metrics.accuracy_score(y_test, test_predict)
        print('Test Accuracy: ', accuracy)
        print('\n-------------------------------------\n')
        train_predict = mlp.predict(X_train)
        print('Train Predict: ', train_predict, len(train_predict))

        accuracy = metrics.accuracy_score(y_train, train_predict)
        print('Train Accuracy: ', accuracy)
        print('\n-------------------------------------\n')

        
        print(metrics.classification_report(y_test, test_predict))

        # cofusion matrix
        labels = list(range(len(test_predict)))
        print("\n\n ------------------------------------------------------------- \n Confustion matrix:\n",metrics.confusion_matrix(y_test, test_predict, labels))


    except:
        print("\n\n ------------------------------------------------------------- \n Unexpected Error:\n")
        raise

    finally:
        print('\n\n ------------------------------------------------------------- \n Process Completed \n')


'''
    Final Exicution of Main Program
'''
if __name__ == "__main__":
    main()