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
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
#from sklearn import metrics


'''
    Load Data
'''

def read_files(gender, person):

    try:
        face_data = []
        with open(os.path.join('points_22/'+gender+'-'+str('{0:03}'.format(person)), gender+'-'+str('{0:03}'.format(person))+'-01.pts'), 'r') as file_object:
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
    dataset = pd.DataFrame(columns=['F1','F2','F3','F4','F5','F6','F7'])
    person_class = 0
    
    if person_class < 76:
        counter = 0
        while counter < 75:
            try:
                # Read the files
                face_array = read_files('m', counter + 1)
                # Convert to nupmy array
                vector_pts = np.array(face_array)
                vector_pts = vector_pts.reshape(22, 2)
                # Add Features to the dataset
                dataset = dataset.append(pd.DataFrame(feature_extraction(vector_pts), columns=['F1','F2','F3','F4','F5','F6','F7']),ignore_index=True)
            except:
                #print('Person: ', person_class, ' was skipped.')
                pass

            # Count up
            person_class += 1
            counter += 1
     
    if person_class > 74:
        counter = 0
        while counter < 60:
            try:
                # Read the files
                face_array = read_files('w', counter + 1)
                # Convert to nupmy array
                vector_pts = np.array(face_array)
                vector_pts = vector_pts.reshape(22, 2)
                # Add Features to the dataset
                dataset = dataset.append(pd.DataFrame(feature_extraction(vector_pts), columns=['F1','F2','F3','F4','F5','F6','F7']),ignore_index=True)
            except:
                #print('Person: ', person_class, ' was skipped.')
                pass
            # Count up
            person_class += 1
            counter += 1
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
    features = [np.around([feature_0,feature_1,feature_2,feature_3,feature_4,feature_5,feature_6], decimals=5)]
    #print('features: ',features)
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
        all_data = import_points()
        y = list(all_data.index)
        X_train, X_test, y_train, y_test = train_test_split(all_data, y, test_size=0.3, random_state=1)

        neigh = KNeighborsClassifier(n_neighbors=7)
        model = neigh.fit(X_train, y_train)

        predictions = neigh.predict(X_test)
        #print(neigh.predict_proba([[0.9]]))

        plt.scatter(y_test, predictions)
        plt.xlabel('True Values')
        plt.ylabel('Predictions')

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