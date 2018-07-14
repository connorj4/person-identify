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
from sklearn.neighbors import KNeighborsClassifier

'''
    Load Data
'''

def read_files(gender, person):

    try:
        face_data = []
        with open(os.path.join('points_22/'+gender+'-'+str('{0:03}'.format(person)), gender+'-'+str('{0:03}'.format(person))+'-01.pts'), 'r') as file_object:
            data = file_object.read().split()
            for element in data:
                try:
                    face_data.append(float(element))
                except ValueError:
                    pass
            face_data = face_data[2:] 
        file_object.close()
        return face_data

    except IOError as e:
        print('File: ' + gender+'-'+str('{0:03}'.format(person))+'-01.pts - was not found')
        pass


def import_points():
    dataset = []
    #note just reading one file
    #need to loop men and women and build the dataset
    face_array = read_files('m', 1)
    print('face m 1:', face_array, len(face_array))

    vector_pts = np.array(face_array)
    vector_pts = vector_pts.reshape(22, 2)
    print('vector pts', vector_pts)
    print('feature_0: ', eye_length_ratio(vector_pts[9, ], vector_pts[10, ], vector_pts[11, ], vector_pts[12, ], vector_pts[8, ], vector_pts[13, ]))
    print('feature_1: ', eye_distance_ratio(vector_pts[0, ], vector_pts[1, ],vector_pts[8, ], vector_pts[13, ]))
    print('feature_2: ', nose_ratio(vector_pts[15, ], vector_pts[16, ], vector_pts[20, ], vector_pts[21, ]))
    print('feature_3: ', lip_size_ratio(vector_pts[2, ], vector_pts[3, ], vector_pts[17, ], vector_pts[18, ]))
    print('feature_4: ', lip_length_ratio(vector_pts[2, ], vector_pts[3, ], vector_pts[20, ], vector_pts[21, ]))
    print('feature_5: ', eye_length_ratio(vector_pts[4, ], vector_pts[5, ], vector_pts[6, ], vector_pts[7, ], vector_pts[8, ], vector_pts[13, ]))
    print('feature_6: ', lip_length_ratio(vector_pts[10, ], vector_pts[19, ], vector_pts[20, ], vector_pts[21, ]))

    dataset.append

    #double check
    #check = math.sqrt((face_array[14]-face_array[24]) ** 2 + (face_array[15]-face_array[25]) ** 2)
    #print('check: ', check)

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
        print('Dataset: ', import_points())

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