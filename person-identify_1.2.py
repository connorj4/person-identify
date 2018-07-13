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
        with open(os.path.join('points_22/'+gender+'-00'+str('{0:03}'.format(person)), 
                gender+'-'+str('{0:03}'.format(person))+'-01.pts'), 'r') as file_object:

            data = file_object.read().split()
            face_data[]
            for element in data:
                try:
                    face_data.append(float(element))
                except ValueError:
                    pass
            array = face_data[2:] 
        file_object.close()

    except IOError as e:
        print('File ' + file_data + ' was not found')
        pass
    finally:
        return array

def import_points():
    dataset = []
    face_array = read_files(m, 1)

    return dataset

'''
    Define Features
'''

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
        print("Program has begun")
        print('Dataset: ', import_points()
        break
    except:
        print("Unexpected error:", sys.exc_info()[0])
        raise
    finally:
        print('Process Completed')


'''
    Final Exicution of Main Program
'''
if __name__ == "__main__":
    main()