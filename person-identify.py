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
import itertools

'''
    Load Data
'''

# Create Database
def ImportPoints():
    # Create person 22pts database
    database = {}
    #face_data = []

    # note Create loop to import all data from database
    
    # Read files and import data into dictionary

    # add men
    file_name_ct = 1
    fine_name_ftr = 1
    file_name_gdr = 'm'
    while file_name_ct < 77:
        file_data = file_name_gdr+'-'+str('{0:03}'.format(file_name_ct))+'-'+str('{0:02}'.format(fine_name_ftr))+'.pts'
        try:
            with open(os.path.join('points_22/'+file_name_gdr+'-'+str('{0:03}'.format(file_name_ct)), file_data), 'r') as file_object:
                data = file_object.read().split()
                face_data = []
                for element in data:
                    try:
                        face_data.append(float(element))
                    except ValueError:
                        pass
        except IOError as e:
            print('File ' + file_data + ' was not found')
            pass

        database[file_name_gdr+'-'+str('{0:03}'.format(file_name_ct))] = face_data[2:] 
        file_object.close()
        file_name_ct += 1

    # add women
    file_name_ct = 1
    fine_name_ftr = 1
    file_name_gdr = 'w'
    print("This is working")
    while file_name_ct < 61:
        file_data = file_name_gdr+'-'+str('{0:03}'.format(file_name_ct))+'-'+str('{0:02}'.format(fine_name_ftr))+'.pts'
        try:
            with open(os.path.join('points_22/'+file_name_gdr+'-'+str('{0:03}'.format(file_name_ct)), file_data), 'r') as file_object:
                data = file_object.read().split()
                face_data = []
                for element in data:
                    try:
                        face_data.append(float(element))
                    except ValueError:
                        pass
        except IOError as e:
            print('File ' + file_data + ' was not found')
            pass

        database[file_name_gdr+'-'+str('{0:03}'.format(file_name_ct))] = face_data[2:] 
        file_object.close()
        file_name_ct += 1

    return database

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
        #testing creating the database
        print('Database Count: ', len(ImportPoints()))

    except IOError:
        print("The file was not found. Check name or path.")
        
    except ValueError:
        print("The data type had an error.")

    finally:
        print('Process Completed')
'''
    Final Exicution of Main Program
'''
if __name__ == "__main__":
    main()