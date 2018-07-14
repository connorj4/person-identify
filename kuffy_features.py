'''
Please add features 5,6,7 using this sample from our dataset
Notes:
    Even and 0 represent the X coordinate
    Odd is the Y coordinate
    Every 2 pair is one point, Eg point 1 = 328.444, 275.496
    Return a number
'''

sample_1 = [328.444, 275.496, 434.921, 275.029, 331.713, 401.121, 427.449, 400.187, 271.936, 270.826, 356.464, 254.014, 388.221, 255.882, 494.698, 268.491, 263.997, 274.095, 301.825, 277.831, 349.459, 277.364, 411.104, 278.298, 459.673, 277.831, 515.246, 276.897, 368.606, 340.41, 355.53, 351.151, 391.957, 350.217, 374.253, 395.527, 374.253, 416.925, 373.276, 483.314, 280.342, 404.39, 499.835, 402.522]


'''
5.	Lip length ratio: Distance between points 
    2 and 3 over distance between 20 and 21.
'''
def lip_length_ratio(x1, y1, x2, y2):
    # note code needs to be developed here
    return feature_5
'''
6.	Eye-brow length ratio: Distance between points 4 and 5 
    (or distance between points 6 and 7 whichever is larger) 
    over distance between 8 and 13.
'''
def eye_brow_length(x1, y1, x2, y2):
    # note code needs to be developed here
    return feature_6

'''
7.	Aggressive ratio: 
'''
def aggressive_ratio():
    # note code needs to be developed here
    return feature_7


'''
    Main Program
'''
def main():
    try:
        print("Program has begun")
        # note change x1, y1, x2, y2 
        print('Aggressive ratio: ', lip_length_ratio(x1, y1, x2, y2))
        print('Eye-brow length ratio: ', eye_brow_length(x1, y1, x2, y2))
        print('Aggressive ratio: ', aggressive_ratio())

        check = math.sqrt((face_array[14]-face_array[24]) ** 2 + (face_array[15]-face_array[25]) ** 2)
       
    except:
        print('Error Detected')
        
    finally:
        print('Process Completed')


'''
    Final Exicution of Main Program
'''
if __name__ == "__main__":
    main()