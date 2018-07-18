'''
Please add features 5,6,7 using this sample from our dataset
Notes:
    Even and 0 represent the X coordinate
    Odd is the Y coordinate
    Every 2 pair is one point, Eg point 1 = 328.444, 275.496
    Return a number
'''
import math

sample_1 = [328.444, 275.496, 434.921, 275.029, 331.713, 401.121, 427.449, 400.187, 271.936, 270.826, 356.464, 254.014, 388.221, 255.882, 494.698, 268.491, 263.997, 274.095, 301.825, 277.831, 349.459, 277.364, 411.104, 278.298, 459.673, 277.831, 515.246, 276.897, 368.606, 340.41, 355.53, 351.151, 391.957, 350.217, 374.253, 395.527, 374.253, 416.925, 373.276, 483.314, 280.342, 404.39, 499.835, 402.522]

'''
    Main Program
'''
def main():
    try:
        print("Program has begun \n -------------------------------- \n")
        # note change x1, y1, x2, y2 
        a = math.sqrt((sample_1[0]-sample_1[2]) ** 2 + (sample_1[1]-sample_1[3]) ** 2)
        b = math.sqrt((sample_1[16]-sample_1[26]) ** 2 + (sample_1[17]-sample_1[27]) ** 2)
        print('feature_1:', a / b)
       
    finally:
        print('Process Completed')


'''
    Final Exicution of Main Program
'''
if __name__ == "__main__":
    main()