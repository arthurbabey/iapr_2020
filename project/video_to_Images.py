import cv2
file_loc='data/robot_parcours_1.avi'
vidcap = cv2.VideoCapture(file_loc)
print(cv2.__version__)
count = 0
success = True
while success:
    success,image = vidcap.read(500)
    if success:
        cv2.imwrite("data/frame%d.jpeg" % count, image)     # save frame as JPEG file in the same location as the script
    #success,image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1
