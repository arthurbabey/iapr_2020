import cv2
file_loc=str(input("Please enter your video location"))
vidcap = cv2.VideoCapture(file_loc)
print(cv2.__version__)
count = 0
success = True
while success:
    success,image = vidcap.read(1000)
    cv2.imwrite("frame%d.tiff" % count, image)     # save frame as JPEG file in the same location as the script
    success,image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1