import numpy as np
import cv2
import sys
import skvideo

vid='vid.mp4'
border_crop=10
show_points=True
inter_frame_delay=20
smoothing_window=100

rolling_trajectory_list=[]


cap = cv2.VideoCapture(vid)

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 200,
                       qualityLevel = 0.4,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 4,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))




# Take first frame and find corners in it
ret, old_frame = cap.read()
# old_frame = old_frame[:1000]

if (ret):
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    #old_gray = clahe.apply(old_gray)
    transformation_matrix_avg = cv2.estimateRigidTransform(old_frame, old_frame, False)


rows,cols = old_gray.shape
print("Video resolution: " + str(cols) + "* "+ str(rows))


points_to_track = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

print("Trackable points detected in first frame:")
print(points_to_track)

frame_mask = np.zeros_like(old_frame)

while(1):
    ret,frame = cap.read()
    
    if not ret:
        break
    # frame = frame[:1000]
    
    #Read a frame and convert it to greyscale
    new_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, new_gray = cv2.threshold(new_gray, 190, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5,5),np.uint8)
    new_gray = cv2.dilate(new_gray,kernel,iterations = 6)
    new_gray = cv2.erode(new_gray,kernel,iterations = 6)
    
    #new_gray = clahe.apply(new_gray)
    #new_gray = cv2.equalizeHist(new_gray)
    #calculate optical flow between the latest frame (new_gray) and the last one we examined
    print("Searching for optical flow between this frame and the last...")
    new_points, matched, err = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, points_to_track, None, **lk_params)

    # Select good tracked points - matched==1 if the point has been found in the new frame
    new = new_points[matched==1]
    old = points_to_track[matched==1]

    print("Old point coordinates:")
    print(old)

    print("New point coordinates:")
    print(new)

    # This should return a transformation matrix mapping the points in "new" to "old"
    transformation_matrix = cv2.estimateRigidTransform(new, old, False)
    print("Transform from new frame to old frame...")
    print(transformation_matrix)
    # Not sure about this...trying to create an smoothed average of the frame movement over the last X frames
    rolling_trajectory_list.append(transformation_matrix)
    if len(rolling_trajectory_list) > smoothing_window:
        rolling_trajectory_list.pop(0)

    transformation_matrix_avg=transformation_matrix

    print("Average transformation over last " + str(smoothing_window) + " frames:")
    print(transformation_matrix_avg)
    #Apply the transformation to the frame
    stabilized_frame = cv2.warpAffine(frame,transformation_matrix_avg,(cols,rows),flags=cv2.INTER_NEAREST|cv2.WARP_INVERSE_MAP)
    stab_gray = cv2.cvtColor(stabilized_frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2,2))
    stab_gray = clahe.apply(stab_gray)
    #stabilized_frame = stabilized_frame[400:1400][100:2100][:]
    
    if show_points:
        for point in new:
            corner_x=point[0]
            corner_y=point[1]
            frame = cv2.circle(frame,(corner_x,corner_y),20,(0,255,0),-1)

        for point in old:
            corner_x=point[0]
            corner_y=point[1]
            frame = cv2.circle(frame,(corner_x,corner_y),2,(0,255,255),-1)
            
            
    numpy_horizontal_concat = np.concatenate((frame, stabilized_frame), axis=1)
    cv2.namedWindow('Video', cv2.WINDOW_NORM)
    cv2.imshow('Video', numpy_horizontal_concat)
    print(stabilized_frame.shape)
    
    cv2.waitKey(inter_frame_delay)

    old_gray = new_gray.copy()
    points_to_track = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)


cv2.destroyAllWindows()
cap.release()