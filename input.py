import stabilizeme
import AddFrames
import cv2

#stabilizer = stabilizeme.StabilizeMe()
#stabilizer.get_transform('vid.mp4')

#stabilizer.applyTransforms('vid.mp4', 'stab_vid2.mp4')
AddFrames.addFrames('stab_vid.avi')


# print(stabilizer.smoothed_trajectory)
# print(stabilizer.trajectory)
# print(stabilizer.transforms)