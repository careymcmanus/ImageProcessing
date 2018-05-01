import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

feature_params = dict( maxCorners = 200,
                       qualityLevel = 0.4,
                       minDistance = 7,
                       blockSize = 7 )

lk_params = dict( winSize  = (50,50),
                  maxLevel = 4,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

class StabilizeMe:

    def __init__(self):
        self.trajectory = None
        self.smoothed_trajectory = None
        self.transforms = None
        self.frameSize = None

    def get_transform(self, input, smoothing_window=30):
        
        capture = cv2.VideoCapture(input)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    

        _gotFrame, prev_frame = capture.read()
        prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        ret, prev_frame_gray = cv2.threshold(prev_frame_gray, 190, 255, cv2.THRESH_BINARY)
        self.frameSize = prev_frame.shape

        prev_to_cur_transform = []

        for i in range(frame_count - 1):

            _gotFrame, cur_frame = capture.read()
            cur_frame_gray = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
            ret, cur_frame_gray = cv2.threshold(cur_frame_gray, 190, 255, cv2.THRESH_BINARY)

            prev_features = cv2.goodFeaturesToTrack(prev_frame_gray, mask=None, **feature_params)
            prev_features = np.array(prev_features, dtype='float32').reshape(-1,1,2)

            cur_features, status, err = cv2.calcOpticalFlowPyrLK(prev_frame_gray, cur_frame_gray, prev_features, None, **lk_params)

            prev_matched_features = []
            cur_matched_features = []
            for i, matched in enumerate(status):
                if matched:
                    prev_matched_features.append(prev_features[i])
                    cur_matched_features.append(cur_features[i])
            
            transform = cv2.estimateRigidTransform(np.array(prev_matched_features),
                                                    np.array(cur_matched_features),
                                                    False)
            
            if transform is not None:
                dx = transform[0,2]
                dy = transform[1,2]
                da = np.arctan2(transform[1,0], transform[0,0])
            else:
                dx = dy = da = 0
            

            prev_to_cur_transform.append([dx, dy, da])

            prev_frame_gray = cur_frame_gray[:]

            raw_transforms = np.array(prev_to_cur_transform)

            trajectory = np.cumsum(prev_to_cur_transform, axis=0)

            self.trajectory = pd.DataFrame(trajectory)

            smoothed_trajectory = self.trajectory.rolling(window=smoothing_window, center=False).mean().fillna(method='bfill')

            self.smoothed_trajectory = smoothed_trajectory.fillna(method='bfill')
            self.transforms = np.array(raw_transforms + (self.smoothed_trajectory -self.trajectory))
    
    def applyTransforms(self, input_path, output, output_fourcc='MJPG', border_type='black', border_size=0):

        transform = np.zeros((2,3))

        capture = cv2.VideoCapture(input_path)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(capture.get(cv2.CAP_PROP_FPS))

        writer = None

        if border_type not in ['black', 'reflect', 'replicate']:
            raise ValueError('Invalid border value')

        border_modes = {'black': cv2.BORDER_CONSTANT,
                        'reflect': cv2.BORDER_REFLECT,
                        'replicate': cv2.BORDER_REPLICATE}
        border_mode = border_modes[border_type]

        if writer is None:
                h,w = self.frameSize[:2]

                write_h = h + 2*border_size
                write_w = w + 2*border_size

                if border_size < 0:
                    neg_border_size = 100 + abs(border_size)
                    border_size = 100
                else:
                    neg_border_size = 0
                

                h += 2 * border_size
                w += 2 * border_size

                writer = cv2.VideoWriter('stab_vid2.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (write_w, write_h), True)

        #Loop through frame count
        for i in range(frame_count - 1):
            _gotFrame, frame = capture.read()
                #build transformation matrix
            transform[0,0] = np.cos(self.transforms[i][2])
            transform[0,1] = -np.sin(self.transforms[i][2])
            transform[1,0] = np.sin(self.transforms[i][2])
            transform[1,1] = np.cos(self.transforms[i][2])
            transform[0,2] = self.transforms[i][0]
            transform[1,2] = self.transforms[i][1]

                #apply the transform
            bordered_frame = cv2.copyMakeBorder(frame,
                                                 border_size*2,
                                                 border_size*2,
                                                 border_size*2,
                                                 border_size*2,
                                                 border_mode)
            transformed = cv2.warpAffine(bordered_frame,
                                            transform,
                                            (w + border_size * 2, h+border_size*2),
                                            borderMode=border_mode)
                
            buffer = border_size +neg_border_size
            transformed = transformed[buffer:(transformed.shape[0] - buffer),
                                          buffer:(transformed.shape[1]- buffer)]
            
            writer.write(transformed)
        capture.release()
        writer.release()
            