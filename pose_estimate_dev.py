import os
import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial.transform import Rotation as R
# import dev.followme_golf as fg
# from google.cloud import storage
# storage_client = storage.Client()
debug_mode = False
RIGHT_TOE_INDEX = 32
LEFT_TOE_INDEX = 31
RIGHT_HEEL_INDEX = 30
LEFT_HEEL_INDEX = 29
RIGHT_ANKLE_INDEX = 28
LEFT_ANKLE_INDEX=27
RIGHT_KNEE_INDEX = 26
LEFT_KNEE_INDEX = 25
RIGHT_HIP_INDEX = 24
LEFT_HIP_INDEX = 23
RIGHT_ELBOW_INDEX = 14
LEFT_ELBOW_INDEX = 13
RIGHT_SHOULDER_INDEX = 12
LEFT_SHOULDER_INDEX = 11

NOSE_INDEX = 0

def landmark_to_vector(landmark):
    return np.array([landmark.x, landmark.y, landmark.z])

def get_frame_reference(frame, results):
    frame_dimensions = np.array([frame.shape[1], frame.shape[0]])

    sum_torso_locs = np.zeros(3)
    for body_loc in [RIGHT_HIP_INDEX, LEFT_HIP_INDEX]:#, RIGHT_SHOULDER_INDEX, LEFT_SHOULDER_INDEX]:
        sum_torso_locs += landmark_to_vector(results.pose_landmarks.landmark[body_loc])
    avg_torso_loc = sum_torso_locs/2

    height = max(
        results.pose_landmarks.landmark[LEFT_TOE_INDEX].y,
        results.pose_landmarks.landmark[RIGHT_ANKLE_INDEX].y
        ) - results.pose_landmarks.landmark[NOSE_INDEX].y

    bottom = min(int(frame_dimensions[0] * (avg_torso_loc[1] + height*1.5)), frame_dimensions[0])
    top = max(int(frame_dimensions[0] * (avg_torso_loc[1] - height*1.5)), 0)
    right = min(int(frame_dimensions[1] * (avg_torso_loc[0] + height*1.5)), frame_dimensions[1])
    left = max(int(frame_dimensions[1] * (avg_torso_loc[0] - height*1.5)), 0)

    return top, bottom, left, right

def center_square_around_centroid(frame, results, margin=0.5):
    frame_width, frame_height = frame.shape[1], frame.shape[0]

    points = []
    for mark in results.pose_landmarks.landmark:
        points.append([mark.x * frame_width, mark.y * frame_height])


    # Calculate centroid of all points
    centroid_x, centroid_y = np.mean(points, axis=0)

    # Determine the largest square size needed to encompass all points
    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)
    square_size = max(max_x - min_x, max_y - min_y)

    # Add margin to the square size
    margin_size = square_size * margin
    square_size += 2 * margin_size  # Add margin to both sides

    # Attempt to center the square around the centroid, adjusting for the added margin
    left = centroid_x - square_size / 2
    top = centroid_y - square_size / 2

    # Adjust for frame boundaries
    if left < 0:
        left = 0
    elif left + square_size > frame_width:
        left = frame_width - square_size

    if top < 0:
        top = 0
    elif top + square_size > frame_height:
        top = frame_height - square_size

    # Ensure the square fits within the frame, adjusting for available space
    if square_size > frame_width:
        left = 0
        square_size = frame_width
        top = max(min(centroid_y - square_size / 2, frame_height - square_size), 0)

    if square_size > frame_height:
        top = 0
        square_size = frame_height
        left = max(min(centroid_x - square_size / 2, frame_width - square_size), 0)

    return int(top), int(top + square_size), int(left), int(left + square_size)


def stack_landmarks(results):
    arr_landmark = np.empty((0, 3))
    for landmark in results.pose_landmarks.landmark:
        arr_landmark = np.vstack((arr_landmark, landmark_to_vector(landmark)))

    return arr_landmark


def show_landmarks(landmark_obj, frame):
    mp.solutions.drawing_utils.draw_landmarks(
        frame,
        landmark_obj,
        mp.solutions.pose.POSE_CONNECTIONS, 
        landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style())
    
    return frame


def normalize_vector(vector, **kwarg):
    return vector/np.linalg.norm(vector,  **kwarg)

def get_rot_mat(original_x, original_y, new_x=np.array([1, 0, 0]), new_y=np.array([0, 1, 0])):
    '''
    generate rotational matrix using the current(real world) reference coordinates
     and video's reference coordinates
    '''
    original_x = normalize_vector(original_x)
    original_y = normalize_vector(original_y)

    new_x = normalize_vector(new_x)
    new_y = normalize_vector(new_y)

    original_z = normalize_vector(np.cross(original_x, original_y))
    new_z = normalize_vector(np.cross(new_x, new_y))

    original_coord = np.column_stack((original_x, original_y, original_z))
    new_coord = np.column_stack((new_x, new_y, new_z))
    R = new_coord @ np.linalg.inv(original_coord)

    if debug_mode:
        print('orthogoanl z:', original_z, new_z)
        print('check orthogonality org:', np.dot(original_z, original_x), np.dot(original_y, original_z))
        print('check orthogonality new:', np.dot(new_z, new_x), np.dot(new_y, new_z))
        print(f'check the rotational matrix converted v_x: {R @ original_x}, v_y: {R @ original_y}')

    return R


def get_original_coord(frame_landmarks, ref_tops, ref_bottoms, vert_ref):
    '''
    calculate the frame landmarks reference coordinates
    '''
    left_bottom, right_bottom = frame_landmarks[ref_bottoms]

    mid_top = frame_landmarks[ref_tops].sum(axis=0)/2
    mid_bottom = frame_landmarks[ref_bottoms].sum(axis=0)/2

    v_vertical_ref = mid_bottom - mid_top
    v_horizontal_ref = left_bottom - right_bottom

    v_ortho_z = np.cross(v_horizontal_ref, v_vertical_ref)
    if vert_ref:
        original_x = np.cross(v_vertical_ref, v_ortho_z)
        original_y = v_vertical_ref
    else:
        original_x = v_horizontal_ref    
        original_y = np.cross(v_ortho_z, v_horizontal_ref)
    if debug_mode:
        print('original x and y:', original_x, original_y)

    return original_x, original_y


def rotate_translate_coordinates(
        frame_landmarks, 
        ref_tops=[LEFT_SHOULDER_INDEX, RIGHT_SHOULDER_INDEX],
        ref_bottoms=[LEFT_HIP_INDEX, RIGHT_HIP_INDEX],
        vert_ref=False
        ):
    '''
    rotate and translate the given coordinate and reference points.
    consist of 
        getting current x and y vectors to align to
        getting rotational matrix to align to standard x and y vectors 
    '''
    original_x, original_y = get_original_coord(frame_landmarks, ref_tops, ref_bottoms, vert_ref)
    R = get_rot_mat(original_x, original_y)

    rotated_landmarks = frame_landmarks @ R.T
    rotated_mid_bottom = rotated_landmarks[ref_bottoms].sum(axis=0)/2
    rotated_translate = np.array([1/2, 3/4, 0]) - rotated_mid_bottom
    rotated_translated_landmarks = rotated_landmarks + rotated_translate


    if debug_mode:
        rotated_left_bottom, rotated_right_bottom = rotated_landmarks[ref_bottoms]
        rotated_mid_top = rotated_landmarks[ref_tops].sum(axis=0)/2
        rotated_v_vertical_ref = rotated_mid_bottom - rotated_mid_top
        rotated_v_horizontal_ref = rotated_left_bottom - rotated_right_bottom
        print(f'check final rotated v_x: {rotated_v_horizontal_ref}, v_y: {rotated_v_vertical_ref}')

    return rotated_translated_landmarks, R

class VideoParamer():
    def __init__(self, filename):
        self.filename = os.path.expanduser(filename)
        self.crop_param = []
        self.results_collection = []

    def _video_process_starter(self):
        '''
        start the necessary objects for processing the video analysis
        '''
        self.cap = cv2.VideoCapture(self.filename)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open {self.filename}")
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=True,
            min_detection_confidence=0.7,
            model_complexity=2
            )

    def _video_process_ender(self):
        '''
        end the necessary objects used for processing the video analysis
        '''
        self.cap.release()
        self.pose.close()
        
    def get_crop_param(self):
        ''' 
        get the four cropping reference points of the frame
        that contains all the landmarks
        '''
        self._video_process_starter()
        no_landmark = True # landmark detection flag: no landmark calculated -> True
        # Until the landmark can be calculated, loop through the video.
        while no_landmark:
            ret, frame = self.cap.read()
            if not ret:
                raise FileExistsError ('Video frames are not available.')
            results = self.pose.process(frame)
            no_landmark = results.pose_landmarks is None
            
            # calculate the boundary window reference points
            if not no_landmark:            

                top, bottom, left, right = center_square_around_centroid(frame, results)
                self.crop_param = [top, bottom, left, right]
                self._video_process_ender()
                if debug_mode:
                    frame = show_landmarks(results.pose_landmarks, frame)
                    cv2.imwrite('cropping_image.jpg', frame[top:bottom, left:right])
                
                return self.crop_param
            
        self._video_process_ender()
        raise ValueError(f"Landmark for '{self.filename}' not found.")
 
    def get_landmark(self):
        '''
        get landmark value on video that in 10 fps
        '''
        self._video_process_starter()
        if len(self.crop_param):
            top, bottom, left, right = self.crop_param
            frame_width = int(512 * (right - left)/(bottom - top))
        else:
            frame_width = int(
                float(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) * 512 
                / int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                )

        fps_down = 10
        fps_og = self.cap.get(cv2.CAP_PROP_FPS)
        sample_freq = max(1, fps_og//fps_down)

        if debug_mode:
            video_filename = f'{os.path.splitext(self.filename)[0]}_annotated.avi'
            processed_video = cv2.VideoWriter(
                    video_filename, 
                    cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                    fps_down,
                    (frame_width, 512)
                )
            
        frame_count = -1
        #TODO: Parallelize
        while True:
            frame_count += 1
            ret, frame = self.cap.read()
            if not ret:
                break

            # ignore all the frames between each sampling period
            if frame_count % sample_freq:
                continue

            if len(self.crop_param):  # crop if crop window references are available
                frame = frame[top:bottom, left:right]
            # estimate the pose
            results_cropped = self.pose.process(frame)
            if results_cropped.pose_landmarks is None:  # if not available in the frame move to the next frame
                continue
            self.results_collection.append(stack_landmarks(results_cropped))
                
            if debug_mode:
                frame = show_landmarks(results_cropped.pose_landmarks, frame)
                cv2.putText(frame, str(frame_count), (frame_width//2 , 500), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
                frame = cv2.resize(frame, (frame_width, 512))
                processed_video.write(frame)

        self._video_process_ender()
        if debug_mode:
            processed_video.release()

    def standardize_landmark(
            self,
            ref_tops=[LEFT_SHOULDER_INDEX, RIGHT_SHOULDER_INDEX],
            ref_bottoms=[LEFT_HIP_INDEX, RIGHT_HIP_INDEX],
            vert_ref=True
            ):
        '''
        standardize the landmark by having the estimated pose facing out of the screen 
        with different references defined in the arguments
        '''
        self.transform_mats = []
        self.transformed_landmarks = []
        
        for frame_landmarks in self.results_collection:
        # Left of the frame has lower x value, Right of the frame has higher x value
        # Top of the frame has lower y value, Bottom of the frame has higher y value
        # Closer to the camera has lower z value, Further to the camera has higher z value
        
            rotated_translated_landmarks, R = rotate_translate_coordinates(
                frame_landmarks,
                ref_tops=ref_tops,
                ref_bottoms=ref_bottoms,
                vert_ref=vert_ref
            )

            self.transform_mats.append(R)
            self.transformed_landmarks.append(rotated_translated_landmarks)

        return self.transform_mats, self.transformed_landmarks

