from ctypes import *
import os, sys
import numpy as np
import cv2

QDEEP = CDLL('./lib/qdeep/libQDEEP.so')
QDEEP_COLORSPACE_TYPE_BGR24 = c_ulong(1)
QDEEP_OBJECT_DETECT_CONFIG_MODEL_HUMAN_SKELETON_17_KEYPOINTS = c_ulong(26) 
QDEEP_OBJECT_DETECT_FLAG_TRAJECTORY_TRACKING = c_ulong(1) 

QDEEP_HUMAN_SKELETON_17KPS_NOSE = c_ulong(0)
QDEEP_HUMAN_SKELETON_17KPS_RIGHT_SHOULDER = c_ulong(1)
QDEEP_HUMAN_SKELETON_17KPS_RIGHT_ELBOW = c_ulong(2)
QDEEP_HUMAN_SKELETON_17KPS_RIGHT_WRIST = c_ulong(3)
QDEEP_HUMAN_SKELETON_17KPS_LEFT_SHOULDER = c_ulong(4)
QDEEP_HUMAN_SKELETON_17KPS_LEFT_ELBOW = c_ulong(5)
QDEEP_HUMAN_SKELETON_17KPS_LEFT_WRIST = c_ulong(6)
QDEEP_HUMAN_SKELETON_17KPS_RIGHT_HIP = c_ulong(7)
QDEEP_HUMAN_SKELETON_17KPS_RIGHT_KNEE = c_ulong(8)
QDEEP_HUMAN_SKELETON_17KPS_RIGHT_ANKLE = c_ulong(9)
QDEEP_HUMAN_SKELETON_17KPS_LEFT_HIP = c_ulong(10)
QDEEP_HUMAN_SKELETON_17KPS_LEFT_KNEE = c_ulong(11)
QDEEP_HUMAN_SKELETON_17KPS_LEFT_ANKLE = c_ulong(12)
QDEEP_HUMAN_SKELETON_17KPS_RIGHT_EYE = c_ulong(13)
QDEEP_HUMAN_SKELETON_17KPS_LEFT_EYE = c_ulong(14)
QDEEP_HUMAN_SKELETON_17KPS_RIGHT_EAR = c_ulong(15)
QDEEP_HUMAN_SKELETON_17KPS_LEFT_EAR = c_ulong(16)

class QCAP_AV_FRAME_T(Structure):
	_fields_ = [("pData", c_ulong*8),
				("nPitch", c_int*8),
				("pPrivateData0", c_ulong),
				("nWidth", c_int),
				("nHeight", c_int),
				("nSamples", c_int),
				("nFormat", c_int)]
	
class QDEEP_OBJECT_DETECT_KEYPOINT(Structure):
	_fields_ = [("nX", c_ulong),
				("nY", c_ulong),
				("nZ", c_ulong),
				("fProbability", c_float)]

class QDEEP_OBJECT_DETECT_BOUNDING_BOX(Structure):
    _fields_ = [("nClassID", c_ulong),
				("nSubClassIDs", c_ulong*8),
				("nObjectID", c_ulong),
				("nX", c_ulong),
				("nY", c_ulong),
				("nWidth", c_ulong),
				("nHeight", c_ulong),
				("fProbability", c_float),
				("fSubProbabilities", c_float*8),
				("fExtraAttributes", c_float*8),
				("nTrajectoryXs", c_ulong*128),
				("nTrajectoryYs", c_ulong*128),
				("sKeypoints", QDEEP_OBJECT_DETECT_KEYPOINT*256),
				("fBehaviorVectors", c_float*32),
				("nBehaviorID", c_ulong),
				("fFeatureVectors", c_float*1024),
                ("nTimeStamp", c_ulonglong),
                ("nDuration", c_ulonglong),
				("pImageResultBuffer", c_void_p)]

class QuickReceiver():
    def __init__(self):

        ### AI ###
        strModelName = './model/Human_Skeleton_17_Keypoints/QDEEP.OD.HUMAN.SKELETON.17KPS.CFG'
        QDEEP.QDEEP_CREATE_OBJECT_DETECT( 0x00000001, 0 , QDEEP_OBJECT_DETECT_CONFIG_MODEL_HUMAN_SKELETON_17_KEYPOINTS, strModelName.encode('utf-8'), byref(self.m_detector), QDEEP_OBJECT_DETECT_FLAG_TRAJECTORY_TRACKING, None)
        QDEEP.QDEEP_START_OBJECT_DETECT(self.m_detector)
        self.m_bStartDetector = 1
        self.m_nWidth = 1920
        self.m_nHeight = 1080
         # Initialize video captur
        video_type = input("Enter video file path or camera (e.g., 0 for video, 1 for webcam): ")

        try:
            video_type = int(video_type)
        except ValueError:
            print("Invalid input. Defaulting to webcam.")
            video_type = 1  # 預設使用攝影機

        if video_type == 0:
            video_path = "./model/Human_Skeleton_17_Keypoints/people-counting.mp4"
             # 確保檔案存在
            if not os.path.exists(video_path):
                print(f"❌ Error: Video file not found at {video_path}")
                exit()
            self.cap = cv2.VideoCapture(video_path)
        else:
            try:
                video_input = int(input("Enter camera ID (e.g., 0 for webcam): "))
            except ValueError:
                print("Invalid camera ID. Defaulting to 0 (webcam).")
                video_input = 0  # 預設使用 0 號攝影機

            self.cap = cv2.VideoCapture(video_input)

        if not self.cap.isOpened():
            print("❌ Unable to open the video source. Check file path or camera connection.")
            exit()

        print("✅ Video source opened successfully!")

        # 設定解析度為 1920x1080
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        # 確認是否成功設定
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Video Resolution: {actual_width}x{actual_height}")

        if actual_width != 1920 or actual_height != 1080:
            print("Warning: Could not set resolution to 1920x1080. Camera may not support it.")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Fail to read")
                break

            ### AI analysis
            self.nObjectSize = c_ulong(1000)
            bufferlen = self.m_nWidth * self.m_nHeight * 3

            image_np = np.array(frame)
            pImageBuffer = cast(image_np.ctypes.data, c_void_p)

            QDEEP.QDEEP_SET_VIDEO_OBJECT_DETECT_UNCOMPRESSION_BUFFER(self.m_detector, QDEEP_COLORSPACE_TYPE_BGR24, c_ulong(self.m_nWidth), c_ulong(self.m_nHeight), pImageBuffer, c_ulong(bufferlen), self.m_pObjectList, byref(self.nObjectSize), 1)
            # print("self.nObjectSize is ", self.nObjectSize)

            for i in range(self.nObjectSize.value):
                for keypoint in range(17):
                    keypoint_x = self.m_pObjectList[i].sKeypoints[keypoint].nX
                    keypoint_y = self.m_pObjectList[i].sKeypoints[keypoint].nY

                    if keypoint_x == 0 and keypoint_y == 0:
                        continue

                    point = (int(keypoint_x),int(keypoint_y))
                    cv2.circle(frame, point, 5, (0, 255, 0), -1)  

                    if keypoint == QDEEP_HUMAN_SKELETON_17KPS_NOSE.value:
                        # NOSE(0) -> RIGHT_EYE(13)
                        right_eye_x = self.m_pObjectList[i].sKeypoints[QDEEP_HUMAN_SKELETON_17KPS_RIGHT_EYE.value].nX
                        right_eye_y = self.m_pObjectList[i].sKeypoints[QDEEP_HUMAN_SKELETON_17KPS_RIGHT_EYE.value].nY

                        if right_eye_x != 0 and right_eye_y != 0:
                            start_point = point
                            end_point = (int(right_eye_x),int(right_eye_y))
                            cv2.line(frame, start_point, end_point, (220, 106, 163), 3) 

                        # NOSE(0) -> LEFT_EYE(14)
                        left_eye_x = self.m_pObjectList[i].sKeypoints[QDEEP_HUMAN_SKELETON_17KPS_LEFT_EYE.value].nX
                        left_eye_y = self.m_pObjectList[i].sKeypoints[QDEEP_HUMAN_SKELETON_17KPS_LEFT_EYE.value].nY

                        if left_eye_x != 0 and left_eye_y != 0:
                            start_point = point
                            end_point = (int(left_eye_x),int(left_eye_y))
                            cv2.line(frame, start_point, end_point, (220, 106, 163), 3)

                    elif keypoint == QDEEP_HUMAN_SKELETON_17KPS_RIGHT_SHOULDER.value:
                        # RIGHT_SHOULDER(1)  -> RIGHT_ELBOW(2)
                        right_elbow_x = self.m_pObjectList[i].sKeypoints[QDEEP_HUMAN_SKELETON_17KPS_RIGHT_ELBOW.value].nX
                        right_elbow_y = self.m_pObjectList[i].sKeypoints[QDEEP_HUMAN_SKELETON_17KPS_RIGHT_ELBOW.value].nY

                        if right_elbow_x != 0 and right_elbow_y != 0:
                            start_point = point
                            end_point = (int(right_elbow_x),int(right_elbow_y))
                            cv2.line(frame, start_point, end_point, (220, 106, 163), 3) 

                        # RIGHT_SHOULDER(1) -> RIGHT_HIP(7) 
                        right_hip_x = self.m_pObjectList[i].sKeypoints[QDEEP_HUMAN_SKELETON_17KPS_RIGHT_HIP.value].nX
                        right_hip_y = self.m_pObjectList[i].sKeypoints[QDEEP_HUMAN_SKELETON_17KPS_RIGHT_HIP.value].nY

                        if right_hip_x != 0 and right_hip_y != 0:
                            start_point = point
                            end_point = (int(right_hip_x),int(right_hip_y))
                            cv2.line(frame, start_point, end_point, (220, 106, 163), 3)
                    
                    elif keypoint == QDEEP_HUMAN_SKELETON_17KPS_RIGHT_ELBOW.value:
                        # RIGHT_ELBOW(2) -> RIGHT_WRIST(3)
                        right_wrist_x = self.m_pObjectList[i].sKeypoints[QDEEP_HUMAN_SKELETON_17KPS_RIGHT_WRIST.value].nX
                        right_wrist_y = self.m_pObjectList[i].sKeypoints[QDEEP_HUMAN_SKELETON_17KPS_RIGHT_WRIST.value].nY

                        if right_wrist_x != 0 and right_wrist_y != 0:
                            start_point = point
                            end_point = (int(right_wrist_x),int(right_wrist_y))
                            cv2.line(frame, start_point, end_point, (220, 106, 163), 3) 

                    elif keypoint == QDEEP_HUMAN_SKELETON_17KPS_LEFT_SHOULDER.value:
                        # LEFT_SHOULDER(4) -> RIGHT_SHOULDER(1)
                        right_shoulder_x = self.m_pObjectList[i].sKeypoints[QDEEP_HUMAN_SKELETON_17KPS_RIGHT_SHOULDER.value].nX
                        right_shoulder_y = self.m_pObjectList[i].sKeypoints[QDEEP_HUMAN_SKELETON_17KPS_RIGHT_SHOULDER.value].nY

                        if right_shoulder_x != 0 and right_shoulder_y != 0:
                            start_point = point
                            end_point = (int(right_shoulder_x),int(right_shoulder_y))
                            cv2.line(frame, start_point, end_point, (220, 106, 163), 3) 

                        # LEFT_SHOULDER(4) -> LEFT_ELBOW(5)
                        left_elbow_x = self.m_pObjectList[i].sKeypoints[QDEEP_HUMAN_SKELETON_17KPS_LEFT_ELBOW.value].nX
                        left_elbow_y = self.m_pObjectList[i].sKeypoints[QDEEP_HUMAN_SKELETON_17KPS_LEFT_ELBOW.value].nY

                        if left_elbow_x != 0 and left_elbow_y != 0:
                            start_point = point
                            end_point = (int(left_elbow_x),int(left_elbow_y))
                            cv2.line(frame, start_point, end_point, (220, 106, 163), 3)

                        # LEFT_SHOULDER(4) -> LEFT_HIP(10) 
                        left_hip_x = self.m_pObjectList[i].sKeypoints[QDEEP_HUMAN_SKELETON_17KPS_LEFT_HIP.value].nX
                        left_hip_y = self.m_pObjectList[i].sKeypoints[QDEEP_HUMAN_SKELETON_17KPS_LEFT_HIP.value].nY

                        if left_hip_x != 0 and left_hip_y != 0:
                            start_point = point
                            end_point = (int(left_hip_x),int(left_hip_y))
                            cv2.line(frame, start_point, end_point, (220, 106, 163), 3)

                    elif keypoint == QDEEP_HUMAN_SKELETON_17KPS_LEFT_ELBOW.value:
                        # LEFT_ELBOW(5) -> LEFT_WRIST(6)
                        left_wrist_x = self.m_pObjectList[i].sKeypoints[QDEEP_HUMAN_SKELETON_17KPS_LEFT_WRIST.value].nX
                        left_wrist_y = self.m_pObjectList[i].sKeypoints[QDEEP_HUMAN_SKELETON_17KPS_LEFT_WRIST.value].nY

                        if left_wrist_x != 0 and left_wrist_y != 0:
                            start_point = point
                            end_point = (int(left_wrist_x),int(left_wrist_y))
                            cv2.line(frame, start_point, end_point, (220, 106, 163), 3) 

                    elif keypoint == QDEEP_HUMAN_SKELETON_17KPS_RIGHT_HIP.value:
                        # RIGHT_HIP(7) -> LEFT_HIP(10)
                        left_hip_x = self.m_pObjectList[i].sKeypoints[QDEEP_HUMAN_SKELETON_17KPS_LEFT_HIP.value].nX
                        left_hip_y = self.m_pObjectList[i].sKeypoints[QDEEP_HUMAN_SKELETON_17KPS_LEFT_HIP.value].nY

                        if left_hip_x != 0 and left_hip_y != 0:
                            start_point = point
                            end_point = (int(left_hip_x),int(left_hip_y))
                            cv2.line(frame, start_point, end_point, (220, 106, 163), 3) 

                        # RIGHT_HIP(7) -> RIGHT_KNEE(8)
                        right_knee_x = self.m_pObjectList[i].sKeypoints[QDEEP_HUMAN_SKELETON_17KPS_RIGHT_KNEE.value].nX
                        right_knee_y = self.m_pObjectList[i].sKeypoints[QDEEP_HUMAN_SKELETON_17KPS_RIGHT_KNEE.value].nY

                        if right_knee_x != 0 and right_knee_y != 0:
                            start_point = point
                            end_point = (int(right_knee_x),int(right_knee_y))
                            cv2.line(frame, start_point, end_point, (220, 106, 163), 3)
                    
                    elif keypoint == QDEEP_HUMAN_SKELETON_17KPS_RIGHT_KNEE.value:
                        # RIGHT_KNEE(8) -> RIGHT_ANKLE(9) 
                        right_ankle_x = self.m_pObjectList[i].sKeypoints[QDEEP_HUMAN_SKELETON_17KPS_RIGHT_ANKLE.value].nX
                        right_ankle_y = self.m_pObjectList[i].sKeypoints[QDEEP_HUMAN_SKELETON_17KPS_RIGHT_ANKLE.value].nY

                        if right_ankle_x != 0 and right_ankle_y != 0:
                            start_point = point
                            end_point = (int(right_ankle_x),int(right_ankle_y))
                            cv2.line(frame, start_point, end_point, (220, 106, 163), 3) 

                    elif keypoint == QDEEP_HUMAN_SKELETON_17KPS_LEFT_HIP.value:
                        # LEFT_HIP(10) -> LEFT_KNEE(11)
                        left_knee_x = self.m_pObjectList[i].sKeypoints[QDEEP_HUMAN_SKELETON_17KPS_LEFT_KNEE.value].nX
                        left_knee_y = self.m_pObjectList[i].sKeypoints[QDEEP_HUMAN_SKELETON_17KPS_LEFT_KNEE.value].nY

                        if left_knee_x != 0 and left_knee_y != 0:
                            start_point = point
                            end_point = (int(left_knee_x),int(left_knee_y))
                            cv2.line(frame, start_point, end_point, (220, 106, 163), 3)
                    
                    elif keypoint == QDEEP_HUMAN_SKELETON_17KPS_LEFT_KNEE.value:
                        # LEFT_KNEE(11) -> LEFT_ANKLE(12)
                        left_ankle_x = self.m_pObjectList[i].sKeypoints[QDEEP_HUMAN_SKELETON_17KPS_LEFT_ANKLE.value].nX
                        left_ankle_y = self.m_pObjectList[i].sKeypoints[QDEEP_HUMAN_SKELETON_17KPS_LEFT_ANKLE.value].nY

                        if left_ankle_x != 0 and left_ankle_y != 0:
                            start_point = point
                            end_point = (int(left_ankle_x),int(left_ankle_y))
                            cv2.line(frame, start_point, end_point, (220, 106, 163), 3)
                            
                    elif keypoint == QDEEP_HUMAN_SKELETON_17KPS_RIGHT_EYE.value:
                        # RIGHT_EYE(13) -> RIGHT_EAR(15)
                        right_ear_x = self.m_pObjectList[i].sKeypoints[QDEEP_HUMAN_SKELETON_17KPS_RIGHT_EAR.value].nX
                        right_ear_y = self.m_pObjectList[i].sKeypoints[QDEEP_HUMAN_SKELETON_17KPS_RIGHT_EAR.value].nY

                        if right_ear_x != 0 and right_ear_y != 0:
                            start_point = point
                            end_point = (int(right_ear_x),int(right_ear_y))
                            cv2.line(frame, start_point, end_point, (220, 106, 163), 3) 

                    elif keypoint == QDEEP_HUMAN_SKELETON_17KPS_LEFT_EYE.value:
                        # LEFT_EYE(14) -> LEFT_EAR(16) 
                        left_ear_x = self.m_pObjectList[i].sKeypoints[QDEEP_HUMAN_SKELETON_17KPS_LEFT_EAR.value].nX
                        left_ear_y = self.m_pObjectList[i].sKeypoints[QDEEP_HUMAN_SKELETON_17KPS_LEFT_EAR.value].nY

                        if left_ear_x != 0 and left_ear_y != 0:
                            start_point = point
                            end_point = (int(left_ear_x),int(left_ear_y))
                            cv2.line(frame, start_point, end_point, (220, 106, 163), 3) 

            cv2.imshow('DEMO', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.close()        
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def close(self):
        if self.m_bStartDetector == 1:
            self.m_bStartDetector = 0
            QDEEP.QDEEP_STOP_OBJECT_DETECT(self.m_detector)
            QDEEP.QDEEP_DESTROY_OBJECT_DETECT(self.m_detector)
            self.m_detector = c_void_p(0)            

    def __del__(self):
        self.close()

    m_detector = c_void_p(0)
    m_nWidth = c_ulong(0)
    m_nHeight = c_ulong(0)
    m_nTrajectoryX = (c_ulong * 128)()
    m_nTrajectoryY = (c_ulong * 128)()
    m_bStartDetector = 0
    nObjectSize = 1000
    PointerArrayType = QDEEP_OBJECT_DETECT_BOUNDING_BOX * nObjectSize
    m_pObjectList = PointerArrayType()

if  __name__ == '__main__':

    quick_receiver = QuickReceiver()

    quick_receiver.close()
