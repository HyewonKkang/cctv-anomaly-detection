import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

mp_pose = mp.solutions.pose

IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture('video2.mp4')
cap.set(cv2.CAP_PROP_FPS, 30)

def plot_line(a, b):
    if (a[0]> 0 and a[1] > 0 and b[0] > 0 and b[1]>0):
      plt.plot([1 - a[0], 1 - b[0]], [1 - a[1], 1 - b[1]], 'k-')

def draw_plot(landmark_pos):
    # draw detected skeleton on the plot
    plt.subplot(122)
    plt.title('skeleton')

    # 각 관절 점 및 텍스트 표시
    for kp in keypoints:
      plt.plot(landmark_pos[kp][0], landmark_pos[kp][1], 'ro')
      plt.text(landmark_pos[kp][0], landmark_pos[kp][1], kp, verticalalignment='bottom' , horizontalalignment='center' )

    # 신체에 맞게 각 관절 선으로 연결
    plot_line(landmark_pos['Nose'], landmark_pos['Neck'])
    plot_line(landmark_pos['Neck'], landmark_pos['LShoulder'])
    plot_line(landmark_pos['Neck'], landmark_pos['RShoulder'])
    plot_line(landmark_pos['LShoulder'], landmark_pos['LElbow'])
    plot_line(landmark_pos['LElbow'], landmark_pos['LWrist'])
    plot_line(landmark_pos['RShoulder'], landmark_pos['RElbow'])
    plot_line(landmark_pos['RElbow'], landmark_pos['RWrist'])
    plot_line(landmark_pos['LHip'], landmark_pos['LKnee'])
    plot_line(landmark_pos['LKnee'], landmark_pos['LAnkle'])
    plot_line(landmark_pos['RKnee'], landmark_pos['RAnkle'])
    plot_line(landmark_pos['RHip'], landmark_pos['RKnee'])
    plot_line(landmark_pos['LHip'], landmark_pos['LShoulder'])
    plot_line(landmark_pos['RHip'], landmark_pos['RShoulder'])
    plot_line(landmark_pos['RHip'], landmark_pos['LHip'])
    plot_line(landmark_pos['LShoulder'], landmark_pos['RShoulder'])

    # show the final output
    plt.show()

def get_keypoints(RGB):
    results = pose.process(RGB)
    if results.pose_landmarks == None:
      return {}

    landmarks = results.pose_landmarks.landmark

    landmark_pos = {
      'Nose': (landmarks[mp_pose.PoseLandmark.NOSE].x, landmarks[mp_pose.PoseLandmark.NOSE].y),
      'RShoulder' : (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y),
      'RElbow' : (landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y),
      'RWrist' : (landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y),
      'LShoulder' : (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y),
      'LElbow' : (landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y),
      'LWrist' : (landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y),
      'RHip' : (landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y),
      'RKnee' : (landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y),
      'RAnkle' : (landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y),
      'LHip' : (landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y),
      'LKnee' : (landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y),
      'LAnkle' : (landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y)
    }

    # calculate Neck
    LShoulder = landmark_pos['LShoulder']
    RShoulder = landmark_pos['RShoulder']
    if LShoulder[0] == 0 and LShoulder[1] == 0:
      landmark_pos['Neck'] = (RShoulder[0], RShoulder[1])
    elif RShoulder[0] == 0 and RShoulder[1] == 0:
      landmark_pos['Neck'] = (LShoulder[0], LShoulder[1])
    else:
      landmark_pos['Neck'] = ((RShoulder[0] + LShoulder[0]) / 2, (RShoulder[1] + LShoulder[1]) / 2)

    # calculate MidHip
    LHip = landmark_pos['LHip']
    RHip = landmark_pos['RHip']
    if LHip[0] == 0 and LHip[1] == 0:
      landmark_pos['MidHip'] = (RHip[0], RHip[1])
    elif RHip[0] == 0 and RHip[1] == 0:
      landmark_pos['MidHip'] = (LHip[0], LHip[1])
    else:
      landmark_pos['MidHip'] = ((RHip[0] + LHip[0]) / 2, (RHip[1] + LHip[1]) / 2)

    return landmark_pos

def keypoint_detector(RGB):
   landmark_pos = get_keypoints(RGB)
   if not landmark_pos:
    return []
   # draw_plot(landmark_pos)

   keypoints_index = ['Nose', 'Neck', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow', 'LWrist', 'MidHip',
        'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle']

   result = []
   for idx in keypoints_index:
      result.append([landmark_pos[idx][0], landmark_pos[idx][1]])

   return result



