import os
import pickle

import cv2
import mediapipe as mp

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "hand_landmarker.task")
CLASSIFIER_PATH = os.path.join(SCRIPT_DIR, "model.pickle")

model_dict = pickle.load(open(CLASSIFIER_PATH, "rb"))
model = model_dict["model"]

labels_dict = {
     "A": "A",
     "B": "B",
     "C": "C",
     "D": "D",
     "E": "E",
     "F": "F",
     "G": "G",
     "I": "I",
     "K": "K",
     "L": "L",
     "M": "M",
     "N": "N",
     "O": "O",
     "P": "P",
     "Q": "Q",
     "R": "R",
     "S": "S",
     "T": "T",
     "U": "U",
}

# Open camera (default)
cap = cv2.VideoCapture(0)

### Objects ###
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Landmarks
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO)


with HandLandmarker.create_from_options(options) as landmarker:
    frame_timestamp_ms = 0

    while True:
        data_aux = []
        xx = []
        yy = []

        ret, frame = cap.read()

        H, W, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # convert from bgr to rgb
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb) # save into mp variable

        hand_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
        frame_timestamp_ms += 1

        if hand_landmarker_result.hand_landmarks:
            for hand_landmarks in hand_landmarker_result.hand_landmarks:
                for landmark in hand_landmarks:
                        xx.append(landmark.x)
                        yy.append(landmark.y)
                    
                for landmark in hand_landmarks: # second loop to avoid problems with min
                    data_aux.append(landmark.x - min(xx))
                    data_aux.append(landmark.y - min(yy))
            
            x1 = int(min(xx) * W) - 10
            y1 = int(min(yy) * H) - 10

            x2 = int(max(xx) * W) + 10
            y2 = int(max(yy) * H) + 10

            # Call trained classifier to predict letter
            prediction = model.predict([np.asarray(data_aux)])
            predicted_char = str(prediction[0])
            predicted_letter = labels_dict.get(predicted_char, predicted_char)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 4)
            cv2.putText(frame, predicted_letter, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 255), 3, cv2.LINE_AA)

        cv2.imshow("LSE detector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q') or key == 27:  # q, Q o ESC
            break





cap.release()
cv2.destroyAllWindows()