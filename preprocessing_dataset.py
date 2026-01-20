import os # para trabajar con archivos y carpetas
import pickle

import mediapipe as mp


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # tomar la ruta de este archivo
DATA_DIR = os.path.join(SCRIPT_DIR, "data") # y se la aplica a data, así usamos path completo

MODEL_PATH = os.path.join(SCRIPT_DIR, "hand_landmarker.task")

data = []
labels = []

### Objects ###
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


# Create landmarker
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE)

with HandLandmarker.create_from_options(options) as landmarker:

    for dir_ in os.listdir(DATA_DIR):
        for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
            
            data_aux = []

            xx = []
            yy = []

            img = os.path.join(DATA_DIR, dir_, img_path)
            mp_image = mp.Image.create_from_file(img)

            hand_landmarker_result = landmarker.detect(mp_image)

            if hand_landmarker_result.hand_landmarks:
                for hand_landmarks in hand_landmarker_result.hand_landmarks:
                    for landmark in hand_landmarks:
                        xx.append(landmark.x)
                        yy.append(landmark.y)
                    
                    for landmark in hand_landmarks: # second loop to avoid problems with min
                        data_aux.append(landmark.x - min(xx))
                        data_aux.append(landmark.y - min(yy))

                data.append(data_aux)
                labels.append(dir_)

# Save de dataset
f = open("data.pickle", "wb")
pickle.dump({"data": data, "labels": labels}, f)
f.close()