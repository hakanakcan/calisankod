import os
from ultralytics import YOLO
import cv2

BASE_DIR = 'C:/Users/Hakan/Desktop/Projetest'

VIDEOS_DIR = os.path.join(BASE_DIR, 'videos')
RUNS_DIR = os.path.join(BASE_DIR, 'runs')

video_path = os.path.join(VIDEOS_DIR, 'deneme1.mp4')
video_path_out = os.path.join(VIDEOS_DIR, 'deneme1test3.mp4')

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

model_path = os.path.join(RUNS_DIR, 'detect', 'yolov8colab', 'weights', 'last.pt')

# Load a model
model = YOLO(model_path)  # load a custom model

threshold = 0.5    # Bu, modelin yalnızca güven puanı 0.5'ten (yani %50 veya daha yüksek) yüksekse obje kutusunu çizip etiket ekleyeceği anlamına gelir.

renk_kodlari = {
    "construction-machine": (0, 255, 255) ,  # Sarı
    "rescue-team": (255, 255, 0),      # Turkuaz
    "collapsed": (0, 0, 255),        # Kırmızı
    "solid": (0, 255, 0),        # Yeşil
    "damaged": (0, 128, 255),      # Turuncu
    "tilted": (0, 64, 255),        # Koyu turuncu
}



while ret:
    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            sinif_adi = results.names[int(class_id)]
            renk = renk_kodlari[sinif_adi]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), renk, 4)
            cv2.putText(frame, sinif_adi.upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, renk, 3, cv2.LINE_AA)

    out.write(frame)
    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()