from ultralytics import YOLO
import cv2
import math
import mediapipe as mp

# Inisialisasi MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Mulai kamera web
cap = cv2.VideoCapture(0)
cap.set(3, 1920)  # Mengatur lebar gambar menjadi 1920 piksel
cap.set(4, 1080)  # Mengatur tinggi gambar menjadi 1080 piksel

focal_length_pixel = 481  # Nilai kalibrasi dari hasil hitung focal length
tinggi_objek_nyata = 181

# Memuat model YOLO
model = YOLO("best.pt")

# Fungsi untuk menghitung jarak
def hitung_jarak(tinggi_bounding_box, focal_length_pixel, tinggi_objek_nyata):
    if tinggi_bounding_box == 0:
        return float('inf')  # Menghindari pembagian dengan nol
    jarak = (tinggi_objek_nyata * focal_length_pixel) / tinggi_bounding_box
    return jarak

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            cls = int(box.cls[0])  # Mengonversi kelas objek terdeteksi menjadi integer
            if cls == 0:  # Memeriksa apakah objek terdeteksi adalah 'person'
                x1, y1, x2, y2 = box.xyxy[0]  # Mengambil koordinat kotak pembatas objek
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  

                # Crop gambar untuk analisis pose dengan MediaPipe
                person_img = img[y1:y2, x1:x2]
                person_img_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
                pose_results = pose.process(person_img_rgb)

                if pose_results.pose_landmarks:
                    # Tarik pose pada gambar yang di-crop
                    mp_drawing.draw_landmarks(person_img, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    
                    # Temukan dan tandai titik paling atas
                    landmarks = pose_results.pose_landmarks.landmark
                    top_landmark = min(landmarks, key=lambda landmark: landmark.y)
                    top_x = int(top_landmark.x * person_img.shape[1])
                    top_y = int(top_landmark.y * person_img.shape[0])
                    global_top_x = x1 + top_x
                    global_top_y = y1 + top_y
                    cv2.circle(img, (global_top_x, global_top_y), 5, (255, 255, 0), -1)

                    # Tampilkan koordinat titik paling atas pada pose
                    teks_koordinat = f"X: {global_top_x}, Y: {global_top_y}"
                    cv2.putText(img, teks_koordinat, (global_top_x + 10, global_top_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    img[y1:y2, x1:x2] = person_img  # Tempelkan kembali ke gambar asli

                confidence = math.ceil((box.conf[0] * 100)) / 100
                if confidence > 0.5:
                    tinggi_bounding_box = y2 - y1
                    jarak = hitung_jarak(tinggi_bounding_box, focal_length_pixel, tinggi_objek_nyata)

                    if jarak < 130 or (x2 - x1) > 310:
                        label = f"Distance(Agung): {jarak:.2f} cm"
                        warna_teks = (0, 0, 255)
                        pesan = "STOP OBJEK DEKAT"
                        cv2.putText(img,pesan,(600, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
                    else:
                        label = f"Distance(Agung): {jarak:.2f} cm"
                        warna_teks = (0, 255, 0)

                    cv2.rectangle(img, (x1, y1), (x2, y2), warna_teks, 3)
                    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, warna_teks, 2)

    cv2.imshow('Webcam', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
