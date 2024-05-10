from ultralytics import YOLO
import cv2
import math
import mediapipe as mp

# Inisialisasi MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.6)
mp_drawing = mp.solutions.drawing_utils

# Mulai kamera web
cap = cv2.VideoCapture(0)
cap.set(3, 1920)  # Mengatur lebar gambar menjadi 1920 piksel
cap.set(4, 1080)  # Mengatur tinggi gambar menjadi 1080 piksel

focal_length_pixel = 481  # Nilai kalibrasi dari hasil hitung focal length
tinggi_objek_nyata = 181  # Tinggi objek nyata dalam cm

# Memuat model YOLO
model = YOLO("last.pt")

# Fungsi untuk menghitung jarak menggunakan YOLO
def hitung_jarak(tinggi_bounding_box, focal_length_pixel, tinggi_objek_nyata):
    if tinggi_bounding_box == 0:
        return float('inf')  # Menghindari pembagian dengan nol
    jarak = (tinggi_objek_nyata * focal_length_pixel) / tinggi_bounding_box
    return jarak / 100  # Konversi ke meter

# Konstanta untuk perhitungan jarak dengan MediaPipe
k = 6.311  # Diperoleh dari kalibrasi, menyesuaikan sesuai kebutuhan

# Fungsi untuk menghitung jarak Euclidean dalam piksel
def hitung_jarak_euclidean(landmark1, landmark2, lebar_img):
    jarak_pix = math.sqrt((landmark1.x - landmark2.x) ** 2 + (landmark1.y - landmark2.y) ** 2) * lebar_img
    return jarak_pix

while True:
    success, img = cap.read()
    if not success:
        break  # Jika tidak berhasil membaca frame, keluar dari loop

    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            confidence = math.ceil((box.conf[0] * 100)) / 100  # Menghitung confidence
            if confidence > 0.6:  # Memeriksa apakah confidence di atas 0.6
                cls = int(box.cls[0])  # Mengonversi kelas objek terdeteksi menjadi integer
                if cls == 0:  # Memeriksa apakah objek terdeteksi adalah 'person'
                    x1, y1, x2, y2 = box.xyxy[0]  # Mengambil koordinat kotak pembatas objek
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                    # Gambar bounding box YOLO
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

                    # Menggunakan YOLO untuk menghitung jarak
                    tinggi_bounding_box = y2 - y1
                    jarak_yolo = hitung_jarak(tinggi_bounding_box, focal_length_pixel, tinggi_objek_nyata)
                    cv2.putText(img, f"Yolo Jarak: {jarak_yolo:.2f} m", (x1, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                    # Crop gambar untuk analisis pose dengan MediaPipe
                    person_img = img[y1:y2, x1:x2]
                    person_img_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
                    pose_results = pose.process(person_img_rgb)

                    if pose_results.pose_landmarks:
                        # Menggambar landmarks pose
                        mp_drawing.draw_landmarks(img[y1:y2, x1:x2], pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                        # Menggunakan MediaPipe untuk estimasi jarak
                        lebar_img = person_img.shape[1]
                        siku_kanan = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
                        pergelangan_kanan = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
                        jarak_pix = hitung_jarak_euclidean(siku_kanan, pergelangan_kanan, lebar_img)
                        
                        if jarak_pix > 0:
                            jarak_mediapipe = (k / jarak_pix) * 10  # Menyesuaikan dengan faktor kalibrasi
                            cv2.putText(img, f"MP Jarak: {jarak_mediapipe:.2f} m", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            # Menambahkan tampilan jarak piksel di sudut kiri atas
                            cv2.putText(img, f"Jarak Piksel Lengan: {jarak_pix:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
