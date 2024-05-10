from ultralytics import YOLO
import cv2
import math
import mediapipe as mp
import numpy as np
import random
from mediapipe.python.solutions.pose import PoseLandmark

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
model = YOLO("yolov8n.pt")

# Fungsi untuk menghitung jarak menggunakan YOLO
def hitung_jarak(tinggi_bounding_box, focal_length_pixel, tinggi_objek_nyata):
    if tinggi_bounding_box == 0:
        return float('inf')  # Menghindari pembagian dengan nol
    jarak = (tinggi_objek_nyata * focal_length_pixel) / tinggi_bounding_box
    return jarak / 100  # Konversi ke meter

def hitung_lebar_objek(lebar_bounding_box, jarak_objek, focal_length_pixel):
    if lebar_bounding_box == 0 or focal_length_pixel == 0:
        return 0  # Menghindari pembagian dengan nol
    lebar_objek = (lebar_bounding_box * jarak_objek) / focal_length_pixel
    return lebar_objek


# Konstanta untuk perhitungan jarak dengan MediaPipe
k = 10.311  # Diperoleh dari kalibrasi, menyesuaikan sesuai kebutuhan
k2 = 24.222

# Fungsi untuk menghitung jarak Euclidean dalam piksel
def hitung_jarak_euclidean(landmark1, landmark2, lebar_img):
    jarak_pix = math.sqrt((landmark1.x - landmark2.x) ** 2 + (landmark1.y - landmark2.y) ** 2) * lebar_img
    return jarak_pix

def hitung_lebar_mediapipe(pose_results, lebar_img):
    if pose_results.pose_landmarks:
        bahu_kiri = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        bahu_kanan = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        jarak_pix = hitung_jarak_euclidean(bahu_kiri, bahu_kanan, lebar_img)
        
        # Kalibrasi untuk konversi piksel ke meter (asumsikan 0.5cm per piksel sebagai contoh)
        # Sesuaikan faktor kalibrasi ini berdasarkan pengukuran nyata atau eksperimen
        faktor_konversi = 0.00087  # 0.087cm per piksel
        lebar_m = jarak_pix * faktor_konversi
        
        return lebar_m
    return 0

def hitung_jarak_vertikal(pose_results, tinggi_img):
    if pose_results.pose_landmarks:
        pusar = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        # Misal kita asumsikan pusar sebagai acuan vertikal (bisa diganti sesuai dengan landmark yang sesuai)
        jarak_vertikal = (1 - pusar.y) * tinggi_img  # Hitung jarak dari atas gambar ke pusar
        return jarak_vertikal
    return 0


def draw_pyramid_grid(img, grid_size=100):
    start_x = img.shape[1] - grid_size - 10  # Posisi X awal grid
    start_y = img.shape[0] - grid_size - 10  # Posisi Y awal grid
    cell_size = grid_size // 10  # Ukuran setiap sel dalam grid

    # Warna latar belakang grid
    cv2.rectangle(img, (start_x, start_y), (start_x + grid_size, start_y + grid_size), (0, 0, 0), -1)

    # Menggambar grid piramida dengan dua sel hijau di puncak
    for i in range(10):
        for j in range(5 - i//2, 5 + i//2 + 1):
            if i == 0 and (j == 4 or j == 5):
                color = (0, 255, 0)  # Sel hijau untuk puncak piramida
            else:
                color = (255, 255, 255)  # Warna lain untuk grid
            cv2.rectangle(img, (start_x + j * cell_size, start_y + i * cell_size),
                          (start_x + (j + 1) * cell_size, start_y + (i + 1) * cell_size), color, -1)

    # Menggambar garis grid
    for i in range(11):
        # Garis horizontal
        cv2.line(img, (start_x, start_y + i * cell_size), (start_x + grid_size, start_y + i * cell_size), (0, 0, 0), 1)
        # Garis vertikal
        cv2.line(img, (start_x + i * cell_size, start_y), (start_x + i * cell_size, start_y + grid_size), (0, 0, 0), 1)

    return img

def initialize_pyramid_grid():
    # Membuat grid dengan dua kotak hijau di puncak
    grid = np.zeros((10, 10), dtype=bool)
    for i in range(10):
        start = max(0, 4 - i)
        end = min(9, 5 + i) + 1
        grid[i, start:end] = True
    return grid

def determine_direction(detection_grid, grid_size=10):
    mid_point = grid_size // 2
    left_count = np.sum(detection_grid[:, :mid_point])
    right_count = np.sum(detection_grid[:, mid_point:])
    
    if left_count > right_count:
        direction = "Turn Right"
    elif right_count > left_count:
        direction = "Turn Left"
    else:
        direction = random.choice(["Turn Left", "Turn Right"])
    
    return direction, abs(right_count - left_count) * 0.2  # Misalkan setiap kotak mewakili 0.2 meter

# Inisialisasi grid deteksi
detection_grid = initialize_pyramid_grid()
detection_grid = np.zeros((10, 10), dtype=bool)
camera_position = [(4, 0), (5, 0)]  # Posisi kamera pada grid

while True:
    success, img = cap.read()
    if not success:
        break  # Jika tidak berhasil membaca frame, keluar dari loop

    detection_grid.fill(False)
    detection_grid[0, 4] = True  # Pos 5,10 dalam 0-index
    detection_grid[0, 5] = True  # Pos 6,10 dalam 0-index

    results = model.predict(img, stream=True, imgsz = 800)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            confidence = math.ceil((box.conf[0] * 100)) / 100  # Menghitung confidence
            if confidence > 0.7:  # Memeriksa apakah confidence di atas 0.6
                cls = int(box.cls[0])  # Mengonversi kelas objek terdeteksi menjadi integer
                if cls == 0:  # Memeriksa apakah objek terdeteksi adalah 'person'
                    x1, y1, x2, y2 = box.xyxy[0]  # Mengambil koordinat kotak pembatas objek
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    color = (255, 0, 0) #inisialisasi warna

                    # Menggunakan YOLO untuk menghitung jarak
                    tinggi_bounding_box = y2 - y1
                    lebar_bounding_box = x2 - x1
                    jarak_yolo = hitung_jarak(tinggi_bounding_box, focal_length_pixel, tinggi_objek_nyata)
                    # Menghitung lebar objek
                    lebar_objek = hitung_lebar_objek(lebar_bounding_box, jarak_yolo, focal_length_pixel)
                    cv2.putText(img, f"BBox Width (Y)px : {lebar_objek:.2f} m", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(img, f"Yolo Distance: {jarak_yolo:.2f} m", (x1, y1 - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    cv2.putText(img, f"Distance Yolo: {jarak_yolo:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    

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
                        bahu_kanan = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                        bahu_kiri = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
                        pusar = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
                        jarak_pixbahu = hitung_jarak_euclidean(bahu_kanan, bahu_kiri, lebar_img)
                        jarak_pix = hitung_jarak_euclidean(siku_kanan, pergelangan_kanan, lebar_img)
                        lebar_img = img.shape[1]
                        tinggi_img = img.shape[0]
                        lebar_m = hitung_lebar_mediapipe(pose_results, lebar_img)
                        grid_size = 10  # Ukuran grid (10x10)
                        jarak_maksimum = 10 * 0.2 
                        #max_jarak = grid_size * 0.2  # Jarak maksimal yang dapat diwakili grid (2 meter dalam kasus ini)

                        cv2.putText(img, f"Human Width: {lebar_m:.2f} m", (10,240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                        if jarak_pix > 0:
                            jarak_mediapipe = (k / jarak_pix) * 10  # Menyesuaikan dengan faktor kalibrasi
                            jarak_mediapipebahu = (k2 / jarak_pixbahu) * 10 

                            # Menghitung posisi grid berdasarkan jarak MediaPipe
                            posisi_horizontal_piksel = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * lebar_img
                            grid_x = int((posisi_horizontal_piksel / lebar_img) * 10)
                            pusar = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
                            posisi_vertikal_piksel = pusar.y * tinggi_img
                            grid_y = int((jarak_maksimum - jarak_mediapipe) / 0.2)
                            #grid_y = grid_size - 1 - int(jarak_mediapipe / 0.2)
                            grid_y = max(0, min(9 - grid_y, 9))
                            #grid_y = max(0, min(grid_y, grid_size - 1))
                            lebar_grid = max(1, int(lebar_m / 0.2))

                            grid_x = min(max(grid_x, 0), 9)
                            # Menghitung jumlah kotak berdasarkan lebar dalam meter
                            # Setidaknya satu kot
                            for i in range(max(0, grid_x - lebar_grid // 2), min(10, grid_x + lebar_grid // 2)):
                                for j in range(max(0, grid_y), min(10, grid_y + lebar_grid // 2)):
                                    detection_grid[j][i] = True
                            #for i in range(grid_size):
                                #if i >= (grid_size - 1 - grid_y):  # Aktifkan dari grid_y sampai yang terdekat dengan kamera
                                   # for j in range(lebar_grid):  # lebar_grid adalah jumlah kotak yang diaktifkan berdasarkan lebar subjek
                                        #detection_grid[i][max(0, grid_x - j)] = True  # Asumsi grid_x sudah dihitung

                            cv2.putText(img, f"MP Hand Distance: {jarak_mediapipe:.2f} m", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            cv2.putText(img, f"MediaPipe Hand Distance: {jarak_mediapipe:.2f}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            cv2.putText(img, f"MP Shoulder Distance: {jarak_mediapipebahu:.2f}", (x1,y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                            cv2.putText(img, f"MediaPipe Shoulder Distance: {jarak_mediapipebahu:.2f}", (10 , 210), cv2.FONT_HERSHEY_SIMPLEX,0.6, (255,255,255),2)

                            # Menambahkan tampilan jarak piksel di sudut kiri atas
                            cv2.putText(img, f"(x,y)px Hand: {jarak_pix:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            cv2.putText(img, f"(x,y)px Shoulder : {jarak_pixbahu:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255,255,255),2 )
                            cv2.putText(img, f"X Pose Center: {posisi_horizontal_piksel:.2f}px", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            

                            # Mengubah warna bounding box menjadi merah jika kedua jarak terpenuhi
                            if jarak_yolo < 1.3 or jarak_mediapipe < 1.3:
                                color = (51, 255, 255)
                                pesan = "OBJEK MENDEKAT"
                                cv2.putText(img,pesan,(600, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)

                                if jarak_mediapipe < 0.5 or lebar_bounding_box > 500:
                                    color = (0, 0, 255)
                                    pesan2 = "STOP OBJEK TERLALU DEKAT"
                                    cv2.putText(img,pesan2,(600, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)


                    # Gambar bounding box YOLO
                    direction, distance = determine_direction(detection_grid)
                    cv2.putText(img, f"{direction} {distance:.2f}m", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    img = draw_pyramid_grid(img)


    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
