from ultralytics import YOLO
import cv2
import math
import mediapipe as mp
import numpy as np
import socket
import time
import datetime
import csv
from mediapipe.python.solutions.pose import PoseLandmark

def delay(seconds):
    start_time = time.time()
    while time.time() - start_time < seconds:
        pass

# Inisialisasi file CSV
csv_file_path = "log.csv"
csv_columns = ["Direction", "Distance", "Timestamp"]
host = "192.168.4.1" # Set to ESP32 Access Point IP Address
port = 80

# Inisialisasi MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.6)
mp_drawing = mp.solutions.drawing_utils

# Mulai kamera web
cap = cv2.VideoCapture(1)
cap.set(3, 1366)  # Mengatur lebar gambar menjadi 1920 piksel
cap.set(4, 768)  # Mengatur tinggi gambar menjadi 1080 piksel
#jika 1920x1080 maka 5 menit
#jika 640x480 < 3mnt
# Konstanta untuk perhitungan jarak dengan MediaPipe
k = 5.900 #120     Diperoleh dari kalibrasi, menyesuaikan sesuai kebutuhan
k2 = 24.222

focal_length_pixel = 481  # Nilai kalibrasi dari hasil hitung focal length
tinggi_objek_nyata = 181  # Tinggi objek nyata dalam cm

# Inisialisasi variabel untuk penghitungan FPS
frame_count = 0
start_time = time.time()
fps = 0

# Memuat model YOLO
model = YOLO("best100epoch.pt")

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

def convert_coordinates(outputs, img_width, img_height):
    boxes = []
    for detection in outputs:
        x_center, y_center, width, height = detection['x_center'], detection['y_center'], detection['width'], detection['height']
        
        x_min = (x_center - width / 2) * img_width
        y_min = (y_center - height / 2) * img_height
        x_max = (x_center + width / 2) * img_width
        y_max = (y_center + height / 2) * img_height
        
        boxes.append((x_min, y_min, x_max, y_max))
    return boxes

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

def draw_grid(img, detection_status, camera_position):
    grid_size = 100  # Ukuran grid di sudut kanan bawah
    start_x = img.shape[1] - grid_size - 10  # Posisi X grid
    start_y = img.shape[0] - grid_size - 10  # Posisi Y grid

    cell_size = int(grid_size / 10)  # Membagi grid menjadi 10x10

    # Menggambar sel grid dengan latar belakang hitam
    for i in range(10):
        for j in range(10):
            cell_color = (0, 0, 255) if detection_status[i][j] else (0, 0, 0)
            cv2.rectangle(img, (start_x + j * cell_size, start_y + i * cell_size),
                          (start_x + (j + 1) * cell_size, start_y + (i + 1) * cell_size), cell_color, -1)

    # Menandai posisi kamera dengan kotak hijau
    for pos in camera_position:
        x, y = pos
        cv2.rectangle(img, (start_x + x * cell_size, start_y + y * cell_size),
                      (start_x + (x + 1) * cell_size, start_y + (y + 1) * cell_size), (0, 255, 0), -1)
        
    # Menggambar garis grid
    for i in range(11):
        cv2.line(img, (start_x, start_y + i * cell_size), (start_x + grid_size, start_y + i * cell_size), (255, 255, 255), 1)
        cv2.line(img, (start_x + i * cell_size, start_y), (start_x + i * cell_size, start_y + grid_size), (255, 255, 255), 1)
    
    text_x2 = start_x - 10
    text_x = start_x - 80
    text_y = start_y - 20  # Menempatkan teks 20 piksel di atas grid
    text_y2 = start_y - 80
    cv2.putText(img, "0.2 Meter per square", (text_x,text_y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.6, (0,0,0), 1)
    cv2.putText(img, f"X: {posisi_horizontal_piksel:.2f}px", (text_x2, text_y2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    return img

def determine_direction(detection_grid, grid_size=10):
    if not np.any(detection_grid):
        return 'Maju', 0, "No detection, moving forward"
    
    mid_point = grid_size // 2
    left_count = np.sum(detection_grid[:, :mid_point])
    right_count = np.sum(detection_grid[:, mid_point:])
    manusia = "manusia terlalu jauh"
    
    #if jarak_mediapipe <0.7:
    if left_count > right_count:
        direction = 'Kanan'
        manusia = "Manusia di Kiri"
    
    elif right_count > left_count:
        direction = 'Kiri'
        manusia = "Manusia di Kanan"
            
    else:
        direction = 'Kanan'
        manusia = "Manusia di tengah"
    #else:
    #    direction = (["Terus Maju"])

        
    return direction, abs(right_count - left_count) * 0.2, manusia  # Misalkan setiap kotak mewakili 0.2 meter


# Inisialisasi grid deteksi
detection_grid = np.zeros((10, 10), dtype=bool)
camera_position = [(4, 0), (5, 0)]  # Posisi kamera pada grid (5,10) dan (6,10)
newtex = None


while True:
    success, img = cap.read()
    if not success:
        break  # Jika tidak berhasil membaca frame, keluar dari loop

    frame_count += 1
    current_time = time.time()

    # Hitung dan tampilkan FPS setiap satu detik
    if current_time - start_time >= 1:
        fps = frame_count / (current_time - start_time)
        frame_count = 0
        start_time = current_time
        #print(f"FPS: {fps:.2f}")

    detection_grid.fill(False)
    detection_grid[0, 4] = True  # Pos 5,10 dalam 0-index
    detection_grid[0, 5] = True  # Pos 6,10 dalam 0-index
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    start_time_yolo = time.time()

    results = model.predict(img, stream=True)

    end_time_yolo = time.time()

    response_time_yolo = end_time_yolo - start_time_yolo
    print(f"Waktu respons YOLO: {response_time_yolo:.5f} detik")
    

#print waktu
    with open('log.csv', mode='a') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
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
                        cv2.putText(img, f"X_min: {x1}px", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                        cv2.putText(img, f"BBox Width (Y)px : {lebar_objek:.2f} m", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                        cv2.putText(img, f"Yolo Distance: {jarak_yolo:.2f} m", (x1, y1 - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                        cv2.putText(img, f"Distance Yolo: {jarak_yolo:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                        
                        start_time_mediapipe = time.time()
                        jarak_mediapipe = None

                        # Crop gambar untuk analisis pose dengan MediaPipe
                        person_img = img[y1:y2, x1:x2]
                        person_img_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
                        pose_results = pose.process(person_img_rgb)

                        end_time_mediapipe = time.time()
                        response_time_mediapipe = end_time_mediapipe - start_time_mediapipe
                        print(f"Waktu respons MediaPipe: {response_time_mediapipe:.5f} detik")
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

                            cv2.putText(img, f"Human Width: {lebar_m:.2f} m", (10,240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                            if jarak_pix > 0:
                                jarak_mediapipe = (k / jarak_pix) * 10  # Menyesuaikan dengan faktor kalibrasi
                                jarak_mediapipebahu = (k2 / jarak_pixbahu) * 10 

                                # Menghitung posisi grid berdasarkan jarak MediaPipe
                                posisi_horizontal_piksel = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * lebar_img
                                grid_x = int(((x1 + 350) / lebar_img) * 10)
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

                                cv2.putText(img, f"MP Hand Distance: {jarak_mediapipe:.2f} m", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                cv2.putText(img, f"MediaPipe Hand Distance: {jarak_mediapipe:.2f}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                print (f"Jarak : {jarak_mediapipe:.2f}")
                                cv2.putText(img, f"MP Shoulder Distance: {jarak_mediapipebahu:.2f}", (x1,y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                                cv2.putText(img, f"MediaPipe Shoulder Distance: {jarak_mediapipebahu:.2f}", (10 , 210), cv2.FONT_HERSHEY_SIMPLEX,0.6, (0,0,0),2)
                                cv2.putText(img, f"FPS: {fps:.2f}", (10, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                                # Menambahkan tampilan jarak piksel di sudut kiri atas
                                cv2.putText(img, f"(x,y)px Hand: {jarak_pix:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                                cv2.putText(img, f"(x,y)px Shoulder : {jarak_pixbahu:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0,0,0),2 )
                                cv2.putText(img, f"X Pose Center: {posisi_horizontal_piksel:.2f}px", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                                # Mengubah warna bounding box menjadi merah jika kedua jarak terpenuhi
                                #if jarak_yolo < 1.3 or jarak_mediapipe < 1.3:
                                #  color = (51, 255, 255)
                                    #pesan = "Valid Object"
                                    #cv2.putText(img,pesan,(600, 700), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)
                        direction, distance, manusia = determine_direction(detection_grid)
                        if direction == 'Maju' and not np.any(detection_grid):
                            pesan = "MAJU"
                            color = (51, 255, 255)  # Color code for moving forward

                        if jarak_mediapipe < 1.0 and direction == 'Kiri': #or lebar_bounding_box > 500:
                            pesan = "BELOK KIRI"
                            color = (0, 0, 255)
                                #pesansend = 'BELOK KIRI'
                                # cv2.putText(img,pesan2,(500, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
                                # pesan = determine_direction(detection_grid, grid_size=10)
                        elif jarak_mediapipe < 1.0 and direction == 'Kanan':
                            pesan = "BELOK KANAN"
                            color = (0,0,255)

                        else:
                            pesan = "MAJU"
                            color = (51,255,255)

                        #pesan = determine_direction(detection_grid, grid_size=10)    
                        if pesan != newtex:
                            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                                s.connect((host, port))

                                delay(0.5)  

                                arah = 'C\n'
                                s.send(arah.encode('utf-8'))
                                time.sleep(1)
                                date = datetime.datetime.now()
                                print(date)
                                agung = ("Stop")
                                data = {
                                    "Direction": agung, 
                                    "Distance": distance,
                                    "Timestamp": date.strftime("%Y-%m-%d %H:%M:%S")  # Format waktu
                                }
                                writer.writerow({"Direction": agung, "Distance": distance, "Timestamp": date.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]})
                                

                                if pesan == "BELOK KIRI":
                                    arah = 'E\n'
                                    date = datetime.datetime.now()
                                    agung = ("Kiri")
                                    print(date)
                                    print("kiri")
                                elif pesan == "BELOK KANAN":
                                    arah = 'A\n'
                                    date = datetime.datetime.now()
                                    agung = ("Kanan")
                                    print(date)
                                    print("kanan")
                                #elif pesan == "MAJU":
                                    #arah = 'B\n'
                                    #date = datetime.datetime.now()
                                    #agung = ("Maju")
                                    #print(date)
                                else:
                                    arah = 'B\n'
                                    date = datetime.datetime.now()
                                    agung = ("Maju")
                                    print(date)
                                s.send(arah.encode('utf-8'))
                                newtex = pesan
                                data = {
                                    "Direction": agung, 
                                    "Distance": distance,
                                    "Timestamp": date.strftime("%Y-%m-%d %H:%M:%S")  # Format waktu
                                }
                                writer.writerow({"Direction": agung, "Distance": distance, "Timestamp": date.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]})

                        print(f"FPS: {fps:.2f}")
                        
                        # Gambar bounding box YOLO
                        direction, distance, manusia = determine_direction(detection_grid)
                        cv2.putText(img, f"{pesan} {distance:.2f}m", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        cv2.putText(img, f"{pesan} for {distance:.2f}",(500,400 ), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 2 )
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        print (manusia)
                        img = draw_grid(img, detection_grid, camera_position)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()