import numpy as np

def draw_grid(img, detection_status):
    grid_size = 100  # Ukuran grid di sudut kanan bawah
    start_x = img.shape[1] - grid_size - 10  # Posisi X grid
    start_y = img.shape[0] - grid_size - 10  # Posisi Y grid

    cell_size = int(grid_size / 5)  # Membagi grid menjadi 5x5

    # Menggambar sel grid
    for i in range(5):
        for j in range(5):
            cell_color = (0, 0, 255) if detection_status[j, i] else (255, 255, 255)
            cv2.rectangle(img, (start_x + j * cell_size, start_y + i * cell_size),
                          (start_x + (j + 1) * cell_size, start_y + (i + 1) * cell_size), cell_color, -1)

    # Menggambar garis grid
    for i in range(6):
        cv2.line(img, (start_x, start_y + i * cell_size), (start_x + grid_size, start_y + i * cell_size), (0, 0, 0), 1)
        cv2.line(img, (start_x + i * cell_size, start_y), (start_x + i * cell_size, start_y + grid_size), (0, 0, 0), 1)

    return img
