import time
import cv2
import numpy as np
import threading

# Mendefinisikan kelas yang akan dikenali, yaitu "helipad-abjad"
classes = ["helipad-abjad"]

# Memuat model YOLOv5 dari file ONNX untuk digunakan dalam deteksi objek
net = cv2.dnn.readNetFromONNX("helipadorange.onnx")

# Inisialisasi variabel global untuk frame, hasil deteksi, kunci threading, dan pusat lingkaran
frame = None
detections = None
lock = threading.Lock()
center = None

# Fungsi untuk memeriksa apakah sebagian besar dari ROI (Region of Interest) berwarna oranye
def is_orange(roi):
    # Mengubah ROI dari BGR ke HSV (Hue, Saturation, Value)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # Mendefinisikan rentang warna oranye dalam HSV
    lower_orange = np.array([10, 100, 100])
    upper_orange = np.array([25, 255, 255])
    # Membuat mask untuk mendeteksi piksel dalam rentang warna oranye
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    # Menghitung persentase area ROI yang berwarna oranye
    orange_percentage = (cv2.countNonZero(mask) / (roi.shape[0] * roi.shape[1])) * 100
    # Mengembalikan True jika lebih dari 50% area ROI berwarna oranye
    return orange_percentage > 50

# Fungsi untuk menangkap frame dari kamera secara terus-menerus
def capture_frames():
    global frame, cap
    # Membuka koneksi ke kamera
    cap = cv2.VideoCapture(1)
    while True:
        ret, img = cap.read()
        if not ret:
            break
        # Mengamankan akses ke variabel frame menggunakan lock untuk threading
        with lock:
            # Menyimpan salinan gambar yang diambil dari kamera
            frame = img.copy()

# Fungsi untuk melakukan deteksi objek menggunakan model YOLOv5
def detect_objects():
    global detections
    while True:
        # Mengamankan akses ke frame
        with lock:
            if frame is None:
                continue
            # Melakukan preprocessing spesifik YOLOv5 (membuat blob dari gambar)
            blob = cv2.dnn.blobFromImage(frame, 1/255, (640, 640), (0, 0, 0), True, False)
        # Memberikan blob sebagai input untuk model
        net.setInput(blob)
        # Melakukan inferensi dan menyimpan hasil deteksi
        detections = net.forward()[0]

# Fungsi untuk menemukan pusat lingkaran helipad di dalam frame
def find_helipad_center(frame):
    # Mengubah gambar ke skala abu-abu
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Melakukan Gaussian blur untuk mengurangi noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    # Menggunakan algoritma HoughCircles untuk mendeteksi lingkaran
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1.2, 100, param1=100, param2=30, minRadius=30, maxRadius=200)

    if circles is not None:
        # Membulatkan koordinat lingkaran yang terdeteksi
        circles = np.round(circles[0, :]).astype("int")
        # Menggambar lingkaran dan pusatnya pada gambar
        for (x, y, r) in circles:
            cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
            # Mengembalikan frame dan koordinat pusat lingkaran
            return frame, (x, y)
    # Jika tidak ada lingkaran ditemukan, mengembalikan frame tanpa perubahan
    return frame, None

# Fungsi untuk menampilkan frame dengan hasil deteksi objek
def display_frame():
    global center
    while True:
        # Mengamankan akses ke frame dan hasil deteksi
        with lock:
            if frame is None or detections is None:
                continue
            img, det = frame.copy(), detections.copy()

        # Inisialisasi daftar untuk kotak, kepercayaan, dan ID kelas objek
        boxes, confidences, class_ids = [], [], []
        img_w, img_h = img.shape[1], img.shape[0]
        # Skala untuk menyesuaikan ukuran bounding box dengan ukuran gambar
        x_scale, y_scale = img_w / 640, img_h / 640

        # Melakukan iterasi pada hasil deteksi
        for row in det:
            confidence = row[4]
            if confidence > 0.2:  # Hanya mengambil deteksi dengan kepercayaan > 0.2
                scores = row[5:]
                class_id = np.argmax(scores)
                if scores[class_id] > 0.2:  # Hanya deteksi dengan probabilitas kelas > 0.2
                    confidences.append(confidence)
                    class_ids.append(class_id)
                    cx, cy, w, h = row[:4]
                    # Menyimpan bounding box yang telah diskalakan
                    boxes.append([int((cx - w / 2) * x_scale), int((cy - h / 2) * y_scale), int(w * x_scale),
                                  int(h * y_scale)])

        # Melakukan Non-Maximum Suppression untuk mengurangi duplikasi kotak deteksi
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.2)
        if len(indices) > 0:
            indices = indices.flatten()
            for i in indices:
                x, y, w, h = boxes[i]
                if class_ids[i] < len(classes):
                    # Pastikan bounding box berada dalam area frame
                    if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
                        continue  # Lewati bounding box yang keluar dari area gambar

                    # Ekstrak area minat (ROI) dari gambar
                    roi = img[y:y + h, x:x + w]
                    if roi.size == 0:
                        continue

                    # Memeriksa apakah ROI mengandung warna oranye
                    if is_orange(roi):
                        label = f"{classes[class_ids[i]]} {confidences[i]:.2f}"
                        # Menggambar bounding box di sekitar objek yang terdeteksi
                        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        cv2.putText(img, label, (x, y - 2), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)

                        print(f"Koordinat x: {x}, y: {y}")

                        # Mencari pusat helipad menggunakan bounding box yang terdeteksi
                        img, center = find_helipad_center(img)
                        if center:
                            print(f"Helipad center: {center}")

        # Menampilkan gambar dengan deteksi objek di jendela GUI
        cv2.imshow("Deteksi Objek", img)
        # Keluar dari loop jika tombol 'Esc' ditekan
        if cv2.waitKey(1) & 0xFF == 27:
            break

# Bagian utama program
if __name__ == "__main__":
    print("connect")

    # Membuat daftar thread yang menjalankan fungsi secara paralel
    threads = [
        threading.Thread(target=display_frame),
        threading.Thread(target=capture_frames),
        threading.Thread(target=detect_objects),
    ]
    # Memulai semua thread
    for t in threads:
        t.start()
    # Menunggu semua thread selesai
    for t in threads:
        t.join()

# Setelah selesai, melepaskan sumber daya dan menutup semua jendela
print("Done!")
cap.release()
cv2.destroyAllWindows()
