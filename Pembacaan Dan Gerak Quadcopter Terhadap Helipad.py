#library yang digunakan
from dronekit import connect, VehicleMode, LocationGlobalRelative 
from pymavlink import mavutil
import time
import cv2
import numpy as np
import threading

# Daftar kelas (objek yang akan dideteksi)
classes = ["helipad-abjad"]

# Memuat model ONNX (model deep learning untuk deteksi helipad)
net = cv2.dnn.readNetFromONNX("helipadorange.onnx")

# Variabel awal untuk frame deteksi, hasil deteksi, dan titik pusat helipad
frame = None
detections = None
lock = threading.Lock() # Digunakan untuk menghindari akses bersamaan ke variabel shared
center = None

# Fungsi untuk mengarming dan take off quadcopter
def arm_and_takeoff(aTargetAltitude):
    print("Basic pre-arm checks")
     # Menunggu hingga drone siap untuk di-arm
    while not vehicle.is_armable:
        print (" Waiting for vehicle to initialise...")
        time.sleep(1)

    print ("Arming motors")
   # Mengatur mode menjadi GUIDED (mode kontrol otomatis)
    vehicle.mode    = VehicleMode("GUIDED") 
    vehicle.armed   = True # jika quadcopter arming = true
    time.sleep(5) # Tunggu sebentar agar motor siap

    # Memastikan drone sudah armed sebelum melakukan take off
    while not vehicle.armed:
        print (" Waiting for arming...")
        time.sleep(1)

    print ("Taking off!")
    vehicle.simple_takeoff(aTargetAltitude) # Melakukan take off ke ketinggian target

    # Menunggu hingga drone mencapai ketinggian yang aman
    while True:
        print (" Altitude: ", vehicle.location.global_relative_frame.alt) #program menyuruh quadcopter untuk membaca ketinggian menggunakan GPS
        if vehicle.location.global_relative_frame.alt>=aTargetAltitude*0.95: # 95% di dapatkan standar program dronekit
            print ("Reached target altitude") # jika ketinggian sudah 95% dari ketinggian yg di tentukan, maka quadcopter akan berhenti.
            break
        time.sleep(1)

# Fungsi untuk mengatur kecepatan drone (gerakan dalam arah x, y, dan z)
def velocity(velocity_x, velocity_y, velocity_z):
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,       
        0, 0,    # Sistem target dan komponen target
        mavutil.mavlink.MAV_FRAME_BODY_NED, # Frame koordinat (lokal pada drone)
        0b0000111111000111,  # Masking tipe gerakan (x, y, z kecepatan)
        0, 0, 0, # Posisi x, y, z (tidak digunakan)
        velocity_x, velocity_y, velocity_z, # m/s (x=maju mundur y=kanan kiri z=atas bawah)
        0, 0, 0, # x, y, z acceleration
        0, 0)
    vehicle.send_mavlink(msg)
# Fungsi untuk gerakan dengan durasi tertentu
def velocityd(velocity_x, velocity_y, velocity_z, duration=0):
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,    
        0, 0,     # Sistem target dan komponen target
        mavutil.mavlink.MAV_FRAME_BODY_NED, # Frame koordinat (lokal pada drone)
        0b0000111111000111, # Masking tipe gerakan (x, y, z kecepatan)
        0, 0, 0, # Posisi x, y, z (tidak digunakan)
        velocity_x, velocity_y, velocity_z, # m/s (x=maju mundur y=kanan kiri z=atas bawah)
        0, 0, 0, # x, y, z acceleration
        0, 0)
    for x in range(0,duration):
        vehicle.send_mavlink(msg)
        time.sleep(1)

# Fungsi untuk menangkap frame dari kamera
def capture_frames():
    global frame, cap
    cap = cv2.VideoCapture(0) # Membuka kamera (index 0 untuk kamera utama)
    while True:
        ret, img = cap.read()
        if not ret:
            break# Jika kamera gagal membaca frame, hentikan
        with lock:# Pastikan hanya satu thread yang mengakses frame
            frame = img.copy()
# Fungsi untuk mendeteksi objek menggunakan model ONNX
def detect_objects():
    global detections
    while True:
        with lock:
            if frame is None:
                continue
            blob = cv2.dnn.blobFromImage(frame, 1/255, (640, 640), (0, 0, 0), True, False)
        net.setInput(blob)
        detections = net.forward()[0]# Hasil deteksi (output dari model)

# Fungsi untuk mencari pusat lingkaran helipad menggunakan metode Hough Circle
def find_helipad_center(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2) # Mengurangi noise dengan Gaussian Blur
    
    circles = cv2.HoughCircles(
        blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=1.5, # Resolusi akumulator lebih rendah (pemrosesan lebih cepat). Nilai lebih kecil (misalnya 1.0) bisa digunakan jika akurasi lebih penting daripada kecepatan.
        minDist=50, # Jarak minimum antar pusat lingkaran, diatur sedikit lebih besar dari diameter (50 piksel) Ini memastikan bahwa lingkaran yang terdeteksi tidak saling tumpang tindih.
        param1=100, # Deteksi tepi lebih selektif.
        param2=30, # Hanya lingkaran dengan pusat yang kuat dideteksi.
        minRadius=18, # Radius minimum (setengah diameter ~20 piksel)
        maxRadius=22  # Radius maksimum (setengah diameter ~20 piksel)
    )
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(frame, (x, y), r, (0, 255, 0), 4)  # Gambar lingkaran
            cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)  # Pusat
            return frame, (x, y)   # Kembalikan frame dan koordinat pusat
    
    return frame, None
# Fungsi untuk menampilkan frame deteksi
def display_frame():
    global center
    while True:
        with lock:
            if frame is None or detections is None:
                continue
            img, det = frame.copy(), detections.copy()
         # Proses deteksi bounding box dan label
        boxes, confidences, class_ids = [], [], []
        img_w, img_h = img.shape[1], img.shape[0]
        x_scale, y_scale = img_w / 640, img_h / 640

        for row in det:
            confidence = row[4]
            if confidence > 0.2: # Ambang batas deteksi
                scores = row[5:]
                class_id = np.argmax(scores)
                if scores[class_id] > 0.2:
                    confidences.append(confidence)
                    class_ids.append(class_id) # rumus mengularkan hasil conffiden
                    cx, cy, w, h = row[:4]
                    boxes.append([int((cx - w / 2) * x_scale), int((cy - h / 2) * y_scale), int(w * x_scale), int(h * y_scale)])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.2)
        if len(indices) > 0: # program untuk conffident jika lebih dari 0 dia akan membaca helipad
            indices = indices.flatten()
            for i in indices:
                x, y, w, h = boxes[i]
                if class_ids[i] < len(classes):
                    label = f"{classes[class_ids[i]]} {confidences[i]:.2f}"
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(img, label, (x, y - 2), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)

        img, center = find_helipad_center(img)
        if center:
            print(f"Helipad center: {center}")

        cv2.imshow("Deteksi Objek", img)# ngeluarin frame di laptop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
# Fungsi untuk mengontrol gerakan drone berdasarkan posisi helipad
def gerakqc():
    global center
    while True:
        if center is None:
            continue
        cX, cY = center
        distx = frame.shape[1] // 2 - cX
        disty = frame.shape[0] // 2 - cY
        x = (distx * (1/320))/2 # di sesuaikan(dibagi dua agar gerakannya tidak terllu cepat)
        y = (disty * ( 1/240))/2
        print("cX:", cX, "cY:", cY, "SpeedX:", x, "SpeedY:", y)

        ######## Gerak Serong
        if cX <= 260 and cX >= 0 and cY >= 260 and cY <= 480:
            print("Bergerak Maju Kiri")
            velocity(x, y, 0)
        elif cX <= 260 and cX >= 0 and cY <= 220 and cY >= 0:
            print("Bergerak Maju Kiri")
            velocity(x, y, 0)
        elif cX >= 300 and cX <= 640 and cY >= 260 and cY <= 480:
            print("Bergerak Mundur Kiri")
            velocity(x, y, 0)
        elif cX >= 300 and cX <= 640 and cY <= 220 and cY >= 0:
            print("Bergerak Mundur Kanan")
            velocity(x, y, 0)

        ######## Gerak Satu Arah ###########################3
        if cX <= 280 and cX >= 0 and cY <= 280 and cY >= 200:
            print("Bergerak Maju")
            velocity(x, 0, 0)
        elif cX >= 360 and cX <= 640 and cY <= 280 and cY >= 200:
            print("Bergerak Mundur")
            velocity(x, 0, 0)
        elif cX <= 360 and cX >= 280 and cY >= 280 and cY <= 480:
            print("Bergerak Kiri")
            velocity(0, y, 0)
        elif cX <= 360 and cX >= 280 and cY <= 200 and cY >= 0:
            print("Bergerak Kanan")
            velocity(0, y, 0)

        if cX <= 360 and cX >= 280 and cY <= 280 and cY >= 200 :
            print("Diatas Helipad, menurunkan Ketinggian")
            velocity(0, 0, 0.2)# kecepatan(0,2 m/s)
            
    # jika pembacaan x dan y sesuai dengan yang di tentukan dan sesuai ketinggian GPS maka quadcopter akan landing
        if cX <= 350 and cX >= 290 and cY <= 270 and cY >= 210 and vehicle.location.global_relative_frame.alt > 1 : 
            vehicle.mode = VehicleMode("LAND")
            print("Diatas Helipad, Landing")
            vehicle.close() # quadcopter mati
            exit(1)

        time.sleep(0.1)
# Program utama
if __name__ == "__main__": #memulai program
    vehicle = connect("COM15", baud=57600) #frekuensi/baudret
    print("connect")

    # QC mode guided
    vehicle.mode = VehicleMode("GUIDED")

    # Take off
    arm_and_takeoff(3)
    time.sleep(1)

    print("Maju Awal 1m")
    velocityd(0.5,0,0,2) # maju 0,5 meter untuk 2 detik
    time.sleep(1)

    #untuk menjalankan program berjalan bersamaan
    threads = [
        threading.Thread(target=display_frame),
        threading.Thread(target=capture_frames),
        threading.Thread(target=detect_objects),
        threading.Thread(target=gerakqc)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print("Done!")
    cap.release()
    cv2.destroyAllWindows()#nutup smua program

