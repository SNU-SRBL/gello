import serial
import time
import csv
import os

# === 사용자 설정 ===
Serial_port = "COM5"                  # 시리얼 포트
csv_filename = "rft_data_log.csv"     # 저장 파일명
DF = 50.0                             # Force Divider
DT = 1000.0                           # Torque Divider

# === 시리얼 포트 초기화 ===
serLoadCell = serial.Serial(Serial_port, 115200, timeout=0.1)
time.sleep(1)

# === 센서 설정 명령 ===
serLoadCell.write(bytes([0x55, 0x0C, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0xAA]))  # Stop Output
serLoadCell.write(bytes([0x55, 0x11, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x12, 0xAA]))  # Set Bias
serLoadCell.write(bytes([0x55, 0x0F, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0F, 0xAA]))  # Output: 200 Hz
serLoadCell.write(bytes([0x55, 0x08, 0x01, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x11, 0xAA]))  # LPF: 30 Hz
time.sleep(0.1)

# === 명령 패킷 ===
cmdData = bytes([0x55, 0x0A, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0xAA])  # Get Data

# === 오프셋 보정 ===
offsetForce = [0, 0, 0]
offsetMoment = [0, 0, 0]
for _ in range(100):
    serLoadCell.write(cmdData)
    data = serLoadCell.read(19)
    if len(data) != 19:
        continue
    offsetForce[0] += int.from_bytes(data[2:4], byteorder='big', signed=True)
    offsetForce[1] += int.from_bytes(data[4:6], byteorder='big', signed=True)
    offsetForce[2] += int.from_bytes(data[6:8], byteorder='big', signed=True)
    offsetMoment[0] += int.from_bytes(data[8:10], byteorder='big', signed=True)
    offsetMoment[1] += int.from_bytes(data[10:12], byteorder='big', signed=True)
    offsetMoment[2] += int.from_bytes(data[12:14], byteorder='big', signed=True)

for i in range(3):
    offsetForce[i] /= 100
    offsetMoment[i] /= 100

# === CSV 저장 함수 ===
def save_to_csv(filename, row):
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Time", "Fx (N)", "Fy (N)", "Fz (N)",
                             "Mx (Nm)", "My (Nm)", "Mz (Nm)"])
        writer.writerow(row)

# === 메인 루프 ===
print("Start reading force/torque data. Press Ctrl+C to stop.")
start_time = time.time()

try:
    while True:
        serLoadCell.write(cmdData)
        data = serLoadCell.read(19)
        if len(data) != 19:
            continue

        # Raw 값 추출
        Fx_raw = int.from_bytes(data[2:4], byteorder='big', signed=True)
        Fy_raw = int.from_bytes(data[4:6], byteorder='big', signed=True)
        Fz_raw = int.from_bytes(data[6:8], byteorder='big', signed=True)
        Mx_raw = int.from_bytes(data[8:10], byteorder='big', signed=True)
        My_raw = int.from_bytes(data[10:12], byteorder='big', signed=True)
        Mz_raw = int.from_bytes(data[12:14], byteorder='big', signed=True)

        # 단위 환산
        Fx = (Fx_raw - offsetForce[0]) / DF
        Fy = (Fy_raw - offsetForce[1]) / DF
        Fz = (Fz_raw - offsetForce[2]) / DF
        Mx = (Mx_raw - offsetMoment[0]) / DT
        My = (My_raw - offsetMoment[1]) / DT
        Mz = (Mz_raw - offsetMoment[2]) / DT

        # 시간 기록 및 출력
        timestamp = time.time() - start_time
        print(f"[{timestamp:.2f} s] Fx: {Fx:.2f} N, Fy: {Fy:.2f} N, Fz: {Fz:.2f} N | "
              f"Mx: {Mx:.3f} Nm, My: {My:.3f} Nm, Mz: {Mz:.3f} Nm")

        # CSV 저장
        save_to_csv(csv_filename, [timestamp, Fx, Fy, Fz, Mx, My, Mz])

        time.sleep(0.01)  # 약 100 Hz 속도

except KeyboardInterrupt:
    print("\nTerminated by user.")
    serLoadCell.close()
