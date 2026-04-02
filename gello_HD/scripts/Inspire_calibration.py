import serial
import time

class SJ_Inspire:
    def __init__(self, device_name="/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_BG027RUV-if00-port0", baudrate=115200):
        self.ser = serial.Serial(device_name, baudrate, timeout=0.1)
        self.sleep_time = 0.002
    
    def __del__(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("Destructor called")
    
    # def close(self):
    #     pass

    def _write(self, id, add, num, val):
        bytes = [0xEB, 0x90]            
        bytes.append(id)                
        bytes.append(num + 3)           
        bytes.append(0x12)              
        bytes.append(add & 0xFF)        
        bytes.append((add >> 8) & 0xFF)
        # print(val)
        for i in range(num):
            bytes.append(val[i])
        checksum = 0x00                 
        for i in range(2, len(bytes)):
            checksum += bytes[i]        
        checksum &= 0xFF                
        bytes.append(checksum)
        
        self.ser.reset_input_buffer()
        self.ser.write(bytes)    
        self.ser.flush()
        recv = self.ser.read(9)
        time.sleep(self.sleep_time)
        if len(recv) < 9:
            print("[Inspire] No response")

    def sensor_calibration(self):
        self._write(1, 1007, 2, [0,1])
        print("[Inspire] Sensor calibration")

def main():
    hand = SJ_Inspire()
    hand.sensor_calibration()

if __name__ == "__main__":
    main()