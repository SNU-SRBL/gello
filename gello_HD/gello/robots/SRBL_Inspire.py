"""
File containing the SRBL Inspire gripper class for controlling the gripper and obtaining sensor data.
Written by Seongjun Koh (Soft Robotics and Bionics Lab, Seoul National University)
"""

import serial
import time

INSPIRE_regdict = {
    'ID'         : 1000,
    'baudrate'   : 1001,
    'mode'       : 1100,
    'clearErr'   : 1003,
    'forceClb'   : 1007,
    'angleSet'   : 1040, # Use this to set finger position
    'forceSet'   : 1046,
    'speedSet'   : 1052,
    'angleAct'   : 1064, # Use this to get finger position : angle Actual
    'forceAct'   : 1070,
    'errCode'    : 1082,
    'statusCode' : 1088,
    'temp'       : 1094,
    'ip'         : 1700,
    'actionSeq'  : 2160,
    'actionRun'  : 2162 
}

SRBL_INSPIRE_FINGER_LOWER_LIMIT = 900 # Lower limit of the finger joint position, units of 0.1 degrees
SRBL_INSPIRE_FINGER_UPPER_LIMIT = 1740 # Upper limit of the finger joint position, units of 0.1 degrees

SRBL_INSPIRE_FINGER_NUMBER = 4 # Number of the finger to control

class SRBL_Inspire_gripper:
    def __init__(self):
        self.ser = serial.Serial('/dev/ttyUSB1', 115200, timeout=0.1)

    def __del__(self):
        self.ser.close()
    
    def writeRegister(self, id, add, num, val):
        bytes = [0xEB, 0x90]            
        bytes.append(id)                
        bytes.append(num + 3)           
        bytes.append(0x12)              
        bytes.append(add & 0xFF)        
        bytes.append((add >> 8) & 0xFF) 
        for i in range(num):
            bytes.append(val[i])
        checksum = 0x00                 
        for i in range(2, len(bytes)):
            checksum += bytes[i]        
        checksum &= 0xFF                
        bytes.append(checksum)
        
        self.ser.write(bytes)                
        time.sleep(0.01)                
        self.ser.read_all() # flush the response
    
    def readRegister(self, id, add, num, mute=True):
        bytes = [0xEB, 0x90]            
        bytes.append(id)                
        bytes.append(0x04)              
        bytes.append(0x11)              
        bytes.append(add & 0xFF)        
        bytes.append((add >> 8) & 0xFF) 
        bytes.append(num)
        checksum = 0x00                 
        for i in range(2, len(bytes)):
            checksum += bytes[i]        
        checksum &= 0xFF                
        bytes.append(checksum)          
        
        # print("Writing:", [hex(b) for b in bytes])
        
        self.ser.write(bytes)           
        time.sleep(0.01)                
        recv = self.ser.read_all()      
        # print(recv)
        if len(recv) == 0:              
            return []
        num = (recv[3] & 0xFF) - 3      
        val = []
        for i in range(num):
            value = (recv[7 + i])
            val.append(value)
        if not mute:
            print('Read:', end='')
            for i in range(num):
                print(val[i], end=' ')
            print()
        return val

    def get_current_position(self):
        val = self.readRegister(1, INSPIRE_regdict['angleAct'], 12, True)
        if len(val) < 12:
            raise RuntimeError("Failed to read gripper position")
        val_act = []
        for i in range(6):
            value_act = val[i*2] + (val[i*2+1] << 8)
            if value_act > 32767:
                value_act -= 65536
            val_act.append(value_act)
        pos = float(val_act[SRBL_INSPIRE_FINGER_NUMBER - 1])
        pos = min(SRBL_INSPIRE_FINGER_UPPER_LIMIT, max(SRBL_INSPIRE_FINGER_LOWER_LIMIT, pos))
        pos = (pos - SRBL_INSPIRE_FINGER_LOWER_LIMIT) / (SRBL_INSPIRE_FINGER_UPPER_LIMIT - SRBL_INSPIRE_FINGER_LOWER_LIMIT) # normalize to [0, 1]
        return pos

    def move(self, target):
        target = target * (SRBL_INSPIRE_FINGER_UPPER_LIMIT - SRBL_INSPIRE_FINGER_LOWER_LIMIT) + SRBL_INSPIRE_FINGER_LOWER_LIMIT
        target = int(min(SRBL_INSPIRE_FINGER_UPPER_LIMIT, max(SRBL_INSPIRE_FINGER_LOWER_LIMIT, target)))
        targets = [-1, -1, -1, target, -1, -1]
        val_reg = []
        for i in range(6):
            val_reg.append(targets[i] & 0xFF)
            val_reg.append((targets[i] >> 8) & 0xFF)
        self.writeRegister(1, INSPIRE_regdict['angleSet'], 12, val_reg)

    def get_sensor_values(self):
        # cf) 25 ms delay in the sample code
        val = self.readRegister(1, 3000, 68, True)
        if len(val) < 68:
            raise RuntimeError("Failed to read gripper sensor data")
        idx = 10 * (SRBL_INSPIRE_FINGER_NUMBER - 1)
        normal_val = val[idx] + (val[idx + 1] << 8)
        tangential_val = val[idx + 2] + (val[idx + 3] << 8)
        tangential_dir = val[idx + 4] + (val[idx + 5] << 8)
        # Proximity NOT implemented yet
        if normal_val > 32767:
            normal_val -= 65536
        if tangential_val > 32767:
            tangential_val -= 65536
        if tangential_dir > 32767:
            tangential_dir -= 65536
        normal_val /= 100.0
        tangential_val /= 100.0
        return list(normal_val, tangential_val, tangential_dir)