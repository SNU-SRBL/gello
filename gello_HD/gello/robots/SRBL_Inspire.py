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
    'angleSet'   : 1040, # Use this to set finger position, units of 0.1 degrees, -1 for no change
    'forceSet'   : 1046,
    'speedSet'   : 1052,
    'angleAct'   : 1064, # Use this to get finger position : angle Actual, units of 0.1 degrees
    'forceAct'   : 1070,
    'currAct'    : 1076, # Use this to get current data, mA
    'errCode'    : 1082,
    'statusCode' : 1088,
    'temp'       : 1094,
    'ip'         : 1700,
    'actionSeq'  : 2160,
    'actionRun'  : 2162, 
    'sensorData' : 3000
}

SRBL_INSPIRE_FINGER_NUMBER = 4 # Number of the finger to control
# little, ring, middle, index, thumb bending, thumb rotation

SRBL_INSPIRE_FINGER_LOWER_LIMIT = [900, 900, 900, 900, 1100, 600] # Lower limit of the finger joint position, units of 0.1 degrees
# 4 fingers : 900, thunmb bending : 1100, thumb rotation : 600
SRBL_INSPIRE_FINGER_UPPER_LIMIT = [1740, 1740, 1740, 1740, 1350, 1800] # Upper limit of the finger joint position, units of 0.1 degrees
# 4 fingers : 1740, thunmb bending : 1350, thumb rotation : 1800

class SRBL_Inspire_gripper:
    def __init__(self, finger=SRBL_INSPIRE_FINGER_NUMBER, device_name="/dev/ttyUSB1", baudrate=115200):
        self.ser = serial.Serial(device_name, baudrate, timeout=0.1)
        self.finger = finger
        self.upper_limit = SRBL_INSPIRE_FINGER_UPPER_LIMIT[finger - 1]
        self.lower_limit = SRBL_INSPIRE_FINGER_LOWER_LIMIT[finger - 1]
        self._SRBL_initialize()

    def __del__(self):
        if self.ser and self.ser.is_open:
            self.ser.close()

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
    
    def _writeRegister(self, id, add, num, val):
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
    
    def _readRegister(self, id, add, num, mute=True):
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
    
    def _SRBL_bytes_to_int16(self, val):
        if len(val) < 2:
            raise ValueError("Not enough bytes to convert to int16")
        value = val[0] + (val[1] << 8)
        if value > 32767:
            value -= 65536
        return value

    # region Control with full parameters
    def get_current_position_full(self):
        val = self._readRegister(1, INSPIRE_regdict['angleAct'], 12, True)
        if len(val) < 12:
            raise RuntimeError("Failed to read gripper position")
        val_act = []
        for i in range(6):
            value_act = self._SRBL_bytes_to_int16(val[i*2:(i*2)+2])
            val_act.append(value_act)
        pos = float(val_act[self.finger - 1])
        pos = min(self.upper_limit, max(self.lower_limit, pos))
        pos = (pos - self.lower_limit) / (self.upper_limit - self.lower_limit) # normalize to [0, 1]
        return pos

    def move_full(self, target):
        target = target * (self.upper_limit - self.lower_limit) + self.lower_limit
        target = int(min(self.upper_limit, max(self.lower_limit, target)))
        targets = [-1, -1, -1, target, -1, -1]
        val_reg = []
        for i in range(6):
            val_reg.append(targets[i] & 0xFF)
            val_reg.append((targets[i] >> 8) & 0xFF)
        self._writeRegister(1, INSPIRE_regdict['angleSet'], 12, val_reg)

    def get_sensor_values_full(self):
        # cf) 25 ms delay in the sample code
        val = self._readRegister(1, INSPIRE_regdict['sensorData'], 68, True) # finger 5*10 + palm 3*6
        if len(val) < 68:
            raise RuntimeError("Failed to read gripper sensor data")
        idx = 10 * (self.finger - 1)
        normal_val = self._SRBL_bytes_to_int16(val[idx:idx+2])
        tangential_val = self._SRBL_bytes_to_int16(val[idx+2:idx+4])
        tangential_dir = self._SRBL_bytes_to_int16(val[idx+4:idx+6])
        # Proximity NOT implemented yet
        normal_val /= 100.0
        tangential_val /= 100.0
        return list(normal_val, tangential_val, tangential_dir)
    
    def get_current_values_full(self):
        val = self._readRegister(1, INSPIRE_regdict['currentData'], 12, True)
        if len(val) < 12:
            raise RuntimeError("Failed to read gripper current data")
        idx = 2 * (self.finger - 1)
        current_val = self._SRBL_bytes_to_int16(val[idx:idx+2])
        current_val /= 1000.0
        return current_val

    def get_velocity_values_full(self):
        """
        Inspire does not provide velocity data, so this function is not implemented.
        If needed, velocity can be estimated by numerical differentiation of position data.
        """
        pass

    def get_position_values_full(self):
        pass

    def get_observation_values_full(self):
        pass
    # endregion

    # region Control one joint at a time

    def _SRBL_initialize(self):
        # Initialize gripper to initial pose
        targets = [-1] * 6
        targets[self.finger - 1] = self.upper_limit
        val_reg = []
        for i in range(6):
            val_reg.append(targets[i] & 0xFF)
            val_reg.append((targets[i] >> 8) & 0xFF)
        self._writeRegister(1, INSPIRE_regdict['angleSet'], 12, val_reg)
    
    def get_current_position(self):
        val = self._readRegister(1, INSPIRE_regdict['angleAct'] + (self.finger - 1), 2, True)
        if len(val) < 2:
            raise RuntimeError("Failed to read gripper position")
        value_act = self._SRBL_bytes_to_int16(val)
        pos = float(value_act)
        pos = min(self.upper_limit, max(self.lower_limit, pos))
        pos = (pos - self.lower_limit) / (self.upper_limit - self.lower_limit) # normalize to [0, 1]
        return pos
    
    def move(self, target):
        target = target * (self.upper_limit - self.lower_limit) + self.lower_limit
        target = int(min(self.upper_limit, max(self.lower_limit, target)))
        val_reg = [target & 0xFF, (target >> 8) & 0xFF]
        self._writeRegister(1, INSPIRE_regdict['angleSet'] + (self.finger - 1), 2, val_reg)
    
    def get_sensor_values(self):
        val = self._readRegister(1, INSPIRE_regdict['sensorData'] + (self.finger - 1) * 10, 6, True)
        if len(val) < 6:
            raise RuntimeError("Failed to read gripper sensor data")
        normal_val = self._SRBL_bytes_to_int16(val[0:2])
        tangential_val = self._SRBL_bytes_to_int16(val[2:4])
        tangential_dir = self._SRBL_bytes_to_int16(val[4:6])
        # Proximity NOT implemented yet
        normal_val /= 100.0
        tangential_val /= 100.0
        return list(normal_val, tangential_val, tangential_dir)
    
    def get_current_values(self):
        val = self._readRegister(1, INSPIRE_regdict['currAct'] + (self.finger - 1), 2, True)
        if len(val) < 2:
            raise RuntimeError("Failed to read gripper current data")
        value_act = self._SRBL_bytes_to_int16(val)
        current_val = value_act / 1000.0
        return current_val
    
    def get_velocity_values(self):
        """
        Inspire does not provide velocity data, so this function is not implemented.
        If needed, velocity can be estimated by numerical differentiation of position data.
        """
        pass

    def get_position_values(self):
        val = self._readRegister(1, INSPIRE_regdict['angleAct'] + (self.finger - 1), 2, True)
        if len(val) < 2:
            raise RuntimeError("Failed to read gripper position")
        value_act = self._SRBL_bytes_to_int16(val)
        pos = float(value_act)
        pos /= 10.0
        return pos

    def get_observation_values(self):
        observation = {}
        observation['position'] = self.get_position_values()
        observation['current'] = self.get_current_values()
        observation['sensor'] = self.get_sensor_values
        return observation
    # endregion

    
    
    
    
