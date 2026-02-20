import math
import time
import rtde_control
import rtde_receive

def deg2rad(deg_list):
    return [math.radians(d) for d in deg_list]

if __name__ == "__main__":
    # Replace with your robot's IP
    robot_ip = "192.168.0.1"
    
    # Connect to robot
    rtde_c = rtde_control.RTDEControlInterface(robot_ip)
    rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
    # Define joint target in degrees
    target_deg = [0, -90, 90, -90, -90, 0]
    
    # Convert to radians
    target_rad = deg2rad(target_deg)
    
    # Use moveJ with default velocity and acceleration
    velocity = 0.5      # rad/s
    acceleration = 0.5  # rad/s²
    # rtde_c.moveJ(target_rad, velocity, acceleration)
    target_pose = [-0.12260326491614308, -0.520207445205494, 0.1321324828699519, -0.4659463465803204, -3.1025569071791397, -0.03652051729753818]
    # rtde_c.moveL(target_pose, speed=0.03)

    print("Target joint moveJ command sent.")
    current_pose = rtde_r.getActualTCPPose() 
    print(current_pose)
    time.sleep(1)
    
    # Disconnect
    rtde_c.servoStop()
    rtde_c.disconnect()
