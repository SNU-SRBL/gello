# GELLO with Force-Torque Sensor (`gello_ft`)

This folder contains a customized version of the [GELLO](https://wuphilipp.github.io/gello_site/) system developed by the SRBL lab, extended with 6-axis force-torque sensing and vision-based tactile sensing capabilities. It integrates the following hardware:

* UR5e Robot Arm
* Robotis Gripper (RH-P12-RN)
* Robotous 6-axis Force-Torque Sensor
* Digit Vision-based Tactile Sensor
* Intel RealSense Camera
* GELLO Dynamixel-based Teleoperation Device

---

## 🔧 System Setup

### 1. Required Hardware Connections

* Connect 5V power to the **Dynamixel Power Hub Board** to activate the GELLO device after connecting the **U2D2** USB device to your desktop.
* Power and connect the **FT sensor**, **DIGIT sensor**, and **camera** to the PC.
* Connect the **UR5e robot** via Ethernet.

---

### 2. Network Setup

Set the IP address of your **desktop** to match the UR robot’s subnet:

* **Robot IP**: `192.168.0.10`
* **Desktop IP example**: `192.168.0.1`

---

### 3. Identify Serial Devices

To verify that all USB devices (GELLO, gripper, FT sensor) are correctly recognized, run:
```bash
sudo dmesg -w
dmesg | grep tty
ls /dev/serial/by-id
```
The expected USB device assignments are:
| Device        | Port           | Notes                               |
| ------------- | -------------- | ----------------------------------- |
| **GELLO**     | `/dev/ttyUSB0` | U2D2 controller for Dynamixels      |
| **Gripper**   | `/dev/ttyUSB1` | Robotis RH-P12-RN gripper           |
| **FT Sensor** | `/dev/ttyUSB2` | Robotous 6-axis Force-Torque sensor |

Expected output (example):

```
/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA7NLK4-if00-port0
```
Use this path in the `--port` argument below.<br>
Verify the video index for **Camera** and **DIGIT sensor** with `sudo dmesg -w`.

---

## ⚙️ Software Setup

### 1. Conda Environment & Python Path

Activate your conda environment:

```bash
conda activate <your-conda-env>
```

Add the current directory to the `PYTHONPATH`:

```bash
cd gello/gello_ft/
export PYTHONPATH="$PYTHONPATH:$(pwd)"
```

---

### 2. Set Initial Robot Joint Position

Move the UR5e robot to the following known joint configuration before launching GELLO:

```
[0, -90, 90, -90, -90, 0]  (degrees)
```

Or in radians: `[0, -1.57, 1.57, -1.57, -1.57, 0]`

---

### 3. Run GELLO with FT Sensor

#### (1) Update Offset Before Use (Every Time)

```bash
python scripts/gello_get_offset.py \
    --start-joints 0 -1.57 1.57 -1.57 -1.57 0 \
    --joint-signs 1 1 -1 1 1 1 \
    --port /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA7NLK4-if00-port0
```

> ⚠️ After running, open `gello/agents/gello_agent.py` and update the printed offset values in the `PORT_CONFIG_MAP`.

---

#### (2) Camera Test (Optional)

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
python experiments/launch_camera_nodes.py --hostname 127.0.0.1

python experiments/launch_camera_clients.py --hostname 127.0.0.1
```
> ⚠️ Check the video port number and update the port number in `experiments/run_evn.py`.

---

#### (3) Launch GELLO

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
python experiments/launch_camera_nodes.py --hostname 127.0.0.1
```

```bash
python experiments/launch_nodes.py --robot ur --robot_ip=192.168.0.10
```

```bash
python experiments/run_env.py --agent=gello --use_save_interface --sensor_ft
```

