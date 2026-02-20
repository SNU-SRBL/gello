# Gello Agent GUIDE

## Check usb
dmesg | grep tty
ls /dev/serial/by-id
# usb-FTDI_USB__-__Serial_Converter_FTA7NLK4-if00-port0

## Add python path
# cd ~/gello_ft/
export PYTHONPATH="$PYTHONPATH:$(pwd)"

## Check gello offset (Must to Everytime when you start again)
python scripts/gello_get_offset.py \
    --start-joints 0 -1.57 1.57 -1.57 -1.57 0 \
    --joint-signs 1 1 -1 1 1 1 \
    --port /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA7NLK4-if00-port0
# and then Go to ~/agents/gello_agents.py
# change the offset in the python file

# Camera Test
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
python experiments/launch_camera_nodes.py --hostname 127.0.0.1

python experiments/launch_camera_clients.py --hostname 127.0.0.1

## Run gello
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
python experiments/launch_camera_nodes.py --hostname 127.0.0.1

python experiments/launch_nodes.py --robot ur --robot_ip=192.168.0.10

python experiments/run_env.py --agent=gello --use_save_interface --sensor_ft