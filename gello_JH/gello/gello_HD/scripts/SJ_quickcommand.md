# Quick Command List for Execution
This file is a collection of the command line commands to run gello_HD. It only lists the commands to make the paste and copy easy. Refer to the README.md for specific details on how to use the commands.

```bash
export PYTHONPATH="$PYTHONPATH:$(pwd)"

python scripts/gello_get_offset.py \
    --start-joints 0 -1.57 1.57 -1.57 -1.57 0 \
    --joint-signs 1 1 -1 1 1 1 \
    --port /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA7NLK4-if00-port0

python experiments/launch_nodes.py --robot urT --robot_ip=192.168.10.2

python experiments/run_env.py --agent=gello --use_save_interface
```