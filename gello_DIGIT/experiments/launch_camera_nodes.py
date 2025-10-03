from dataclasses import dataclass
from multiprocessing import Process
from typing import Tuple

import tyro

from gello.cameras.realsense_camera import RealSenseCamera, get_device_ids
from gello.cameras.digit_camera import DIGITCamera, get_digit_camera_indices
from gello.zmq_core.camera_node import ZMQServerCamera


@dataclass
class Args:
    # hostname: str = "127.0.0.1"
    hostname: str = "128.32.175.167"
    digit_indices: Tuple[int, ...] = (0, 2)  # Default DIGIT camera indices
    use_digit: bool = True  # Enable DIGIT cameras by default

def launch_server(port: int, camera_id: int, args: Args):
    camera = RealSenseCamera(camera_id)
    server = ZMQServerCamera(camera, port=port, host=args.hostname)
    print(f"Starting camera server on port {port}")
    server.serve()

def launch_digit_server(port: int, camera_index: int, args: Args):
    camera = DIGITCamera(camera_index)
    server = ZMQServerCamera(camera, port=port, host=args.hostname)
    print(f"Starting DIGIT camera server on port {port} for camera index {camera_index}")
    server.serve()

def main(args):
    ids = get_device_ids()
    camera_port = 5000
    camera_servers = []
    for camera_id in ids:
        # start a python process for each camera
        print(f"Launching camera {camera_id} on port {camera_port}")
        camera_servers.append(
            Process(target=launch_server, args=(camera_port, camera_id, args))
        )
        camera_port += 1

    # Launch DIGIT cameras if enabled
    if args.use_digit:
        available_digit_indices = get_digit_camera_indices()
        print(f"Available camera indices: {available_digit_indices}")
        
        for digit_index in args.digit_indices:
            if digit_index in available_digit_indices:
                print(f"Launching DIGIT camera {digit_index} on port {camera_port}")
                camera_servers.append(
                    Process(target=launch_digit_server, args=(camera_port, digit_index, args))
                )
                camera_port += 1
            else:
                print(f"Warning: DIGIT camera index {digit_index} not available")

    for server in camera_servers:
        server.start()


if __name__ == "__main__":
    main(tyro.cli(Args))
