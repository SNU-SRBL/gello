"""
Modified from the original launch_nodes.py in gello repository.
Removed unused robots leaving only the UR robot for simplicity.
Modified by Seongjun Koh (Soft Robotics and Bionics Lab, Seoul National University)
"""

from dataclasses import dataclass
from pathlib import Path

import tyro

from gello.robots.robot import BimanualRobot, PrintRobot
from gello.zmq_core.robot_node import ZMQServerRobot


@dataclass
class Args:
    robot: str = "xarm"
    robot_port: int = 6001
    hostname: str = "127.0.0.1"
    robot_ip: str = "192.168.1.10"


def launch_robot_server(args: Args):
    port = args.robot_port
    if args.robot == "ur":
        from gello.robots.urHD import URRobot # May modify this part and the urHD file to enable the Inspire gripper
        robot = URRobot(robot_ip=args.robot_ip)
    else:
        raise NotImplementedError(
            f"Robot {args.robot} not implemented, choose one of: sim_ur, xarm, ur, bimanual_ur, none"
        )
    server = ZMQServerRobot(robot, port=port, host=args.hostname)
    print(f"Starting robot server on port {port}")
    server.serve()


def main(args):
    launch_robot_server(args)


if __name__ == "__main__":
    main(tyro.cli(Args))
