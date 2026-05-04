"""
Probe the Inspire gripper at common baudrates to diagnose silent-bus problems.

Sends a readRegister request for the angleAct register at each baudrate and
reports how many bytes come back. If every baud is silent -> power/wiring.
If exactly one responds -> the gripper's baudrate register is set to that.
"""
import argparse
import serial
import time

from gello.robots.SRBL_Inspire import INSPIRE_regdict

BAUDRATES = [115200, 57600, 19200, 921600]
GRIPPER_ID = 1
EXPECTED_LEN = 12 + 8  # 12 data bytes (6 fingers * int16) + 8 frame overhead


def build_read_frame(id_, add, num):
    frame = [0xEB, 0x90, id_, 0x04, 0x11, add & 0xFF, (add >> 8) & 0xFF, num]
    checksum = sum(frame[2:]) & 0xFF
    frame.append(checksum)
    return bytes(frame)


def probe(device, baud, settle=0.1, timeout=0.2):
    ser = serial.Serial(device, baud, timeout=timeout)
    time.sleep(settle)
    ser.reset_input_buffer()
    frame = build_read_frame(GRIPPER_ID, INSPIRE_regdict['angleAct'], 12)
    ser.write(frame)
    ser.flush()
    recv = ser.read(EXPECTED_LEN)
    ser.close()
    return recv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("device", type=str, help="Serial port, e.g. /dev/ttyUSB1")
    parser.add_argument("--repeat", type=int, default=2, help="Probes per baud")
    args = parser.parse_args()

    print(f"Probing {args.device} (gripper ID={GRIPPER_ID}, expecting {EXPECTED_LEN} bytes)")
    print("-" * 60)
    any_response = False
    for baud in BAUDRATES:
        best = b""
        for _ in range(args.repeat):
            try:
                recv = probe(args.device, baud)
            except serial.SerialException as e:
                print(f"  {baud:>7}  ERROR  {e}")
                best = None
                break
            if len(recv) > len(best):
                best = recv
        if best is None:
            continue
        status = "OK" if len(best) >= EXPECTED_LEN else ("PARTIAL" if best else "silent")
        if best:
            any_response = True
        head = best[:12].hex(" ") if best else "-"
        print(f"  {baud:>7}  {len(best):>2} bytes  {status:<7}  {head}")

    print("-" * 60)
    if not any_response:
        print("No response at any baud. Check 24 V power and RS485 A/B wiring.")
    else:
        print("Got bytes. The baud(s) above match the gripper's current setting.")


if __name__ == "__main__":
    main()
