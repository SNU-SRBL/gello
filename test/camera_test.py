import pyrealsense2 as rs
import numpy as np
import cv2

# 1. List connected devices and get serial numbers
ctx = rs.context()
connected_devices = ctx.query_devices()
serials = [d.get_info(rs.camera_info.serial_number) for d in connected_devices]

for device in ctx.query_devices():
    print("Device:", device.get_info(rs.camera_info.name))
    print("Serial Number:", device.get_info(rs.camera_info.serial_number))

if len(serials) < 2:
    raise RuntimeError("Less than 2 RealSense devices connected")

# 2. Create pipelines and configs for each device
pipelines = []
for serial in serials[:2]:  # Use the first two cameras
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    pipelines.append(pipeline)

try:
    while True:
        frames = []
        for pipeline in pipelines:
            frameset = pipeline.wait_for_frames()
            color_frame = frameset.get_color_frame()
            if not color_frame:
                frames.append(None)
            else:
                color_image = np.asanyarray(color_frame.get_data())
                frames.append(color_image)

        # 3. Display both frames side by side
        if all(f is not None for f in frames):
            both = np.hstack(frames)
            cv2.imshow('Dual RealSense RGB', both)

        if cv2.waitKey(1) == 27:  # ESC
            break

finally:
    for pipeline in pipelines:
        pipeline.stop()
    cv2.destroyAllWindows()
