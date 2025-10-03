import cv2
import numpy as np
from typing import Optional, Tuple

from gello.cameras.camera import CameraDriver


class DIGITCamera(CameraDriver):
    """DIGIT tactile sensor camera driver that implements the CameraDriver protocol."""
    
    def __init__(self, camera_index: int = 0, frame_width: int = 320, frame_height: int = 240):
        """
        Initialize DIGIT camera
        
        Args:
            camera_index: USB camera index for the DIGIT sensor
            frame_width: Width of captured frames
            frame_height: Height of captured frames
        """
        self.camera_index = camera_index
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Initialize camera
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open DIGIT camera at index {camera_index}")
            
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"DIGIT Camera initialized at index {camera_index} ({frame_width}x{frame_height})")
    
    def read(self, img_size: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read a frame from the DIGIT camera
        
        Args:
            img_size: Optional target size for the image. If None, original size is returned.
            
        Returns:
            Tuple of (color_image, depth_image). 
            For DIGIT sensors, depth_image is a dummy array since DIGIT provides only RGB.
        """
        ret, frame = self.cap.read()
        
        if not ret:
            # Return black frames if read fails
            if img_size is not None:
                h, w = img_size
            else:
                h, w = self.frame_height, self.frame_width
            
            color_img = np.zeros((h, w, 3), dtype=np.uint8)
            depth_img = np.zeros((h, w, 1), dtype=np.uint16)
            return color_img, depth_img
        
        # Convert BGR to RGB for consistency with other cameras
        color_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize if requested
        if img_size is not None:
            color_img = cv2.resize(color_img, (img_size[1], img_size[0]))
        
        # Create dummy depth image (DIGIT sensors don't provide depth)
        height, width = color_img.shape[:2]
        depth_img = np.zeros((height, width, 1), dtype=np.uint16)
        
        return color_img, depth_img
    
    def close(self):
        """Release the camera"""
        if self.cap:
            self.cap.release()
    
    def __del__(self):
        """Ensure camera is released when object is destroyed"""
        self.close()


def get_digit_camera_indices(max_check: int = 10) -> list:
    """
    Get available camera indices that could be DIGIT sensors
    
    Args:
        max_check: Maximum number of camera indices to check
        
    Returns:
        List of available camera indices
    """
    available = []
    for i in range(max_check):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available


if __name__ == "__main__":
    # Test the DIGIT camera driver
    print("Available cameras:", get_digit_camera_indices())
    
    digit_cam = None
    try:
        # Test with camera index 0 (adjust as needed)
        digit_cam = DIGITCamera(camera_index=0)
        
        print("Press 'q' to quit")
        while True:
            color_img, depth_img = digit_cam.read()
            
            # Display the color image
            display_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
            cv2.imshow('DIGIT Camera Test', display_img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except RuntimeError as e:
        print(f"Error: {e}")
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        if digit_cam is not None:
            digit_cam.close()
        cv2.destroyAllWindows()