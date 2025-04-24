"""
Video capture and processing module for roulette prediction system.
"""

import cv2
import numpy as np
import sys
import os

# Add the project root to the path to be able to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src import config

class VideoCapture:
    """Handles video capture and basic processing."""
    
    def __init__(self, source=None):
        """Initialize the video capture with the given source.
        
        Args:
            source: Camera index or video file path. Default is from config.
        """
        self.source = source if source is not None else config.VIDEO_SOURCE
        self.cap = None
        self.frame_width = config.FRAME_WIDTH
        self.frame_height = config.FRAME_HEIGHT
        self.fps = config.FPS
    
    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
    
    def open(self):
        """Open the video capture."""
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise Exception(f"Failed to open video source: {self.source}")
        
        # Set properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        return self
    
    def read(self):
        """Read a frame from the video source.
        
        Returns:
            tuple: (success, frame) where success is a boolean and frame is the image
        """
        if self.cap is None:
            self.open()
        return self.cap.read()
    
    def release(self):
        """Release the video capture resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def get_fps(self):
        """Get the FPS of the video source."""
        if self.cap is None:
            self.open()
        return self.cap.get(cv2.CAP_PROP_FPS)


def test_camera(source=None):
    """Test the camera capture functionality.
    
    Args:
        source: Camera index or video file path. Default is from config.
    """
    with VideoCapture(source) as cap:
        print(f"Testing camera with source: {cap.source}")
        print("Press 'q' to exit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Display the resulting frame
            cv2.imshow('Camera Test', frame)
            
            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cv2.destroyAllWindows()
    print("Camera test completed")


if __name__ == "__main__":
    # If run directly, parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Test camera capture')
    parser.add_argument('--source', type=str, default=None,
                        help='Camera index or video file path')
    
    args = parser.parse_args()
    
    # Convert source to integer if it's a digit string (webcam index)
    if args.source and args.source.isdigit():
        args.source = int(args.source)
    
    test_camera(args.source) 