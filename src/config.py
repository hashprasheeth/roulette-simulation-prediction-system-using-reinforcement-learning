"""
Configuration settings for the Roulette Prediction system.
"""

# Video capture settings
VIDEO_SOURCE = 0  # 0 for webcam, or path to video file
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS = 30

# Vision processing parameters
BLUR_KERNEL_SIZE = (5, 5)
CANNY_THRESHOLD1 = 50
CANNY_THRESHOLD2 = 150
HOUGH_CIRCLE_DP = 1.2
HOUGH_CIRCLE_MIN_DIST = 50
HOUGH_CIRCLE_PARAM1 = 50
HOUGH_CIRCLE_PARAM2 = 30
MIN_WHEEL_RADIUS = 200
MAX_WHEEL_RADIUS = 400
MIN_BALL_RADIUS = 5
MAX_BALL_RADIUS = 15

# RL model parameters
STATE_SIZE = 10  # Number of state parameters
ACTION_SIZE = 37  # 0-36 for European roulette
LEARNING_RATE = 0.001
DISCOUNT_FACTOR = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 64
MEMORY_SIZE = 10000
UPDATE_TARGET_EVERY = 5

# Paths
MODEL_SAVE_PATH = "models/roulette_model"
TRAINING_DATA_PATH = "data/training_data"
LOG_DIR = "logs"

# Training parameters
EPISODES = 1000
MAX_STEPS_PER_EPISODE = 200
SAVE_MODEL_EVERY = 100 