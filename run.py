#!/usr/bin/env python
"""
Launcher script for the Roulette Prediction System.
"""

import os
import sys
import argparse


def main():
    """Main function to parse arguments and launch the appropriate script."""
    parser = argparse.ArgumentParser(
        description='Roulette Prediction System using Reinforcement Learning'
    )
    
    # Define subparsers for different actions
    subparsers = parser.add_subparsers(dest='action', help='Action to perform')
    
    # Train subparser
    train_parser = subparsers.add_parser('train', help='Train the prediction model')
    train_parser.add_argument(
        '--episodes', type=int, default=1000,
        help='Number of episodes to train for'
    )
    train_parser.add_argument(
        '--visualize', action='store_true',
        help='Visualize the training process'
    )
    train_parser.add_argument(
        '--visualize-interval', type=int, default=100,
        help='How often to visualize during training (every N episodes)'
    )
    train_parser.add_argument(
        '--fast', action='store_true',
        help='Use fast mode (skips some physics steps for faster training)'
    )
    
    # Test subparser
    test_parser = subparsers.add_parser('test', help='Test the prediction model on simulated data')
    test_parser.add_argument(
        '--tests', type=int, default=10,
        help='Number of tests to run'
    )
    test_parser.add_argument(
        '--visualize', action='store_true',
        help='Visualize the testing process'
    )
    
    # Predict subparser
    predict_parser = subparsers.add_parser('predict', help='Run live prediction on video feed')
    predict_parser.add_argument(
        '--source', type=str, default=None,
        help='Video source (webcam index or video file path)'
    )
    predict_parser.add_argument(
        '--record', action='store_true',
        help='Record the prediction session'
    )
    
    # Camera test subparser
    camera_parser = subparsers.add_parser('camera', help='Test camera setup')
    camera_parser.add_argument(
        '--source', type=str, default=None,
        help='Video source (webcam index or video file path)'
    )
    
    # Simulation subparser
    sim_parser = subparsers.add_parser('simulation', help='Run roulette wheel simulation')
    sim_parser.add_argument(
        '--wheel-speed', type=float, default=3.0,
        help='Initial wheel angular velocity (rad/s)'
    )
    sim_parser.add_argument(
        '--ball-speed', type=float, default=10.0,
        help='Initial ball angular velocity (rad/s)'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # If no action specified, show help
    if not args.action:
        parser.print_help()
        return
    
    # Execute the appropriate script based on the action
    if args.action == 'train':
        from src.train import train_model
        train_model(
            episodes=args.episodes,
            visualize_interval=args.visualize_interval if args.visualize else float('inf'),
            fast_mode=args.fast
        )
    
    elif args.action == 'test':
        from src.train import test_model
        test_model(
            num_tests=args.tests,
            visualize=args.visualize
        )
    
    elif args.action == 'predict':
        from src.predict import run_live_prediction
        # Convert source to integer if it's a digit string (webcam index)
        source = args.source
        if source and source.isdigit():
            source = int(source)
        run_live_prediction(source=source, record=args.record)
    
    elif args.action == 'camera':
        from src.vision.video_capture import test_camera
        # Convert source to integer if it's a digit string (webcam index)
        source = args.source
        if source and source.isdigit():
            source = int(source)
        test_camera(source)
    
    elif args.action == 'simulation':
        from src.simulation.roulette_sim import RouletteSimulation
        sim = RouletteSimulation()
        sim.reset(
            wheel_velocity=args.wheel_speed,
            ball_velocity=args.ball_speed
        )
        outcome, _ = sim.run_simulation(visualize=True)
        print(f"Simulation outcome: {outcome}")


if __name__ == "__main__":
    main() 