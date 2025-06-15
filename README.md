# Pursuit-Evasion Simulation in AirSim (DDP)

This repository contains the final implementation of a vision-based pursuer drone that tracks an evader in 3D using monocular/depth/LiDAR-based strategies. The code is part of Siddharth Anand's Dual Degree Project (DDP2) at IIT Bombay.

## Project Goals
- Simulate real-time drone pursuit in a 3D AirSim environment
- Depth estimate for chase using 2 sensing strategies
     - Strategy 1: monocular camera and drone width ratio
     - Strategy 2: using depth perspective airsim-API
- Maintain evader visibility using PID-based gimbal control
- Handle edge cases like gimbal lock with recovery logic

## Folder Structure
- `/controllers_3d/` – Control logic for 3D chase
- `ACGC_3d*.py` – Main simulation scripts
- `drone_boundingbox.py` – Object localization utilities
- `drone_positions.xlsx` – Drone spawn layout
- `requirements.txt` – Python packages used

## Requirements
- Python 3.8+
- AirSim + Unreal Engine (Windows)
- OpenCV, NumPy, matplotlib

## Running the Code
```bash
python ACGC_3d-depth_cam.py
