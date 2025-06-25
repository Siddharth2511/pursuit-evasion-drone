# Pursuit-Evasion Simulation in AirSim (Thesis)

This repository contains the implementation of a vision-based pursuer drone that tracks an evader in 3D using monocular/depth based strategies. The code is part of Siddharth Anand's Final Thesis Project at IIT Bombay under the guidence of Prof. Debraj Chakraborty.

## Project Goals
- Simulate real-time drone pursuit in a 3D AirSim environment
- Depth estimate for chase using 2 sensing strategies
     - Strategy 1: monocular camera and drone width ratio
     - Strategy 2: using depth perspective airsim-API
- Maintain evader visibility using PID-based gimbal control
- Handle edge cases like gimbal lock with recovery logic

## Folder Structure
- `ACGC_3d*.py` – Main simulation scripts
- `drone_boundingbox.py` – Object localization utilities
- `requirements.txt` – Python packages used

## Requirements
- Python 3.8+
- AirSim + Unreal Engine (Windows)
- OpenCV, NumPy, matplotlib

## Running the Code
```bash
python ACGC_3d-depth_cam.py --EVADER_MOTION --TRACKING_METRICS --STRATEGY_FOR_DEPTH
```bash
EVADER_MOTIONS : --v_straight (Moves along vertical line) 
                 --circle (Moves in circular path with constant upward velocity [Spiral])
                 --depth (Moves into the plane [Depth axis])


TRACKING_METRICS : --plot (Plots angle error of heading vector and distance between evader and persuer)
                   --3d_track (3D Mapping  of chase)
(Can use both together)

STRATEGY_FOR_DEPTH : --strategy 1 (Monocular Camera with metric scaling based approach)
                     --strategy 2 (Depth camera based approach, Uses Airsim depth perspective API) [DEFAULT]
