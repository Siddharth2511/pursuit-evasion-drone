import cosysairsim as airsim
import cv2
import numpy as np
import time
import math
import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import threading

class PIDController:
        def __init__(self, Kp, Ki, Kd, setpoint=0):
            self.Kp = Kp
            self.Ki = Ki
            self.Kd = Kd
            self.setpoint = setpoint
            self.prev_error = 0
            self.integral = 0
            self.last_time = time.time()

        def compute(self, measurement):
            current_time = time.time()
            dt = current_time - self.last_time
            error = self.setpoint - measurement
            self.integral += error * dt
            derivative = (error - self.prev_error) / dt if dt > 0 else 0

            output = (self.Kp * error) + (self.Ki * self.integral) + (self.Kd * derivative)

            self.prev_error = error
            self.last_time = current_time

            return output
        
e_x_vals, e_y_vals, time_vals = [], [], []
normalized_e_x_vals = []
normalized_e_y_vals = []
e_x_vals, e_y_vals = [], []
normalized_e_x_vals, normalized_e_y_vals = [], []

def live_plot_e_x():
    fig, ax = plt.subplots()
    line1, = ax.plot([], [], label='e_x', color='blue')

    ax.set_xlabel("Frame")
    ax.set_ylabel("Normalized Pixel Offset")
    ax.set_title("Live Plot of Horizontal Pixel Offset")
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 0.5)
    ax.legend()

    def init():
        return line1,

    def update(frame):
        if len(normalized_e_x_vals) == 0:
            return line1,

        line1.set_data(range(len(normalized_e_x_vals)), normalized_e_x_vals)

        if len(normalized_e_x_vals) <= 1000:
            ax.set_xlim(0, 1000)
        else:
            ax.set_xlim(len(normalized_e_x_vals) - 1000, len(normalized_e_x_vals))

        return line1,

    ani = animation.FuncAnimation(fig, update, init_func=init, blit=True, interval=100)
    plt.tight_layout()
    plt.show()

# Start the plot
threading.Thread(target=live_plot_e_x, daemon=True).start()

# Define vehicle names
EVADER = "Drone0"
PURSUER = "Drone1"
camera_name = "0"  # Front center camera

# Fns defining evader motion
def evader_motion_circle(client, angle, vel=1, vz=-1, omega=1, dt=0.1):
    """Move evader in a horizontal circle."""
    angle += omega * dt
    vx = vel * math.cos(angle)
    vy = vel * math.sin(angle)
    client.moveByVelocityAsync(vx, vy, vz, dt, vehicle_name=EVADER)
    return angle

def evader_motion_v_straight(client, vz=-1, dt=0.1):
    """Move evader straight up or down."""
    client.moveByVelocityAsync(0, 0, vz, dt, vehicle_name=EVADER)

def main(args):

    # Initialize the AirSim client
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.reset()
    # Reset camera orientation on pursuer
    client.simSetCameraPose(camera_name, airsim.Pose(), vehicle_name=PURSUER)
    

    # Enable API control and arm both drones
    client.enableApiControl(True, vehicle_name=EVADER)
    client.enableApiControl(True, vehicle_name=PURSUER)
    client.armDisarm(True, vehicle_name=EVADER)
    client.armDisarm(True, vehicle_name=PURSUER)

    # Take off both drones
    client.takeoffAsync(vehicle_name=EVADER).join()
    client.takeoffAsync(vehicle_name=PURSUER).join()

    ### Move the evader 

    client.moveToPositionAsync(12, 15, -8.5, 2, vehicle_name=EVADER)

    # Position the pursuer 
    client.moveToPositionAsync(-5, 10, -10, 2, vehicle_name=PURSUER)

    # Allow some time for the drones to stabilize
    time.sleep(10)

    # Camera and detection settings
    
    image_type = airsim.ImageType.Scene
    detection_radius_cm = 500 * 100  # 500 meters

    # Set detection radius for the pursuer's camera
    client.simSetDetectionFilterRadius(camera_name, image_type, detection_radius_cm, vehicle_name=PURSUER)

    # Add the evader drone's mesh name to the detection filter for the pursuer's camera
    client.simAddDetectionFilterMeshName(camera_name, image_type, EVADER, vehicle_name=PURSUER)

    
    # PID controllers for yaw and camera tilt
    yaw_pid = PIDController(Kp=0.5, Ki=0.02, Kd=0)
    tilt_pid = PIDController(Kp=0, Ki=0, Kd=0)


    angle = 0
    time_step = 0.1
    prev_time = time.time()

    while True:
        if args.circle:
            angle = evader_motion_circle(client, angle, vel=1.2, vz=0, omega=0.3, dt=time_step)

        elif args.v_straight:
            evader_motion_v_straight(client, vz=-1.0, dt=time_step)


        #time.sleep(time_step)

        # Capture image from the pursuer's camera
        raw_image = client.simGetImage(camera_name, image_type, vehicle_name=PURSUER)
        
        if raw_image is None:
            print("Failed to capture image from the pursuer's camera.")
            continue
        

        # Retrieve detections from the pursuer's camera
        detections = client.simGetDetections(camera_name, image_type, vehicle_name=PURSUER)
        drone_detected = False #Flag
        
        tilt_angle = 0 # Tilt angle init
      
        
        # Process detections
        for detection in detections:
            if detection.name == EVADER:
                print("Detected Enemy!")
                drone_detected = True

                bbox_min = detection.box2D.min
                bbox_max = detection.box2D.max
                x_min, y_min = int(bbox_min.x_val), int(bbox_min.y_val)
                x_max, y_max = int(bbox_max.x_val), int(bbox_max.y_val)
                
                # Decode the image
                np_img = np.frombuffer(raw_image, dtype=np.uint8)
                img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

                y_c = img.shape[0] / 2 # height
                x_c = img.shape[1] / 2 # width
                    
                print(f"x_c:{x_c}, y_c:{y_c}")
                focal_len = img.shape[0] / 2 #FOV == 90 deg

                # Draw bounding box with red color and thickness of 1
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 1)
                
                x_e = (x_min + x_max) // 2
                y_e = (y_min + y_max) // 2
                
                e_x = x_c - x_e 
                e_x_vals.append(abs(e_x))
                normalized_e_x_vals.append(abs(e_x) / (256 + len(e_x_vals)))

                e_y = y_e - y_c
                e_y_vals.append(abs(e_y))
                normalized_e_y_vals.append(abs(e_y) / 144)

                yaw_rate = yaw_pid.compute(e_x)
                tilt_rate = tilt_pid.compute(e_y)

                # Get loop time delta
                current_time = time.time()
                dt = current_time - prev_time
                prev_time = current_time

                orientation = client.simGetCameraInfo(camera_name, vehicle_name=PURSUER).pose.orientation
                tilt_angle = airsim.quaternion_to_euler_angles(orientation)[1]
                tilt_angle += tilt_rate*dt

                # Camera Pose to be fed to SetCameraPose Function
                camera_pose = airsim.Pose(
                    airsim.Vector3r(0, 0, 0),
                    airsim.euler_to_quaternion(0, tilt_angle, 0)  # tilt = pitch
                )
                
                print(f"tilt_angle: {tilt_angle}, yaw_rate: {yaw_rate}" )
                
                # Apply control inputs
                client.moveByVelocityAsync(0, 0, 0, duration=1, yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate), vehicle_name=PURSUER)
                client.simSetCameraPose(camera_name, camera_pose, vehicle_name=PURSUER)


                # Display coordinates
                # print(f"Bounding Box: ({x_min}, {y_min}) to ({x_max}, {y_max})")
                print(f"Evader Coordinates: ({x_e}, {y_e})")

        if not drone_detected:
            print("Evader not detected. Exiting program.")
            # sys.exit()

        # Display the image
        cv2.namedWindow("Pursuer's View", cv2.WINDOW_KEEPRATIO)
        cv2.imshow("Pursuer's View", img)
        cv2.resizeWindow("Pursuer's View", 384, 216)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cv2.destroyAllWindows()
    client.armDisarm(False, vehicle_name=EVADER)
    client.armDisarm(False, vehicle_name=PURSUER)
    client.enableApiControl(False, vehicle_name=EVADER)
    client.enableApiControl(False, vehicle_name=PURSUER)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--circle", action="store_true", help="Evader moves in circular path")
    parser.add_argument("--v_straight", action="store_true", help="Evader moves in vertical line")
    args = parser.parse_args()
    main(args)