import cosysairsim as airsim
import cv2
import numpy as np
import time
import math
import argparse
import matplotlib.pyplot as plt
from collections import deque
from mpl_toolkits.mplot3d import Axes3D
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


# vehicle names
EVADER = "Drone0"
PURSUER = "Drone1"
camera_name = "0"  # Front center camera

### Fn for live plots ###

angle_errors_plot = deque(maxlen=100)
true_depths_plot = deque(maxlen=100)
pursuer_positions = []
evader_positions = []

## compares ANGLE of (visually) estimated and true heading vector AND true distance ||true_dg||
def live_plot_dual():
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    
    line1, = ax1.plot([], [], 'r-', label='Angle Error')
    ax1.set_ylim(0, 30)
    ax1.set_xlim(0, 100)
    ax1.set_ylabel("Angle Error (deg)")
    ax1.set_title("Live Angle Error")

    line2, = ax2.plot([], [], 'b-', label='True Depth')
    ax2.set_ylim(0, 50)  # adjust as needed
    ax2.set_xlim(0, 100)
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("True Distance to Evader")
    ax2.set_title("Live True Distance")

    plt.tight_layout()
    plt.legend()

    while True:
        if len(angle_errors_plot) > 1:
            line1.set_ydata(angle_errors_plot)
            line1.set_xdata(range(len(angle_errors_plot)))
            ax1.relim()
            ax1.autoscale_view()

            line2.set_ydata(true_depths_plot)
            line2.set_xdata(range(len(true_depths_plot)))
            ax2.relim()
            ax2.autoscale_view()

            fig.canvas.draw()
            fig.canvas.flush_events()

### 3D plot of evader (blue) and pursuer (red) motions in XYZ coordinates
def live_plot_3d():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Real-Time 3D Trajectories")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    pursuer_line, = ax.plot([], [], [], label='Pursuer', color='blue')
    evader_line, = ax.plot([], [], [], label='Evader', color='red')
    ax.legend()

    def init():
        return pursuer_line, evader_line

    def update(frame):
        if len(pursuer_positions) == 0 or len(evader_positions) == 0:
            return pursuer_line, evader_line

        px, py, pz = zip(*pursuer_positions)
        ex, ey, ez = zip(*evader_positions)

        pursuer_line.set_data(px, py)
        pursuer_line.set_3d_properties(pz)

        evader_line.set_data(ex, ey)
        evader_line.set_3d_properties(ez)

        ax.set_xlim(min(px+ex), max(px+ex))
        ax.set_ylim(min(py+ey), max(py+ey))
        ax.set_zlim(min(pz+ez), max(pz+ez))

        return pursuer_line, evader_line

    ani = animation.FuncAnimation(fig, update, init_func=init, blit=False, interval=100)
    plt.tight_layout()
    plt.show()

# Fns defn for Math involved

def R_y(theta):
    """Rotation matrix about the y-axis (pitch)."""
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    return np.array([
        [ cos_t, 0, sin_t],
        [     0, 1,     0],
        [-sin_t, 0, cos_t]
    ])

def quaternion_to_rotation_matrix(q):
    w, x, y, z = q.w_val, q.x_val, q.y_val, q.z_val
    return np.array([
        [1 - 2*(y**2 + z**2),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x**2 + z**2),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
    ])


# Fns defining evader motion
def evader_motion_circle(client, angle, vel=1, vz=-1, omega=1, dt=0.1):
    """Move evader in a horizontal circle, with vertical velocity."""
    angle += omega * dt
    vx = vel * math.cos(angle)
    vy = vel * math.sin(angle)
    client.moveByVelocityAsync(vx, vy, vz, dt, vehicle_name=EVADER)
    return angle

def evader_motion_v_straight(client, vz=-1, dt=0.1):
    """Move evader straight up or down."""
    client.moveByVelocityAsync(0, 0, vz, dt, vehicle_name=EVADER)

def evader_motion_depth(client, vx=5, vy = 0, dt=0.1):
    """Move evader straight up or down."""
    client.moveByVelocityAsync(vx, vy, 0, dt, vehicle_name=EVADER)

### STRATEGIES 1, 2 are monochrome_cam, depth_cam ###


def main(args, strategy):

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

    client.moveToPositionAsync(18, 5, -6, 2, vehicle_name=EVADER)

    # Position the pursuer 
    client.moveToPositionAsync(0, 0, -10, 2, vehicle_name=PURSUER)


    # Allow some time for the drones to stabilize
    time.sleep(15)

    if args.plot:
        threading.Thread(target=live_plot_dual, daemon=True).start()

    ### Check threading
    # Camera and detection settings
    
    image_type = airsim.ImageType.Scene
    detection_radius_cm = 500 * 100  # 500 meters

    # Set detection radius for the pursuer's camera
    client.simSetDetectionFilterRadius(camera_name, image_type, detection_radius_cm, vehicle_name=PURSUER)

    # Add the evader drone's mesh name to the detection filter for the pursuer's camera
    client.simAddDetectionFilterMeshName(camera_name, image_type, EVADER, vehicle_name=PURSUER)

    
    # PID controllers for yaw and camera tilt
    yaw_pid = PIDController(Kp=1, Ki=0.5, Kd=0)
    tilt_pid = PIDController(Kp=0.02, Ki=0.002, Kd=0)


    angle = 0
    time_step = 0.1
    prev_time = time.time()

    
    # ----------------------------------------------------

    while True:
        if args.circle:
            angle = evader_motion_circle(client, angle, vel=1.5, vz=-0.5, omega=2, dt=time_step)

        elif args.v_straight:
            evader_motion_v_straight(client, vz=-1, dt=time_step)
        
        elif args.depth:
            evader_motion_depth(client, vx = 1, vy = 0, dt=time_step)

        # Capture image from the pursuer's camera
        raw_image = client.simGetImage(camera_name, image_type, vehicle_name=PURSUER)
        
        if raw_image is None:
            print("Failed to capture image from the pursuer's camera.")
            continue

        # Retrieve detections from the pursuer's camera
        detections = client.simGetDetections(camera_name, image_type, vehicle_name=PURSUER)
        gimbal_recovery_mode = False
      
        # Process detections
        for detection in detections:
            if detection.name == EVADER:
                print("Detected Enemy!")


                bbox_min = detection.box2D.min
                bbox_max = detection.box2D.max
                x_min, y_min = int(bbox_min.x_val), int(bbox_min.y_val)
                x_max, y_max = int(bbox_max.x_val), int(bbox_max.y_val)
                
                # Decode the image
                np_img = np.frombuffer(raw_image, dtype=np.uint8)
                img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
                
                # Draw bounding box with red color and thickness of 1
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 1)
                
                #Extracion of useful orientation(s) using Airsim APIs
                camera_info = client.simGetCameraInfo(camera_name, vehicle_name=PURSUER)
                q_cam = camera_info.pose.orientation
                position_cam = camera_info.pose.position
                tilt_angle_cam = airsim.quaternion_to_euler_angles(q_cam)[1]

                q = client.getMultirotorState(vehicle_name=PURSUER).kinematics_estimated.orientation
                pursuer_pos = client.getMultirotorState(vehicle_name=PURSUER).kinematics_estimated.position
                evader_pos = client.getMultirotorState(vehicle_name=EVADER).kinematics_estimated.position

                ##Subtract initial offsets for correct plotting of 3D trajectories
                pursuer_positions.append((abs(pursuer_pos.x_val), abs(pursuer_pos.y_val), abs(pursuer_pos.z_val)))
                evader_positions.append((abs(evader_pos.x_val)-10, abs(evader_pos.y_val), abs(evader_pos.z_val))) 

                # Compute true global direction from pursuer to evader (only for comparision with estimated heading vector)
                true_vec = np.array([
                    [evader_pos.x_val - pursuer_pos.x_val],
                    [evader_pos.y_val - pursuer_pos.y_val],
                    [evader_pos.z_val - pursuer_pos.z_val]
                ])
                true_vec = true_vec.flatten()
                


                y_c = img.shape[0] / 2 # height/2
                x_c = img.shape[1] / 2 # width/2
               
                f_cam_y = img.shape[0] / 2 # FOV == 90 deg ; Focal len of front camera (along vertical axis)
                f_cam_x = img.shape[1] / 2
                a = img.shape[1]/img.shape[0] #Aspect Ratio

                ## Evader Coordinates Calculation:
                x_e = (x_min + x_max) / 2
                y_e = (y_min + y_max) / 2
                
                ##Horizontal and vertical pixel offset
                e_x = x_e - x_c 
                e_y = y_e - y_c

                if strategy == 1:
                    # Strategy 1: Monochrome camera + scaling factor
                    w_bb = abs(x_max - x_min)
                    gamma = w_bb / 1.36  # based on drone width
                    d_c = np.array([[f_cam_x / gamma],
                                     [e_x / gamma],
                                     [-e_y / gamma*a]]) 

                elif strategy == 2:
                    # Strategy 2: Depth camera
                    depth_response = client.simGetImages([
                        airsim.ImageRequest(camera_name, airsim.ImageType.DepthPerspective, pixels_as_float=True)
                    ], vehicle_name=PURSUER)[0]

                    depth_img = np.array(depth_response.image_data_float, dtype=np.float32)
                    print(f"depth_img_len: {len(depth_img)}")
                    depth_img = depth_img.reshape((img.shape[0], img.shape[1]))  
                    depth_at_bbox = depth_img[int(y_e)][int(x_e)]
                
                    depth_at_bbox = depth_at_bbox
                    
                    d_c = np.array([[depth_at_bbox],
                                     [e_x * depth_at_bbox / f_cam_x],
                                     [-e_y * depth_at_bbox / f_cam_y]])
                    
                    
                
               
                #### FRAME TRANSFORMATIONS OF HEADING VECTOR
                print(f"d_c: {d_c.flatten()}")

                R_cb = airsim.euler_to_rotation_matrix(0, tilt_angle_cam, 0)
                d_b = R_cb @ d_c # wrt body frame

                R_bg = quaternion_to_rotation_matrix(q)
                d_g = R_bg @ d_b # wrt global frame

                # comparison
                print("d_g:  ", d_g.flatten())
                print("Actual direction to evader:", true_vec)
                d_g_norm = d_g.flatten()/ np.linalg.norm(d_g.flatten())
                cos_theta = np.dot(d_g_norm, true_vec/ np.linalg.norm(true_vec))

                 # --- Get true distance to evader ---
                true_distance = np.linalg.norm(true_vec)
                # ------------------------------------

                angle_error = np.arccos(np.clip(cos_theta, -1.0, 1.0)) * 180 / np.pi
                print(f"Angle error between visual and true direction: {angle_error:.2f}°")
                print("-" * 50)
                
                if args.plot:
                    angle_errors_plot.append(angle_error)
                    true_depths_plot.append(np.linalg.norm(true_vec))
                
                ## DECOUPLED DRONE BODY YAW AND CAMERA PITCH CONTROL
                yaw_rate_body = yaw_pid.compute(e_x)
                tilt_rate_cam = tilt_pid.compute(e_y)

                # Get loop time delta
                current_time = time.time()
                dt = current_time - prev_time
                prev_time = current_time

                tilt_angle_cam += tilt_rate_cam*dt

                # Camera Pose to be fed to SetCameraPose Function
                camera_pose = airsim.Pose(
                    airsim.Vector3r(0, 0, 0),
                    airsim.euler_to_quaternion(0, tilt_angle_cam, 0)  # tilt = pitch
                ) 
                
                
                # Apply control inputs
                client.moveByVelocityAsync(0, 0, 0, duration=1, yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=-yaw_rate_body), vehicle_name=PURSUER) 
                ##YAW RATE should work w the minus sign here 
                client.simSetCameraPose(camera_name, camera_pose, vehicle_name=PURSUER)

                #Moves drone along dg vector while gimbal lock has not occured

                ### ADAPTIVE POWER ###
                magnitude = np.linalg.norm(d_g.flatten())

                # Logarithmic scaling
                scaled = np.log1p(magnitude)           # log(1 + magnitude)
                normalized = scaled / (scaled + 1)     # in (0, 1)

                # Scale power from 1 to 2.5
                power = 1 + normalized * (2.5 - 1)
                d_g_norm = power*d_g_norm                 

                # Check for GIMBAL LOCK condition

                gimbal_lock_threshold = math.radians(70)
                gimbal_unlock_threshold = math.radians(60)

                if abs(tilt_angle_cam) > gimbal_lock_threshold:
                    gimbal_recovery_mode = True
                if abs(tilt_angle_cam) < gimbal_unlock_threshold:
                    gimbal_recovery_mode = False

                ### GIMBAL LOCKING COMPENSATION (GLC) ###
                if gimbal_recovery_mode:
                    if tilt_angle_cam > 0:
                        print("Gimbal locked looking UP. Ascending to recover.")
                        client.moveByVelocityAsync(0, 0, -power, dt, vehicle_name=PURSUER)  # DRONE BODY descend
                    else:
                        print("Gimbal locked looking DOWN. Descending to recover.")
                        client.moveByVelocityAsync(0, 0, power, dt, vehicle_name=PURSUER)  # DRONE BODY ascend
                    continue
                
                else:
                    # PURSUER MOVING ALONG ESTIMATED HEADING VECTOR
                    client.moveByVelocityAsync(d_g_norm[0], d_g_norm[1], d_g_norm[2], dt, vehicle_name=PURSUER)
                
                # --------------------------------------------------------------------

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
    parser.add_argument("--depth", action="store_true", help="Evader moves in x axis (depth)")
    parser.add_argument("--strategy", type=int, default=2, help="Choose strategy: 1=monochrome_cam, 2=depth_cam")
    parser.add_argument("--plot", action="store_true", help="Enable live plotting of angle error and depth")
    parser.add_argument("--3d_track", action="store_true", help="Plot 3D trajectory of pursuer and evader")

    args = parser.parse_args()
    if args.__dict__.get("3d_track"):
        threading.Thread(target=live_plot_3d, daemon=True).start()
    main(args, args.strategy)

