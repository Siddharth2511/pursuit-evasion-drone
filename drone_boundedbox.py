import cosysairsim as airsim
import cv2
import numpy as np
import time

# Initialize the AirSim client
client = airsim.MultirotorClient()
client.confirmConnection()
client.reset()

# Define vehicle names
EVADER = "Drone0"
PURSUER = "Drone1"

# Enable API control and arm both drones
client.enableApiControl(True, vehicle_name=EVADER)
client.enableApiControl(True, vehicle_name=PURSUER)
client.armDisarm(True, vehicle_name=EVADER)
client.armDisarm(True, vehicle_name=PURSUER)

# Take off both drones
client.takeoffAsync(vehicle_name=EVADER).join()
client.takeoffAsync(vehicle_name=PURSUER).join()

### Move the evader 

# client.moveToPositionAsync(0, 0, -5, 2, vehicle_name=EVADER).join()
client.moveToPositionAsync(0, 2.5, -6.5, 2, vehicle_name=EVADER)
# Position the pursuer 
client.moveToPositionAsync(-10, 0, -5, 2, vehicle_name=PURSUER).join()
# client.moveByVelocityAsync(1, 0.5, -0.5, duration=1000, vehicle_name=EVADER)

# Allow some time for the drones to stabilize
time.sleep(2)

# Camera and detection settings
camera_name = "0"  # Front center camera
image_type = airsim.ImageType.Scene
detection_radius_cm = 1000 * 100  # 1000 meters

# Set detection radius for the pursuer's camera
client.simSetDetectionFilterRadius(camera_name, image_type, detection_radius_cm, vehicle_name=PURSUER)

# Add the evader drone's mesh name to the detection filter for the pursuer's camera
client.simAddDetectionFilterMeshName(camera_name, image_type, "Drone0", vehicle_name=PURSUER)

# Continuously capture images and detect the evader drone
while True:
    # Capture image from the pursuer's camera
    raw_image = client.simGetImage(camera_name, image_type, vehicle_name=PURSUER)
    if raw_image is None:
        print("Failed to capture image from the pursuer's camera.")
        continue

    # Decode the image
    np_img = np.frombuffer(raw_image, dtype=np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # Retrieve detections from the pursuer's camera
    detections = client.simGetDetections(camera_name, image_type, vehicle_name=PURSUER)

    # Process detections
    for detection in detections:
        if detection.name == "Drone0":
            print("Detected Drone0!")
            bbox_min = detection.box2D.min
            bbox_max = detection.box2D.max
            x_min, y_min = int(bbox_min.x_val), int(bbox_min.y_val)
            x_max, y_max = int(bbox_max.x_val), int(bbox_max.y_val)

            # Draw bounding box with red color and thickness of 1
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 1)

            # Draw center point with blue color
            x_center = (x_min + x_max) // 2
            y_center = (y_min + y_max) // 2
            # cv2.circle(img, (x_center, y_center), 5, (255, 0, 0), -1)

            # Display coordinates
            print(f"Bounding Box: ({x_min}, {y_min}) to ({x_max}, {y_max})")
            print(f"Center: ({x_center}, {y_center})")

    # Display the image with bounding box
    cv2.namedWindow("Pursuer's View", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Pursuer's View", img)
    cv2.resizeWindow("Pursuer's View", 512, 288)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
client.armDisarm(False, vehicle_name=EVADER)
client.armDisarm(False, vehicle_name=PURSUER)
client.enableApiControl(False, vehicle_name=EVADER)
client.enableApiControl(False, vehicle_name=PURSUER)
