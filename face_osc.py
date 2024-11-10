import cv2
import mediapipe as mp
import time
import argparse
from pythonosc import udp_client

# Initialize webcam capture
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Face Mesh
mpFaceMesh = mp.solutions.face_mesh
face_mesh = mpFaceMesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

# Initialize OSC client arguments
parser = argparse.ArgumentParser()
parser.add_argument("--ip", default="127.0.0.1", help="The IP of the OSC server")
parser.add_argument("--port", type=int, default=6448, help="The port the OSC server is listening on")
args = parser.parse_args()

client = udp_client.SimpleUDPClient(args.ip, args.port)

# Time variables for FPS calculation
pTime = 0

# Indices of 10 selected landmarks
selected_landmark_indices = [33, 133, 362, 263, 1, 9, 78, 308, 13, 14]  # For example: nose, eyes, mouth

while True:
    success, img = cap.read()
    if not success:
        break
    
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(imgRGB)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            ih, iw, _ = img.shape
            landmark_info = []

            # Loop over each selected landmark and normalize its coordinates
            for idx in selected_landmark_indices:
                lm = face_landmarks.landmark[idx]
                x = lm.x
                y = lm.y
                landmark_info.append(x)
                landmark_info.append(y)
                
                # Draw the selected landmarks on the image
                cx, cy = int(x * iw), int(y * ih)
                cv2.circle(img, (cx, cy), 2, (255, 0, 0), cv2.FILLED)

            # Send the 10 selected landmarks to Wekinator in a single message
            client.send_message("/wek/inputs", landmark_info)

            # Draw face mesh on the image (optional)
            mpDraw.draw_landmarks(img, face_landmarks, mpFaceMesh.FACEMESH_CONTOURS)

    # Calculate and display FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    # Display the result
    cv2.imshow("Image", img)

    # ESC interrupt
    if cv2.waitKey(5) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
