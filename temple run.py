import sys
sys.path.append(r"C:\Users\SIVA KIRAN\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\LocalCache\local-packages\Python312\site-packages")
import time
import cv2
import mediapipe as mp
import PostEstimationModule as pem
from pynput.keyboard import Key, Controller

# Initialize camera, pose detector, and keyboard controller
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Lower resolution for better performance
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 20)  # Reduce frame rate if needed

detector = pem.poseDetector()
keyboard = Controller()

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.resize(img, (640, 480))
    img = cv2.flip(img, 1)
    
    # Pose detection
    detector.findPose(img, draw=False)
    lmList = detector.getPosition(img, draw=False)

    if lmList:
        p1, p2 = lmList[1][1:], lmList[23][1:]  # Head and mid-hip points
        left, right = lmList[18][1:], lmList[19][1:]  # Left and right wrists
        l, _, _ = detector.findDistance(p1, p2)
        l1, _, _ = detector.findDistance(left, right)

        # Move forward if both hands are raised above a threshold
        if left[1] < 200 and right[1] < 200:  # Both hands above head height
            keyboard.press(Key.up)
            print("Moving Forward")
        else:
            keyboard.release(Key.up)

        # Move backward if hands are lowered below a threshold
        if left[1] > 300 and right[1] > 300:  # Both hands below waist level
            keyboard.press(Key.down)
            print("Moving Backward")
        else:
            keyboard.release(Key.down)

        # Move left if left hand is stretched out to the left
        if left[0] < 150:  # Left hand far to the left side
            keyboard.press(Key.left)
            print("Moving Left")
        else:
            keyboard.release(Key.left)

        # Move right if right hand is stretched out to the right
        if right[0] > 500:  # Right hand far to the right side
            keyboard.press(Key.right)
            print("Moving Right")
        else:
            keyboard.release(Key.right)

    # Stop the character if no movement condition is met
    if not (left[0] < 200 and right[1] < 200) and not (left[1] > 300 and right[1] > 300) and not (left[0] < 150) and not (right[0] > 500):
        print("No Movement, Stopping Character")
        keyboard.release(Key.up)
        keyboard.release(Key.down)
        keyboard.release(Key.left)
        keyboard.release(Key.right)

    # Display the game window
    cv2.imshow("Temple Run", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
