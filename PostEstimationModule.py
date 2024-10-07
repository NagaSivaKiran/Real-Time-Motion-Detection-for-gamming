import cv2
import mediapipe as mp
import time
import math
from pynput.keyboard import Key, Controller

class poseDetector:
    def __init__(self):
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=False,
                                     model_complexity=0,
                                     smooth_landmarks=True,
                                     enable_segmentation=False,
                                     smooth_segmentation=True,
                                     min_detection_confidence=0.5,
                                     min_tracking_confidence=0.5)
        self.mpDraw = mp.solutions.drawing_utils

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks and draw:
            self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def getPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
        return self.lmList

    def findDistance(self, p1, p2, img=None, color=(255, 0, 255), scale=5):
        x1, y1 = p1
        x2, y2 = p2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        if img is not None:
            cv2.circle(img, (x1, y1), 10, color, cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, color, cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), color, max(1, scale // 3))
            cv2.circle(img, (cx, cy), 10, color, cv2.FILLED)
        return length, (x1, y1, x2, y2, cx, cy), img

    def findAngle(self, img, p1, p2, p3, draw=True):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        # Calculate the angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 5, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 5, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x3, y3), 5, (255, 0, 0), cv2.FILLED)
            cv2.putText(img, str(int(angle)), (x2 - 20, y2 + 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
        return angle

def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = poseDetector()
    keyboard = Controller()

    while True:
        success, img = cap.read()
        img = cv2.resize(img, (900, 600))
        img = detector.findPose(img)
        lmList = detector.getPosition(img)

        if lmList:
            # Head and hip positions
            p1, p2 = lmList[1][1:], lmList[23][1:]  # Head and mid-hip points
            left_wrist, right_wrist = lmList[18][1:], lmList[19][1:]  # Left and right wrists

            # Movement logic for character control
            if left_wrist[1] < 200 and right_wrist[1] < 200:  # Both hands raised
                keyboard.press(Key.up)
                print("Moving Forward")
            else:
                keyboard.release(Key.up)

            if left_wrist[1] > 300 and right_wrist[1] > 300:  # Both hands lowered
                keyboard.press(Key.down)
                print("Moving Backward")
            else:
                keyboard.release(Key.down)

            if left_wrist[0] < 150:  # Left hand far left
                keyboard.press(Key.left)
                print("Moving Left")
            else:
                keyboard.release(Key.left)

            if right_wrist[0] > 500:  # Right hand far right
                keyboard.press(Key.right)
                print("Moving Right")
            else:
                keyboard.release(Key.right)

        # FPS calculation for smoother output
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.imshow("Image", img)

        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
