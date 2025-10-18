import random
import cv2
import mediapipe as mp#detects 21 landmarks
import util
import pyautogui
from pynput.mouse import Button, Controller
mouse = Controller()


screen_width, screen_height = pyautogui.size()
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,#cuz we r capvideo
    model_complexity=1,
    min_detection_confidence=0.7,#the min score is req is 0.7
    min_tracking_confidence=0.7,#if more than 70 than only track
    max_num_hands=1
)

def move_mouse(index_finger_tip):
    if index_finger_tip is not None:
        x = int(index_finger_tip.x*screen_width)#multiply with actual screen width
        y = int(index_finger_tip.y*screen_height)#multiply with actual screen width
        pyautogui.moveTo(x,y)
def find_finger_tip(processed):
    if processed.multi_hand_landmarks:
        hand_landmarks = processed.multi_hand_landmarks[0]
        return hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
    return None

def is_left_click(landMarks_list,  thumb_index_dist):
    return(util.get_angle(landMarks_list[5], landMarks_list[6], landMarks_list[8])<50 and 
           util.get_angle(landMarks_list[9], landMarks_list[10], landMarks_list[12])>90 and
           thumb_index_dist>50
           )

def is_right_click(landMarks_list,  thumb_index_dist):
    return(util.get_angle(landMarks_list[5], landMarks_list[6], landMarks_list[8])>90 and 
           util.get_angle(landMarks_list[9], landMarks_list[10], landMarks_list[12])<50 and
           thumb_index_dist>50
           )

def is_screenshot(landMarks_list, thumb_index_dist):
    return(
        util.get_angle(landMarks_list[5], landMarks_list[6], landMarks_list[8])<50 and 
        util.get_angle(landMarks_list[9], landMarks_list[10], landMarks_list[12])<50 and
        thumb_index_dist<50
    )


def is_double_click(landMarks_list, thumb_index_dist):
    return(
        util.get_angle(landMarks_list[5], landMarks_list[6], landMarks_list[8])<50 and 
        util.get_angle(landMarks_list[9], landMarks_list[10], landMarks_list[12])<50 and
        thumb_index_dist>50
    )




def detect_gesture(frame, landMarks_list, processed):
    if len(landMarks_list)>=21:
        index_finger_tip = find_finger_tip(processed)
        #print(index_finger_tip)
        thumb_index_dist = util.get_distance([landMarks_list[4],landMarks_list[5]])

        if thumb_index_dist < 50 and util.get_angle(landMarks_list[5], landMarks_list[6], landMarks_list[8])>90:
            move_mouse(index_finger_tip)

        #left_click
        elif is_left_click(landMarks_list,thumb_index_dist):#thumb and middle finger remains open
            mouse.press(Button.left)
            mouse.release(Button.left)
            cv2.putText(frame,"Left Click",(50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        elif is_right_click(landMarks_list,thumb_index_dist):#thumb and index finger remains open
            mouse.press(Button.right)
            mouse.release(Button.right)
            cv2.putText(frame,"Right Click",(50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        elif is_double_click(landMarks_list,thumb_index_dist):#only thumb remains open
            pyautogui.doubleClick()
            cv2.putText(frame,"Double Click",(50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        elif is_screenshot(landMarks_list,thumb_index_dist):#all close
            im1 = pyautogui.screenshot()
            label = random.randit(1,1000)
            im1.save(f'my_screenshot_{label}.png')
            cv2.putText(frame,"Screen Shot Taken",(50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            
        


def main():
    capture = cv2.VideoCapture(0)
    draw = mp.solutions.drawing_utils
    try:
        while capture.isOpened():
            ret, frame = capture.read()

            if not ret:
                break
            frame = cv2.flip(frame, 1)

            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed = hands.process(frameRGB)

            landMarks_list = []
            if processed.multi_hand_landmarks:
                hand_landmarks = processed.multi_hand_landmarks[0]
                draw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)

                for lm in hand_landmarks.landmark:
                    landMarks_list.append((lm.x, lm.y))
                    #print(landMarks_list)


            detect_gesture(frame, landMarks_list, processed)

            cv2.imshow('Frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):#wait for 1msec after is each frame and keybord inp is q break
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':#used when imported to other fiels
    main()
