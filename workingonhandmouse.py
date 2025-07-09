import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize Mediapipe Hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2)

# Get screen size
screen_width, screen_height = pyautogui.size()
smooth_factor = 5  # Adjust for smoother movement

# Previous cursor position for smoothing
prev_x, prev_y = 0, 0

# OpenCV video capture
cap = cv2.VideoCapture(0)

# Hand tracking lock variables
locked_hand = None  # Store locked hand landmark data
lock_threshold = 0.05  # Allowable deviation for the same hand

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convert to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        if locked_hand is None:  # Lock onto the first detected hand
            nearest_hand = None
            min_depth = float('inf')

            for hand_landmarks in results.multi_hand_landmarks:
                wrist_depth = hand_landmarks.landmark[0].z
                if wrist_depth < min_depth:
                    min_depth = wrist_depth
                    nearest_hand = hand_landmarks  # Track the nearest hand

            if nearest_hand:
                locked_hand = nearest_hand  # Lock the hand

        # Check if the locked hand is still in the frame
        if locked_hand:
            wrist_x = locked_hand.landmark[0].x
            wrist_y = locked_hand.landmark[0].y

            # Verify if the same hand is still present in the frame
            matching_hand = None
            for hand_landmarks in results.multi_hand_landmarks:
                if abs(hand_landmarks.landmark[0].x - wrist_x) < lock_threshold and \
                   abs(hand_landmarks.landmark[0].y - wrist_y) < lock_threshold:
                    matching_hand = hand_landmarks
                    break  # Stop checking once a match is found

            if matching_hand:
                locked_hand = matching_hand  # Update locked hand position

                # Draw landmarks
                mp_drawing.draw_landmarks(frame, locked_hand, mp_hands.HAND_CONNECTIONS)

                # Get index fingertip (Landmark 8) for cursor movement
                x_index = int(locked_hand.landmark[8].x * w)
                y_index = int(locked_hand.landmark[8].y * h)

                # Adjust cursor to the top of the fingertip
                fingertip_offset = 20  # Adjust as needed
                y_index -= fingertip_offset  # Move cursor slightly above

                # Convert to screen coordinates
                screen_x = np.interp(x_index, [0, w], [0, screen_width])
                screen_y = np.interp(y_index, [0, h], [0, screen_height])

                # Apply smoothing
                curr_x = (prev_x * (smooth_factor - 1) + screen_x) / smooth_factor
                curr_y = (prev_y * (smooth_factor - 1) + screen_y) / smooth_factor

                # Move cursor
                pyautogui.moveTo(curr_x, curr_y)

                # Update previous position
                prev_x, prev_y = curr_x, curr_y

                # Draw a circle at fingertip
                cv2.circle(frame, (x_index, y_index + fingertip_offset), 10, (0, 255, 0), -1)

                # Get key landmarks
                thumb_x = locked_hand.landmark[4].x
                index_base_x = locked_hand.landmark[2].x  # Base of index finger
                middle_tip = locked_hand.landmark[12]  # Middle finger tip
                middle_knuckle = locked_hand.landmark[10]  # Middle finger knuckle

                # Left Click -> If thumb moves forward (crosses the index base)
                if thumb_x > index_base_x + 0.02:
                    pyautogui.click()
                    cv2.putText(frame, "Left Click", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Right Click -> If middle finger moves significantly below its knuckle
                if middle_tip.y > middle_knuckle.y + 0.02:
                    pyautogui.rightClick()
                    cv2.putText(frame, "Right Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            else:
                locked_hand = None  # Release lock if the hand disappears

    # Show the video feed
    cv2.imshow("Hand Tracking Mouse", frame)

    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
