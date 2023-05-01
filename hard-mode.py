import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def is_thumbs_up(hand_landmarks):
    def distance(a, b):
        return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5

    landmarks = hand_landmarks.landmark
    closed_distance_threshold = 0.1
    thumb_distance_threshold = 0.125
    thumb_tip, thumb_mcp = landmarks[mp_hands.HandLandmark.THUMB_TIP], landmarks[mp_hands.HandLandmark.THUMB_MCP]

    # Please see the commented code appended at the end of this script for more understandable logic.
    # Wanted something more concise. Downside is the magic numbers and illegibility, etc.
    if thumb_tip.y < thumb_mcp.y and distance(thumb_tip, thumb_mcp) > thumb_distance_threshold:
        for i in range(4):
            finger_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP + i * 4]
            finger_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP + i * 4]
            if distance(finger_tip, finger_mcp) >= closed_distance_threshold:
                return False
        return True

    return False

def main():
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb_frame)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    if is_thumbs_up(hand_landmarks):
                        cv2.putText(frame, 'Thumbs Up!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        print('👍!')

            cv2.imshow('Thumbs Up Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


######################################################################
# "Unabridged" version of `is_thumbs_up()`
######################################################################
# def is_thumbs_up(hand_landmarks):
#     def distance(a, b):
#         return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5
#
#     # Set distance thresholds...
#     # Whether or not a finger is closed:
#     closed_distance_threshold = 0.1
#     # Whether or not the thumb tip is "away" from the thumb MCP
#     thumb_distance_threshold = 0.125

#     # Get coordinates for thumb and other finger landmarks
#     thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
#     thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
#     index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
#     index_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
#     middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
#     middle_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
#     ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
#     ring_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
#     pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
#     pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
#
#
#     # Check if thumb tip is above thumb MCP joint; if distance between thumb tip and MCP is greater than threshold
#     if thumb_tip.y < thumb_mcp.y and distance(thumb_tip, thumb_mcp) > thumb_distance_threshold:
#         # Check if the other fingers are closed
#         if (distance(index_finger_tip, index_finger_mcp) < closed_distance_threshold and
#             distance(middle_finger_tip, middle_finger_mcp) < closed_distance_threshold and
#             distance(ring_finger_tip, ring_finger_mcp) < closed_distance_threshold and
#             distance(pinky_tip, pinky_mcp) < closed_distance_threshold):
#             return True
#     return False
######################################################################