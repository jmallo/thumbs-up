import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def print_result(result: vision.GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    # Couldn't resist this ridiculous syntax, ha.
    if any(gesture.category_name == 'Thumb_Up' for gesture_list in result.gestures for gesture in gesture_list):
        print('👍!')

def get_gesture_recognizer_options():
    return vision.GestureRecognizerOptions(
        base_options=python.BaseOptions(model_asset_path='./gesture_recognizer.task'),
        running_mode=vision.RunningMode.LIVE_STREAM,
        result_callback=print_result
    )

def main():
    cap = cv2.VideoCapture(0)
    options = get_gesture_recognizer_options()

    with vision.GestureRecognizer.create_from_options(options) as recognizer:
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # Frame/timestamp oddness to bridge OpenCV with Mediapipe
            timestamp_ms = int(round(frame_count * 1000 / cap.get(cv2.CAP_PROP_FPS)))
            recognizer.recognize_async(mp_image, timestamp_ms=timestamp_ms)

            cv2.imshow('Thumbs Up Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()