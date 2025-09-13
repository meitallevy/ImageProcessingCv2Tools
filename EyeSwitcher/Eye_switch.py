import cv2
import numpy as np


def overlay_eyes(background, overlay, x, y):
    """Overlay a small RGBA image (overlay) on a larger background at position (x, y)."""
    bg_h, bg_w = background.shape[:2]
    h, w = overlay.shape[:2]

    # Ensure overlay is within background bounds
    if x >= bg_w or y >= bg_h:
        return background
    w = min(w, bg_w - x)
    h = min(h, bg_h - y)
    overlay = overlay[:h, :w]

    # Add alpha channel if missing
    if overlay.shape[2] < 4:
        alpha_channel = np.ones((h, w, 1), dtype=overlay.dtype) * 255
        overlay = np.concatenate([overlay, alpha_channel], axis=2)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)

    # Create circular mask
    mask = np.zeros((h, w), dtype=np.uint8)
    mask = cv2.ellipse(
        mask, (w // 2, h // 2), (int(w * 0.75) // 2, int(h * 0.75) // 2),
        0, 0, 360, 1, -1
    )

    # Overlay each color channel
    for c in range(3):
        background[y:y+h, x:x+w, c] = (
            mask * overlay[:, :, c] + (1 - mask) * background[y:y+h, x:x+w, c]
        )

    return background


def detect_faces_and_eyes(frame, face_cascade, eye_cascade, to_frame):
    """Detect faces and eyes and return list of face and eye info."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3)

    detected = []
    for (x, y, w, h) in faces:
        if to_frame:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes_info = []
        eyes = eye_cascade.detectMultiScale(
            roi_gray, 1.3, 1, minSize=(40, 40), maxSize=(90, 90)
        )
        for (ex, ey, ew, eh) in eyes:
            eye_img = roi_color[ey:ey+eh, ex:ex+ew]
            eyes_info.append((eye_img, ex, ey))
            if to_frame:
                cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (100, 7, 90), 2)

        detected.append((x, y, w, h, eyes_info))
    return detected


def switch_eyes(frame, faces_info, mode):
    """Switch eyes between faces if mode is enabled."""
    for face in faces_info:
        x, y, w, h, eyes_info = face
        if len(eyes_info) < 2 or not mode:
            continue

        first_eye, second_eye = eyes_info[0], eyes_info[1]
        fx, fy = x + second_eye[1], y + second_eye[2]
        sx, sy = x + first_eye[1], y + first_eye[2]
        frame = overlay_eyes(overlay_eyes(frame, first_eye[0], fx, fy), second_eye[0], sx, sy)

    return frame


def create_title_section(frame_shape):
    """Create a section with instructions for keys."""
    title_section = np.zeros((frame_shape[0] // 4, frame_shape[1], 3), np.uint8)
    cv2.putText(title_section, "Click space to show / hide switched eyes",
                (title_section.shape[1] // 30, title_section.shape[0] // 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(title_section, "Click Esc to exit",
                (title_section.shape[1] // 30, title_section.shape[0] // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(title_section, "Click 1 to frame / remove frame from faces",
                (title_section.shape[1] // 30, title_section.shape[0] * 3 // 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return title_section


def main():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_EXPOSURE, -4)

    mode = True      # Switch eyes on/off
    to_frame = False # Draw rectangles around faces/eyes

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        new_frame = frame.copy()

        faces_info = detect_faces_and_eyes(new_frame, face_cascade, eye_cascade, to_frame)
        new_frame = switch_eyes(new_frame, faces_info, mode)

        title_section = create_title_section(frame.shape)
        cv2.imshow("CrazyEyes!", np.vstack((new_frame, title_section)))

        key = cv2.waitKey(1)
        if key == 27:   # ESC
            break
        elif key == 32: # Space
            mode = not mode
        elif key == 49: # '1'
            to_frame = not to_frame

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
