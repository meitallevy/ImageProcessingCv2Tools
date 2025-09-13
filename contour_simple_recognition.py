import cv2
import numpy as np


def process_frame(frame, lower_hsv, upper_hsv, min_area=80):
    """
    Detect objects within a specific HSV range and draw their convex hulls.

    :param frame: BGR image from the camera
    :param lower_hsv: lower HSV boundary (list/np.array)
    :param upper_hsv: upper HSV boundary (list/np.array)
    :param min_area: minimum contour area to consider as target
    :return: frame with detected targets drawn
    """
    # Convert to HSV and threshold
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower_hsv), np.array(upper_hsv))
    filtered = cv2.bitwise_and(frame, frame, mask=mask)

    # Convert to grayscale and threshold to find contours
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw convex hulls for sufficiently large contours
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            hull = cv2.convexHull(contour)
            cv2.drawContours(filtered, [hull], -1, (200, 0, 200), 3)

    return filtered


def main():
    cap = cv2.VideoCapture(0)
    lower_hsv = [0, 42, 0]
    upper_hsv = [179, 255, 255]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        processed = process_frame(frame, lower_hsv, upper_hsv)
        cv2.imshow("Detection", processed)

        if cv2.waitKey(1) & 0xFF == 32:  # space bar to exit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
