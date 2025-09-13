import cv2
import numpy as np
from imutils import grab_contours

# HSV color boundaries for green
LOWER_GREEN = np.array([0, 206, 72])
UPPER_GREEN = np.array([27, 255, 114])

GOAL_MAX_CONTOUR = 0  # 0: max contour only, 1: process all contours


class Cam:
    """Camera processing for detecting green objects and calculating angles."""

    def __init__(self, port, fov_horizontal, video_width, focal_length, exposure):
        self.port = port
        self.fov_horizontal = fov_horizontal
        self.video_width = video_width
        self.focal_length = focal_length
        self.cap = cv2.VideoCapture(port)
        self.exposure = exposure
        self.cap.set(cv2.CAP_PROP_EXPOSURE, self.exposure)

    def run(self):
        """Main processing loop."""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue

            display_frame = frame.copy()
            filtered_mask = self.apply_filter(frame)

            if filtered_mask is None:
                self.show(display_frame, text="No target")
                continue

            filled_mask = self.fill_mask(filtered_mask)
            contours = self.find_contours(filled_mask)

            if not contours:
                self.show(display_frame, filled_mask, text="No target")
                continue

            if GOAL_MAX_CONTOUR == 0:
                self.process_max_contour(contours, display_frame, filled_mask)
            else:
                self.process_all_contours(contours, display_frame, filled_mask)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.release()

    def apply_filter(self, frame):
        """Apply Gaussian blur and HSV mask for green color."""
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
        return mask

    def find_contours(self, mask):
        """Find contours in a binary mask."""
        contours = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        return grab_contours(contours)

    def find_center(self, contour):
        """Compute center coordinates of a contour."""
        m = cv2.moments(contour)
        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])
        return cx, cy

    def fill_mask(self, mask):
        """Fill holes in the binary mask."""
        _, threshed = cv2.threshold(mask, 220, 255, cv2.THRESH_BINARY_INV)
        floodfilled = threshed.copy()
        h, w = threshed.shape[:2]
        mask_flood = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(floodfilled, mask_flood, (0, 0), 255)
        return cv2.bitwise_not(floodfilled)

    def calculate_angle(self, cx):
        """Calculate horizontal angle of the target from center."""
        return (1 - 2 * cx / self.video_width) * self.fov_horizontal / 2

    def draw_contours_and_center(self, frame, contour, cx, cy, angle):
        """Draw target contours, center, and angle on the frame."""
        convex_hull = cv2.convexHull(contour)
        cv2.drawContours(frame, [convex_hull], 0, (200, 0, 200), 3)
        cv2.circle(frame, (cx, cy), 10, (200, 200, 0), -1)
        cv2.putText(
            frame, f"{round(angle, 2)}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, 255
        )

    def process_max_contour(self, contours, frame, mask):
        """Process only the largest contour."""
        target = max(contours, key=cv2.contourArea)
        if cv2.contourArea(target) <= 80:
            self.show(frame, mask, text="Too small")
            return
        cx, cy = self.find_center(target)
        angle = self.calculate_angle(cx)
        self.draw_contours_and_center(frame, target, cx, cy, angle)
        self.show(frame, mask)

    def process_all_contours(self, contours, frame, mask):
        """Process all contours larger than a threshold."""
        count_valid = 0
        for c in contours:
            if cv2.contourArea(c) >= 80:
                count_valid += 1
                cx, cy = self.find_center(c)
                angle = self.calculate_angle(cx)
                self.draw_contours_and_center(frame, c, cx, cy, angle)
        if count_valid == 0:
            self.show(frame, mask, text="All targets are too small")
        else:
            self.show(frame, mask)

    def show(self, frame, mask=None, text=""):
        """Display results with optional mask overlay and text."""
        if mask is not None:
            cv2.imshow("Mask", mask)
        cv2.putText(frame, text, (5, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
        cv2.imshow("Result Frame", frame)

    def release(self):
        """Release camera and destroy windows."""
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    cam_instance = Cam(0, 55, 640, 678.5, -4)
    try:
        cam_instance.run()
    except Exception as e:
        cam_instance.release()
        raise e
