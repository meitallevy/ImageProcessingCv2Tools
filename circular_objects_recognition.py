import cv2
import numpy as np
import math
import consts


class Cam:
    """Camera wrapper with exposure control and basic capture functionality."""

    def __init__(
        self,
        port,
        fov_horizontal=consts.FOV_HORIZONTAL,
        video_width=consts.VIDEO_WIDTH,
        focal_length=consts.FOCAL_LENGTH,
        exp=-4,
    ):
        self.port = port
        self.fov_horizontal = fov_horizontal
        self.video_width = video_width
        self.focal_length = focal_length
        self.exp = exp

    def capture(self):
        """Initialize video capture with specified exposure."""
        self.cap = cv2.VideoCapture(self.port)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, self.exp)

    def get_frame(self):
        """Capture a single frame."""
        ret, frame = self.cap.read()
        return frame if ret else np.zeros((1, 1, 3), dtype=np.uint8)

    def release(self):
        """Release camera resources."""
        self.cap.release()

    def set_exp(self, exp):
        """Set camera exposure value."""
        self.exp = exp


class Location:
    """Represents the spatial properties of a detected target."""

    def __init__(self, target, cam: Cam):
        self.cam = cam
        self.target = target
        self.cx, self.cy = self.center()
        self.angle = self.horizontal_angle()
        self.distance = self.calculate_distance()
        self.radius = self.get_radius()

    def center(self):
        """Compute the center of the target contour."""
        (x, y), _ = cv2.minEnclosingCircle(self.target)
        return x, y

    def horizontal_angle(self):
        """Calculate horizontal angle relative to camera center."""
        return (1 - 2 * self.cx / self.cam.video_width) * self.cam.fov_horizontal / 2

    def calculate_distance(self):
        """Estimate distance to target using contour area."""
        return (
            self.cam.focal_length
            * (consts.CIRC_TARGET_AREA / cv2.contourArea(self.target)) ** 0.5
        )

    def get_radius(self):
        """Get the radius of the minimum enclosing circle of the target."""
        (_, _), radius = cv2.minEnclosingCircle(self.target)
        return radius

    def compare(self):
        """Placeholder for further target shape validation."""
        pass


class NTV:
    """Process frames to detect, filter, and locate targets."""

    def __init__(self, lower, upper, frame, cam: Cam):
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        denoised = self.denoise(hsv_frame)
        filtered = self.apply_hsv_filter(lower, upper, denoised)
        self.target = self.find_target(filtered)
        self.location = Location(self.target, cam) if self.target is not None else None
        if self.location:
            self.location.compare()

    def denoise(self, frame):
        """Apply morphological opening and median blur to remove noise."""
        morph = cv2.morphologyEx(frame, cv2.MORPH_OPEN, (5, 5))
        return cv2.medianBlur(morph, 5)

    def apply_hsv_filter(self, lower, upper, frame):
        """Filter frame using HSV bounds."""
        lower = np.array(lower)
        upper = np.array(upper)
        mask = cv2.inRange(frame, lower, upper)
        res = cv2.bitwise_and(frame, frame, mask=mask)
        cv2.imshow("Filtered HSV", res)
        return res

    def find_contours(self, filtered):
        """Retrieve contours from a filtered frame."""
        contours = cv2.findContours(
            filtered.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        return contours[0] if len(contours) == 2 else contours[1]

    def find_target(self, frame):
        """Identify the most circular and largest target contour."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, 0)
        filled = self.fill_contours(thresh)
        contours = self.find_contours(filled)

        if not contours:
            print("No target detected")
            return None

        circular_contours = [c for c in contours if self.is_circular(c)[0]]
        if circular_contours:
            target = max(circular_contours, key=cv2.contourArea)
            if cv2.contourArea(target) <= 80:
                return None
            return target
        return None

    def is_circular(self, contour):
        """Determine if a contour approximates a circle."""
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return False, 0
        area = cv2.contourArea(contour)
        circularity = 4 * math.pi * (area / (perimeter**2))
        return 0.7 < circularity < 1.2, circularity

    def fill_contours(self, frame):
        """Fill holes in binary thresholded contours."""
        _, threshed = cv2.threshold(frame, 220, 255, cv2.THRESH_BINARY_INV)
        flood_filled = threshed.copy()
        h, w = threshed.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(flood_filled, mask, (0, 0), 255)
        return cv2.bitwise_not(flood_filled)


class OnComp:
    """Main camera processing loop to visualize targets."""

    def __init__(self, port):
        self.cam = Cam(port)
        self.cam.capture()
        self.run()

    def run(self):
        """Main loop: capture frames, detect targets, display results."""
        while True:
            frame = self.cam.get_frame()
            lower = consts.LOWER_BOUNDARY_HSV
            upper = consts.UPPER_BOUNDARY_HSV
            vision = NTV(lower, upper, frame, self.cam)

            if vision.location:
                frame = self.draw_location(frame, vision.target, vision.location)

            cv2.imshow("Camera Feed", frame)
            if cv2.waitKey(1) == 32:  # Space to exit
                break

        self.cam.release()
        cv2.destroyAllWindows()

    def draw_location(self, frame, target, loc: Location):
        """Draw target contour, center, angle, and estimated distance on frame."""
        convex_hull = cv2.convexHull(target)
        cv2.drawContours(frame, [convex_hull], 0, (200, 0, 200), 3)
        center_coords = (int(loc.cx), int(loc.cy))
        cv2.circle(frame, center_coords, 10, (200, 200, 0), -1)
        cv2.putText(
            frame,
            f"{round(loc.angle, 2)}",
            center_coords,
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            255,
        )
        cv2.putText(
            frame,
            f"{round(loc.distance, 2)}",
            (int(loc.cx), int(loc.cy + 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            255,
        )
        cv2.circle(frame, center_coords, int(loc.radius), (0, 255, 0), 2)
        return frame


# Run the main vision processing
if __name__ == "__main__":
    vision_system = OnComp(0)
