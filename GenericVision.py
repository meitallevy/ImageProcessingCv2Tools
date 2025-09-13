import cv2
import numpy as np


class TrackBar:
    """
    Wrapper class for OpenCV trackbars with callback execution.
    """

    def __init__(self, name, window_name, max_value, on_change="pass", initial_pos=0):
        """
        Initialize a trackbar.

        :param name: Trackbar name.
        :param window_name: Window to attach trackbar to.
        :param max_value: Maximum value of trackbar.
        :param on_change: Code to execute when trackbar value changes.
        :param initial_pos: Initial position of the trackbar.
        """
        self.name = name
        self.window_name = window_name
        self.value = 0
        self.max_value = max_value
        self.on_change = on_change

        cv2.createTrackbar(
            self.name, self.window_name, self.value, self.max_value, self._on_change
        )
        self.set_value(initial_pos)

    def _on_change(self, val):
        self.value = val
        exec(self.on_change)

    def get_value(self):
        return self.value

    def set_value(self, value):
        cv2.setTrackbarPos(self.name, self.window_name, value)


class Camera:
    """
    Camera wrapper for capturing frames and adjusting exposure.
    """

    def __init__(
        self, port, fov_horizontal=55, video_width=640, focal_length=678.5, exposure=-5
    ):
        self.port = port
        self.fov_horizontal = fov_horizontal
        self.video_width = video_width
        self.focal_length = focal_length
        self.exposure = exposure
        self.cap = None

    def capture(self):
        self.cap = cv2.VideoCapture(self.port)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, self.exposure)

    def get_frame(self):
        ret, frame = self.cap.read()
        return frame

    def release(self):
        if self.cap:
            self.cap.release()

    def set_exposure(self, exposure):
        self.exposure = exposure


class Results:
    """
    Utility class for color detection, contour extraction, and visualization.
    """

    @staticmethod
    def show_color(lower, upper, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        return cv2.bitwise_and(frame, frame, mask=mask)

    def show_max_contour_center(self, lower, upper, frame, cam=None):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv = cv2.GaussianBlur(hsv, (5, 5), 0)
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        res = cv2.bitwise_and(frame, frame, mask=mask)
        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, 0)
        filled = self.fill_contours(thresh)
        contours = self.find_contours(filled)

        if not contours:
            return [frame.copy(), res]

        target = max(contours, key=cv2.contourArea)
        if cv2.contourArea(target) <= 80:
            return [frame.copy(), res]

        cx, cy = self.find_center(target)
        angle = self.calc_angle(cx, cam) if cam else -1

        frame_copy = frame.copy()
        res = self.draw_contours_and_center(res, target, cx, cy, angle)
        frame_copy = self.draw_contours_and_center(frame_copy, target, cx, cy, angle)

        return [frame_copy, res]

    def show_all_contours_center(self, lower, upper, frame, cam=None):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        res = cv2.bitwise_and(frame, frame, mask=mask)
        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, 0)
        filled = self.fill_contours(thresh)
        contours = self.find_contours(filled)

        if not contours:
            return [frame.copy(), res]

        frame_copy = frame.copy()
        for c in contours:
            if cv2.contourArea(c) >= 80:
                cx, cy = self.find_center(c)
                angle = self.calc_angle(cx, cam) if cam else -1
                res = self.draw_contours_and_center(res, c, cx, cy, angle)
                frame_copy = self.draw_contours_and_center(frame_copy, c, cx, cy, angle)

        return [frame_copy, res]

    @staticmethod
    def find_contours(filtered):
        contours = cv2.findContours(
            filtered.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        return contours[0] if len(contours) == 2 else contours[1]

    @staticmethod
    def find_center(contour):
        m = cv2.moments(contour)
        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])
        return cx, cy

    @staticmethod
    def fill_contours(frame):
        _, threshed = cv2.threshold(frame, 220, 255, cv2.THRESH_BINARY_INV)
        flood_filled = threshed.copy()
        h, w = threshed.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(flood_filled, mask, (0, 0), 255)
        return cv2.bitwise_not(flood_filled)

    @staticmethod
    def calc_angle(cx, cam):
        return (1 - 2 * cx / cam.video_width) * cam.fov_horizontal / 2

    @staticmethod
    def draw_contours_and_center(frame, contour, cx, cy, angle=-1):
        convex_hull = cv2.convexHull(contour)
        cv2.drawContours(frame, [convex_hull], 0, (200, 0, 200), 3)
        cv2.circle(frame, (cx, cy), 10, (200, 200, 0), -1)
        text = f"{round(angle, 2)}" if angle != -1 else " "
        cv2.putText(frame, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
        return frame


class Display:
    """
    Main display loop for camera feed and color tracking visualization.
    """

    def __init__(self, cam: Camera):
        self.cam = cam
        cam.capture()

    def run(self):
        results = Results()
        play = True

        # HSV Trackbars
        cv2.namedWindow("HSV")
        cv2.resizeWindow("HSV", 550, 700)
        tlH = TrackBar("Low H", "HSV", 360)
        tlS = TrackBar("Low S", "HSV", 255)
        tlV = TrackBar("Low V", "HSV", 255)
        tuH = TrackBar("Up H", "HSV", 360, initial_pos=360)
        tuS = TrackBar("Up S", "HSV", 255, initial_pos=255)
        tuV = TrackBar("Up V", "HSV", 255, initial_pos=255)

        # HSV visualization
        try:
            hsv_image = cv2.imread("hsvimage.png")
        except Exception:
            hsv_image = np.zeros((125, 550, 3), np.uint8)
            cv2.putText(
                hsv_image,
                "H - Hue, S - Saturation, V - Value",
                (100, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

        key_state = 48  # initial key state

        while True:
            if not play:
                key = cv2.waitKey(1) & 0xFF
                if key == 32:  # space to resume
                    play = True
                continue

            frame = self.cam.get_frame()
            lower_color = np.array([tlH.value, tlS.value, tlV.value])
            upper_color = np.array([tuH.value, tuS.value, tuV.value])

            # Select mode
            res = results.show_color(lower_color, upper_color, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 32:
                play = False
            elif key == 27:  # ESC to quit
                self._confirm_quit()
                break
            elif key in [33, 64, 49, 50, 48]:
                res, frame = self._select_mode(
                    key, results, lower_color, upper_color, frame
                )
                key_state = key
            else:
                res, frame = self._select_mode(
                    key_state, results, lower_color, upper_color, frame
                )

            self._show_combined(frame, res, hsv_image)

    def _select_mode(self, key, results, lower_color, upper_color, frame):
        if key == 33:  # !
            return (
                results.show_max_contour_center(lower_color, upper_color, frame)[1],
                results.show_max_contour_center(lower_color, upper_color, frame)[0],
            )
        elif key == 64:  # @
            return (
                results.show_all_contours_center(lower_color, upper_color, frame)[1],
                results.show_all_contours_center(lower_color, upper_color, frame)[0],
            )
        elif key == 49:  # 1
            return (
                results.show_max_contour_center(
                    lower_color, upper_color, frame, self.cam
                )[1],
                results.show_max_contour_center(
                    lower_color, upper_color, frame, self.cam
                )[0],
            )
        elif key == 50:  # 2
            return (
                results.show_all_contours_center(
                    lower_color, upper_color, frame, self.cam
                )[1],
                results.show_all_contours_center(
                    lower_color, upper_color, frame, self.cam
                )[0],
            )
        elif key == 48:  # 0
            return results.show_color(lower_color, upper_color, frame), frame

    def _show_combined(self, frame, res, hsv_image):
        h_combined = np.hstack((frame, res))
        width = h_combined.shape[1]

        upper = np.zeros((50, width, 3), np.uint8)
        lower = np.zeros((350, width, 3), np.uint8)

        cv2.putText(
            upper,
            "camera",
            (250, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            upper,
            "filtered",
            (950, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        instructions = [
            "0 - initial state 1 - shows the max contour, its center and its angle",
            "2 - shows all contours bigger than 80px, their center and angle",
            "Shift+1/2 (!/@) - show contours without angle",
            "Space - pause/resume, ESC - quit",
            "Change HSV values in the HSV window to choose the color range",
        ]

        for idx, text in enumerate(instructions):
            cv2.putText(
                lower,
                text,
                (10, 30 + 30 * idx),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        combined = np.vstack((upper, h_combined, lower))
        cv2.imshow("window title", combined)
        cv2.imshow("HSV", hsv_image)

    def _confirm_quit(self):
        win = np.zeros((100, 550, 3), np.uint8)
        cv2.putText(
            win,
            "Are you sure you want to quit?",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            win,
            "ESC - yes, any other key - no",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        cv2.imshow("Quit Confirmation", win)
        key = cv2.waitKey(0) & 0xFF
        if key == 27:
            self.cam.release()
            cv2.destroyAllWindows()
        else:
            cv2.destroyWindow("Quit Confirmation")


# Camera exposure selection
cv2.namedWindow("choose exposure")
cv2.resizeWindow("choose exposure", 550, 200)
texp = TrackBar("exp = -", "choose exposure", 13, initial_pos=4)

win = np.zeros((100, 550, 3), np.uint8)
cv2.putText(
    win,
    "0 >= exp >= -13 click any key to continue",
    (10, 50),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (255, 255, 255),
    2,
)
cv2.putText(
    win,
    "For regular picture choose -4 to -6",
    (10, 70),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.55,
    (255, 255, 255),
    2,
)
cv2.putText(
    win, "(default -4)", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
)
cv2.imshow("choose exposure", win)
cv2.waitKey(0) & 0xFF

camera = Camera(0, exposure=-texp.value)

# Run the display loop with error handling
while True:
    try:
        display_instance = Display(camera)
        display_instance.run()
        break
    except Exception as e:
        error = np.zeros((200, 1550, 3), np.uint8)
        messages = [
            "An error has occurred :(",
            "Check if cv2.VideoCapture(port) succeeded (common issue).",
            "Is the port correct?",
            "Is the camera used by another program?",
            "If not, unknown error. Report if this occurs.",
            f"Error details: {e}",
            "Click any key to retry.",
        ]
        for idx, msg in enumerate(messages):
            cv2.putText(
                error,
                msg,
                (10, 30 + idx * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5 + 0.05 * (idx == 0),
                (255, 255, 255) if idx != 0 else (0, 0, 255),
                2,
            )
        cv2.imshow("Error!", error)
        cv2.waitKey(0) & 0xFF
