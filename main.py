import numpy
import cv2


def drawCube(
    frame: numpy.ndarray,
    rvec: numpy.ndarray,
    tvec: numpy.ndarray,
    color=(0, 255, 0),
    corners: bool = True,
):
    """draws cube on aruco marker

    Args:
        frame (numpy.ndarray): input frame
        rvec (numpy.ndarray): rotation vectors
        tvec (numpy.ndarray): translation vectors
        color (tuple, optional): color of cube. Defaults to (0, 255, 0).
        corners (bool, optional): draw corner or not. Defaults to True.
    """
    projected_points, _ = cv2.projectPoints(
        CUBE, rvec, tvec, calibration["matrix"], calibration["distCoeffs"]
    )

    projected_points = numpy.int32(projected_points).reshape(-1, 2)

    for index in range(4):
        cv2.line(
            frame,
            projected_points[index],
            projected_points[index + 1 if index < 3 else 0],
            color,
            3,
        )
        cv2.line(
            frame,
            projected_points[index + 4],
            projected_points[(index + 1 if index < 3 else 0) + 4],
            color,
            3,
        )
        cv2.line(frame, projected_points[index], projected_points[index + 4], color, 3)

    if corners:
        for point in projected_points:
            cv2.circle(frame, point, 5, (0, 0, 255), -1)


def drawPyramid(
    frame: numpy.ndarray,
    rvec: numpy.ndarray,
    tvec: numpy.ndarray,
    color=(0, 255, 0),
    corners: bool = True,
):
    """draws pyramid on aruco marker

    Args:
        frame (numpy.ndarray): input frame
        rvec (numpy.ndarray): rotation vectors
        tvec (numpy.ndarray): translation vectors
        color (tuple, optional): color of pyramid. Defaults to (0, 255, 0).
        corners (bool, optional): draw corner or not. Defaults to True.
    """
    projected_points, _ = cv2.projectPoints(
        PYRAMID, rvec, tvec, calibration["matrix"], calibration["distCoeffs"]
    )

    projected_points = numpy.int32(projected_points).reshape(-1, 2)

    for index in range(4):
        cv2.line(
            frame,
            projected_points[index],
            projected_points[index + 1 if index < 3 else 0],
            color,
            3,
        )
        cv2.line(frame, projected_points[index], projected_points[-1], color, 3)

    if corners:
        for point in projected_points:
            cv2.circle(frame, point, 5, (0, 0, 255), -1)


CUBE = numpy.array(
    [
        [0.05, 0.05, 0.0],
        [0.05, -0.05, 0.0],
        [-0.05, -0.05, 0.0],
        [-0.05, 0.05, 0.0],
        [0.05, 0.05, 0.1],
        [0.05, -0.05, 0.1],
        [-0.05, -0.05, 0.1],
        [-0.05, 0.05, 0.1],
    ]
)

PYRAMID = numpy.array(
    [
        [0.05, 0.05, 0.00],
        [0.05, -0.05, 0.00],
        [-0.05, -0.05, 0.00],
        [-0.05, 0.05, 0.00],
        [0.00, 0.00, 0.10],
    ]
)

# use your own distortion coeffs and camera matrix
calibration = numpy.load(r"calibration.npz")

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)

shape = (1280, 720)

cap = cv2.VideoCapture(0)

cap.set(3, shape[0])
cap.set(4, shape[1])

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        continue

    corners, ids, _ = cv2.aruco.detectMarkers(
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), dictionary
    )

    # cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    if ids is not None:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, 0.1, calibration["matrix"], calibration["distCoeffs"]
        )

        for rvec, tvec in zip(rvecs, tvecs):
            drawCube(frame, rvec, tvec)
            cv2.drawFrameAxes(
                frame,
                calibration["matrix"],
                calibration["distCoeffs"],
                rvec,
                tvec,
                0.05,
            )
            drawPyramid(frame, rvec, tvec)

    cv2.flip(frame, 1, frame)

    cv2.imshow("Aruco Pose Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        cap.release()

cv2.destroyAllWindows()
