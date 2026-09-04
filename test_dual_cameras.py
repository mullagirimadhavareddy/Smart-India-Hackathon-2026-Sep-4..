import cv2
import sys
import time
import os
import numpy as np

# Configurable camera indices (environment variables or defaults)
MAC_CAM_INDEX = int(os.environ.get("MAC_CAM_INDEX", "0"))
OBS_CAM_INDEX = int(os.environ.get("OBS_CAM_INDEX", "1"))

def create_placeholder(width, height, title, subtitle, hint):
    """Creates a clean standby screen when a camera is offline."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (width - 10, height - 10), (50, 50, 60), 2)
    cv2.putText(img, title, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    cv2.putText(img, subtitle, (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    cv2.putText(img, hint, (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return img

def main():
    mac_idx = int(sys.argv[1]) if len(sys.argv) > 1 else MAC_CAM_INDEX
    obs_idx = int(sys.argv[2]) if len(sys.argv) > 2 else OBS_CAM_INDEX

    print("=" * 65)
    print(" DUAL CAMERA TEST UTILITY")
    print(f" Camera 1 (Mac Webcam): Index {mac_idx}")
    print(f" Camera 2 (OBS Virtual Camera / Phone): Index {obs_idx}")
    print("=" * 65)

    mac_cam = cv2.VideoCapture(mac_idx)
    obs_cam = cv2.VideoCapture(obs_idx)

    # Allow AVFoundation backend to initialize
    time.sleep(1.0)

    mac_opened = mac_cam.isOpened()
    obs_opened = obs_cam.isOpened()

    print(f"Mac Webcam (Index {mac_idx}) isOpened: {mac_opened}")
    print(f"OBS Virtual Camera (Index {obs_idx}) isOpened: {obs_opened}")

    if not mac_opened and not obs_opened:
        print("\n❌ Error: Neither camera could be opened!")
        print("Please check camera permissions and ensure OBS Virtual Camera is started.")
        return

    if not obs_opened:
        print("\n⚠️ Note: OBS Virtual Camera (Index 1) is not active yet.")
        print("   To start it: Open OBS Studio -> Click 'Start Virtual Camera' (bottom right dock).")
        print("   Showing Mac Webcam on Left and Standby on Right.\n")

    # Set common resolution & FPS
    for cap in (mac_cam, obs_cam):
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)

    print("Opening dual camera preview window. Press 'q' inside window to quit.")

    fps_timer = time.time()
    frames = 0
    fps = 0.0

    while True:
        ret_mac = False
        frame_mac = None
        if mac_cam.isOpened():
            ret_mac, frame_mac = mac_cam.read()

        ret_obs = False
        frame_obs = None
        if obs_cam.isOpened():
            ret_obs, frame_obs = obs_cam.read()

        # Handle Mac webcam frame
        if ret_mac and frame_mac is not None:
            f_mac = cv2.resize(frame_mac, (640, 480))
            cv2.putText(f_mac, f"Mac Webcam (Index {mac_idx}) | FPS: {fps:.1f}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            f_mac = create_placeholder(640, 480, f"MAC WEBCAM (Index {mac_idx})", "Signal not ready", "Check camera permissions")

        # Handle OBS webcam frame
        if ret_obs and frame_obs is not None:
            f_obs = cv2.resize(frame_obs, (640, 480))
            cv2.putText(f_obs, f"OBS Phone Camera (Index {obs_idx}) | FPS: {fps:.1f}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            f_obs = create_placeholder(640, 480, f"OBS PHONE CAMERA (Index {obs_idx})",
                                       "Standby: Camera not detected",
                                       "Click 'Start Virtual Camera' in OBS Studio")

        # Combine side-by-side
        combined = cv2.hconcat([f_mac, f_obs])

        # Master title bar
        header = np.zeros((35, combined.shape[1], 3), dtype=np.uint8)
        header[:] = (25, 25, 30)
        cv2.putText(header, "DUAL CAMERA MONITOR: Mac Webcam (Left) | OBS Phone Camera (Right)",
                    (20, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        display = cv2.vconcat([header, combined])

        # FPS calculation
        frames += 1
        now = time.time()
        if now - fps_timer >= 1.0:
            fps = frames / (now - fps_timer)
            frames = 0
            fps_timer = now

        cv2.imshow("Dual Camera Live Monitor (Press 'q' to quit)", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    mac_cam.release()
    obs_cam.release()
    cv2.destroyAllWindows()
    print("Dual camera preview closed.")

if __name__ == "__main__":
    main()
