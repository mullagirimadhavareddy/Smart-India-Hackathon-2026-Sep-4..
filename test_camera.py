import cv2
import sys
import time

def scan_cameras(max_to_test=5):
    print("=" * 50)
    print(" Scanning Available Camera Devices...")
    print("=" * 50)
    available = []
    for i in range(max_to_test):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                h, w, _ = frame.shape
                print(f" [✓] Camera Index {i}: AVAILABLE (Resolution: {w}x{h})")
                available.append(i)
            else:
                print(f" [?] Camera Index {i}: Opened but could not grab frame")
            cap.release()
        else:
            print(f" [x] Camera Index {i}: Not available")
    print("=" * 50)
    return available

def preview_stream(source):
    # Try converting to int if it's a numeric index
    try:
        src = int(source)
    except ValueError:
        src = source

    print(f"\nOpening stream preview for source: {src}")
    print("Press 'q' inside the preview window to exit.\n")
    
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"❌ Error: Failed to open camera source: {src}")
        return

    fps_timer = time.time()
    frame_count = 0
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Warning: Empty frame received or stream disconnected.")
            time.sleep(0.1)
            continue

        frame_count += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            fps_timer = time.time()

        # Overlay info
        cv2.putText(frame, f"Source: {src} | FPS: {fps:.1f}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Camera Stream Test (Press 'q' to quit)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Preview closed successfully.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        preview_stream(sys.argv[1])
    else:
        cameras = scan_cameras()
        if not cameras:
            print("No local cameras detected. If using DroidCam OBS, make sure you clicked 'Start Virtual Camera' in OBS Studio!")
        else:
            print(f"\nFound cameras: {cameras}")
            target = cameras[-1] if len(cameras) > 1 else cameras[0]
            print(f"Auto-selected default camera: Index {target}")
            print("Launching camera preview now (Press 'q' inside window to quit)...\n")
            preview_stream(target)
