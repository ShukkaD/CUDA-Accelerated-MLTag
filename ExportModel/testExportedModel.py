import cv2
import numpy as np
import os
from ultralytics import YOLO
from pathlib import Path
from yaml import safe_load


BASE_MODEL_PATH = Path(__file__).resolve().parent / "yolo26nApriltag.pt"
INFERENCE_SIZE = 640
CAMERA_FPS = 60


class AprilTagDetector():
    def __init__(self, model_path:str, webcam_index:int, show: bool = True):
        self.model_path = model_path
        self.webcam_index = webcam_index
        self.show = show
        self.imgsz = INFERENCE_SIZE
        self.class_names = self._load_class_names()
        self.model = YOLO(self.model_path, task="detect")
        api_preference = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(self.webcam_index, api_preference)
        self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.predict_kwargs = {
            "imgsz": self.imgsz,
            "conf": 0.20,
            "iou": 0.65,
            "max_det": 32,
            "agnostic_nms": False,
            "verbose": False,
            "stream": False,
        }

    def _load_class_names(self):
        metadata_path = Path(self.model_path).with_name(f"{Path(self.model_path).stem}.metadata.yaml")
        if metadata_path.exists():
            metadata = safe_load(metadata_path.read_text(encoding="utf-8")) or {}
            names = metadata.get("names")
            if isinstance(names, list):
                return {index: name for index, name in enumerate(names)}
            if isinstance(names, dict):
                return {int(index): name for index, name in names.items()}

        if BASE_MODEL_PATH.exists():
            return YOLO(str(BASE_MODEL_PATH), task="detect").names

        return {}

    def _apply_class_names(self, results):
        if not self.class_names:
            return

        if self.model.predictor is not None and hasattr(self.model.predictor, "model"):
            self.model.predictor.model.names = self.class_names
        results[0].names = self.class_names

    def warmup(self, count: int = 3):
        # Use a grayscale dummy to match runtime input (converted to 3-channel for engine)
        dummy_frame = np.zeros((INFERENCE_SIZE, INFERENCE_SIZE, 1), dtype=np.uint8)
        dummy_frame = cv2.cvtColor(dummy_frame, cv2.COLOR_GRAY2BGR)
        for _ in range(count):
            results = self.model(dummy_frame, **self.predict_kwargs)
            self._apply_class_names(results)

    def detect(self):

        while self.cap.isOpened() and self.captureFrames:
            ret, frame = self.cap.read()
            if not ret:
                print("Webcam couldn't be initialized")
                self.captureFrames = False
                break
            
            # Real-life deployment is always grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            results = self.model(gray_3ch, **self.predict_kwargs)
            self._apply_class_names(results)

            if self.show:
                annotated_frame = results[0].plot()
                cv2.imshow("AprilTag Detection", annotated_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or cv2.getWindowProperty("AprilTag Detection", cv2.WND_PROP_VISIBLE) < 1:
                    print("Exiting...")
                    self.captureFrames = False
    
    def cleanup(self):
        self.cap.release()
        if self.show:
            cv2.destroyAllWindows()

    def run(self):
        self.captureFrames = True
        try:
            self.warmup()
            self.detect()
        finally:
            self.cleanup()

if __name__ == "__main__":
    engine_path = Path(__file__).resolve().parent / "TensorRTExports/yolo26nApriltagMixedINT8.engine"
    detector = AprilTagDetector(str(engine_path), 0)
    detector.run()
