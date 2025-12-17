import cv2
import numpy as np
from keras.layers import TFSMLayer
from keras import Input, Model

LABELS = [
    "gaaf", "meem", "taa", "ra", "dhad", "ta", "seen", "thaa", "yaa", "laam",
    "dha", "dal", "sheen", "zay", "haa", "waw", "ya", "bb", "al", "ghain",
    "toot", "ain", "saad", "thal", "aleff", "fa", "khaa", "jeem", "la",
    "nun", "ha", "kaaf"
]

class SignLanguageDetector:
    def __init__(self, saved_model_path, input_size=(224, 224), confidence_threshold=0.5):
        print("Loading model...")
        # Wrap the SavedModel as a TFSMLayer
        self.model_layer = TFSMLayer(saved_model_path, call_endpoint="serving_default")
        
        # Create a Keras model for inference
        inp = Input(shape=(input_size[0], input_size[1], 3))
        out = self.model_layer(inp)
        self.model = Model(inputs=inp, outputs=out)
        print("Model loaded successfully!")

        self.input_width, self.input_height = input_size
        self.conf_threshold = confidence_threshold

    def preprocess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_width, self.input_height))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)
        return img

    def detect(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("ERROR: Could not open camera")
            return

        print("Camera opened. Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Skin color range
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)

            mask = cv2.inRange(hsv, lower_skin, upper_skin)
            mask = cv2.GaussianBlur(mask, (5, 5), 0)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                c = max(contours, key=cv2.contourArea)
                if cv2.contourArea(c) > 1000:
                    x, y, w, h = cv2.boundingRect(c)
                    hand_roi = frame[y:y+h, x:x+w]

                    processed = self.preprocess(hand_roi)
                    pred = self.model.predict(processed, verbose=0)[0]
                    class_id = np.argmax(pred)
                    confidence = pred[class_id]

                    if confidence > self.conf_threshold:
                        label = LABELS[class_id]
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(frame, f"{label} ({confidence:.2f})",
                                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8, (0, 255, 0), 2)

            cv2.imshow("Arabic Sign Language - Real-Time Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


saved_model_path = "best_saved_model"
detector = SignLanguageDetector(saved_model_path)
detector.detect()
