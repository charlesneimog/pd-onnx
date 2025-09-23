import os
import numpy as np
import librosa
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from catboost import CatBoostClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


class AudioClassifier:
    def __init__(
        self,
        folders,
        model_type="catboost",  # "catboost" or "randomforest"
        n_mfcc=19,
        window_size=2048,
        max_offset=256,
        n_windows=10,
        test_fraction=0.2,
        iterations=150,  # for CatBoost
        n_estimators=150,  # for RandomForest
        random_state=42,
    ):
        self.folders = folders
        self.model_type = model_type.lower()
        self.n_mfcc = n_mfcc
        self.window_size = window_size
        self.max_offset = max_offset
        self.n_windows = n_windows
        self.test_fraction = test_fraction
        self.iterations = iterations
        self.n_estimators = n_estimators
        self.random_state = random_state

        # -------- Model toggle --------
        if self.model_type == "catboost":
            self.clf = CatBoostClassifier(
                iterations=self.iterations,
                depth=6,
                learning_rate=0.1,
                loss_function="MultiClass",
                random_seed=self.random_state,
                verbose=100,
                early_stopping_rounds=20,
            )
        elif self.model_type == "randomforest":
            self.clf = RandomForestClassifier(
                n_estimators=self.n_estimators, random_state=self.random_state
            )
        else:
            raise ValueError("model_type must be 'catboost' or 'randomforest'")

    # -------- FEATURE EXTRACTION --------
    def extract_window_features(self, y, sr):
        if len(y) < self.window_size + self.max_offset:
            return None
        start = random.randint(0, self.max_offset)
        segment = y[start : start + self.window_size]
        segment = segment * np.hanning(len(segment))
        mfcc = librosa.feature.mfcc(
            y=segment,
            sr=sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.window_size,
            hop_length=self.window_size,
        )
        return mfcc[:, 0]

    def augment_audio(self, y, sr, choice):
        if choice == "noise":
            noise_amp = 0.005 * np.random.uniform() * np.amax(y)
            y = y + noise_amp * np.random.normal(size=y.shape)
        elif choice == "pitch":
            n_steps = np.random.uniform(-2, 2)
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
        elif choice == "stretch":
            rate = np.random.uniform(0.8, 1.2)
            y = librosa.effects.time_stretch(y=y, rate=rate)
        return y

    def extract_features_from_file(self, file_path):
        y, sr = librosa.load(file_path, sr=None)
        y_trimmed, _ = librosa.effects.trim(y, top_db=60)
        if len(y_trimmed) == 0:
            return []

        feats = []
        for _ in range(self.n_windows):
            for aug in ["noise", "pitch", "stretch", "none"]:
                y_aug = self.augment_audio(y_trimmed, sr, aug)
                y_windowed = y_aug * np.hanning(len(y_aug))
                mfcc = librosa.feature.mfcc(
                    y=y_windowed,
                    sr=sr,
                    n_mfcc=self.n_mfcc,
                    n_fft=self.window_size,
                    hop_length=self.window_size,
                )
                for i in range(mfcc.shape[1]):
                    feats.append(mfcc[:, i])
        return feats

    # -------- DATA LOADING --------
    def load_data(self):
        X_train, y_train = [], []
        X_test, y_test = [], []

        for label, folder in self.folders.items():
            all_files = [
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.endswith((".aif", ".aiff"))
            ]
            n_test = max(1, int(len(all_files) * self.test_fraction))
            test_files = random.sample(all_files, n_test)
            train_files = [f for f in all_files if f not in test_files]

            print(f"Processing class '{label}'...")

            # Training
            for idx, f in enumerate(train_files, 1):
                feats = self.extract_features_from_file(f)
                for feat in feats:
                    X_train.append(feat)
                    y_train.append(label)
                print(f"  Train: processed {idx}/{len(train_files)} files", end="\r")

            # Testing
            for idx, f in enumerate(test_files, 1):
                feats = self.extract_features_from_file(f)
                for feat in feats:
                    X_test.append(feat)
                    y_test.append(label)
                print(f"  Test: processed {idx}/{len(test_files)} files", end="\r")

            print()  # new line after finishing class

        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.X_test = np.array(X_test)
        self.y_test = np.array(y_test)
        print("Data loading complete!")

    # -------- TRAINING --------
    def train(self):
        if self.model_type == "catboost":
            self.clf.fit(
                self.X_train, self.y_train, eval_set=(self.X_test, self.y_test)
            )
        else:
            self.clf.fit(self.X_train, self.y_train)

    # -------- EVALUATION --------
    def evaluate(self):
        y_pred = self.clf.predict(self.X_test)
        print(classification_report(self.y_test, y_pred))

    # -------- PCA VISUALIZATION --------
    def pca_visualize(self, n_components=2):
        pca = PCA(n_components=n_components)
        X_all = np.vstack([self.X_train, self.X_test])
        y_all = np.hstack([self.y_train, self.y_test])
        X_2d = pca.fit_transform(X_all)
        return X_2d, y_all

    # -------- EXPORT TO ONNX --------
    def export_onnx(self, filename="model.onnx"):
        if self.model_type == "randomforest":
            initial_type = [("mfcc", FloatTensorType([None, self.X_train.shape[1]]))]
            result = convert_sklearn(
                self.clf, initial_types=initial_type, options={"zipmap": False}
            )
            if isinstance(result, tuple):
                onnx_model, _topology = result
            else:
                onnx_model = result
            with open(filename, "wb") as f:
                f.write(onnx_model.SerializeToString())
        elif self.model_type == "catboost":
            self.clf.save_model(filename, format="onnx")
        print(f"Model exported to {filename}")

    # -------- HASH FUNCTION --------
    def shash(self, s: str) -> str:
        v = 5381
        if s:
            for c in s:
                v = ((v << 5) + v) + ord(c)
                v &= 0xFFFFFFFF
        return f"0x{v:08X}"


if __name__ == "__main__":
    folders = {
        "attack-percurssivo": "./../Flute/attack-percurssivo",
        "jet-whistle": "./../Flute/jet-whistle",
        "som+ar": "./../Flute/som+ar",
        # "silence": "./../Flute/silence",
    }

    # Toggle model here: "catboost" or "randomforest"
    classifier = AudioClassifier(folders=folders, model_type="catboost")

    classifier.load_data()
    print("Feature dimension:", classifier.X_train.shape[1])
    classifier.train()
    classifier.evaluate()
    classifier.export_onnx("model.onnx")
