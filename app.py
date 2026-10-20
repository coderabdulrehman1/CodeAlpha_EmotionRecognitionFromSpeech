
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import gradio as gr

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE / "src"))
from features import extract_mfcc, cmn      

MODELS = HERE / "models"
config = json.load(open(MODELS / "final_config.json"))
classes = json.load(open(MODELS / "classes.json"))
stats = np.load(MODELS / config["norm_file"])                    # mean/std of the TRAINING set
USE_CMN = config["use_cmn"]
nets = [tf.keras.models.load_model(MODELS / f) for f in config["model_files"]]
print(f"Loaded {len(nets)} model(s). Per-clip normalisation: {USE_CMN}")


def predict(audio_path):
    """Audio file -> {emotion: probability} (average over the models in the ensemble)."""
    if audio_path is None:
        return None
    x = extract_mfcc(audio_path)                                 # (40, 130)
    if USE_CMN:
        x = cmn(x)
    x = (x - stats["mean"]) / stats["std"]                       # same normalisation as training
    x = x[np.newaxis, ..., np.newaxis].astype(np.float32)        # (1, 40, 130, 1)
    probs = np.mean([net.predict(x, verbose=0)[0] for net in nets], axis=0)
    return {c: float(p) for c, p in zip(classes, probs)}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Audio(sources=["upload", "microphone"], type="filepath", label="Upload or record speech (3+ seconds)"),
    outputs=gr.Label(num_top_classes=4, label="Predicted emotion"),
    title="Speech Emotion Recognition",
    description="Trained on RAVDESS (acted speech). Works best on clear, English speech with a clear emotion.",
)

if __name__ == "__main__":
    demo.launch()
