# Speech Emotion Recognition

A speech emotion recognition demo that predicts one of eight emotions from an audio clip using MFCC features and a three-model CNN ensemble. The models were trained and evaluated on the acted speech portion of the [RAVDESS dataset](https://zenodo.org/records/1188976).

The project includes a Gradio app for trying the trained models, plus Jupyter notebooks documenting data preparation, model training, and evaluation.

## Emotions

The model predicts probabilities for:

- Angry
- Calm
- Disgust
- Fearful
- Happy
- Neutral
- Sad
- Surprised

## Try the web app

The trained model files are included in `models/`, so the app can be run without downloading the training dataset.

### Requirements

- Python
- The packages listed in `requirements.txt`

### Install and run

From the project root, create and activate a virtual environment, then install the dependencies.

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

**macOS / Linux**

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

Open the local URL printed by Gradio. Upload an audio file or record speech with your microphone. The app expects a clip of at least a few seconds; it uses up to three seconds starting 0.5 seconds into the clip and displays the four most likely emotions.

## Reproduce training

### Get the dataset

1. Download the RAVDESS **Audio_Speech_Actors_01-24.zip** file from the [RAVDESS Zenodo record](https://zenodo.org/records/1188976).
2. Extract the archive under `data/ravdess/`. The resulting layout should include folders such as `data/ravdess/Actor_01/` containing WAV files.
3. Install the requirements using the steps above.

The dataset contains 1,440 recordings from 24 actors. It is required for the notebooks, but not for running the already-trained app.

### Run the notebooks

Open the notebooks in Jupyter and run them in order. In `01_setup.ipynb`, make sure `DATA_DIR` points to `data/ravdess/`. The later notebooks define a `BASE` project path near the start; update it to the absolute path of your local project directory before running them.

| Notebook | Purpose |
| --- | --- |
| `notebooks/01_setup.ipynb` | Inspect RAVDESS recordings and create `data/metadata.csv`. |
| `notebooks/02_features.ipynb` | Extract MFCC features into `features/X_mfcc.npy`. |
| `notebooks/03_baseline.ipynb` | Create labels and actor-based splits; evaluate the SVM baseline. |
| `notebooks/04_cnn.ipynb` | Train and evaluate the initial CNN. |
| `notebooks/05_lstm_augmentation.ipynb` | Train an LSTM and create augmented training features. |
| `notebooks/05b_improvements.ipynb` | Evaluate CNN improvements including per-clip normalization and masking. |
| `notebooks/06_tune_and_test.ipynb` | Tune the final CNN ensemble, run the final test evaluation, and inspect errors. |

Feature arrays under `features/` and the raw audio under `data/ravdess/` can be regenerated or downloaded, respectively. Keep the final model files and normalization statistics under `models/`; the app needs them to make predictions.

## Project structure

```text
app.py                    Gradio application and ensemble inference
src/features.py           MFCC extraction and per-clip normalization used by the app
notebooks/                Data preparation, training, tuning, and evaluation notebooks
data/metadata.csv         Recording metadata used in the notebook workflow
data/ravdess/             Downloaded RAVDESS audio (not included in source control)
features/                 Generated feature arrays (not included in source control)
models/
  classes.json            Emotion class labels
  final_config.json       Inference configuration and ensemble members
  final_1.keras           Trained ensemble model
  final_2.keras           Trained ensemble model
  final_3.keras           Trained ensemble model
  norm_cmn.npz            Training-set normalization statistics used by the app
results.csv               Validation results for model variants
results_tuning.csv        Final-model tuning results
requirements.txt          Python dependencies
```

## How inference works

1. Audio is loaded and resampled to 22,050 Hz.
2. The first 0.5 seconds are skipped, and up to three seconds are kept.
3. Forty MFCC coefficients are extracted and padded or cropped to 130 frames.
4. Per-clip normalization and the training-set normalization statistics are applied.
5. The three CNN models predict emotion probabilities; their outputs are averaged.

The app and training workflow use the same feature settings in `src/features.py`.

## Results

Validation results for model variants are recorded in `results.csv`:

| Model | Validation accuracy | Validation macro-F1 |
| --- | ---: | ---: |
| Baseline SVM | 47.08% | 0.4536 |
| CNN | 53.33% | 0.4837 |
| LSTM | 45.83% | 0.4519 |
| CNN + augmentation | 64.58% | 0.6357 |
| LSTM + augmentation | 52.92% | 0.5218 |
| CNN + augmentation + masking | 66.25% | 0.6589 |
| CNN + augmentation + per-clip normalization | 67.08% | 0.6645 |
| CNN + augmentation + per-clip normalization + masking | 65.83% | 0.6559 |

The final three-model CNN ensemble achieved **69.58% validation accuracy** and **63.75% test accuracy** with a **0.6156 test macro-F1** (`models/final_metrics.json`). The test split contains actors 21–24; validation and test figures describe this dataset and split, not expected performance in general use.

## Limitations and responsible use

- RAVDESS consists of acted speech recorded under controlled conditions. Real-world audio, different accents, languages, microphones, and background noise may substantially reduce performance.
- The test set contains only four actors, so its results have high uncertainty.
- Emotion labels are subjective, and the model may confuse similar-sounding emotions.
- Predictions are fallible and can reflect biases in the training data. Do not use them to make consequential decisions about people.
