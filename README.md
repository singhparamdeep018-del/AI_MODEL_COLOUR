# Micro-Mend AI

**AI-Assisted Colorimetric Wound Monitoring Prototype**

## 1. Project overview

Micro-Mend AI is a research prototype that pairs a future color-responsive
wound dressing with computer vision and machine learning. As a
color-responsive hydrogel sensing layer reacts to the wound environment, its
color shift is intended to be captured with a phone camera, measured with
computer vision, and classified into a defined prototype state by a trained
model.

## 2. Problem

Manual wound assessment is subjective and infrequent. A low-cost, color-based
sensing layer paired with automated image analysis could give patients and
clinicians a consistent, frequent way to observe changes in a wound
environment between checkups.

## 3. Proposed solution

A biodegradable dressing containing a color-responsive hydrogel layer, imaged
with a phone camera and interpreted by a computer-vision + machine-learning
pipeline that classifies the observed color into a defined state.

## 4. Current prototype

The physical hydrogel sensing layer is **still under development**. This
software prototype demonstrates the image-to-color-state pipeline using
standardized synthetic reference color images as a stand-in for the future
hydrogel output. It does not process real wound images and makes no clinical
claims.

Six prototype color states are defined:

| State | Color |
|---|---|
| S1_yellow | Yellow |
| S2_yellow_green | Yellow-Green |
| S3_green | Green |
| S4_blue_green | Blue-Green |
| S5_blue | Blue |
| S6_purple | Purple |

## 5. AI pipeline

1. Image upload (PNG / JPG / JPEG)
2. Preprocessing (RGB/HSV conversion)
3. Color-region detection (masks out background/near-neutral pixels)
4. RGB + HSV statistical feature extraction (24-dimensional feature vector)
5. Dominant-hue histogram detection
6. Machine-learning classification
7. Confidence scoring across the six prototype states
8. Rule-based color-engine cross-check and MobileNetV2 experimental
   cross-check, following the multi-model comparison approach validated
   during prototyping

Three complementary approaches are combined, mirroring what was already
validated in the original prototype scripts:

- **RGB + HSV Random Forest** (`models/micromend_color_ml_model.pkl`) -
  primary model. This is a colorimetric prototype at heart, and the Random
  Forest operates directly on RGB/HSV statistics, which is the most
  defensible signal for a color-responsive sensor. It also exposes a clean
  6-class probability distribution for the confidence display.
- **MobileNetV2** (`models/micromend_color_model.keras`) - experimental deep
  learning cross-check, preserved from the original prototype. During
  inspection this model occasionally disagreed with the color engine on
  real (non-synthetic) images, so it is shown as a supporting signal rather
  than the primary result.
- **Rule-based color engine** - a deterministic hue/saturation/value
  classifier with no learned parameters. Always available even if both
  model files are missing.

The app's "System cross-check" panel shows whether all three approaches
agree, the same majority-vote idea used in the original `test_real_image.py`
script.

## 6. Dataset

`micromend_color_dataset/` (not included in this package to avoid
duplicating a large binary dataset) contains synthetic reference-color
images split into `train/validation/test`, 6 classes, ~400/100/100 images
per class. Place it at the project root if you want to retrain either model
with the scripts in `scripts/`.

## 7. Model

- `models/micromend_color_ml_model.pkl` - `RandomForestClassifier`
  (scikit-learn), 300 trees, trained on the 24-dimensional RGB+HSV feature
  vector described above.
- `models/micromend_color_model.keras` - MobileNetV2 (ImageNet-pretrained
  backbone, frozen) with a small classification head, fine-tuned on the
  synthetic color dataset.

Both are loaded once and cached via `st.cache_resource` so the app does not
reload them on every interaction.

## 8. How to install

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 9. How to run

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (typically `http://localhost:8501`).

To retrain either model (requires the dataset described in section 6):

```bash
python scripts/train_random_forest.py   # RGB + HSV Random Forest
python scripts/train_mobilenet.py       # MobileNetV2 experimental model
```

## 10. Current limitations

- Uses standardized/synthetic reference color images, not real hydrogel or
  wound images.
- MobileNetV2 has occasionally disagreed with the color engine on real
  images; it is shown as an experimental cross-check, not the primary
  result.
- No healing-time, biomarker, or clinical output is produced.
- Small, synthetic dataset (6 classes, ~600 images/class across splits).

## 11. Future development

- Integrate real hydrogel imagery as the material becomes available
- Experimental colorimetric calibration against physical reference cards
- Biomarker-state mapping
- Larger, real-world dataset
- Healing-time trajectory modelling
- Longitudinal wound observation and advanced predictive modelling

## 12. Disclaimer

Prototype demonstration only. This system is not a medical diagnostic
device, and its current color-state output should not be used to make
clinical decisions. It does not predict wound-healing time, biomarker
concentrations, or any clinical outcome.

## Project structure

```
micro_mend_ai/
├── app.py                          # Streamlit application
├── requirements.txt
├── README.md
│
├── models/
│   ├── micromend_color_model.keras       # MobileNetV2 (experimental)
│   └── micromend_color_ml_model.pkl      # RGB+HSV Random Forest (primary)
│
├── src/
│   ├── color_analysis.py           # color masking, RGB/HSV stats, feature vector
│   ├── prediction.py               # model loading + inference orchestration
│   └── utils.py                    # image validation / loading helpers
│
├── scripts/
│   ├── train_random_forest.py      # retrain the RGB+HSV Random Forest
│   └── train_mobilenet.py          # retrain the MobileNetV2 model
│
└── assets/
    └── samples/                    # small reference images for quick testing
```
