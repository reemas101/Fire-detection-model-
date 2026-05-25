# Fire and Smoke Detection — Project README

ARTI 404 — Image Processing — Group 3  
Imam Abdulrahman Bin Faisal University · Spring 2026

This repository implements an image-based fire and smoke detection system that combines classical image processing with a transfer-learned MobileNetV2 CNN. It is the complete deliverable for the ARTI 404 course project.

---

# Deliverables in this repository

| File | What it is |
|------|-------------|
| `Fire_Smoke_Detection.ipynb` | The Jupyter notebook with the complete implementation. Run it top-to-bottom to reproduce the results. |
| `Fire_Smoke_Detection_Paper.docx` | The IEEE-style research paper. Open in Word and fill in actual experimental numbers in Table II once you have results. |
| `Fire_Smoke_Detection_Slides.pptx` | The presentation deck. 13 slides. |
| `realtime.py` | Standalone webcam / video inference script. |
| `README.md` | This file. |

---

# Quick-start

## 1. Set up the environment

```bash
# Python 3.9+ recommended. Use a virtualenv or conda.

python -m venv venv

# Windows
venv\Scripts\activate

# Linux / Mac
source venv/bin/activate

pip install tensorflow opencv-python scikit-learn matplotlib seaborn pillow tqdm jupyter
```

---

## 2. Get the dataset

Download the Kaggle fire/smoke dataset  
(example: `phylake1337/fire-dataset` or any similar fire/smoke dataset).

Arrange it like this:

```text
data/fire_dataset/
├── fire/
├── smoke/
└── non_fire/
```

If your dataset has only two classes (`fire / non_fire`), change the notebook config to:

```python
CLASSES = ["non_fire", "fire"]
NUM_CLASSES = 2
```

---

## 3. Run the notebook

```bash
jupyter notebook Fire_Smoke_Detection.ipynb
```

Run all cells in order.

- Colab GPU: about 20–30 minutes
- CPU: about 2–3 hours

---

## 4. Run the live demo

After training, the model is saved as:

```text
outputs/models/mobilenetv2_best.keras
```

Run:

```bash
python realtime.py --source 0
```

or:

```bash
python realtime.py --source data/test_video.mp4
```

Press `q` to quit.

The script prints:
- FPS
- CNN call count
- skipped frames

This helps show how the pre-filter saves computation.

---

# Recommended repository structure

```text
fire-smoke-detection/
├── data/
│   └── fire_dataset/
│       ├── fire/
│       ├── smoke/
│       └── non_fire/
│
├── notebooks/
│   └── Fire_Smoke_Detection.ipynb
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── classical.py
│   ├── models.py
│   ├── train.py
│   └── realtime.py
│
├── outputs/
│   ├── models/
│   ├── figures/
│   └── reports/
│
├── paper/
│   └── Fire_Smoke_Detection_Paper.docx
│
├── slides/
│   └── Fire_Smoke_Detection_Slides.pptx
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

# Minimal `.gitignore`

```gitignore
venv/
__pycache__/
*.pyc
data/
outputs/
.ipynb_checkpoints/
.DS_Store
```

---

# Minimal `requirements.txt`

```txt
tensorflow>=2.12
opencv-python>=4.5
scikit-learn>=1.0
matplotlib>=3.5
seaborn>=0.11
pillow>=9.0
tqdm>=4.60
jupyter>=1.0
```

---

# Implementation Timeline (10 Days)

| Day | Task | Owner |
|----|------|------|
| 1 | Set up repo + environment. Download and inspect dataset. | Everyone |
| 2 | Run preprocessing pipeline. Generate sample images. | 1 person |
| 3 | Run classical baseline. Record scores. | 1 person |
| 4 | Train MobileNetV2 (stage 1). Save best checkpoint. | 1 person |
| 5 | Fine-tune model (stage 2). Plot training curves. | Same person |
| 6 | Run evaluation script. Generate confusion matrix and ROC curves. | 1 person |
| 7 | Update paper with real results. | Writers |
| 8 | Update presentation with charts and screenshots. | Same |
| 9 | Rehearse demo and test realtime.py. | Everyone |
| 10 | Final review and GitHub submission. | Everyone |

---

# Before Submission

1. Run the notebook with the real dataset.
2. Replace placeholder values in Table II.
3. Insert figures into the paper and slides.
4. Test the live demo.
5. Push everything to GitHub and submit.

---

# ARTI 404 Rubric Mapping

| Rubric Line | Where it is satisfied | Marks |
|-------------|----------------------|------|
| First draft | Submitted PDF | 2 |
| Presentation clarity | Slides + delivery | 2 |
| Conference style | IEEE paper template | 1 |
| Paper organization | Sections I–VII | 2 |
| Research quality | Hybrid classical + DL system | 5 |
| Source code quality | Modular notebook + realtime.py | 3 |

**Total: 15**

---

# Image Processing vs Deep Learning

## Image Processing
- Resize
- Normalize
- Median filter
- Gaussian blur
- CLAHE
- HSV/RGB rules
- Morphological operations
- Canny edges
- Classical pre-filter

## Deep Learning
- Custom CNN
- MobileNetV2
- Transfer learning
- Callbacks

## Hybrid Runtime
- Classical pre-filter + CNN decision system

---

# Risk Analysis and Limitations

| Risk | Mitigation |
|------|------------|
| Dataset too small | Use augmentation and class weights |
| False positives from sunsets/orange colors | CNN reduces this issue |
| Out-of-memory during training | Reduce batch size |
| TensorFlow/OpenCV installation errors | Use Google Colab |
| Live demo failure | Prepare backup recorded video |

---

# References

See the References section inside:

```text
Fire_Smoke_Detection_Paper.docx
```

---

# Authors

- Amal Saad Alghamdi
- Sadeem Ayman AlTurki
- Layan Abdulaziz Alomair
- Rimas Yousif Alajmi
- Maryam Amr Alghamdi
- Eman Yassen Alsadh

Imam Abdulrahman Bin Faisal University  
College of Computer Science and Information Technology