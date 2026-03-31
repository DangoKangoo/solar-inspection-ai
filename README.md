# Solar Inspection AI

Local Streamlit app for reviewing a solar panel image and showing a model-based quality/risk assessment.

## Repo Layout

```text
backend/                     Shared inference logic and tests
dashboard/                   Streamlit dashboard UI
dashboard/results/runs/      Saved inspection runs the dashboard can open
models/                      Local model artifact guidance
Deeplearning_FinalProject_ResNetModel.py
DeepLearning_FinalProject_TransferLearn_ModelTest.py
requirements.txt
```

## Main Entry Points

- `streamlit run dashboard/app.py`
- `python DeepLearning_FinalProject_TransferLearn_ModelTest.py <image_path>`
- `python Deeplearning_FinalProject_ResNetModel.py`

## Model Files

Place the trained files here:

- `models/checkpoints/TestModel.pth`
- `models/checkpoints/classes.txt`

`classes.txt` must list one class name per line in the same order the model outputs logits.

## Install

```bash
python -m pip install -r requirements.txt
```

## Current Workflow

1. Launch the dashboard.
2. Upload a single cropped panel image.
3. Click `Analyze panel`.
4. The backend writes a run into `dashboard/results/runs/`.
5. The dashboard reloads that run and shows class, confidence, fault probability, and risk level.
