# Models

Store local model artifacts here.

## Expected files

- `models/checkpoints/TestModel.pth`
- `models/checkpoints/classes.txt`

`TestModel.pth` is intentionally ignored by Git, so copy it in locally once you receive it.
`classes.txt` should contain one class label per line in model-output order.

If you retrain with [Deeplearning_FinalProject_ResNetModel.py](/d:/Projects/solar-inspection-ai/Deeplearning_FinalProject_ResNetModel.py), it now writes both files into `models/checkpoints/`.
