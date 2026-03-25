import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2 as cv

from PIL import Image
import numpy as np

categories = [] # ADD SOLAR PANEL CATEGORIES IN ALPHABETICAL ORDER

# Load the model
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(categories))
model.load_state_dict(torch.load("./TestModel.pth")) # ADD PATH TO WHERE MODEL IS SAVED INCLUDING THE MODEL NAME
model.eval()
print(model)

# Create function to prepare the image for prediction

def prepareImage(pathToImage):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    image = Image.open(pathToImage).convert("RGB")
    imgResult = transform(image)
    imgResult = imgResult.unsqueeze(0)

    return imgResult

testImage = '' # ADD PATH TO A TEST IMAGE

imgForModel = prepareImage(testImage)

with torch.no_grad():
    resultArray = model(imgForModel)
    resultArray = torch.softmax(resultArray, dim=1)
    resultArray = resultArray.numpy()

answer = np.argmax(resultArray, axis = 1)

print(answer)

index = answer[0]

print("The predicted damage to the solar panel is: " + categories[index])

img = cv.imread(testImage)
cv.putText(img, categories[index], (10, 100), cv.FONT_HERSHEY_COMPLEX, 1.6, (255, 0, 0), 3, cv.LINE_AA)
cv.imshow("image", img)
cv.waitKey(0)