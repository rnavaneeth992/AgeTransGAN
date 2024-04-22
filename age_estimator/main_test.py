import os
import torch
import numpy as np
import cv2
import torch.nn as nn
from model.resnet50_ft_dims_2048 import resnet50_ft
import matplotlib.pyplot as plt
import argparse

START_AGE = 0
END_AGE = 100

def predict(model, image):
    model.eval()
    with torch.no_grad():
        image = np.transpose(image, (2, 0, 1))  # Transpose to (C, H, W)
        img = torch.from_numpy(image).cuda()
        img = img.type('torch.FloatTensor').cuda()
        
        output = model(img[None])
        m = nn.Softmax(dim=1)
        output = m(output)
        
        a = torch.arange(START_AGE, END_AGE + 1, dtype=torch.float32).cuda()
        mean = (output * a).sum(1, keepdim=True).cpu().data.numpy()
        pred = np.around(mean)[0][0]
    return pred

def get_args():
    parser = argparse.ArgumentParser(description='Age Prediction from a Single Image')
    parser.add_argument('-pm', '--pred_model', type=str, required=True, help='Path to the trained model')
    parser.add_argument('-i', '--image_path', type=str, required=True, help='Path to the input image')
    return parser.parse_args()

def main():
    args = get_args()
    
    model = resnet50_ft()
    model.load_state_dict(torch.load(args.pred_model))
    model.eval()
    model.cuda()
    
    img = cv2.imread(args.image_path)
    if img is None:
        print("Image not found.")
        return

    resized_img = cv2.resize(img, (224, 224))  # Ensure your input image is resized to the model's expected input size
    pred_age = predict(model, resized_img)

    print(f"Predicted Age: {pred_age}")

    # Optionally display the image
    img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.title(f"Prediction: {pred_age}")
    plt.show()

if __name__ == "__main__":
    main()