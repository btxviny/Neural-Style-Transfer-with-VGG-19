import numpy as np
import cv2
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

device = 'cuda'

def parse_args():
    parser = argparse.ArgumentParser(description='Neural Style Transfer using VGG-19')
    parser.add_argument('--content_image_path', type=str, help='Directory of content image')
    parser.add_argument('--style_image_path', type=str, help='Directory of style image')
    parser.add_argument('--output_image_path', type=str, help='Directory to store output')
    return parser.parse_args()

def load_image(image_path, shape):
    image = Image.open(image_path)
    if shape is not None:
        image = image.resize((shape[0], shape[1]))
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image).unsqueeze(0)

def compute_style_transfer_loss(content_image, style_image, target_image, content_weight = 1.0, style_weight = 1e6):
    content_features = get_features(content_image)
    style_features = get_features(style_image)
    
    target_features = get_features(target_image)
    content_loss = torch.mean((target_features['conv4_1'] - content_features['conv4_1'])**2)
    
    style_loss = 0
    for layer in style_features:
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)
        style_gram = gram_matrix(style_features[layer])
        layer_style_loss = torch.mean((target_gram - style_gram)**2)
        style_loss += layer_style_loss
    
    total_loss = content_weight * content_loss + style_weight * style_loss
    return total_loss
# Define a function to get features from VGG network
def get_features(image):
    layers = {
        '0': 'conv1_1', '5': 'conv2_1',
        '10': 'conv3_1', '19': 'conv4_1',
        '28': 'conv5_1'
    }
    features = {}
    x = image
    for name, layer in vgg19._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

# Define a function to compute the Gram matrix
def gram_matrix(input):
    batch, channels, h, w = input.size()
    features = input.view(batch * channels, h * w)
    gram_matrix = torch.mm(features, features.t())
    return gram_matrix

if __name__ == '__main__':
    vgg19 = models.vgg19(weights='VGG19_Weights.IMAGENET1K_V1').features.to(device).eval()
    args = parse_args()
    # assert valid inputs
    if not os.path.exists(args.content_image_path):
        print(f"Error: Content image file '{args.content_image_path}' does not exist.")
        exit()
    if not os.path.exists(args.style_image_path):
        print(f"Error: Style image file '{args.style_image_path}' does not exist.")
        exit()
    _, content_ext = os.path.splitext(args.content_image_path)
    _, style_ext = os.path.splitext(args.style_image_path)
    if (content_ext.lower() not in ['.jpg', '.png']) or (style_ext.lower() not in ['.jpg', '.png']):
        print(f"Error: Invalid image file extension. Supported extensions are {', '.join(['.jpg', '.png'])}.")
        exit()

    content_image = load_image(args.content_image_path, shape = (512,256)).to(device)
    style_image = load_image(args.style_image_path, shape = (512,256)).to(device)
    target_image = content_image.clone().requires_grad_(True).to(device)
    optimizer = optim.Adam([target_image], lr=0.05)
    num_iterations = 1000
    cv2.namedWindow('window',cv2.WINDOW_NORMAL)
    # Optimization loop
    for iteration in range(num_iterations):
        torch.cuda.empty_cache()
        optimizer.zero_grad()
        loss = compute_style_transfer_loss(content_image, style_image, target_image, content_weight= 1, style_weight= 1e3)
        loss.backward()
        optimizer.step()

        image = target_image.clone().cpu().detach().squeeze(0).permute(1,2,0).numpy()
        image = (image * np.array([0.229, 0.224, 0.225],dtype = np.float32)) + np.array([0.485, 0.456, 0.406],dtype = np.float32)
        image = (np.clip(image,0,1) * 255.0).astype(np.uint8)
        style = style_image.clone().cpu().squeeze(0).permute(1,2,0).numpy()
        style = (style * np.array([0.229, 0.224, 0.225],dtype = np.float32)) + np.array([0.485, 0.456, 0.406],dtype = np.float32)
        style = (np.clip(style,0,1) * 255.0).astype(np.uint8)
        concatenated = cv2.cvtColor(cv2.hconcat([image,style]),cv2.COLOR_BGR2RGB)
        cv2.imshow('window',concatenated)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break
        print(f"\rIteration {iteration}: Loss {loss.item()}", end = "")
    cv2.destroyAllWindows()
    Image.fromarray(image, "RGB").save(args.output_image_path)