# Neural Style Transfer using VGG-19 style loss
This project implements Neural Style Transfer (NST) using the VGG-19 deep learning model. Unlike traditional deep learning approaches that train a model to generate artistic images, this method formulates style transfer as an **optimization problem**.
By leveraging a pre-trained convolutional neural network (VGG-19), the algorithm extracts **content features** from a content image and **style features** from a style image. It then optimizes a target image to minimize a loss function, which balances content preservation and style adaptation.

![Example](https://github.com/btxviny/Neural-Style-Transfer-with-VGG-19/blob/main/result.gif)

This implementation is inspired by the paper:
[A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)
## Inference Instructions
- Install the necessary requirements:
     ```bash
     python -m pip install -r requirements.txt
     ```

- Run the following command:
     ```bash
     python neural_style.py --content_image_path ./content1.jpg --style_image_path ./style1.jpg --output_image_path ./result.jpg
     ```
-Replace ./content1.jpg with the path to your content image.
-Replace ./style1.jpg with the path to your style image.
-Replace ./result.jpg with the path where you want to store the result.
