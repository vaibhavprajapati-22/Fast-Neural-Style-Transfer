# Fast Neural Style Transfer

## Introduction 
Neural Style Transfer (NST) is an advanced technique in artificial intelligence that
merges the content of one image with the artistic style of another using convolutional
neural networks. This report explores the implementation and evaluation of fast Neural Style Transfer using the TransformerNet architecture, highlighting its applications,
methodology, results, and potential improvements. Through detailed analysis, the
report demonstrates how NST can transform images, maintaining the structural elements of the content image while adopting the stylistic features of the reference image.
Key findings include high-quality image generation and flexibility in style application,
though challenges such as computational demands and handling complex styles remains.

To view the project report,[Click Here](https://drive.google.com/file/d/1T6ZPwOEbs9g0smvQbUrEXco4StRbMfgQ/view?usp=sharing) or demo video, [Click Here](https://drive.google.com/file/d/1HcGsahvrPKm6-p8mgTMTgp0A_JbSovBE/view?usp=sharing).

## Table of Contents :bar_chart:
- Requirements
- Datasets Used
- Data Preprocessing
- Models Architecture
- Loss Functions
- Model Training and Hyperparameters
- Model Evaluation
- How To Run
- Improvements
- References

## Requirements :bell:
```
pillow
torch
torchvision
tqdm
collections
```
## Dataset Used :school_satchel:
I have used COCO 2017 dataset for training the TransformerNet. The COCO (Common
Objects in Context) dataset is a large-scale object detection, segmentation, and captioning
dataset widely used in the computer vision community. By selecting 40,000 test images from
the COCO 2017 dataset, we ensured a diverse and comprehensive training set that includes
a wide range of content types and visual scenarios. This diversity is crucial for enhancing
the robustness and generalization capabilities of the TransformerNet model.
8 Styles images were used to train TransformerNets. Each style need a saparate TransformetNet to train.


For COCO 2017 dataset [Click Here]([https://www.kaggle.com/datasets/soumikrakshit/lol-dataset](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset))

## Data Preprocessing :scissors:
1. Data Preprocessing for COCO 2017 Images
   ```ptython
   def TrainImagesPath(path):
    images_path = []
    for file in os.listdir(path):
        if file == "test2017":
            file_path = os.path.join(path, file)
            for f in os.listdir(file_path):
                image_path = os.path.join(file_path, f)
                images_path.append(image_path)
                if len(images_path) == 40000:
                    break
    return images_path
   ```
   The code focuses on a specific folder named ”test2017” within the provided path. This
  suggests it’s only interested in a particular subset of images for training, possibly due
  to limitations or training efficiency.
    ```python
    train_transform = transforms.Compose([
                                transforms.Resize((256, 256)),
                                transforms.CenterCrop((256, 256)),
                                transforms.ToTensor(),
                                transforms.Lambda(lambda x: x.mul(255))
                                    ])
    ```
    By applying transforms.Resize and transforms.CenterCrop, it enforces a uniform size
    (256x256 pixels) on all images. This ensures the model receives data with consistent dimensions. The transforms.ToTensor function converts the loaded images (likely in PIL
    format) into tensors, which is a format commonly used by machine learning frameworks like PyTorch. A lambda function to multiply each pixel value by 255. This
    normalizes the pixel intensities to a range between 0 and 1, which can be beneficial for
    certain machine learning algorithms. This is a lambda function that takes a tensor x
    (representing an image) as input. The mul function (short for multiply) element-wise
    multiplies each value in the tensor by 255.
2. Data Preprocessing for Style Images:
   ```python
   def LoadStyleImage(path, batch_size, device):
    style = Image.open(path).convert('RGB')
    style = style.resize((256, 256))
    style = style_transform(style)
    style = style.repeat(batch_size, 1, 1, 1)
    style = style.to(device) 
    return style
   ```
   The image is opened using Image.open and converted to RGB format with convert('RGB'), ensuring it has three color channels. It's resized to 256x256 pixels with style.resize((256, 256)) for model consistency.    The preprocessed style image is then replicated batchsize times using style.repeat(batchsize, 1, 1, 1), which is useful for batch processing. Finally, the tensor is transferred to the specified device (CPU or    GPU) with .to(device).
   ```python
   style_transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Lambda(lambda x: x.mul(255))
                            ])
   ```
   Converts the PIL image format into a PyTorch tensor suitable for computations. Normalizes the pixel values between 0 and 255.

## Models Architecture :triangular_ruler:

### 1. ConvLayer :
This class represents a convolutional layer, a fundamental building block in Convolutional
Neural Networks (CNNs). It performs the core operation of a CNN - extracting features
from the input data using learnable filters (kernels).
### 2. ResidualBlock :
This class defines a residual block, a commonly used architecture in deep residual networks.
It introduces a ”shortcut connection” that allows the gradient to flow directly through the
layers, addressing the vanishing gradient problem that can hinder training deep neural networks.

### 3. UpsampleConvLayer :
This class defines a convolutional layer with upsampling capabilities. It’s often used in decoder parts of networks where the goal is to increase the spatial resolution of the feature maps.

### 4. TransformerNet
It combines convolutional layers, residual blocks, and upsampling convolutions to achieve
the desired style transfer effect.

## Loss Functions :
1. Feature reconstruction loss encourages the model to capture similar features rather than exact pixel values. It uses a pre-trained network to extract key features like edges and textures from both the target and output images. The loss is calculated based on the difference between these features, ensuring the model captures the essence of the target image, even if pixel values differ.
   ![image](https://github.com/vaibhavprajapati-22/Fast-Neural-Style-Transfer/assets/148644657/263a46e9-6c3c-4cb4-af04-6de59963da95)

2. Style reconstruction loss analyzes feature co-occurrences using a pre-trained network, calculating the Gram matrices to capture stylistic elements like colors and textures. By comparing these matrices between the generated and reference images across multiple layers, it ensures the model captures stylistic fingerprints accurately, from small-scale brushstrokes in lower layers to larger-scale color harmonies in higher layers
   
   ![image](https://github.com/vaibhavprajapati-22/Fast-Neural-Style-Transfer/assets/148644657/4bde0d5b-3421-464e-a37e-b7b4ca3368c3)

## Model Training and Hyperparameters :dart:
All the model were trained on GPU P100 that is available on kaggle. Model training and
hyperparametrs of different models is given below :
###  Autoencoder :
  - Patch size : 256 x 256
  - Batch size : 32
  - Loss : Mean squared error
  - Optimizer : Adam
  - Learning rate : 0.001
  - Learning rate scheduler : learning rate becomes 0.9 after every 10 epochs
  - Epochs : 100
### CDBNet :
  - Patch size : 256 x 256
  - Batch size : 32
  - Loss : Mean squared error
  - Optimizer : Adam
  - Learning rate : 0.001
  - Learning rate scheduler : learning rate becomes 0.9 after every 50 epochs
  - Epochs : 2000
### RIDNet :
  - Patch size : 256 x 256
  - Batch size : 8
  - Loss : L1 loss
  - Optimizer : Adam
  - Number of feactures : 128
  - Learning rate : 0.001
  - Learning rate scheduler : learning rate becomes 0.9 times after every 10 epochs
  - Epochs : 100

### Training Function : 
```python
for epoch in range(epochs):
    loss_e = 0
    for batch in tqdm(dataloader):
        Model.train()
        optimizer.zero_grad()
        
        low_img, high_img = batch
        low_img = low_img.to(device)
        high_img = high_img.to(device)
        
        denoished_img = Model(low_img)
        
        error = criterion(denoished_img, high_img) 
        
        error.backward()
        optimizer.step()
        loss_e += error.item()

    scheduler.step()
    loss_e /= len(dataloader)
    loss.append(loss_e)
    val_loss, val_psnr = validation(Model)
    train_loss, train_psnr = training(Model)
    print(f"{epoch+1} / {epochs} Runnung Training loss : {loss_e}")
    print(f"Training loss : {train_loss:.4f} Training PSNR : {train_psnr:.4f} Validation Loss : {val_loss:.4f} Validation PSNR : {val_psnr:.4f}")
```
### Model weights 
Download weights and config files form hugging face :
[Autoencoder](https://huggingface.co/vaibhavprajapati22/Image_Denoising_Autoencoder)
[CBDNet](https://huggingface.co/vaibhavprajapati22/Image_Denoising_CBDNet)
[RIDNet](https://huggingface.co/vaibhavprajapati22/Image_Denoising_RIDNet)
### Training PSNR :
  - Autoencoder : 16.4439
  -  CBDNet : 35.2989
  -  RIDNet : 26.6179
### Loss :
![image](https://github.com/vaibhavprajapati-22/Image-Denoising/assets/148644657/f85551a0-c0e6-456d-99a8-f31ea93bfced)

## Model Evaluation :loudspeaker:
### Validation PSNR :
  - Autoencoder : 15.391
  - CBDNet : 22.016
  - RIDNet : 22.379
### Validation SSIM :
  - Autoencoder : 0.594
  - CBDNet : 0.789
  - RIDNet : 0.746

### Testing On Some Images :
![image](https://github.com/vaibhavprajapati-22/Image-Denoising/assets/148644657/84f76c61-43d8-4af8-8c87-9604b907276b)

## How To Run :gun:
  1. Clone the repository:
    <pre>
    <code class="python">
    git clone https://github.com/vaibhavprajapati-22/Image-Denoising
    </code>
    </pre>
  2. Install the required dependencies:
     <pre>
      <code class="python">
        cd Image-Denoising
        pip install -r requirements.txt
      </code>
     </pre>
  3. Run main script
     <pre>
      <code class="python">
        python main.py
      </code>
     </pre>

After running the main.py file images in test/low will be read and correspoing outputs will be stored in test/predicted. Make sure that images are in png format.

## Improvements :mag_right:
  - Increase Computational Resources: Utilize platforms with longer session limits or more powerful GPUs for more extensive training, particularly benefiting CBDNet.
  - Data Augmentation: Enhance training data diversity with techniques like random cropping, flipping, and adding noise patterns.
  - Early Stopping: Prevent overfitting by monitoring validation loss and stopping training when it increases.
## References :paperclip:
  1. https://www.ni.com/en-in/innovations/white-papers/11/peak-signal-to-noise-ratio-as-an-image-quality-metric.html
  2. https://keras.io/examples/vision/autoencoder/
  3. https://arxiv.org/pdf/1807.04686v2
  4. https://arxiv.org/pdf/1904.07396v2
