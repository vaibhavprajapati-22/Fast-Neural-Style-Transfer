# Fast Neural Style Transfer

## Introduction 
Neural Style Transfer (NST) is an advanced technique in artificial intelligence that
merges the content of one image with the artistic style of another using convolutional
neural networks. This report explores the implementation and evaluation of fast Neural Style Transfer using the TransformerNet architecture, highlighting its applications,
methodology, results, and potential improvements. Through detailed analysis, the
report demonstrates how NST can transform images, maintaining the structural elements of the content image while adopting the stylistic features of the reference image.
Key findings include high-quality image generation and flexibility in style application,
though challenges such as computational demands and handling complex styles remains.

To view the project report,[Click Here]([https://github.com/user-attachments/files/15887289/Image_Denoising.pdf](https://drive.google.com/file/d/1T6ZPwOEbs9g0smvQbUrEXco4StRbMfgQ/view?usp=sharing)) or demo video, [Click Here](https://drive.google.com/file/d/1HcGsahvrPKm6-p8mgTMTgp0A_JbSovBE/view?usp=sharing).

## Table of Contents :bar_chart:
- Requirements
- Datasets Used
- Data Preprocessing
- Models Architecture
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

For the Autoencoder model, data preprocessing involves loading and resizing images to 1024x1024 pixels, separating low and high-quality images for training, and noisy and high-quality pairs for validation. Images are also split into smaller patches to aid feature learning. They're converted to tensors and normalized. CBDNet follows similar steps minus patch generation, as it doesn't enhance performance with input-sized patches. RIDNet shares preprocessing steps with the Autoencoder.
```python
def create_patches ( images , patch_size =(256 , 256 , 3) ) :
  combined_patches = []
  for image in images :
  H , W , C = image . shape
  patch_height , patch_width , patch_channels = patch_size
  patches = []
  for i in range (0 , H , patch_height ) :
    for j in range (0 , W , patch_width ) :
     patch = image [ i : i + patch_height , j : j + patch_width , :]
    patches . append ( patch )
   combined_patches . extend ( patches )
 return combined_patches
```
## Models Architecture :triangular_ruler:

### 1. Autoencoder :
The Autoencoder is a neural network tailored for image denoising tasks, comprising an encoder and decoder. The encoder compresses the input image into a lower-dimensional representation, while the decoder reconstructs the denoised image from this compressed form. This architecture efficiently removes noise by learning to preserve essential features while discarding noise through convolutional and deconvolutional layers. It ensures accurate preservation and reconstruction of spatial information critical for image quality enhancement.

### 2. CBDNet :
CBDNet consists of two subnetworks: a noise estimation network and a denoising network. The noise estimation network uses five convolutional layers with ReLU activations to estimate the noise level map in a noisy image. The denoising network takes the noisy image and the estimated noise map as inputs, employing convolutional, ReLU, and average pooling layers to extract features and reduce noise. It includes transposed convolutional layers for upsampling and additional convolutional layers for refinement, incorporating skip connections for enhanced denoising performance. The final denoised image is obtained by adding the denoised output from the network to the original noisy image.

### 3. RIDNet :
RIDNet is a convolutional neural network designed for real image denoising, addressing challenges posed by spatially variant noise in photographs. It features a modular architecture with a Channel Attention (CA) module to enhance channel dependencies, an Effective Attention Module (EAM) utilizing dilated convolutions for multi-scale feature capture, and Residual-on-Residual connections to facilitate information flow, particularly for low-frequency details.

All three model's architecture can be found in Models directory.

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
