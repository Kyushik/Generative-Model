# Generative Models

This repository is for implementation of `Generative Models` using [Tensorflow](https://www.tensorflow.org). 

[Description of the Papers (Korean)](https://www.notion.so/Code-Description-53c93afd0b9740728143ffab1b2caa2f)

The structure of the code is based on the [Hwalsuk Lee's Generative Model github repository](https://github.com/hwalsuklee/tensorflow-generative-model-collections)  



## Implemented Paper List

1. [Auto-Encoding Variational Bayes (VAE)](https://arxiv.org/abs/1312.6114)
2. [Generative Adversarial Networks (GAN)](https://arxiv.org/abs/1406.2661)
3. [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks (DCGAN)](https://arxiv.org/abs/1511.06434)
4. [Least Squares Generative Adversarial Networks (LSGAN)](https://arxiv.org/abs/1611.04076)
5. [Wasserstein GAN (WGAN)](https://arxiv.org/abs/1701.07875)
6. [Improved Training of Wasserstein GANs (WGAN GP)](https://arxiv.org/abs/1704.00028)
7. [Conditional Generative Adversarial Nets (CGAN)](https://arxiv.org/abs/1411.1784)
8. [Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets (InfoGAN)](https://arxiv.org/abs/1606.03657)
9. [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (CycleGAN)](https://arxiv.org/abs/1703.10593)
10. [Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation (StarGAN)](https://arxiv.org/abs/1711.09020)
11. [Attention-Guided Generative Adversarial Networks for Unsupervised Image-to-Image Translation (AGGAN)](https://arxiv.org/abs/1903.12296)
12. [Neural Discrete Representation Learning(VQ-VAE)](https://arxiv.org/abs/1711.00937)
13. [Adherent Raindrop Removal with Self-Supervised Attention Maps andSpatio-Temporal Generative Adversarial Networks](http://openaccess.thecvf.com/content_ICCVW_2019/papers/ADW/Alletto_Adherent_Raindrop_Removal_with_Self-Supervised_Attention_Maps_and_Spatio-Temporal_Generative_ICCVW_2019_paper.pdf)

14. [Beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl)

15. [HoloGAN: Unsupervised Learning of 3D Representations From Natural Images](https://arxiv.org/abs/1904.01326)



## GAN Results

### 1. GAN

**MNIST**

<img src="./image/GAN_MNIST.png" width="400">



### 2. DCGAN

|                      MNIST                      |                      CelebA                      |
| :---------------------------------------------: | :----------------------------------------------: |
| <img src="./image/DCGAN_MNIST.png" width="500"> | <img src="./image/DCGAN_CelebA.png" width="500"> |



### 3. LSGAN

|                      MNIST                      |                      CelebA                      |
| :---------------------------------------------: | :----------------------------------------------: |
| <img src="./image/LSGAN_MNIST.png" width="500"> | <img src="./image/LSGAN_CelebA.png" width="500"> |



### 4. WGAN

|                     MNIST                      |                     CelebA                      |
| :--------------------------------------------: | :---------------------------------------------: |
| <img src="./image/WGAN_MNIST.png" width="500"> | <img src="./image/WGAN_CelebA.png" width="500"> |



### 5. WGAN-GP

|                       MNIST                       |                       CelebA                       |
| :-----------------------------------------------: | :------------------------------------------------: |
| <img src="./image/WGAN-GP_MNIST.png" width="500"> | <img src="./image/WGAN-GP_CelebA.png" width="500"> |



### 6. Conditional GAN

**MNIST**

<img src="./image/ConditionalGAN_MNIST.png" width="400">

### 7. InfoGAN

**MNIST**

<img src="./image/InfoGAN_MNIST.png" width="400">

### 8. HoloGAN

**CelebA**

<img src="./image/HoloGAN_CelebA.png" width="800">



## Image-to-Image Translation Results

### 1. CycleGAN

|                      Monet to Photo                      |                      Photo to Monet                      |
| :------------------------------------------------------: | :------------------------------------------------------: |
| <img src="./image/CycleGAN_Monet2Photo.png" width="400"> | <img src="./image/CycleGAN_Photo2Monet.png" width="400"> |

|                      Horse to Zebra                      | Zebra to Horse                                           |
| :------------------------------------------------------: | -------------------------------------------------------- |
| <img src="./image/CycleGAN_Horse2Zebra.png" width="400"> | <img src="./image/CycleGAN_Zebra2Horse.png" width="400"> |



### 2. AGGAN

|                    Horse to Zebra                     |                    Zebra to Horse                     |
| :---------------------------------------------------: | :---------------------------------------------------: |
| <img src="./image/AGGAN_Horse2Zebra.png" width="400"> | <img src="./image/AGGAN_Zebra2Horse.png" width="400"> |



### 3. StarGAN

**CelebA**

<img src="./image/StarGAN_CelebA.png" width="800">

## VAE Results

### 1. VAE

**Reconstruction**

|                        MNIST                        |                        CelebA                        |
| :-------------------------------------------------: | :--------------------------------------------------: |
| <img src="./image/VAE_MNIST_Recon.png" width="400"> | <img src="./image/VAE_CelebA_Recon.png" width="400"> |



**Latent Space Interpolation (MNIST)** 

<img src="./image/VAE_Latent_MNIST.png" width="800">



**Latent Space Interpolation (CelebA)** 

<img src="./image/VAE_Latent_CelebA.png" width="800">



### 2. Beta-VAE

**Latent Space Interpolation: Beta = 10 (CelebA)** 

<img src="./image/betaVAE_10.png" width="800">



**Latent Space Interpolation: Beta = 200 (CelebA)** 

<img src="./image/betaVAE_200.png" width="800">



### 3. VQ-VAE

**Reconstruction (MNIST)** 

|                         Input                          | Reconstruction                                         |
| :----------------------------------------------------: | ------------------------------------------------------ |
| <img src="./image/VQ-VAE_Input_MNIST.png" width="400"> | <img src="./image/VQ-VAE_Recon_MNIST.png" width="400"> |



**Reconstruction (CelebA)** 

|                          Input                          |                     Reconstruction                      |
| :-----------------------------------------------------: | :-----------------------------------------------------: |
| <img src="./image/VQ-VAE_Input_CelebA.png" width="400"> | <img src="./image/VQ-VAE_Recon_CelebA.png" width="400"> |



**PixelCNN Trained Latent Decoding**

|                          MNIST                          |                          CelebA                          |
| :-----------------------------------------------------: | :------------------------------------------------------: |
| <img src="./image/VQ-VAE_Decode_MNIST.png" width="400"> | <img src="./image/VQ-VAE_Decode_CelebA.png" width="400"> |



## Application Results

### 1. Raindrop Removal

<img src="./image/Raindrop_Removal.png" width="800">



