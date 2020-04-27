# Generative Models

- This repository is for implementation of `Generative Models` using [Tensorflow](https://www.tensorflow.org) 1.12

- [Description of the Papers (Korean)](https://www.notion.so/Code-Description-53c93afd0b9740728143ffab1b2caa2f)

- The structure of the code is based on the [Hwalsuk Lee's Generative Model github repository](https://github.com/hwalsuklee/tensorflow-generative-model-collections)  

<br>

## Contributors

[MMC Lab](http://mmc.hanyang.ac.kr/) GAN Study Group members

- [Kyushik Min](https://github.com/Kyushik), [Gihoon Kim](https://github.com/GihoonKim), [Hyukju Sohn](https://github.com/Hyukju-Sohn), [Yoonyong Ahn](https://github.com/YoonyongAhn), [Seungwon-Choi](https://github.com/seungwon-Choi), [Jongyoon Baek](https://github.com/whd2345)

<br>

## Implemented Paper List (18 Papers)

### GAN

1. [[GAN] Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
2. [[DCGAN] Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)
3. [[LSGAN] Least Squares Generative Adversarial Networks](https://arxiv.org/abs/1611.04076)
4. [[WGAN] Wasserstein GAN](https://arxiv.org/abs/1701.07875)
5. [[WGAN_GP] Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028)
6. [[CGAN] Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784)
7. [[InfoGAN] Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets](https://arxiv.org/abs/1606.03657)
8. [[HoloGAN] Unsupervised Learning of 3D Representations From Natural Images](https://arxiv.org/abs/1904.01326)
9. [[SinGAN] Learning a Generative Model from a Single Natural Image](https://arxiv.org/pdf/1905.01164.pdf)
10. [[PGGAN] Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/pdf/1710.10196.pdf)

### Image-to-Image Translation

1. [[CycleGAN] Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)
2. [[AGGAN] Attention-Guided Generative Adversarial Networks for Unsupervised Image-to-Image Translation](https://arxiv.org/abs/1903.12296)
3. [[StarGAN] Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation](https://arxiv.org/abs/1711.09020)
4. [[DMIT] Multi-mapping Image-to-Image Translation via Learning Disentanglement](https://arxiv.org/pdf/1909.07877.pdf)

### Interpretable GAN Latent

1. [Unsupervised Discovery of Interpretable Directions in the GAN Latent Space](https://arxiv.org/abs/2002.03754)

### VAE

1. [Auto-Encoding Variational Bayes (VAE)](https://arxiv.org/abs/1312.6114)
2. [Beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl)
3. [Neural Discrete Representation Learning(VQ-VAE)](https://arxiv.org/abs/1711.00937)

### Application

1. [Adherent Raindrop Removal with Self-Supervised Attention Maps andSpatio-Temporal Generative Adversarial Networks](http://openaccess.thecvf.com/content_ICCVW_2019/papers/ADW/Alletto_Adherent_Raindrop_Removal_with_Self-Supervised_Attention_Maps_and_Spatio-Temporal_Generative_ICCVW_2019_paper.pdf)

<br>

# Our Results

## GAN Results

### 1. GAN

**MNIST**

<img src="./image/GAN_MNIST.png" width="400">

<br>

### 2. DCGAN

|                      MNIST                      |                      CelebA                      |
| :---------------------------------------------: | :----------------------------------------------: |
| <img src="./image/DCGAN_MNIST.png" width="500"> | <img src="./image/DCGAN_CelebA.png" width="500"> |

<br>

### 3. LSGAN

|                      MNIST                      |                      CelebA                      |
| :---------------------------------------------: | :----------------------------------------------: |
| <img src="./image/LSGAN_MNIST.png" width="500"> | <img src="./image/LSGAN_CelebA.png" width="500"> |

<br>

### 4. WGAN

|                     MNIST                      |                     CelebA                      |
| :--------------------------------------------: | :---------------------------------------------: |
| <img src="./image/WGAN_MNIST.png" width="500"> | <img src="./image/WGAN_CelebA.png" width="500"> |

<br>

### 5. WGAN-GP

|                       MNIST                       |                       CelebA                       |
| :-----------------------------------------------: | :------------------------------------------------: |
| <img src="./image/WGAN-GP_MNIST.png" width="500"> | <img src="./image/WGAN-GP_CelebA.png" width="500"> |

<br>

### 6. Conditional GAN

**MNIST**

<img src="./image/ConditionalGAN_MNIST.png" width="400">

<br>

### 7. InfoGAN

**MNIST**

<img src="./image/InfoGAN_MNIST.png" width="400">

<br>

### 8. HoloGAN

**CelebA**

<img src="./image/HoloGAN_CelebA.png" width="800">

<br>

### 9. SinGAN

**Balloon**

<img src="./image/SinGAN_Ballon_1.png" width="600">
<img src="./image/SinGAN_Ballon_0.png" width="600">

**Mountain**

<img src="./image/SinGAN_Mountain_1.png" width="600">
<img src="./image/SinGAN_Mountain_0.png" width="600">

**Starry Night**

<img src="./image/SinGAN_Starry_0.png" width="600">
<img src="./image/SinGAN_Starry_1.png" width="600">

<br>

### 10. PGGAN
**Cherry picked images**
<img src="./image/PGGAN_03.png" width="100"><img src="./image/PGGAN_04.png" width="100"><img src="./image/PGGAN_07.png" width="100"><img src="./image/PGGAN_21.png" width="100"><img src="./image/PGGAN_13.png" width="100"><img src="./image/PGGAN_17.png" width="100"><img src="./image/PGGAN_01.png" width="100">
\
**Latent interpolation**
<img src="./image/PGGAN_inter.png" width="700">
\
**Fixed latent**
<img src="./image/PGGAN_latent.gif" width="500">
\
**No cherry picked images**
<img src="./image/PGGAN_NC.png" width="500">
<br>

## Image-to-Image Translation Results

### 1. CycleGAN

|                      Monet to Photo                      |                      Photo to Monet                      |
| :------------------------------------------------------: | :------------------------------------------------------: |
| <img src="./image/CycleGAN_Monet2Photo.png" width="400"> | <img src="./image/CycleGAN_Photo2Monet.png" width="400"> |

|                      Horse to Zebra                      | Zebra to Horse                                           |
| :------------------------------------------------------: | -------------------------------------------------------- |
| <img src="./image/CycleGAN_Horse2Zebra.png" width="400"> | <img src="./image/CycleGAN_Zebra2Horse.png" width="400"> |

<br>

### 2. AGGAN

|                    Horse to Zebra                     |                    Zebra to Horse                     |
| :---------------------------------------------------: | :---------------------------------------------------: |
| <img src="./image/AGGAN_Horse2Zebra.png" width="400"> | <img src="./image/AGGAN_Zebra2Horse.png" width="400"> |

<br>

### 3. StarGAN

**CelebA**

<img src="./image/StarGAN_CelebA.png" width="800">

<br>

### 4. DMIT

**Summer2Winter**

<img src="./image/DMIT.png" width="800">

<br>

## Interpretable GAN Latent

### 1. Unsupervised Discovery of Interpretable Directions in the GAN Latent Space

#### 1) MNIST

<img src="./image/unsupervised_interpretable_latent.png" width="800">



<br>

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

<br>

### 2. Beta-VAE

**Latent Space Interpolation: Beta = 10 (CelebA)**

<img src="./image/betaVAE_10.png" width="800">



**Latent Space Interpolation: Beta = 200 (CelebA)**

<img src="./image/betaVAE_200.png" width="800">

<br>

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

<br>

## Application Results

### 1. Raindrop Removal

<img src="./image/Raindrop_Removal.png" width="800">
