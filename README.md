# LiDAR-based Face Anti-Spoofing

Welcome to the Repository for LiDAR-based Face Anti-Spoofing Model. This repository is for experimenting with the **'CloudNet: A LiDAR-based Face Anti-Spoofing model that is robust against light variation'** paper published in *IEEE Access(URL not yet)*.

[Samples from Our Own Built Dataset]
![Fig2_LDFAS](https://user-images.githubusercontent.com/14557402/216536362-b3c4895d-310b-4d34-9302-3ac6b170a226.JPG)

* Dataset Link : [LiDAR Dataset for Face Anti-Spoofing(LDFAS) - IEEE DataPort](https://ieee-dataport.org/documents/lidar-dataset-face-anti-spoofingldfas)    


# Introduction
![Fig1_Framework](https://user-images.githubusercontent.com/14557402/216536485-6d208fd0-88a4-4e42-97cf-45e14f937a33.JPG)

We devised a multi-modal FAS model using a LiDAR sensor, which is generalized for the light domain. All you need to do is install the LiDAR camera app on your iPad Pro, and then capture the face of the target. This model has strong identification ability **even if the light domain of the test set is different from the training dataset**. This model distinguishes print attacks, replay attacks, and 3D mask attacks as spoofing.


# Architecture of CloudNet
![Fig3_Architecture](https://user-images.githubusercontent.com/14557402/216536648-622a5cf2-ac7e-455e-a434-307a88ce8870.JPG)

The architecture of CloudNet is a binary classifier based on Resnet34. The structure is composed of a RGB space and LiDAR space networks. Each network extracts facial features from the RGB and LiDAR data (point cloud and depth). CloudNet performs both early fusion and late fusion to classify bonafide and spoofing images. Herein, binary cross-entropy was used as the loss function. 

# Experimental results



# Quick Start

> ### Version
> ##### * python : 3.6.13
> ##### * matplotlib :  3.3.4
> ##### * torchvision : 0.2.1
> ##### * cudatoolkit : 11.3 (pytorch-nightly)
> ##### * scikit-learn : 0.24.2
> ##### * opencv-python : 4.5.4
