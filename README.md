# LiDAR-based Face Anti-Spoofing

Welcome to the Repository for LiDAR-based Face Anti-Spoofing Model. This repository is for experimenting with the **'CloudNet: A LiDAR-based Face Anti-Spoofing model that is robust against light variation'** paper published in ***IEEE Access***. [[Paper Link]](https://ieeexplore.ieee.org/document/10038600) 

[Samples from Our Own Built Dataset]
![Fig2_LDFAS](https://user-images.githubusercontent.com/14557402/216536362-b3c4895d-310b-4d34-9302-3ac6b170a226.JPG)

* Dataset Link : [LiDAR Dataset for Face Anti-Spoofing(LDFAS) - IEEE DataPort](https://ieee-dataport.org/documents/lidar-dataset-face-anti-spoofingldfas)    

# Introduction
![Fig1_Framework](https://user-images.githubusercontent.com/14557402/216536485-6d208fd0-88a4-4e42-97cf-45e14f937a33.JPG)

We devised a multi-modal FAS model using a LiDAR sensor, which is generalized for the light domain. All you need to do is install the LiDAR camera app on your iPad Pro, and then capture the face of the target. This model has strong identification ability **even if the light domain of the test set is different from the training dataset**. This model distinguishes print attacks, replay attacks, and 3D mask attacks as spoofing.

* How to install the LiDAR camera app: [Download the code from this Git repository](https://github.com/kyoungmingo/ARKit_extract_PT). Then, open the code in XCode and run it to complete the installation process.

# Architecture of CloudNet
![Fig3_Architecture](https://user-images.githubusercontent.com/14557402/216536648-622a5cf2-ac7e-455e-a434-307a88ce8870.JPG)

The CloudNet is a binary classifier based on Resnet34. The structure is composed of a RGB space and LiDAR space networks. Each network extracts facial features from the RGB and LiDAR data (point cloud and depth). The CloudNet performs both early fusion and late fusion to classify bonafide and spoofing images. Herein, binary cross-entropy was used as the loss function. 

# Experimental results
![image](https://user-images.githubusercontent.com/14557402/216754585-7aea855c-1a2f-4cad-8218-953586e51e2e.png)

### Evaluation Metrics
The bonafide presentation classification error rate (**BPCER**), attack presentation classification error rate (**APCER**), and average classification error rate (**ACER**) were used as the evaluation metrics. These metrics were proposed in *ISO/IEC 30107-3:2017* for performance assessment of presentation attack detection mechanisms. BPCER is the proportion of bonafides incorrectly rejected as an attack. APCER is the percentage of attacks incorrectly accepted as bonafides. ACER is the average of BPCER and APCER. Additionally, a receiver operating characteristic (**ROC**) curve was also used in the paper. [[Paper Link]](https://ieeexplore.ieee.org/document/10038600) 

### Evaluation Protocols
The goal of this study was to develop a generalized FAS model considering light variations. Three protocols were designed for this purpose. Protocol 1 corresponds to when the learning and test datasets are in the same light conditions. By contrast, protocols 2 and 3 used different light conditions. The indoor, outdoor, and indoor (dark) sets were tested while training only the indoor sets. Details of each protocol are listed in the paper. [[Paper Link]](https://ieeexplore.ieee.org/document/10038600) 

### Discussion & Conclusion  
Experimental results indicate that for protocols 2 and 3, CloudNet error rates increase by 0.1340 and 0.1528, whereas the error rates of the RGB model increase by 0.3951 and 0.4111, respectively, as compared with protocol 1. These results demonstrate that the LiDAR-based FAS model with CloudNet has a more generalized performance compared with the RGB model. You can find more details in the paper. [[Paper Link]](https://ieeexplore.ieee.org/document/10038600) 

# Quick Start

### Version 
* python : 3.6.13
* matplotlib :  3.3.4
* torchvision : 0.2.1
* cudatoolkit : 11.3 (pytorch-nightly)
* scikit-learn : 0.24.2
* opencv-python : 4.5.4

### Command 

#### Install Pytorch
~~~
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
~~~

#### Install Packages
~~~
pip install -r requirements.txt
~~~

#### Train

~~~
python train.py --model rgbdp_v2 --attacktype rpm --epochs 1000 --cuda 0 --message 0828_total_rgbdp_v2
~~~

#### Test

~~~
python test.py --model rgbdp_v2 --attacktype rpm --epochs 1000 --cuda 0 --message 0828_total_rgbdp_v2
~~~

#### Using Tensorboard in Local 
~~~
[Remote]: tensorboard --logdir=runs --port=6006
[Local]:  ssh -L localhost:6006:localhost:6006 id@ip_address
[Local]:  localhost:6006 
~~~
