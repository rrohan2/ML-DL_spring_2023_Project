# ML-DL_spring_2023_Project

###Group Number: 27

### Project Name: Domain-supervised learning of generalizable models

Team CA Mentor: Yuta Kobayashi

Team TA Mentor: Jingfeng Wu

Team Members:

i) Marcelo Morales (lmoral10@jhu.edu)

ii) Bhupendara Mahar (bmahar1@jh.edu)

iii) Vikram Shivakumar (vshivak1@jh.edu)

iii) Ritwik Rohan (rrohan2@jh.edu)


### Using the code:

Mount the google drive in the jupyter notebook using this command:
```bash
from google.colab import drive
drive.mount('/content/drive')
```

- Clone this repository using these command in jupyter notebook:

```bash
git clone https://github.com/rrohan2/ML-DL_spring_2023_Project.git
```


The code is stable using Python 3.6.10, Pytorch 1.4.0


To install all the dependencies using pip, write the following command:
```
!pip install -r requirements.txt
```
### Codes in this git repo

i) resnet50_imagenet.py : This code is training the undistorted imagenet dataset on Resnet50 model.

ii) domain_adaptation.py: This code is training the undistorted imagenet dataset on domain adaptation model so that it can improve the accuracy for unseen distorted images while testing.

iii) train.py: This code is used to train the model and the model can be selected as per the user's choice. The weighted model is saved in the google drive (link attached below this section).

iv) test_distorted.py: This code is used to test the trained model by using the "res50model_checkpoint.pth" or "dom_ada_checkpoint.pth" (automatically saved in "imagenet-mini" folder after training) for distorted images. This code is run for Vanilla Resnet50 and Domain Adaptation model to compare the accuracy on distorted images

v) test_undistorted.py: This code is used to test the trained model using the trained model by using the "res50model_checkpoint.pth" (automatically saved in "imagenet-mini" folder after training) for undistorted images. This is run on Vanilla Resnet50 to show the state of the art model.

vi) data_manipulation.py: This code is used to add 12 types of distortion to create distorted test images.

### Dataset
You can access the datasets from the google drive link. We have given the drive access to anyone with this link. Google drive link: https://drive.google.com/drive/folders/1a8yqwGKm5Jo7cPLHw69rjfCaQRH5Rfff?usp=sharing

## Reproducing the Code for your dataset

### Dataset Preparation

The datasets are already present in the google drive (link above). In case you need to run with different datasets, follow the procedure below:

Create an "imagenet-mini" folder and prepare the dataset in the following format for easy use of the code. 

```bash
"train" Folder-----
          WNID1
             img1
             img2
             ...
          WNID2
             img1
             img2
             ...          
          .......
"val" Folder-----
          WNID1
             img1
             img2
             ...
          WNID2
             img1
             img2
             ...
          .......
"test_undistorted" Folder-----
          WNID1
             img1
             img2
             ...
          WNID2
             img1
             img2
             ...
          .......
"test_distorted" Folder-----
          WNID1
             img1
             img2
             ...
          WNID2
             img1
             img2
             ...
          .......

```

### Training Command for vanilla Resnet50 in jupyter notebook:

```bash 
!python resnet50_imagenet.py --train_path "content/drive/MyDrive/imagenet-mini/train"--val_path "content/drive/MyDrive/imagenet-mini/val"
```

### Training Command for Domain adaptation in jupyter notebook:

```bash 
!python domain_adaptation.py --train_path "content/drive/MyDrive/imagenet-mini/train"--val_path "content/drive/MyDrive/imagenet-mini/val"
```


### Testing Command for UNDISTORTED image using Vanilla Resnet50 model in jupyter notebook:

```bash 
!python test_undistorted.py --test_path "/content/drive/MyDrive/imagenet-mini/test_undistorted" --checkpoint_dir "/content/drive/MyDrive/imagenet-mini/res50model_checkpoint.pth/res50model_checkpoint.pth"
```

### Testing Command for DISTORTED image using Vanilla Resnet50 model in jupyter notebook:

```bash 
!python test_distorted.py --test_path "/content/drive/MyDrive/imagenet-mini/test_distorted" --checkpoint_dir "/content/drive/MyDrive/imagenet-mini/res50model_checkpoint.pth/res50model_checkpoint.pth"
```

### Testing Command for DISTORTED image using Domain adaptation model in jupyter notebook:

```bash 
!python test_distorted.py --test_path "/content/drive/MyDrive/imagenet-mini/test_distorted" --checkpoint_dir "/content/drive/MyDrive/imagenet-mini/dom_ada_checkpoint.pth"
```


