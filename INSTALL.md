# Installation Tutorial

We provide step-by-step installation instructions to create the corresponding environment for using DKUNet and demonstrates the corresponding folder structure to input data samples.


## Conda Environment Setup
Create your own conda environment 
```
conda create -n dkunet python=3.9
conda activate dkunet
```

Install [Pytorch](https://pytorch.org/) == 1.12.1, [torchvision](https://pytorch.org/vision/stable/index.html) == 0.13.1, cudatooltookit == 11.6.0 
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```
Install [monai](https://github.com/Project-MONAI/MONAI)
```
pip install monai
```
Clone this repository and install other required packages:
```
git clone git@github.com:MASILab/3DUX-Net.git
pip install -r requirements.txt
```

## Input Folder Format
We initially divide different datasets in the following structure:

    path to all data directory/
    ├── LA2018
    ├── ...

We further sub-divide the samples into training, validation and testing as follow:

    root_dir/
    ├── imagesTr
    ├── labelsTr
    ├── imagesVal
    ├── labelsVal
    ├── imagesTs
For the input of both training and inference, our code currently only allows data samples in Nifti format. Feel free to provide suggestions on adapting other data formats for both training and inference.
