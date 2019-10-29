# DGL Sagemaker GCN Examples
We show how to run GCN with Amazon Sagemaker here.

For more information about DGL and GCN please refer to docs.dgl.ai

## Setup conda env for DGL (MXNet backend)
Curently we can only install conda env for DGL with MXNet backend with CPU-build.

See following steps:
```
# Clone python3 environment
conda create --name DGL_py36_mxnet1.5 --clone python3

# Install mxnet and DGL (This is only CPU version)
source activate DGL_py36_mxnet1.5
conda install -c anaconda scipy
conda install -c anaconda numpy
conda install -c anaconda numexpr
conda install -c anaconda blas=1.0=mkl mkl-service 
conda install -c anaconda mkl_fft==1.0.1 mkl_random==1.0.1
conda install -c anaconda numpy-base==1.16.0 scikit-learn mxnet=1.5.0
conda install -c dglteam dgl
```
You can select DGL_py36_mxnet1.5 conda env now.

## Setup conda env for DGL (Pytorch backend)
We can install conda env for DGL with Pytorch backend with GPU-build.

See following steps:
```
# Clone python3 environment
conda create --name DGL_py36_pytorch1.2 --clone python3

# Install pytorch and DGL
conda install --name DGL_py36_pytorch1.2 pytorch=1.2 torchvision -c pytorch
conda install --name DGL_py36_pytorch1.2 -c dglteam dgl-cuda10.0
```
You can select DGL_py36_pytorch1.2 conda env now.
