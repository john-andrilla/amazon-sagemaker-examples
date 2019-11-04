# DGL SageMaker GCN Examples for Molecular Property Prediction

We show how to run GCN for molecular property prediction with Amazon SageMaker here.

With atoms being nodes and bonds being edges, molecules have been an important type of data in the application of 
graph neural networks. In this example, we consider the dataset of **Tox21**. The 
"Toxicology in the 21st Century (Tox21)" initiative created a public database measuring toxicity of compounds. The 
dataset contains qualitative toxicity measurement for 8014 compounds on 12 different targets, including nuclear 
receptors and stress response pathways. Each target yields a binary classification problem. Therefore, we model the 
problems as graph classification problems. MoleculeNet [1] randomly splits the dataset for into a training, validation 
and test set with a 80/10/10 ratio and we follow their approach.

We use atom descriptors as initial node features. After updating node features as in usual GCN, we combine the sum and
maximum of the updated node (atom) representations for graph (molecule) representations. Finally, we use a FNN to 
make the prediction from the representations.

For more information about DGL and GCN please refer to docs.dgl.ai

## Setup conda env for DGL (Pytorch backend)

We can install conda env for DGL with GPU enabled PyTorch backend.

See the following steps:
```
# Clone python3 environment

conda create --name DGL_py36_pytorch1.2_chem --clone python3

# Install pytorch and DGL
conda install --name DGL_py36_pytorch1.2_chem pytorch=1.2 torchvision -c pytorch
conda install --name DGL_py36_pytorch1.2_chem -c dglteam dgl-cuda10.0
```
You can select DGL_py36_pytorch1.2_chem conda env now.
