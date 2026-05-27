## HG-CAD: Hierarchical Graph Learning for CAD Material Recommendation

### Section 1: Usage

- **Recommended runtime**
  - Ubuntu 22.04 LTS
  - NVIDIA GeForce RTX 5090D
  - Python == 3.11
  - PyTorch == 2.10.0+cu128
  - NVIDIA driver compatible with CUDA 12.8 (`nvidia-smi` should report a recent enough driver)

```bash
##################################
# --- Ubuntu 22.04 LTS (X86) --- #
##################################

# Basics
sudo apt-get update
sudo apt-get install -y build-essential git cmake ninja-build

# Install Conda if needed
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc

# Create and activate the project environment
conda env create -f environment.yml
conda activate hg_cad

# Verify the requested torch build
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

# DGL note:
# The official DGL GPU wheel matrix currently documents PyTorch/CUDA builds only
# up to the 2.4 / CUDA 12.4 generation, so for torch 2.10.0+cu128 you should
# build DGL from source inside this environment. If this machine does not
# already provide CUDA compiler tools, install a CUDA 12.8 toolkit first.
git clone --recurse-submodules https://github.com/dmlc/dgl.git
cd dgl
bash script/build_dgl.sh -g
cd python
python setup.py install
python setup.py build_ext --inplace
cd ../..

# Make sure DGL uses the PyTorch backend
export DGLBACKEND=pytorch
```

- **Obtaining the data used in paper**:
  - See [link](https://drive.google.com/file/d/10dHBssZEMqE7-Q-mbzDH-pQ2PYnvFuqX/view?usp=sharing)
  - If you want to download from command line, try with `gdown` installed (`conda install -c conda-forge gdown`)
  - Unzip (`unzip <file>.zip`) and place inside a folder (for example `./dataset/`)
  - **If `zipfile corrupt` appears**: try `zip -FF Corrupted.zip --out New.zip`

- **Training**: trains on the train split and automatically tests after training finishes.
```bash
# Additional tuning knobs are inside classification.py
python classification.py train \
  --dataset_path /path/to/dataset \
  --max_epochs 100 \
  --batch_size 16 \
  --accelerator gpu \
  --devices 1
```

- **Testing**: evaluates the test split from a saved checkpoint.
```bash
python classification.py test \
  --dataset_path /path/to/dataset \
  --checkpoint /path/to/best.ckpt \
  --random_seed 1234
```

- **Inference**: on a *single* sample assembly, producing material predictions for *all* bodies.
```bash
python inference.py single_sample \
  --inference_sample /path/to/inference/sample \
  --checkpoint /path/to/best.ckpt \
  --vocab /path/to/vocab.pickle
```

- **Inference**: on *multiple* sample assemblies, producing a prediction for *one random* body per assembly.
```bash
python inference.py multiple_sample \
  --inference_sample /path/to/inference/samples \
  --checkpoint /path/to/best.ckpt \
  --vocab /path/to/vocab.pickle
```

---

### Section 2: Baselines

| Model | Data Format | Repository | Data Processing Tools | Results |
| -- | -- | -- | -- | -- |
| PointNet [*](https://doi.org/10.48550/arXiv.1612.00593) | PointCloud OFF Files | [Adapted Version](https://github.com/BrandonBian/pointnet-tensorflow) | [OBJ to OFF](TODO) | [Link](https://drive.google.com/drive/folders/1gD5NRNyzzHVVn0mfhRETty_dEB-Wut0N?usp=share_link) |
| UV-Net [*](https://doi.org/10.48550/arXiv.2006.10211) | DGL Graph BIN Files | [Adapted Version](https://github.com/BrandonBian/UV-Net) | [STEP to BIN](TODO); [Visualization](TODO) | [Link](https://drive.google.com/drive/folders/1GS14bYIzT5ut42Tr50X6nOTdyKpmDXdQ?usp=share_link) |
| Human Baseline | OBJ and PNG files | [Original Version](https://github.com/BrandonBian/Human-Baseline) | [Prompt Generation](TODO) | - |
