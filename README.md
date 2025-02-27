The official implementation of the paper "Effective Multi-Agent Collaborative Perception with Robust Representation Learning.".![DSRC_Overview](https://github.com/easy2star/DSRC_plus_plus/blob/main/structures_trans.png)

**Effective Multi-Agent Collaborative Perception with
Robust Representation Learning**, <br>Jingyu Zhang, Yilei Wang, Hanqi Wang, Xu Deng, Peng Sun, Liang Song† <br>

## Abstract

As a promising application of Vehicle-to-Everything (V2X) communication, multi-agent collaborative perception has shown significant progress in 3D object detection. While these methods have achieved remarkable performance on standard benchmarks, their robustness in complex, real-world environments remains to be further validated.
To address this gap, we introduce the first comprehensive benchmark specifically tailored to assess the robustness of collaborative perception methods in the presence of real-world natural corruptions.
Furthermore, we propose DSRC++, a robustness-enhanced collaborative perception method against corruptions. DSRC++ consists of four key designs.
First, we employ a sparse-to-dense learning mechanism to extract reliable 3D features in the latent space, addressing the challenges posed by sparse point cloud dcloudata under adverse environmental conditions. Second, we leverage a semantic-guided learning strategy to preserve semantic consistency in corrupted scenarios. This strategy uses ground truth labels to impart category-specific semantic information to teacher point cloud, enabling the student model to effectively learn comprehensive semantic representations. Third, we introduce a cross-scale attention fusion module to integrate contextual information across multiple scales. Finally, our method incorporates a feature-to-point cloud reconstruction module, which imposes regularization constraints on the feature learning process.
To thoroughly evaluate DSRC++, we conduct extensive experiments on real-world and simulated datasets. The results demonstrate that our method outperforms SOTA collaborative perception methods in both clean and corrupted conditions.



## Installation

```bash
# Setup conda environment
conda create -n dsrc python=3.7 -y
conda activate dsrc

pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

# spconv 2.0 install, choose the correct cuda version for you
pip install spconv-cu113

# Install dependencies
pip install -r requirements.txt
# Install bbx nms calculation cuda version
python opencood/utils/setup.py build_ext --inplace

# install opencood into the environment
python setup.py develop
```

## Data Downloading
### 1. OPV2V
All the data can be downloaded from [google drive](https://drive.google.com/drive/folders/1dkDeHlwOVbmgXcDazZvO6TFEZ6V_7WUu). If you have a good internet, you can directly
download the complete large zip file such as `train.zip`. In case you suffer from downloading large files, we also split each data set into small chunks, which can be found 
in the directory ending with `_chunks`, such as `train_chunks`. After downloading, please run the following command to each set to merge those chunks together:

```python
cat train.zip.part* > train.zip
unzip train.zip
```
### 2. DAIR-V2X
Download the raw data of [DAIR-V2X](https://thudair.baai.ac.cn/index) and  the [complemeted annotations](https://siheng-chen.github.io/dataset/dair-v2x-c-complemented/).


## Quick Start


### **Train** 
### Step1: Train the teacher model

First, ensure that the `root_dir` in the YAML file (e.g., `opencood/hypes_yaml/point_pillar_base_multi_scale_teacher.yaml`) is set to the training dataset path, such as `opv2v/train`.
Then, you can use the following command to train your teacher model from scratch or a continued checkpoint:

```python
python opencood/tools/train.py --hypes_yaml ${CONFIG_FILE} [--model_dir  ${CHECKPOINT_FOLDER}]
```
The explanation of the optional arguments are as follows:

- `hypes_yaml`: the path of the training configuration file, e.g.  `opencood/hypes_yaml/point_pillar_base_multi_scale_teacher.yaml`. You can change the configuration parameters in this provided yaml file.
- `model_dir` (optional) : the path of the checkpoints. This is used to fine-tune the trained models. When the `model_dir` is
given, the trainer will discard the `hypes_yaml` and load the `config.yaml` in the checkpoint folder.


### Step2: Train the student model

Copy the checkpoint folder of the teacher model and rename it as `student_train_folder`, keep only the last checkpoint and rename it as epoch_1.pth, then change the `core_method` in config.yaml under the checkpoint folder to `point_pillar_base_multi_scale_student`.

```python
python opencood/tools/train.py --model_dir student_train_folder
```


### **Test** 
Before you run the following command, first make sure the `validation_dir` in config.yaml under your checkpoint folder
refers to the testing dataset path, e.g. `opv2v/test`.

```python
python opencood/tools/inference.py --model_dir ${student_train_folder} 
```
If you want to test the performance of the model in different environments, you can uncomment the application code for each environment simulation function in the file `opencood/data_utils/datasets/basedataset.py`, such as `apply_motion_blur_to_numpy`.

## Citation
 If you are using our DSRC for your research, please cite the following paper:
  ```bibtex
@article{zhang2024dsrc,
  title={DSRC: Learning Density-insensitive and Semantic-aware Collaborative Representation against Corruptions},
  author={Zhang, Jingyu and Wang, Yilei and Qian, Lang and Sun, Peng and Li, Zengwen and Jiang, Sudong and Liu, Maolin and Song, Liang},
  journal={arXiv preprint arXiv:2412.10739},
  year={2024}
}
  ```
## Acknowledgment
Many thanks to the high-quality dataset and codebase, including [Robo3D](https://github.com/ldkong1205/Robo3D), [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD) and  [Where2comm](https://github.com/MediaBrain-SJTU/Where2comm.git).
