# yolo_v8_and_environment

### 安装:

` git clone https://github.com/wangyimaotongxue/yolo_v8_and_environment.git `


### 文件目录
```bash
.
└── src
    ├── dataset
    │   ├── images
    │   │   ├── en_train
    │   │   ├── en_val
    │   │   ├── train
    │   │   └── val
    │   └── labels
    │       ├── en_train
    │       ├── en_val
    │       ├── train
    │       └── val
    ├── inference
    │   └── __pycache__
    ├── runs
    │   └── detect
    │       └── train
    │           └── weights
    └── train
        ├── __pycache__
        └── runs
            └── detect
                ├── train
                ├── train2
                ├── train3
                ├── train4
                ├── train5
                └── train6
                    └── weights

29 directories

```
*因为github存储空间问题，上传文件时，删除掉了所有图片和标签，请根据列表进行还原*


### 环境CUDA配置

* 安装nvidia显卡驱动550以上的版本
* 安装CUDA 11.8
```bash
# 下载CUDA11.8
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run

# 安装CUDA
sudo sh cuda_11.8.0_520.61.05_linux.run 

```

添加CUDA环境变量

```bash
# 在~/.bashrc 行末添加环境变量

# >>> cuda initialize >>>
export PATH=$PATH:/usr/local/cuda-11.8/bin
export LD_LIBRARY_PATH=$LD_LIBARY_PATH:/usr/local/cuda-11.8/lib64
export LD_LIBRARY_PATH=$LD_LIBARY_PATH:/usr/local/cuda-11.8/extras/CUPTI/lib64
# <<< cuda initialize <<<
```

* 下载并安装cuDNN
    * 下载连接：https://developer.nvidia.com/rdp/cudnn-archive
    * 下载适配CUDA 11.X 的ubuntu22.04 '.deb'格式文件
    ```bash
    sudo apt install ./cudnn-local-repo-ubuntu2204-8.9.7.29_1.0-1_amd64.deb

    sudo cp /var/cudnn-local-repo-ubuntu2204-8.9.7.29/cudnn-local-8AE81B24-keyring.gpg /usr/share/keyrings/
    ```
    * 安装依赖
    ```bash
    sudo apt update && sudo apt upgrade -y

    sudo apt dist-upgrade 

    sudo apt install python3-pip libopenblas-base libopenmpi-dev
    # 可选择安装
    # libopenblas-dev
    ```

### 在Anconda3中安装yolo_v8 和 pytorch 2.6.0
* 创建conda虚拟环境
```bash
# 启动conda
conda activate

#创建一个虚拟环境
conda create -n yolo_v8 python=3.11

#查看当前全部虚拟环境
conda env list

#进入到指定的虚拟环境
conda activate yolo_v8

```

* 安装pytorch

下载连接：https://pytorch.org/
版本选择| Stable | Linux | Conda | python | CUDA 11.8 

```bash
# 在当前虚拟环境安装pytorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 再补充点其他的依赖包
sudo apt install libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev
```

* 安装Ultralytics
```bash
conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=11.8 ultralytics
```
