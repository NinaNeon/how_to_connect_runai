# how_to_connect_runai

# Run:ai UNet+VAE分割模型训练与推理指南
```bash
nina@nina-X550VX:~$ cd runai-isaac
nina@nina-X550VX:~/runai-isaac$ source secrets/env.sh
lftp -u ${FTP_USER},${FTP_PASS} ${STORAGE_NODE_IP}
lftp elsalab@10.225.0.35:~> cd /mnt/nfs/nina
put process.py
cd: Fatal error: Certificate verification: Not trusted (B8:A9:2A:44:F9:60:4F:D6:D1:7F:BA:35:54:EF:F1:69:7E:6F:A6:CC)
put: /home/nina/runai-isaac/process.py: No such file or directory
lftp elsalab@10.225.0.35:~> cd /mnt/nfs/nina
put process.py
cd: Fatal error: Certificate verification: Not trusted (B8:A9:2A:44:F9:60:4F:D6:D1:7F:BA:35:54:EF:F1:69:7E:6F:A6:CC)
put: /home/nina/runai-isaac/process.py: No such file or directory
lftp elsalab@10.225.0.35:~> echo "set ssl:verify-certificate no" >> ~/.lftprc
lftp elsalab@10.225.0.35:~> set ssl:verify-certificate no
lftp elsalab@10.225.0.35:~> cd /mnt/nfs/nina
cd: Access failed: 550 Failed to change directory. (/mnt/nfs/nina)
lftp elsalab@10.225.0.35:/> put ~/runai-isaac/hello/process.py
put: /home/nina/runai-isaac/hello/process.py: Access failed: 553 Could not create file. (process.py)
lftp elsalab@10.225.0.35:/> cd /mnt/nfs
mkdir nina
cd nina
cd ok, cwd=/mnt/nfs
mkdir ok, `nina' created
cd ok, cwd=/mnt/nfs/nina
lftp elsalab@10.225.0.35:/mnt/nfs/nina> put ~/runai-isaac/hello/process.py
118 bytes transferred
lftp elsalab@10.225.0.35:/mnt/nfs/nina> ls
-rw------- 1 ftp ftp 118 May 13 03:47 process.py
lftp elsalab@10.225.0.35:/mnt/nfs/nina> ls
-rw------- 1 ftp ftp 118 May 13 03:47 process.py
lftp elsalab@10.225.0.35:/mnt/nfs/nina> ls
-rw------- 1 ftp ftp 118 May 13 03:47 process.py
lftp elsalab@10.225.0.35:/mnt/nfs/nina> put process.py
put: /home/nina/runai-isaac/process.py: No such file or directory
lftp elsalab@10.225.0.35:/mnt/nfs/nina> put ~/runai-isaac/hello/process.py
88 bytes transferred
lftp elsalab@10.225.0.35:/mnt/nfs/nina> ls
-rw-r--r-- 1 ftp ftp 39 May 13 04:05 output.txt
-rw------- 1 ftp ftp 88 May 13 04:08 process.py
lftp elsalab@10.225.0.35:/mnt/nfs/nina> get output.txt
39 bytes transferred
lftp elsalab@10.225.0.35:/mnt/nfs/nina> cat output.txt
This is a file written from process.py
39 bytes transferred
lftp elsalab@10.225.0.35:/mnt/nfs/nina>
```


ikea_purewater — 5/17/25, 9:20 PM
https://github.com/j3soon/runai-isaac
GitHub
GitHub - j3soon/runai-isaac: Tools and Scripts for running Isaac Si...
Tools and Scripts for running Isaac Sim workloads on Run:ai - j3soon/runai-isaac
Tools and Scripts for running Isaac Sim workloads on Run:ai - j3soon/runai-isaac
陳政錡    alex    zqchen020719@gmail.com    楊老師    O    23810367b1Ii2!
https://docs.google.com/spreadsheets/d/1nsghCMsiS0Gl_LCTboJ-JZn6w9Ww6nr1rNJv_cB8ANw/edit?gid=0#gid=0
Google Docs
OVX L40 User Data (2024-11)
Image
Run:ai README,
VPN Profile is under secrets/<YOUR_USERNAME>.ovpn

<RUNAI_USER_EMAIL> & <RUNAI_USER_PASSWORD> is in the "OVX L40 User Data" spreadsheet.

secrets/env.sh:

export RUNAI_URL="https://runai.tpe1.local"
export STORAGE_NODE_IP="10.225.0.35"
export FTP_USER="elsalab"
export FTP_PASS="CaHaDdAUtifpthtoDbDy"


Documentation at:

https://github.com/j3soon/runai-isaac,

Replace every lab1 to elsalab and you should be good to go.

Note that <YOUR_USERNAME> and <RUNAI_USER_EMAIL> is different. For example my username is johnsons and email is johnsons@nvidia.com.

If you encountered any questions, or have ideas to improve the GitHub README, please let me know. Thanks!
GitHub
GitHub - j3soon/runai-isaac: Tools and Scripts for running Isaac Si...
Tools and Scripts for running Isaac Sim workloads on Run:ai - j3soon/runai-isaac
Tools and Scripts for running Isaac Sim workloads on Run:ai - j3soon/runai-isaac
ikea_purewater — 5/17/25, 9:28 PM
https://drive.google.com/drive/u/0/folders/1OYtNgODgzxY_r7XiMBMtBIQXQ6uDTjth
Google Drive: Sign-in
Access Google Drive with a Google account (for personal use) or Google Workspace account (for business use).
36232305
Image
Image
djd4b5123N
ikea_purewater — 5/17/25, 9:36 PM
https://runai.tpe1.local/
陳政錡    alex    zqchen020719@gmail.com    楊老師    O    23810367b1Ii2!
https://docs.google.com/presentation/d/1G9SBX6WeM9oiN1si909Cy2YLJcWyw6BM/edit?slide=id.p8#slide=id.p8
ikea_purewater — 5/17/25, 9:46 PM
https://drive.google.com/drive/folders/1EPzkAYBd0TxuYO3lbGNlgb9_PnYznk49?usp=sharing
Google Drive



## 概述

1. 上传预训练的UNet和VAE模型
2. 上传train_segmentation_fixed.py训练脚本
3. 上传测试数据（图像和COCO格式的标注文件）
4. 在Run:ai上创建训练任务
5. 下载训练好的模型进行评估

## 1. 准备目录结构

首先，我们需要在Run:ai的存储节点上创建以下目录结构：

```
/mnt/nfs/nina/
├── models/
│   ├── unet/         # 预训练的UNet模型
│   └── vae/          # 预训练的VAE模型
├── data/
│   ├── images/       # 训练和测试图像
│   └── coco.json     # COCO格式的标注文件
├── train_segmentation_fixed.py  # 训练脚本
└── process.py        # 推理测试脚本
```

## 2. 上传文件

### 2.1 连接到存储节点

```bash
source secrets/env.sh
echo "set ssl:verify-certificate no" >> ~/.lftprc
lftp -u ${FTP_USER},${FTP_PASS} ${STORAGE_NODE_IP}
```

### 2.2 创建目录结构

在LFTP会话中执行：

```bash
cd /mnt/nfs/nina
mkdir -p models/unet models/vae data/images segmentation_output
```

### 2.3 上传预训练模型

预训练模型上传有两种方式：

#### A. 直接上传完整模型文件夹（如果模型较小）

```bash
# 退出lftp，在本地执行
lftp -u ${FTP_USER},${FTP_PASS} -e "set ssl:verify-certificate no" ${STORAGE_NODE_IP}
cd /mnt/nfs/nina/models/
mirror -R /本地路径/unet unet
mirror -R /本地路径/vae vae
```

#### B. 压缩后上传（如果模型较大）

```bash
# 在本地执行
cd /本地路径/
tar -czvf unet.tar.gz unet/
tar -czvf vae.tar.gz vae/

# 上传压缩文件
lftp -u ${FTP_USER},${FTP_PASS} -e "set ssl:verify-certificate no" ${STORAGE_NODE_IP}
cd /mnt/nfs/nina/models/
put unet.tar.gz
put vae.tar.gz

# 在Run:ai任务中解压文件（训练脚本执行前）
tar -xzvf /mnt/nfs/nina/models/unet.tar.gz -C /mnt/nfs/nina/models/
tar -xzvf /mnt/nfs/nina/models/vae.tar.gz -C /mnt/nfs/nina/models/
```

### 2.4 上传训练脚本和数据

```bash
# 上传训练脚本
cd /mnt/nfs/nina/
put train_segmentation_fixed.py

# 上传COCO标注文件
cd /mnt/nfs/nina/data/
put coco.json

# 上传图像
cd /mnt/nfs/nina/data/images/
mput /本地路径/images/*.jpg  # 或者其他格式的图像
```

### 2.5 创建一个简单的测试推理脚本

```bash
cat > /tmp/process.py << 'EOF'
import os
import torch
import json
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL
import sys
import cv2

# 从train_segmentation_fixed.py导入模型定义
sys.path.append('/mnt/nfs/nina')
from train_segmentation_fixed import SegmentationModel, UNet2DConditionModel, AttnProcessor

# 设置路径
MODEL_PATH = "/mnt/nfs/nina/segmentation_output/best_segmentation_model.pth"
VAE_PATH = "/mnt/nfs/nina/models/vae"
UNET_PATH = "/mnt/nfs/nina/models/unet"
OUTPUT_FILE = "/mnt/nfs/nina/output.txt"
TEST_IMAGE = "/mnt/nfs/nina/data/images/test_image.jpg"  # 替换为实际测试图像
INPUT_SIZE = (256, 256)

def main():
    with open(OUTPUT_FILE, 'w') as f:
        f.write("U-Net分割模型推理测试\n")
        f.write("===================\n\n")
        
        # 检查GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        f.write(f"使用设备: {device}\n\n")
        
        try:
            # 加载VAE
            f.write("加载VAE模型...\n")
            vae = AutoencoderKL.from_pretrained(VAE_PATH).to(device)
            vae.eval()
            
            # 加载UNet
            f.write("加载UNet模型...\n")
            unet = UNet2DConditionModel.from_pretrained(UNET_PATH).to(device)
            unet.set_attn_processor(AttnProcessor())
            
            # 构建分割模型 - 假设有5个类别，实际应根据训练时的类别数量调整
            num_classes = 5  # 根据您的实际类别数进行调整
            f.write(f"创建分割模型，类别数: {num_classes}...\n")
            model = SegmentationModel(unet, num_classes=num_classes).to(device)
            
            # 加载训练好的权重
            f.write(f"加载模型权重: {MODEL_PATH}...\n")
            checkpoint = torch.load(MODEL_PATH, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            f.write("模型加载成功\n\n")
            
            # 记录分割头输入/输出形状
            # 创建一个随机输入进行测试
            f.write("测试分割头...\n")
            test_input = torch.randn(1, 4, 32, 32).to(device)
            with torch.no_grad():
                # 获取编码器特征
                feat, skip_connections = model.encoder(test_input, t=0)
                f.write(f"分割头输入形状: {feat.shape}\n")
                
                # 获取分割头输出
                out = model.decoder(feat, skip_connections)
                f.write(f"分割头输出形状: {out.shape}\n\n")
            
            # 实际图像测试
            if os.path.exists(TEST_IMAGE):
                f.write(f"测试实际图像: {TEST_IMAGE}\n")
                
                # 加载和预处理图像
                from torchvision import transforms
                transform = transforms.Compose([
                    transforms.Resize(INPUT_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5]*3, [0.5]*3)
                ])
                
                img = Image.open(TEST_IMAGE).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device)
                
                # 推理
                with torch.no_grad():
                    # 编码图像
                    latent = vae.encode(img_tensor).latent_dist.sample() * 0.18215
                    f.write(f"VAE编码后潜在表示形状: {latent.shape}\n")
                    
                    # 分割推理
                    output = model(latent, t=0)
                    f.write(f"分割模型输出形状: {output.shape}\n")
                    
                    # 获取预测的类别
                    pred = torch.argmax(output, dim=1)[0].cpu().numpy()
                    
                    # 统计各类别的像素数量
                    for class_idx in range(output.shape[1]):
                        pixel_count = (pred == class_idx).sum()
                        f.write(f"类别 {class_idx} 像素数量: {pixel_count}\n")
                    
                    f.write("\n推理成功完成!\n")
            else:
                f.write(f"警告: 测试图像不存在: {TEST_IMAGE}\n")
                f.write("仅进行了模型加载和形状验证\n")
            
        except Exception as e:
            f.write(f"推理过程中发生错误: {e}\n")
            import traceback
            f.write(traceback.format_exc())

if __name__ == "__main__":
    main()
EOF

lftp -u ${FTP_USER},${FTP_PASS} -e "set ssl:verify-certificate no" ${STORAGE_NODE_IP}
cd /mnt/nfs/nina/
put /tmp/process.py
```

## 3. 在Run:ai上创建训练任务

### 3.1 创建环境

1. 前往Workload manager > Assets > Environments并点击+ NEW ENVIRONMENT
2. 填写以下信息：
   - Scope: runai/runai-cluster/<YOUR_LAB>/<YOUR_PROJECT>
   - Environment name: nina-segmentation
   - Workspace: (选中)
   - Training: (选中)
   - Inference: (不选)
   - Image URL: pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
   - Image pull policy: Always pull the image from the registry
   - Tool: Jupyter
   - Command: /run.sh "pip install diffusers transformers pycocotools opencv-python" "jupyter lab --ip=0.0.0.0 --no-browser --allow-root --NotebookApp.base_url=/${RUNAI_PROJECT}/${RUNAI_JOB_NAME} --NotebookApp.token='' --notebook-dir=/"
3. 点击CREATE ENVIRONMENT

### 3.2 创建训练工作负载

1. 前往Workload manager > Workloads并点击+ NEW WORKLOAD > Training
2. 填写以下信息：
   - Workspace name: nina-segmentation-train
   - Environment: nina-segmentation
   - Command: /run.sh "cd /mnt/nfs/nina && pip install diffusers transformers pycocotools opencv-python && python train_segmentation_fixed.py"
   - Compute resource: gpu1
   - Data sources: <YOUR_LAB>-nfs
   - Attempts: 1
3. 点击CREATE WORKSPACE

### 3.3 创建推理工作负载（训练完成后）

1. 前往Workload manager > Workloads并点击+ NEW WORKLOAD > Batch
2. 填写以下信息：
   - Workspace name: nina-segmentation-inference
   - Environment: nina-segmentation
   - Command: /run.sh "cd /mnt/nfs/nina && pip install diffusers transformers pycocotools opencv-python && python process.py"
   - Compute resource: gpu1
   - Data sources: <YOUR_LAB>-nfs
   - Attempts: 1
3. 点击CREATE WORKSPACE

## 4. 监控训练和查看结果

1. 在Workload manager > Workloads中监控训练进度
2. 等待训练完成
3. 下载训练好的模型和输出文件：

```bash
lftp -u ${FTP_USER},${FTP_PASS} -e "set ssl:verify-certificate no" ${STORAGE_NODE_IP}
cd /mnt/nfs/nina/segmentation_output
get best_segmentation_model.pth
get final_segmentation_model.pth

# 获取推理结果
cd /mnt/nfs/nina
get output.txt
```

## 注意事项

1. 根据您的实际环境调整路径和参数
2. 确保所有依赖包都已安装
3. 查看日志文件了解训练和推理过程中的详细信息
4. 如果训练时间较长，可以使用信号处理器功能（Ctrl+C）安全地终止训练并保存检查点
5. 任务完成后请删除工作负载以释放资源
