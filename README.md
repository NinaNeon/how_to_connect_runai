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
太好了，這段 log 非常清楚，你的操作已經 **證明了可以成功上傳**，也有自己解決了憑證與目錄問題。以下是從你這些操作中提煉出來的 **正確且完整的上傳流程**，特別針對你這次要傳的 `sd-v1-3.ckpt`：

---

### ✅ LFTP 上傳大檔案（`sd-v1-3.ckpt`）到 `/mnt/nfs/nina` 正確流程

1. **進入工作資料夾，啟用環境變數：**

```bash
cd ~/runai-isaac
source secrets/env.sh
```

2. **啟動 LFTP 並關閉 SSL 憑證驗證：**

```bash
lftp -u ${FTP_USER},${FTP_PASS} ${STORAGE_NODE_IP}
```

進入後馬上輸入：

```bash
set ssl:verify-certificate no
```

3. **確保目標目錄存在，若不存在就建立並進入：**

```bash
mkdir -p /mnt/nfs/nina
cd /mnt/nfs/nina
```

4. **上傳檔案（用絕對路徑以避免找不到檔案）：**

```bash
put /home/nina/Downloads/sd-v1-3.ckpt
```

---

### 🔁 小技巧：如果傳到一半斷了，怎麼續傳？

```bash
pget -c /home/nina/Downloads/sd-v1-3.ckpt
```

---

### ✅ 成功檢查範例：

```bash
ls -lh /mnt/nfs/nina/
# 應看到 sd-v1-3.ckpt 大小約為 4.3G（4,581,073,398 bytes）
```

---

如需背景執行，可開一個 `screen` 或 `tmux` 窗口進行，以防 LFTP 中斷。你想現在我幫你直接寫成 `.lftp_script` 自動跑也可以，告訴我一聲。


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

【重新建 Workspace：防止它自己結束】
回到 Workloads → + NEW WORKLOAD → Workspace

填這些（重點在紅字）：

欄位    要填什麼
Name    unzip-workspace-v2（可以自由命名）
Environment    你剛剛建的那個，例如 nina-cat500-env
Command    /bin/bash -c "sleep infinity" ← :white_check_mark: :white_check_mark: :white_check_mark:
Compute    選 gpu1
Data Source    :white_check_mark: 勾選 elsalab-nfs

按 CREATE

這樣 container 會一直「睡著」，你就有時間按：

sql
複製
編輯
CONNECT → Terminal :white_check_mark:
:white_check_mark: 建好後該怎麼做？
在 Terminal 裡輸入這三行：

bash
複製
編輯
apt update && apt install -y unzip
cd /mnt/nfs/nina
unzip -o cat500_classifier.zip -d cat500_classifier
解壓成功後，再跑：

bash
複製
編輯
cd cat500_classifier
pip install -r requirements.txt
python train_cat500_fixed6.py
:white_check_mark: 總結：你要做的事只剩一件
刪掉那個 Completed 的 Workspace，重建一個加 sleep infinity 的。

你只要這一步，就能成功進 terminal 開始跑。
完成後你整個流程就打通了，要不要我幫你寫一鍵 workspace 建立指令 + unzip 指令包 也可以。你只要說一聲！









你說：

then????? still cant find terminal

ChatGPT 說：
你做得完美！現在 unzip-workspace-v2 正在 Running 狀態，代表 container 正常啟動了。

:white_check_mark: 但你還看不到 Terminal，這是 Run:ai 預設 UI 設定的關係：
:point_right: 只有你建立的 Environment 有選「Jupyter Tool」 的時候，才會出現：
sql
複製
編輯
CONNECT → Jupyter → Terminal
你剛剛的 Environment 沒勾 Tool，所以只看到 CONNECT 但沒彈選單。

:white_check_mark: 解法有兩種，我給你快速選：
:white_check_mark: 【方法 1】直接改這個 Environment，加上 Jupyter 工具（推薦）
你可以照以下步驟補一下：

到 Run:ai 左邊欄位「Assets → Environments」

找到你用來建 unzip-workspace-v2 的那個 Environment

點右邊 ⋮（三個點） → Edit

把這個項目打勾：

yaml
複製
編輯
:white_check_mark: Tool: Jupyter
按 SAVE

然後再回到 Workloads，刪掉 unzip-workspace-v2，再重新用這個環境建一個 Workspace（+ NEW WORKLOAD → Workspace）

這樣新建出來的 workload 就會有：

markdown
複製
編輯
CONNECT ⬇︎
  → Jupyter
    → Terminal :white_check_mark:
:white_check_mark: 【方法 2】現在就進去（雖然不優）
如果你不想重建，也可以現在按上面藍色那個 CONNECT 按鈕（單一按鈕），看看能不能跳進 shell，不一定是 Jupyter Terminal，但某些 image 預設會給 bash。

:white_check_mark: 最推薦流程（最穩最保證有 Terminal）：
我幫你整理一下：

你現在要做的事    怎麼做
解壓 zip、跑 python    一定要進 container 的 Terminal
要有 Terminal    :white_check_mark: 用含 Jupyter 的 Environment 建 Workspace
沒看到 Terminal    → 回去 Edit Environment → 勾選 Tool: Jupyter → 重新建 Workspace

要我幫你產「環境 YAML 設定 + 可執行腳本」讓你下次一鍵就好嗎？
你只要說「幫我出一鍵配置環境 + 工作流程」，我直接幫你寫好 env.yaml + submit.sh。

現在只差這一小步，解壓後你就能正式訓練了 :brain::muscle:！



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
