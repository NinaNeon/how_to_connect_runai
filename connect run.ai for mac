那非常適合你！你目前的目的是：

> ✅ **在 MacBook 上開發 → 經由 VPN + FTP 上傳 → 透過 Run\:ai 執行於 GPU 叢集 → 回傳結果**

這正是這份 `run-ai-isaac` 文件設計的主要使用情境！以下是根據你目標整理的完整執行步驟（針對 MacBook）：

---

## ✅【一鍵概覽】你的流程是：

1. **連上 VPN**：取得 Run\:ai 網頁介面與內部 FTP 權限
2. **開發程式與 Dockerfile**：本地寫好模型與推論腳本
3. **打包 Docker image 並上傳到 Docker Hub**
4. **用 FTP 上傳資料與程式碼到 NFS（叢集儲存空間）**
5. **在 Run\:ai 網頁介面建立環境與工作負載**
6. **連線到 Jupyter Lab 或查看 Batch logs**
7. **下載訓練結果、刪除工作負載以釋放資源**

---

## 🧰 一、前置：環境與 VPN 設定

### ✅ 1. 安裝必要工具（Homebrew 都可）

```bash
brew install openvpn3 lftp
brew install --cask docker
```

> Docker Desktop for Mac 要先登入 Docker Hub 才能推送 image。

### ✅ 2. 下載 repo 並填寫 VPN 設定

```bash
git clone https://github.com/j3soon/run-ai-isaac.git
cd run-ai-isaac
```

將你的 `.ovpn` 配置檔命名為 `client.ovpn` 並放到 `secrets/` 資料夾
編輯 `secrets/env.sh` 填入叢集資訊（向管理員索取）：

```bash
export RUNAI_URL="https://runai.local"
export STORAGE_NODE_IP="10.xxx.xxx.xxx"
export FTP_USER="yourname"
export FTP_PASS="yourpassword"
```

### ✅ 3. 連線 VPN

```bash
source secrets/env.sh
scripts/vpn/install_config.sh client.ovpn
scripts/vpn/connect.sh
```

---

## 🧪 二、開發與 Docker Image 準備

### ✅ 4. 撰寫模型程式與 `requirements.txt`

例如放在 `./mnist/` 或你的專案資料夾中，記得加一個 `main.py`

### ✅ 5. 建立 Dockerfile（範例）

```Dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
WORKDIR /
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY mnist/ /mnist/
COPY run.sh /
COPY omnicli/ /omnicli/
```

> 注意 `run.sh` 必須放在根目錄 `/`，否則系統無法呼叫。

### ✅ 6. Build 並 Push 到 Docker Hub

```bash
docker build -t yourdockerhub/runai-mnist .
docker push yourdockerhub/runai-mnist
```

---

## 🚀 三、上傳資料與程式碼到 NFS

```bash
source secrets/env.sh
lftp -u ${FTP_USER},${FTP_PASS} ${STORAGE_NODE_IP}
> cd /mnt/nfs
> mkdir nina
> cd nina
> mirror --reverse mnist mnist
```

⚠️ 請先 `rm -r mnist` 再上傳，以防版本殘留！

---

## 🖥️ 四、Run\:ai 上建立環境與工作負載

登入 `https://runai.local`

> 密碼初次登入會要求更改

### ✅ 建立 Environment：

> Workload manager > Assets > Environments > `+ NEW ENVIRONMENT`

填寫：

* 名稱：`nina-mnist`
* Image URL：`yourdockerhub/runai-mnist`
* Tools：勾選 `Jupyter`
* Command：

```bash
/run.sh "pip install jupyterlab" "jupyter lab --ip=0.0.0.0 --no-browser --allow-root --NotebookApp.base_url=/${RUNAI_PROJECT}/${RUNAI_JOB_NAME} --NotebookApp.token='' --notebook-dir=/"
```

---

### ✅ 建立 Workload（Jupyter Lab）

> Workload manager > Workloads > `+ NEW WORKLOAD > Workspace`

* 名稱：`nina-mnist-test1`
* Compute：選 GPU1
* Data Sources：選 `<YOUR_LAB>-nfs`
* Environment：選剛剛創建的 `nina-mnist`

建立後，點 CONNECT > Jupyter Lab → 進入終端執行：

```bash
cd /mnt/nfs/nina/mnist
pip install -r requirements.txt
python main.py --save-model
```

---

## 📥 五、下載結果

回到 lftp：

```bash
cd /mnt/nfs/nina/mnist
get mnist_cnn.pt
rm mnist_cnn.pt
```

---

## 🧹 六、刪除 Workload 釋放資源

回到 Workload manager，刪除剛剛建立的 workload。

---

## ✅ 補充建議

* 如果你未來常會更新模型程式碼，可考慮：

  * 用 FTP 上傳而不是 Docker build（速度快）
  * 或加入版本號路徑如 `/nina/mnist-v2`
* 可整合 **WandB** 做實驗追蹤

---

## 如需我幫你：

✅ 幫你寫 Dockerfile
✅ 幫你設定 `env.sh`、run.sh
✅ 幫你整理你目前的 `main.py` 結構

都可以直接傳給我，我可以為你客製化接下來的步驟。是否要我幫你開始整理一份最小範例包？
