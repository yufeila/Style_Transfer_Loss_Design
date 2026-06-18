# 一次性 sanity check 记录

记录日期：2026-06-18

## 结论

sanity check 已成功完成，且只执行了一次最小模型流程。

说明：本记录是重新设计正式实验前的历史 sanity check；旧 `configs/baseline.json` 已废弃。后续正式实验使用 `configs/strong_baseline.json`，不要重复小型 sanity check。

已确认：

- 依赖安装成功。
- `torch` / `torchvision` / `PIL` / `numpy` / `matplotlib` / `pandas` / `tqdm` 可导入。
- VGG19 预训练权重可下载并加载。
- 远程内容图和风格图可读取。
- GPU 可用。
- 一次前向传播、反向传播和优化步可完成。
- 临时输出、日志和中间产物已删除。

## 环境信息

```text
Python: 3.10.20
torch: 2.12.1+cu130
torchvision: 0.27.1+cu130
CUDA available: True
GPU: NVIDIA A800 80GB PCIe
```

## 依赖安装

执行位置：

```text
/data1/yyf/Style_Transfer_Loss_Design/code
```

安装命令：

```bash
source /workspace/yyf/env/miniconda3/etc/profile.d/conda.sh
conda activate /workspace/yyf/env/conda/style-transfer
cd /data1/yyf/Style_Transfer_Loss_Design/code
pip install -r requirements.txt
```

## sanity check 命令

只执行一次：

注意：下面命令仅作为历史记录保留，不再复用；旧 `configs/baseline.json` 已删除。

```bash
source /workspace/yyf/env/miniconda3/etc/profile.d/conda.sh
conda activate /workspace/yyf/env/conda/style-transfer
cd /data1/yyf/Style_Transfer_Loss_Design/code
CUDA_VISIBLE_DEVICES=1 python run_style_transfer.py \
  --config configs/baseline.json \
  --output-dir /data1/yyf/Style_Transfer_Loss_Design/outputs/_sanity_check \
  --steps 1 \
  --optimizer adam \
  --device cuda
```

## 临时输出记录

sanity check 临时输出目录：

```text
/data1/yyf/Style_Transfer_Loss_Design/outputs/_sanity_check
```

成功时该目录包含：

```text
config.json
metrics.csv
result.png
loss_curve.png
```

临时指标：

```text
total_loss,content_loss,style_loss,tv_loss,step
149.40597534179688,0.0,0.0014940598048269749,0.11449332535266876,1
```

## 清理结果

已删除：

```text
/data1/yyf/Style_Transfer_Loss_Design/outputs/_sanity_check
```

清理验证：

```bash
test ! -e /data1/yyf/Style_Transfer_Loss_Design/outputs/_sanity_check
```

返回成功。

## 下一步

可以进入 TODO 第 4 节“正式实验配置”。由于 sanity check 已经成功，后续不再重复小型试探实验，应直接使用正式配置推进。
