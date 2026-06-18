# Style Transfer Code

本目录是本地核心代码副本；完整运行在远程：

```text
/data1/yyf/Style_Transfer_Loss_Design/code/
```

## 环境

远程激活环境：

```bash
source /workspace/yyf/env/miniconda3/etc/profile.d/conda.sh
conda activate /workspace/yyf/env/conda/style-transfer
```

安装依赖：

```bash
pip install -r requirements.txt
```

## Strong Baseline

当前 baseline 是 Gatys-aligned 配置：

```text
init: noise
pooling: avg
content_weight: 3.0
style_weight: 300000.0
tv_weight: 0.1
```

建议优先使用空闲 GPU，例如 GPU 1：

```bash
cd /data1/yyf/Style_Transfer_Loss_Design/code
CUDA_VISIBLE_DEVICES=1 python run_style_transfer.py --config configs/strong_baseline.json
```

输出目录由配置文件指定，默认：

```text
/data1/yyf/Style_Transfer_Loss_Design/outputs/strong_baseline/
  config.json
  metrics.csv
  result.png
  loss_curve.png
```

## 结果汇总

```bash
python evaluate_results.py /data1/yyf/Style_Transfer_Loss_Design/outputs/strong_baseline \
  --output /data1/yyf/Style_Transfer_Loss_Design/outputs/summary.csv
```

## SAM 语义 Mask

当前 6.2 正在使用 SAM 改进语义 mask，已生成 mask，但还没有重新跑 3000-step 风格迁移。

```bash
cd /data1/yyf/Style_Transfer_Loss_Design/code
CUDA_VISIBLE_DEVICES=1 python create_sam_semantic_masks.py \
  --content-image /data1/yyf/Style_Transfer_Loss_Design/data/raw/content/golden_gate_bridge.jpg \
  --style-image /data1/yyf/Style_Transfer_Loss_Design/data/raw/style/composition_vii_kandinsky.jpg \
  --checkpoint /data1/yyf/Style_Transfer_Loss_Design/models/sam/sam_vit_b_01ec64.pth \
  --output-dir /data1/yyf/Style_Transfer_Loss_Design/outputs/semantic_mask_style_transfer \
  --image-size 1024
```

当前核心输出：

```text
/data1/yyf/Style_Transfer_Loss_Design/outputs/semantic_mask_style_transfer/masks.png
```

## 画对比图

```bash
python plot_results.py grid \
  --images content.jpg style.jpg result.png \
  --labels content style result \
  --output comparison.png
```

## 注意

- 本 README 只给运行方式；一次性 sanity check 已完成，后续不要重复小型试探实验。
- 旧 `baseline`、旧候选中间产物和上一轮 `strong_baseline` 结果已删除；正式结果从新的 `strong_baseline` 重新开始。
