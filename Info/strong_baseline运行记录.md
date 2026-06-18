# 旧 strong baseline 运行记录（已废弃）

记录日期：2026-06-18

## 结论

本记录对应上一轮 `content_noise + max pooling + tv_weight=1.0` 的 strong baseline。该结果因 Kandinsky 风格仍偏弱，已被废弃；对应本地结果、远程输出、中间图和远程日志已删除。

## 远程运行信息

远程工作区：

```text
/data1/yyf/Style_Transfer_Loss_Design
```

远程代码目录：

```text
/data1/yyf/Style_Transfer_Loss_Design/code
```

conda 环境：

```text
/workspace/yyf/env/conda/style-transfer
```

GPU：

```text
CUDA_VISIBLE_DEVICES=1
NVIDIA A800 80GB PCIe
```

运行命令：

```bash
source /workspace/yyf/env/miniconda3/etc/profile.d/conda.sh
conda activate /workspace/yyf/env/conda/style-transfer
cd /data1/yyf/Style_Transfer_Loss_Design/code
CUDA_VISIBLE_DEVICES=1 python run_style_transfer.py --config configs/strong_baseline.json
```

日志路径：

```text
/data1/yyf/Style_Transfer_Loss_Design/logs/strong_baseline.log
```

远程输出路径：

```text
/data1/yyf/Style_Transfer_Loss_Design/outputs/strong_baseline/
```

本地同步路径：

```text
/Users/yufei/course/third_year_sem2/AI_class/Final project/Style_Transfer_Loss_Design/Experiments/results/strong_baseline/
```

## 正式配置

- 内容图：`golden_gate_bridge.jpg`
- 风格图：`composition_vii_kandinsky.jpg`
- 分辨率：$1024\times1024$
- 初始化：`content_noise`
- `init_noise_std = 0.05`
- 优化器：`L-BFGS`
- 迭代步数：`3000`
- 内容层：`conv4_2`
- 风格层：`conv1_1`、`conv2_1`、`conv3_1`、`conv4_1`、`conv5_1`
- 内容权重：$\alpha=10.0$
- 风格权重：$\beta=100000.0$
- 平滑权重：$\gamma=1.0$
- 中间图保存间隔：每 `300` step

## 输出文件

核心输出：

```text
config.json
metrics.csv
result.png
loss_curve.png
strong_baseline.log
```

中间图：

```text
step_0300.png
step_0600.png
step_0900.png
step_1200.png
step_1500.png
step_1800.png
step_2100.png
step_2400.png
step_2700.png
step_3000.png
```

## 最终指标

`metrics.csv` 最后一行：

```text
total_loss,content_loss,style_loss,tv_loss,step
12.766976356506348,0.7598769664764404,4.987589636584744e-05,0.18061770498752594,3000
```

按总损失权重换算：

```text
content term = 10.0 * 0.7598769664764404 = 7.598769664764404
style term   = 100000.0 * 4.987589636584744e-05 = 4.987589636584744
tv term      = 1.0 * 0.18061770498752594 = 0.18061770498752594
total        = 12.766976356506348
```

## 视觉检查

- 漩涡形噪声：相比旧 500-step baseline 明显减少，但天空和水面仍有细碎纹理。
- Kandinsky 风格：色彩和局部线条感更明显，但整体抽象程度仍偏弱。
- 内容保持：Golden Gate Bridge 主结构清楚，桥塔、桥面和缆线仍可辨认。
- TV 约束：`tv_loss` 前期上升后进入平台，最终约为 `0.1806`；更强 TV 权重确实参与总损失，但没有完全消除细碎纹理。

## 后续判断

该结果不再作为正式 baseline。新的 baseline 已改为 `noise + avg pooling + alpha=3.0 + beta=300000.0 + gamma=0.1`，并额外报告 weighted loss。
