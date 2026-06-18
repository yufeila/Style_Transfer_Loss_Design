# Gatys-aligned strong baseline 运行记录

记录日期：2026-06-18

## 结论

更新后的 `strong_baseline` 已完成。相比上一轮，Kandinsky 抽象色块、斜向纹理和高饱和色彩明显增强；Golden Gate Bridge 主结构仍可辨认。代价是局部高频纹理更重，尤其在天空、水面和山体区域。

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

运行耗时：约 `5:13`。

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
- 初始化：`noise`
- pooling：`avg`
- 优化器：`L-BFGS`
- 迭代步数：`3000`
- 内容层：`conv4_2`
- 风格层：`conv1_1`、`conv2_1`、`conv3_1`、`conv4_1`、`conv5_1`
- 内容权重：$\alpha=3.0$
- 风格权重：$\beta=300000.0$
- 平滑权重：$\gamma=0.1$
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
total_loss,content_loss,style_loss,tv_loss,weighted_content_loss,weighted_style_loss,weighted_tv_loss,step
0.42214536666870117,0.07444623112678528,6.173493147798581e-07,0.1360190212726593,0.22333869338035583,0.18520478904247284,0.013601901941001415,3000
```

weighted loss 分解：

```text
content term = 0.2233
style term   = 0.1852
tv term      = 0.0136
total        = 0.4221
```

解释：raw style loss 很小，但乘以 $\beta=300000.0$ 后，weighted style term 与 weighted content term 已经接近；这解释了为什么这轮风格强度明显提升。

## 与 Gatys 2015 的关系

相同点：

- 内容层使用 `conv4_2`。
- 风格层使用 `conv1_1` 到 `conv5_1` 五层。
- 直接优化生成图像像素。
- 使用 average pooling 更接近论文 Methods。
- 使用噪声初始化更接近论文生成流程。

差异点：

- 当前仍加入 TV loss，论文原始目标没有该项。
- 当前 $\alpha/\beta=10^{-5}$，比论文 Kandinsky 示例的 $10^{-4}$ 更偏风格；这是为了补偿本实现中的 Gram/MSE 归一化差异。
- 当前报告 weighted loss，避免只看 raw loss 误判优化压力。

## 视觉检查

- Kandinsky 抽象风格：明显增强，出现大面积高饱和色块和斜向抽象纹理。
- 内容保持：桥塔、桥面、缆线主体仍可辨认。
- 高频噪声：比上一轮更明显，但没有完全破坏主体结构。
- TV 约束：weighted TV term 最终仅约 `0.0136`，说明平滑约束较弱，主要目标转向内容与风格匹配。

## 后续判断

这轮结果更适合作为最终 baseline。报告中应强调：从上一轮保守结果到本轮强风格结果，关键不只是“增加 step”，而是把初始化、pooling、权重比例和 weighted loss 诊断都向 Gatys 2015 的方法逻辑靠拢。
