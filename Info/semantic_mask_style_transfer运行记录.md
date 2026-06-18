# semantic_mask_style_transfer 运行记录

> 当前状态提示（2026-06-19）：本文件记录的是旧版粗 mask 3000-step 实验。按当前 6.2 方案，旧粗 mask 结果已删除或不再作为正式结论；当前正在使用 `create_sam_semantic_masks.py` 生成 SAM 语义 mask，已得到 `Experiments/results/semantic_mask_style_transfer/masks.png`，但尚未重新跑 3000-step 风格迁移。

## 结论

实验 6.2 已完成。相比 `strong_baseline`，语义区域约束明显减少了水面和天空中的跨区域 Kandinsky 斜向色块泄漏：水面主要接收蓝绿色块，天空主要保留浅色/线条纹理；但结果更粗粝，content loss 更高，整体 Kandinsky 风格没有被削弱。

## 方法依据

- `Controlling Perceptual Factors in Neural Style Transfer`：使用 spatial guidance channel，在每个区域内计算 guided Gram matrix，避免天空等区域被地面风格污染。
- `Semantic Style Transfer and Turning Two-Bit Doodles into Fine Artworks`：语义图可以来自手工标注或分割模型，并用于约束风格匹配区域。
- `Deep Photo Style Transfer`：将内容图和风格图的 semantic segmentation mask 下采样到每层 feature map，在每个 label 内分别计算 Gram style loss，减少 spillover。

## 代码与配置

- 新增配置：`Experiments/code/configs/semantic_mask_style_transfer.json`
- 新增 mask 模块：`Experiments/code/src/masks.py`
- 新增 mask 生成入口：`Experiments/code/create_semantic_masks.py`
- 修改 loss：`Experiments/code/src/losses.py`
- 修改 runner：`Experiments/code/run_style_transfer.py`

核心设置：

```json
{
  "style_loss_mode": "masked_gram",
  "image_size": 1024,
  "optimizer": "lbfgs",
  "steps": 3000,
  "content_weight": 3.0,
  "style_weight": 300000.0,
  "tv_weight": 0.1
}
```

## Mask 记录

Mask 数量：4。

内容图 mask 来源：固定 Golden Gate Bridge 多边形 + HSV 桥体/建筑红橙区域补充。

风格图 mask 来源：Kandinsky 图像 HSV 弱规则，分为浅色背景、蓝绿色区域、暖色/黑色线条、剩余陆地色块。

区域占比：

| image | sky | water | bridge_building | mountain_land |
|---|---:|---:|---:|---:|
| content | 0.437844 | 0.161149 | 0.345891 | 0.055116 |
| style | 0.162601 | 0.130704 | 0.530853 | 0.175842 |

## 远程运行

远程目录：`/data1/yyf/Style_Transfer_Loss_Design`

conda 环境：`/workspace/yyf/env/conda/style-transfer`

运行命令：

```bash
cd /data1/yyf/Style_Transfer_Loss_Design/code
CUDA_VISIBLE_DEVICES=0 nohup /workspace/yyf/env/conda/style-transfer/bin/python run_style_transfer.py --config configs/semantic_mask_style_transfer.json > /data1/yyf/Style_Transfer_Loss_Design/logs/semantic_mask_style_transfer.log 2>&1 &
```

运行信息：

- 代码状态：当前本地工作区未检测到 Git 仓库，按已同步的代码文件、配置文件和本记录追溯。
- 远程 PID：`866999`
- GPU：`CUDA_VISIBLE_DEVICES=0`
- 用时：约 9 分 34 秒
- 日志：`/data1/yyf/Style_Transfer_Loss_Design/logs/semantic_mask_style_transfer.log`
- 远程输出：`/data1/yyf/Style_Transfer_Loss_Design/outputs/semantic_mask_style_transfer/`
- 本地结果：`Experiments/results/semantic_mask_style_transfer/`

## 最终指标

| metric | value |
|---|---:|
| total_loss | 2.671537399291992 |
| content_loss | 0.38185644149780273 |
| style_loss / masked_style_loss | 0.000004627988801075844 |
| tv_loss | 1.3757127523422241 |
| weighted_content_loss | 1.1455693244934082 |
| weighted_style_loss | 1.3883966207504272 |
| weighted_tv_loss | 0.1375712752342224 |
| style_loss_sky | 0.0000023558859538752586 |
| style_loss_water | 0.0000008015997536858777 |
| style_loss_bridge_building | 0.0000012265489885976422 |
| style_loss_mountain_land | 0.00000024395433229074115 |

对照 `strong_baseline`：

| experiment | total_loss | content_loss | style_loss | weighted_style_loss |
|---|---:|---:|---:|---:|
| strong_baseline | 0.42214536666870117 | 0.07444623112678528 | 0.0000006173493147798581 | 0.18520478904247284 |
| semantic_mask_style_transfer | 2.671537399291992 | 0.38185644149780273 | 0.000004627988801075844 | 1.3883966207504272 |

## 输出文件

- `result.png`
- `loss_curve.png`
- `metrics.csv`
- `config.json`
- `masks.png`
- `baseline_vs_semantic_mask.png`
- `semantic_mask_style_transfer.log`

## 视觉观察

- 水面：baseline 中水面混入了大量红黄高饱和斜向块；semantic mask 后水面主要是蓝绿色 Kandinsky 色块，跨区域泄漏明显减少。
- 天空：baseline 天空有大块彩色斜向纹理；semantic mask 后天空以浅色背景和细碎线条为主，异常高饱和色块减少。
- 山体与桥体：桥体更集中接收暖色/黑色线条，山体保留部分绿色/黄色纹理，但山体 mask 占比较小，区域效果不如水面明显。
- 风格强度：没有过度削弱。相反，区域内风格更强，整体更抽象、更粗粝，代价是内容结构和自然感下降。
