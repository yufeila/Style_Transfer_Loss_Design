# color_preservation 运行记录

记录日期：2026-06-18

## 结论

颜色保持实验已按 Gatys et al. 的 `luminance-only transfer` 修正版完成。修正版只优化单通道亮度 $Y$，先对风格图亮度做论文中的 luminance-histogram matching，再将生成亮度 $Y_T$ 与内容图色度 $I_C,Q_C$ 合成最终图像。

结果表明：相比首版，loss 不再在前几步后异常卡死，最终图像也不再像单纯的色度通道图；但由于该方法主动丢弃风格图颜色，Kandinsky 的高饱和彩色几何风格仍明显弱于 `strong_baseline`。

## 方法

实现方式对应论文 Figure 5 的流程：

1. 将内容图和风格图转换到 YIQ 颜色空间。
2. 提取内容图亮度 $L_C$、内容图色度 $I_C,Q_C$ 和风格图亮度 $L_S$。
3. 对风格图亮度做二阶统计匹配：

   $$
   L'_S=\frac{\sigma_C}{\sigma_S}(L_S-\mu_S)+\mu_C
   $$

4. 直接优化单通道 $Y_{\text{generated}}$；送入 VGG19 前复制为 3 通道。
5. 最终合成：

   $$
   T=(Y_{\text{generated}}, I_C, Q_C)
   $$

这不是颜色损失函数，而是论文式颜色保持预处理/后处理方法。

## 修正点

首版问题：使用 3 通道生成图，并在每个 L-BFGS step 后执行 `luminance_projection=true`，导致优化几步后进入平台，最终图像接近内容图色度通道叠加颗粒纹理。

修正版改动：

- 新增 `optimize_luminance_channel: true`，真正优化单通道 $Y$。
- 新增 `luminance_histogram_matching: true`，补上论文 Figure 5(e)(f) 的亮度匹配。
- 关闭 `luminance_projection`，避免每步硬投影破坏 L-BFGS。
- 将 `style_weight` 从 `300000.0` 降到 `200000.0`。
- 将 `tv_weight` 从 `0.1` 提高到 `1.0`。
- 额外保存 `content_luminance_reference.png`、`content_chrominance_reference.png`、`style_luminance_matched_reference.png`。

## 远程运行信息

远程工作区：

```text
/data1/yyf/Style_Transfer_Loss_Design
```

运行命令：

```bash
cd /data1/yyf/Style_Transfer_Loss_Design/code
CUDA_VISIBLE_DEVICES=1 /workspace/yyf/env/conda/style-transfer/bin/python run_style_transfer.py --config configs/color_preservation.json
```

日志路径：

```text
/data1/yyf/Style_Transfer_Loss_Design/logs/color_preservation_fixed.log
```

远程输出路径：

```text
/data1/yyf/Style_Transfer_Loss_Design/outputs/color_preservation/
```

本地同步路径：

```text
/Users/yufei/course/third_year_sem2/AI_class/Final project/Style_Transfer_Loss_Design/Experiments/results/color_preservation/
```

## 配置

```text
experiment_name: color_preservation
color_preservation: luminance_only
optimize_luminance_channel: true
luminance_histogram_matching: true
luminance_projection: false
image_size: 1024
init: noise
pooling: avg
optimizer: lbfgs
steps: 3000
content_layers: conv4_2
style_layers: conv1_1, conv2_1, conv3_1, conv4_1, conv5_1
content_weight alpha: 3.0
style_weight beta: 200000.0
tv_weight gamma: 1.0
```

## 输出文件

```text
config.json
metrics.csv
result.png
loss_curve.png
luminance_result.png
content_color_reference.png
content_luminance_reference.png
content_chrominance_reference.png
style_luminance_reference.png
style_luminance_matched_reference.png
baseline_vs_color_preservation.png
step_0300.png ... step_3000.png
color_preservation_fixed.log
```

## 最终指标

`metrics.csv` 最后一行：

```text
total_loss,content_loss,style_loss,tv_loss,weighted_content_loss,weighted_style_loss,weighted_tv_loss,color_mean_loss,color_std_loss,color_loss,step
0.13983578979969025,0.028336860239505768,2.558886080805678e-07,0.003647490171715617,0.0850105807185173,0.051177721470594406,0.003647490171715617,4.276370418665465e-06,0.005802157800644636,0.005806433968245983,3000
```

解释：

- `total_loss = 0.1398`，明显低于首版异常平台值 `5.1920`。
- `color_loss = 0.0058`，颜色统计比首版 `0.0122` 更接近内容图。
- `weighted_content_loss = 0.0850`，内容结构保持较强。
- `weighted_style_loss = 0.0512`，亮度风格被迁移，但彩色风格被保色机制压制。

## 视觉检查

- 颜色保持：成功，天空、水面、桥体主要保持内容图颜色。
- 亮度迁移：成功，`luminance_result.png` 中可见明显亮暗笔触和块状纹理。
- 内容结构：桥塔、缆线和桥面清楚。
- 方法局限：Kandinsky 的彩色抽象色块明显弱于 `strong_baseline`，这是 luminance-only 方法主动丢弃风格颜色导致的。

## 后续用途

该实验可作为报告中的“颜色可控性”模块，说明经典 NST 可以通过颜色空间分解实现可控风格迁移：颜色由内容图色度通道控制，风格主要作用在亮度通道。
