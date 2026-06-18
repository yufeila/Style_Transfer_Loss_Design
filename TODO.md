# TODO: 图像风格迁移中的多损失设计

## 项目目标

- [ ] 完成经典 Neural Style Transfer 复现。
- [ ] 围绕内容损失、风格损失、总变分损失解释 baseline 结果。
- [ ] 完成与 Gatys 2015 论文设置对齐的高质量 baseline。
- [ ] 输出可用于报告和 PPT 的核心结果图、loss 曲线和指标表格。
- [ ] 完成最终实验报告、汇报 PPT 和提交压缩包。

## 0. 工作区与远程环境

- [x] 确认远程服务器可连接：`ssh gpu-server`。
- [x] 确认远程工作区存在：`/data1/yyf/Style_Transfer_Loss_Design`。
- [x] 若远程工作区不存在，创建项目目录，并包含 `code/`、`data/`、`outputs/`、`logs/`、`models/`。
- [x] 确认远程 conda 可用。
- [x] 确认或创建 conda 环境，优先放在 `/workspace/yyf/env/conda/`。
- [x] 确认 GPU 状态、显存、磁盘空间和网络可用性。
- [x] 如需下载依赖、模型或数据，先确认是否需要本地代理和 SSH 反向隧道。
- [x] 在本地 `Experiments/` 仅保留核心代码、核心配置、核心结果和实验摘要。

## 1. 资料与数据准备

- [x] 阅读并标注核心论文：
  - [x] Gatys et al., A Neural Algorithm of Artistic Style
  - [x] Gatys et al., Image Style Transfer Using Convolutional Neural Networks
  - [x] Johnson et al., Perceptual Losses for Real-Time Style Transfer and Super-Resolution
  - [x] Zhang et al., LPIPS
- [x] 在 `Reference/` 记录论文链接、核心公式和可引用段落。
- [x] 准备内容图像，建议至少 3 张，覆盖人像、建筑、风景等不同结构。
- [x] 准备风格图像，建议至少 3 张，覆盖油画、水彩、素描等明显风格。
- [x] 记录所有图片来源和许可信息。
- [x] 统一实验输入分辨率，正式实验推荐 $512\times512$ 或更高。

## 2. 代码实现

- [x] 设计远程代码结构：

```text
code/
  run_style_transfer.py
  evaluate_results.py
  plot_results.py
  configs/
  src/
```

- [x] 实现图像读取、resize、归一化和反归一化。
- [x] 加载 VGG19 预训练模型，并冻结参数。
- [x] 实现内容特征提取，默认内容层为 `conv4_2`。
- [x] 实现 Gram 矩阵计算。
- [x] 实现风格损失，默认风格层为 `conv1_1`、`conv2_1`、`conv3_1`、`conv4_1`、`conv5_1`。
- [x] 实现内容损失：

$$
\mathcal{L}_{content}
$$

- [x] 实现风格损失：

$$
\mathcal{L}_{style}
$$

- [x] 实现总变分损失：

$$
\mathcal{L}_{tv}
$$

- [x] 实现总损失：

$$
\mathcal{L}_{total}=\alpha\mathcal{L}_{content}+\beta\mathcal{L}_{style}+\gamma\mathcal{L}_{tv}
$$

- [x] 支持 Adam 或 L-BFGS 优化生成图像。
- [x] 保存生成图、loss 曲线、配置文件和指标文件。
- [x] 所有正式实验输出必须包含 `config.json`、`metrics.csv`、`result.png`、`loss_curve.png`。

## 3. 一次性 sanity check

- [x] 只执行一次最小 sanity check。
- [x] 检查依赖能否导入。
- [x] 检查 VGG19 能否加载。
- [x] 检查一张内容图和一张风格图能否读取。
- [x] 检查 GPU 能否使用。
- [x] 检查一次前向、反向传播和输出保存是否正常。
- [x] sanity check 成功后，删除该检查产生的临时输出、临时日志和中间产物。
- [x] sanity check 成功后，直接进入正式实验配置。

## 4. 正式实验配置

- [x] 删除旧 `baseline` 结果和旧 500-step 中间产物。
- [x] 废弃旧 `configs/baseline.json`，改用 `configs/strong_baseline.json`。
- [x] 删除上一轮 `content_noise + max pooling + tv_weight=1.0` 的 strong baseline 结果、日志和中间图。
- [x] 删除 `configs/ablation_plan.json`，不再按消融网格推进。
- [x] 修改代码支持论文式 `avg` pooling。
- [x] 修改 `noise` 初始化为像素空间白噪声再归一化。
- [x] 指标文件同时报告 raw loss 和 weighted loss。
- [x] 配置新的 Gatys-aligned strong baseline，而不是继续使用旧 baseline。
- [x] strong baseline 参数：
  - [x] 内容图：`golden_gate_bridge.jpg`
  - [x] 风格图：`composition_vii_kandinsky.jpg`
  - [x] 输入分辨率：$1024\times1024$
  - [x] 初始化：`noise`
  - [x] 池化方式：`avg`
  - [x] 优化器：`L-BFGS`
  - [x] 迭代步数：`3000`
  - [x] 内容层：`conv4_2`
  - [x] 风格层：`conv1_1`、`conv2_1`、`conv3_1`、`conv4_1`、`conv5_1`
  - [x] 内容权重 $\alpha = 3.0$
  - [x] 风格权重 $\beta = 300000.0$
  - [x] 平滑权重 $\gamma = 0.1$
  - [x] 每 `300` step 保存一次中间图。
- [x] 运行更新后的 strong baseline：

```bash
source /workspace/yyf/env/miniconda3/etc/profile.d/conda.sh
conda activate /workspace/yyf/env/conda/style-transfer
cd /data1/yyf/Style_Transfer_Loss_Design/code
CUDA_VISIBLE_DEVICES=1 python run_style_transfer.py --config configs/strong_baseline.json
```

- [x] 保存更新后 strong baseline 的结果图、loss 曲线、配置和指标。
- [x] 同步核心结果到本地 `Experiments/results/strong_baseline/`。
- [x] 记录远程运行命令、conda 环境、日志路径、输出路径和最终指标。
- [x] 视觉检查重点：
  - [x] Kandinsky 抽象色块和线条风格是否明显增强。
  - [x] Golden Gate Bridge 主结构是否仍可辨认。
  - [x] weighted style term 是否成为主要优化压力。
  - [x] TV 变弱后是否重新出现不可接受的高频噪声。

## 5. 取消消融，改为 baseline 诊断

- [x] 旧 500-step 候选实验结果已删除，不再作为当前实验依据。
- [x] 原 A-D 消融实验计划已取消。
- [x] 不再用多组参数扫描解释问题，直接用 Gatys 2015 对比修正 baseline。
- [x] 更新后 baseline 完成后，分析 raw loss 与 weighted loss 的差异。
- [x] 将结果与 Gatys 2015 的 `conv4_2`、五层风格层、$\alpha/\beta$ 设置进行文字对比。
- [ ] 若仍不理想，只进行一次目标明确的 baseline 修订，不展开消融网格。

## 6. 可选创新实验

- [ ] 创新实验目标：从经典 NST baseline 过渡到可控风格迁移。
- [ ] 所有创新实验都以当前 `strong_baseline` 为参照，不再做大规模参数消融。

### 实验 6.1：颜色保持实验

- [x] 实验名：`color_preservation`
- [x] 方法来源：Gatys et al., Preserving Color in Neural Artistic Style Transfer。
- [x] 只加入颜色保持机制，不加入结构保持机制。
- [x] 推荐优先实现论文式方法：luminance-only transfer 或 style-to-content color transfer。
- [x] 修正 luminance-only 实现：单通道 $Y$ 优化、亮度统计匹配、关闭每步硬投影。
- [x] 观察颜色偏移是否减少。
- [x] 观察 Kandinsky 风格强度是否变弱。
- [x] 输出 `result.png`、`loss_curve.png`、`metrics.csv`、`config.json`。

### 实验 6.2：语义区域约束实验

- [x] 旧版 `semantic_mask_style_transfer` 已完成一次粗 mask 实验，结果归档在 `Experiments/results/semantic_mask_style_transfer/`，但 mask 过粗，只作为失败分析或对照，不作为最终 6.2 结论。
- [ ] 重做正式版实验名：`semantic_mask_style_transfer`
- [ ] 方法来源：优先对标 Gatys et al., Controlling Perceptual Factors in Neural Style Transfer 的 spatial guidance / guided Gram matrices；参考 Luan et al., Deep Photo Style Transfer 的 semantic segmentation guidance。
- [ ] 只加入 semantic mask / spatially guided style loss，不加入颜色保持机制。
- [ ] 使用分割模型生成内容图语义 mask，不再使用固定多边形作为正式结果。
- [ ] 分割模型优先选择可离线运行、可追溯的现成模型，例如 SAM / Grounded-SAM / Mask2Former / SegFormer；若远程无权重，再记录原因并改用精细人工标注。
- [ ] 内容图 mask 至少包含：`sky`、`water`、`bridge`、`building`、`mountain_land`；桥体和左下角建筑必须分开，避免细节一起被同一区域 Gram 统计覆盖。
- [ ] 风格图 mask 使用对应的风格区域 mask：可用分割模型、SAM 点/框提示、或精细人工标注，将浅色背景、蓝绿色块、暖色块、黑色线条等区域分开。
- [ ] 保存每个单独 mask 和总览图，人工检查边界是否贴合桥索、桥塔、桥面、建筑、水面、天空和山体。
- [ ] 在每个 mask 内分别计算 Gram style loss；对过小区域设置最小面积阈值，避免不稳定 Gram。
- [ ] metrics.csv 增加 `masked_style_loss`、每个区域 style loss、mask 面积占比和最终 total/content/style/tv loss。
- [ ] 观察水面和天空中异常斜向风格块是否减少。
- [ ] 观察桥体、桥索、桥塔和左下角建筑细节是否比粗 mask 版本更清楚。
- [ ] 观察整体 Kandinsky 风格强度是否被过度削弱。
- [ ] 远程正式运行 3000-step 配置，日志保存到 `/data1/yyf/Style_Transfer_Loss_Design/logs/semantic_mask_style_transfer.log`。
- [ ] 输出 `result.png`、`loss_curve.png`、`metrics.csv`、`config.json`、`masks.png`、`baseline_vs_semantic_mask.png`。

### 实验 6.3：颜色保持 + 语义区域约束联合实验

- [ ] 实验名：`color_semantic_mask_joint`
- [ ] 同时加入颜色保持机制和 semantic mask / spatially guided style loss。
- [ ] 目标是同时控制颜色偏移和风格纹理跨语义区域泄漏。
- [ ] 观察颜色保持与语义区域约束是否互补。
- [ ] 观察是否出现风格过弱、区域边界过硬或 mask 痕迹明显的问题。
- [ ] 输出 `result.png`、`loss_curve.png`、`metrics.csv`、`config.json`、`masks.png`。

### 创新实验汇总

- [ ] 明确三个创新实验相对 baseline 的改动点。
- [ ] 生成 `baseline / color_preservation / semantic_mask_style_transfer / color_semantic_mask_joint` 四宫格对比图。
- [ ] 汇总四组实验的最终 raw loss、weighted loss、颜色指标、结构指标和主观评价。
- [ ] 整理成功案例和失败案例，并说明可能原因。

## 7. 评价与结果整理

- [ ] 计算内容保持指标。
- [ ] 计算风格匹配指标。
- [ ] 计算总变分平滑指标。
- [ ] 可选计算 LPIPS。
- [ ] 整理所有实验的 `metrics.csv`。
- [ ] 生成统一对比图：
  - [ ] 原内容图
  - [ ] 风格图
  - [ ] baseline 结果
  - [ ] Gatys 2015 对比说明图或中间过程图
  - [ ] 三组创新实验结果
- [ ] 整理失败案例，并说明可能原因。
- [ ] 从远程同步核心结果到本地 `Experiments/`。
- [ ] 不同步大型中间产物、完整日志和模型权重。

## 8. 报告撰写

- [ ] 在 `Report/` 下创建报告草稿。
- [ ] 写明题目、组员姓名、学号和组内贡献。
- [ ] 写背景：为什么研究风格迁移中的多损失设计。
- [ ] 写算法原理：
  - [ ] CNN 特征表示
  - [ ] 内容损失
  - [ ] Gram 矩阵
  - [ ] 风格损失
  - [ ] 总变分损失
  - [ ] 感知损失
- [ ] 所有公式必须使用 LaTeX。
- [ ] 所有理论和公式必须引用文献。
- [ ] 写实验设置：数据、网络、层选择、超参数、运行环境。
- [ ] 写实验结果：图像对比、loss 曲线、指标表格。
- [ ] 写 baseline 诊断：解释 raw loss 与 weighted loss 如何影响结果。
- [ ] 写与 Gatys 2015 设置的差异分析。
- [ ] 写总结与反思：有效设计、敏感参数、失败案例和改进方向。
- [ ] 检查报告是否覆盖课程要求中的背景、算法原理、实验与结果分析、总结与反思、组内贡献。

## 9. PPT 与最终提交

- [ ] 制作汇报 PPT。
- [ ] PPT 结构建议：
  - [ ] 选题动机
  - [ ] 方法原理
  - [ ] 损失函数设计
  - [ ] 实验设置
  - [ ] 结果对比
  - [ ] baseline 诊断
  - [ ] 与论文设置对齐的改进点
  - [ ] 总结反思
- [ ] 准备演示用核心图片，优先使用清晰并排对比图。
- [ ] 检查 ZIP 提交内容：
  - [ ] 源代码
  - [ ] 汇报 PPT
  - [ ] 实验报告
  - [ ] 必要核心结果
- [ ] 按课程要求命名压缩包：`组号_大作业题目.zip`。
- [ ] 提交前最后检查路径、图片引用、公式渲染和参考文献。

## 10. 当前优先级

1. 先确认远程工作区、conda、GPU、磁盘和网络。
2. 完成 VGG19 风格迁移基础代码。
3. 做一次最小 sanity check，并删除其产物。
4. 直接跑更新后的 Gatys-aligned baseline 正式配置。
5. 同步核心结果到本地并开始报告。
