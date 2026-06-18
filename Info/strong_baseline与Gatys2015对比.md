# strong baseline 与 Gatys 2015 对比

记录日期：2026-06-18

对比论文：`Reference/A Neural Algorithm of Artistic Style.pdf`

更新状态：已根据本文结论修改 baseline。新配置使用 `noise` 初始化、`avg` pooling、$\alpha=3.0$、$\beta=300000.0$、$\gamma=0.1$，并报告 weighted loss。

## 结论

`strong_baseline` 和 Gatys et al. 2015 的 Kandinsky 实验在“表面配置”上很接近，但结果风格不够强，主要不是因为 VGG 模型太小，也不主要是 step 太少，而是因为初始化、池化方式、loss 归一化、TV 正则和内容图本身共同把结果推向了更保守的内容保持解。

## 与论文相同的部分

Gatys et al. 在 Figure 2 / Figure 3 中使用 CNN 内容表示和 Gram 风格表示合成图像。论文 Methods 中给出的总目标为：

$$
\mathcal{L}_{total}(p,a,x)=\alpha\mathcal{L}_{content}(p,x)+\beta\mathcal{L}_{style}(a,x)
$$

其中内容损失来自高层特征，风格损失来自多层 Gram 矩阵。

我们的 `strong_baseline` 与论文 Kandinsky 示例相同或接近的地方：

- 内容层：论文 Figure 2 使用 `conv4_2`；我们也使用 `conv4_2`。
- 风格层：论文使用 `conv1_1`、`conv2_1`、`conv3_1`、`conv4_1`、`conv5_1`；我们也使用这五层。
- 内容/风格比例：论文 Kandinsky 结果使用 $\alpha/\beta=10^{-4}$；我们设置 $\alpha=10.0,\ \beta=100000.0$，比例也是 $10^{-4}$。
- 优化对象：论文直接优化生成图像像素；我们也是固定 VGG，只优化生成图像。

## 关键差异

### 1. 初始化不同

论文方法从白噪声图像开始优化。上一轮 `strong_baseline` 使用：

```text
init = content_noise
init_noise_std = 0.05
```

这会让优化起点非常接近内容图。好处是桥结构更稳定，坏处是更容易停在“照片轻微风格化”的解，而不是论文中更明显的艺术化解。

### 2. 池化方式不同

论文 Methods 明确说明：为了让梯度流动更好、结果更好看，作者把 VGG 中的 max pooling 替换成 average pooling。

上一轮实现直接使用 `torchvision.models.vgg19(...).features`，仍然是默认 max pooling。这个差异会影响风格纹理的连续性，容易让结果更碎、更像局部纹理噪声，而不是连续的绘画结构。当前代码已支持并默认在新 baseline 中使用 `avg` pooling。

### 3. loss 归一化不同，$\alpha/\beta$ 不能直接等价

论文**内容损失**使用特征差的平方和形式：

$$
\mathcal{L}_{content}^{l}(p,x)=\frac{1}{2}\sum_{i,j}(F_{ij}^{l}(x)-P_{ij}^{l})^2
$$

论文**风格层损失**为：

$$
E_l=\frac{1}{4N_l^2M_l^2}\sum_{i,j}(G_{ij}^{l}(x)-A_{ij}^{l})^2
$$

我们的实现使用 PyTorch `mse_loss`，并且 Gram 矩阵先除以 `channels * height * width`：

```python
gram = torch.bmm(flattened, flattened.transpose(1, 2))
return gram / (channels * height * width)
```

因此，即使 $\alpha/\beta$ 都是 $10^{-4}$，两个系统里的“实际内容约束强度”和“实际风格约束强度”并不等价。不能只看比例判断风格强弱。

### 4. 我们加入了 TV loss，原论文没有

我们的目标是：

$$
\mathcal{L}_{total}
=\alpha\mathcal{L}_{content}
+\beta\mathcal{L}_{style}
+\gamma\mathcal{L}_{tv}
$$

其中 $\gamma=1.0$。TV loss 会抑制局部剧烈变化，使图像更平滑。它有助于减少漩涡形噪声，但也会压制 Kandinsky 风格中强烈的线条、边界和色块变化。

### 5. 最终有效损失仍偏向内容

`strong_baseline` 最后一步指标为：

```text
total_loss = 12.766976356506348
content_loss = 0.7598769664764404
style_loss = 4.987589636584744e-05
tv_loss = 0.18061770498752594
```

乘以权重后：

```text
content term = 7.5988
style term   = 4.9876
tv term      = 0.1806
```

也就是说，虽然 $\alpha/\beta=10^{-4}$ 看起来风格权重很大，但在当前实现的归一化下，最终内容项仍然是最大的优化压力。因此结果会保留很清楚的桥结构，风格不会彻底压过内容。

### 6. Gram 风格表示本身不擅长保留 Kandinsky 的全局构图

论文中的风格表示是多层特征相关性，也就是 Gram 矩阵。它擅长迁移颜色、纹理、笔触统计，但会丢掉风格图的空间布局。

Kandinsky 的 `Composition VII` 很大一部分视觉冲击来自全局抽象构图、几何方向和大色块关系。Gram loss 不直接约束这些空间结构，所以它更容易生成“局部有抽象纹理”的桥，而不是“整体像 Kandinsky 抽象画”的图。

## 为什么当前结果风格不够强

当前结果可以理解为论文 Figure 3 中更偏右侧的情况：照片内容很清楚，但风格匹配不够充分。原因是：

1. `content_noise` 初始化把结果锚定在内容图附近。
2. 当前 weighted content term 大于 weighted style term。
3. TV loss 抑制了风格纹理和色块边界。
4. max pooling 与论文的 average pooling 不一致。
5. Golden Gate Bridge 本身高频结构很多，缆线、桥架、水面和山体会强烈吸引内容损失。
6. Gram loss 无法强制生成 Kandinsky 的全局抽象构图。

## 已落实到新 baseline 的修改

为了更接近论文级 Kandinsky 风格，当前不做消融网格，而是直接把以下判断落实到新的唯一 baseline：

- 使用 `noise` 初始化替代 `content_noise`。
- 使用 average pooling 替代 max pooling。
- 降低内容权重到 $\alpha=3.0$。
- 提高风格权重到 $\beta=3\times10^5$。
- 降低 TV 权重到 $\gamma=0.1$。
- 单独报告 weighted content/style/TV term，而不是只报告 raw loss。

这比单纯继续增加 step 更有效，因为当前 loss 曲线已经较早进入平台期，说明主要矛盾不是优化时间不足，而是目标函数把结果约束在内容保持较强的区域。
