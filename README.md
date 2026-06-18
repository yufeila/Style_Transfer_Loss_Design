# Style Transfer Loss Design

AI 课程大作业：图像风格迁移中的多损失设计。

## 当前状态

- `strong_baseline`：3000-step 已完成。
- `color_preservation`：修正版 3000-step 已完成。
- `semantic_mask_style_transfer`：SAM 语义 mask 已生成，尚未重新跑 3000-step 风格迁移。

## 目录

- `Experiments/code/`：核心代码、配置和运行入口。
- `Experiments/results/`：核心结果图、mask、metrics 和 config。
- `Experiments/远程资产清单.md`：远程数据、模型、输出和日志处理记录。
- `Info/`：实验设计、运行记录和同步记录。
- `Reference/`：论文阅读笔记和引用说明；PDF 默认不进 Git。

## 远程运行

远程工作区：

```text
/data1/yyf/Style_Transfer_Loss_Design
```

远程 conda：

```text
/workspace/yyf/env/conda/style-transfer
```

SAM 权重：

```text
/data1/yyf/Style_Transfer_Loss_Design/models/sam/sam_vit_b_01ec64.pth
```

更多恢复步骤见 `Info/GitHub同步准备记录.md`。
