# GitHub 同步准备记录

记录日期：2026-06-19

## 当前结论

GitHub remote：`git@github.com:yufeila/Style_Transfer_Loss_Design.git`。

已完成远程资产盘点、轻量文件准备和大文件清单。提交内容包含代码、配置、Info/TODO、核心结果图、SAM mask、metrics、恢复说明和 6 张正式原始数据图片；SAM 权重等大文件不进普通 Git。

## 已准备入库

- 代码：`Experiments/code/`，与远程 `code/` 一致，排除 `__pycache__`。
- 配置：`Experiments/code/configs/*.json`。
- 数据：`data/raw/content/*.jpg`、`data/raw/style/*.jpg`。
- TODO 与 Info：`TODO.md`、`Info/*.md`。
- 核心结果：`strong_baseline`、`color_preservation` 的 `result.png`、`loss_curve.png`、`metrics.csv`、`config.json`。
- SAM mask：`Experiments/results/semantic_mask_style_transfer/masks.png`、`sam_mask_config.json` 和单类 mask 图。
- 清单：`Experiments/数据清单.md`、`Experiments/远程资产清单.md`。

## 日志摘要

完整日志位于远程 `/data1/yyf/Style_Transfer_Loss_Design/logs/`，不上传普通 Git。

| 日志 | 摘要 |
|---|---|
| `strong_baseline.log` | 3000/3000 完成，用时约 5:13，最终 `total_loss=0.422` |
| `color_preservation_fixed.log` | 3000/3000 完成，用时约 5:13，最终 `total_loss=0.140` |
| `color_preservation.log` | 旧颜色保持运行，最终 `total_loss=5.19`，以 fixed 结果为准 |
| `semantic_mask_style_transfer.log` | 旧粗 mask 3000-step 日志仍在；当前 SAM 版只生成 mask，未重新跑 3000-step |

## Git / LFS 状态

- 本地 `git lfs` 不可用：`git: 'lfs' is not a git command`。
- 已新增 `.gitattributes`，为 `.pth`、模型目录和归档文件预留 LFS 规则。
- 6 张正式数据图片总量约 30M，已按用户要求进入普通 Git。
- 在安装 Git LFS 前，不应把 SAM 权重或大型结果直接加入普通 Git。

## 下一台服务器恢复步骤

```bash
git clone <GITHUB_REPO_URL> Style_Transfer_Loss_Design
cd Style_Transfer_Loss_Design/Experiments/code
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

恢复数据和远程大文件：

```bash
mkdir -p /data1/yyf/Style_Transfer_Loss_Design/{data,models,outputs,logs}
rsync -az data/ /data1/yyf/Style_Transfer_Loss_Design/data/
rsync -az gpu-server:/data1/yyf/Style_Transfer_Loss_Design/models/ /data1/yyf/Style_Transfer_Loss_Design/models/
sha256sum /data1/yyf/Style_Transfer_Loss_Design/models/sam/sam_vit_b_01ec64.pth
```

SAM 权重校验值应为：

```text
ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912
```

继续 6.2：

```bash
cd /data1/yyf/Style_Transfer_Loss_Design/code
CUDA_VISIBLE_DEVICES=1 python create_sam_semantic_masks.py --help
```

确认 mask 后，再运行 3000-step `semantic_mask_style_transfer`。
