# rebuild

这个目录用于把当前牙科蒸馏闭环整理成一个未来可以单独搬走、单独复现的实验工程骨架。目标不是保存“最终论文结论”，而是保存“如何从头重跑每个实验”的最小可执行结构。

## 目录设计原则

- 每个 `教师模型 + Qwen2.5-7B` 组合单独成目录
- 每个目录都保留自己的 `data/`、`configs/`、`scripts/`、`README.md`
- 共享训练脚本统一放在 `shared/`
- 每个目录的配置尽量固定为该教师历史上的最佳或最关键方案
- README 以“迁移后如何运行”为优先，而不是以“开发历史”叙述为优先

## 目录结构

- `shared/`: 通用训练、评估、标签生成、数据构造脚本
- `00_baseline_gt_sft/`: Qwen2.5-7B 纯 GT SFT 基线
- `01_qwen32b_whitebox/`: Qwen2.5-32B → Qwen2.5-7B 白盒蒸馏
- `02_deepseek_v3_choice_head/`: DeepSeek-V3 → Qwen2.5-7B 两阶段蒸馏
- `03_doubao_choice_head/`: Doubao → Qwen2.5-7B 两阶段蒸馏
- `04_kimi_choice_head/`: Kimi → Qwen2.5-7B 两阶段蒸馏
- `05_qwen14b_choice_head/`: Qwen2.5-14B → Qwen2.5-7B 两阶段蒸馏

## 迁移到独立项目后的建议布局

- 把当前整个 `rebuild/` 目录复制出去
- 在新项目中保留如下结构：
  - `models/Qwen2.5-7B-Instruct`
  - `models/Qwen2.5-14B-Instruct`
  - `models/Qwen2.5-32B-Instruct`
- 如果不使用上述默认位置，可以通过环境变量覆盖模型路径

## 独立项目初始化建议

1. 复制 `setup.example.env` 为 `setup.env`
2. 按你的机器修改 Python 路径、模型路径、API key
3. 执行：

```bash
source setup.env
bash check_env.sh
```

4. 自检通过后，再进入各实验目录运行 `scripts/*.sh`

## 统一输入输出约定

- 训练集入口统一为 `data/train.jsonl`
- 验证集入口统一为 `data/val.jsonl`
- 测试集入口统一为 `data/test.jsonl`
- 训练输出通常落在各目录自己的 `outputs/` 或 `runs/`
- 脚本优先读取环境变量，无法读取时再回退到当前仓库中的默认位置

## 环境变量约定

- `EASYEDIT_PY`: 指定 Python 解释器
- `BASE_MODEL_7B`: 指定 Qwen2.5-7B-Instruct 路径
- `BASE_MODEL_14B`: 指定 Qwen2.5-14B-Instruct 路径
- `BASE_MODEL_32B`: 指定 Qwen2.5-32B-Instruct 路径
- `NIGHT_MODE=1`: 启用 API 教师任务的夜间慢速模式

## API 教师任务说明

- 白天手动执行时，默认不主动加大限速
- 通过 `run_night_api_tasks.sh` 调度时，会自动启用 `NIGHT_MODE=1`
- 夜间模式下会启用更长请求间隔、更强退避和周期性冷却，尽量避免全局速率限制

## 推荐复现顺序

1. `00_baseline_gt_sft`
2. `01_qwen32b_whitebox`
3. `02_deepseek_v3_choice_head`
4. `03_doubao_choice_head`
5. `04_kimi_choice_head`
6. `05_qwen14b_choice_head`

## 顶层辅助脚本

- `run_all_readme.sh`: 快速查看各实验目录说明
- `run_night_api_tasks.sh`: 夜间顺序调度 API 教师任务
- `run_module_batch.sh`: 顶层批量执行器，可按动作批量运行 teacher-eval、train、eval、pipeline、start-web、start-best-web
- `shared/schedule_quiet_job.sh`: 通用定时静默后台调度器

## 统一脚本约定

- 每个实验目录都补齐了 `scripts/run_pipeline.sh`，用于串行执行该目录的标准流程
- 每个带教师的目录都补齐了 `scripts/run_teacher_eval.sh`，用于先生成测试集教师标签，再输出教师准确率摘要
- 每个实验目录都补齐了 `scripts/start_web.sh`，用于启动本地 Web 验证页
- Choice-Head 两阶段目录继续保留原有的 `generate_teacher_labels.sh`、`prepare_soft_labels.sh`、`build_head_dataset.sh` 等细粒度脚本

## Web 启动说明

- 默认监听 `127.0.0.1`，端口从 7860 开始按项目递增
- 可通过环境变量覆盖：`HOST=0.0.0.0 PORT=7862 bash scripts/start_web.sh`
- 默认读取该目录最新训练产物；也可以通过 `ADAPTER_DIR` 手动指定某个 LoRA 目录
- Web 页会显示当前基础模型、推理后端、已加载 adapter，并允许在 `outputs/` 或 `runs/` 下切换可发现的 checkpoint
- 每个项目额外提供 `scripts/start_best_web.sh`，用于直接按该项目默认“最佳/最新”路径启动

## 顶层批量执行示例

```bash
bash run_module_batch.sh list
bash run_module_batch.sh teacher-eval 01_qwen32b_whitebox 05_qwen14b_choice_head
bash run_module_batch.sh pipeline 00_baseline_gt_sft 02_deepseek_v3_choice_head
CONTINUE_ON_ERROR=1 bash run_module_batch.sh eval
```

批量执行日志默认写到 `batch_logs/<timestamp>/`，并生成 `summary.txt` 汇总各模块成功/失败情况。
