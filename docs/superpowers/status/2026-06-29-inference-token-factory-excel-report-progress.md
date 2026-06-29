# 推理 token 工厂 Excel 汇报进度保存

更新时间：2026-06-29

## 当前工作区

- 主工作区：`/work/development-code/Resource_Planning_Tool`
- 实施 worktree：`/tmp/Resource_Planning_Tool_worktrees/inference-token-factory-report`
- 分支：`report/inference-token-factory-excel`
- 当前 HEAD：`4fda918 feat: 增加报告 Excel 可视化图表`
- 当前状态：Task 1-6 已完成；Task 7-8 待继续。

## 已完成任务

| 任务 | 状态 | 关键提交 |
| --- | --- | --- |
| Task 1：锁定工作簿结构测试 | 完成 | `b1c3cde`, `eb28aeb` |
| Task 2：汇总上下文与 H200 FP8 数据 | 完成 | `efaa5e7` |
| Task 3：生成汇报工作簿基础结构 | 完成 | `1a135a7`, `80aea96`, `cef3345` |
| Task 4：增加专题原理简图 | 完成 | `129de14`, `d0ef495` |
| Task 5：补充背景、量化结论、资源方案与附录正文 | 完成 | `4ede874`, `9610888`, `0278c27` |
| Task 6：增加 Excel 可视化图表 | 完成 | `4fda918` |

## 已实现内容

- 多 sheet 汇报工作簿结构，包含目录页和 `v2.x` 分层逻辑。
- 背景目标页围绕工服场景 token 工厂展开，写入请求规模、token 规模、长上下文压力和指标口径。
- 专题 sheet 覆盖 PD 分离、缓存命中、KV Cache 量化、模型量化、投机解码、MOE 专家并行等。
- 各专题 sheet 已有简要原理可视化，缺数据项以原理分析、测试方案和待补测状态呈现。
- 模型量化使用 H200 32B/72B BF16 vs FP8 历史数据生成汇总结论。
- 数据缺失或无可比样本时可降级为待补测，不会崩溃，也不会写出不真实收益。
- PD 分离正文包含 H200 做 PD、H200 做 P、H20 做 D、H200+H20 异构方案、GLM5.1/GLM5.2 FP8 资源申请口径、带宽/互联要求和大参数长上下文适用场景。
- 背景页已有上下文分布图；模型量化页已有 H200 FP8 vs BF16 吞吐对比图。

## 当前测试结果

最近一次专项验证：

```bash
python3 -m pytest tests/test_inference_token_factory_report.py -q
```

结果：`13 passed`。

已知环境问题：全量 `python3 -m pytest backend/tests -q` 在实现前就因缺少 `structlog` 失败，属于当前环境依赖问题，不是本次报告生成改动引入。

## 数据依赖

worktree 中已补齐以下被 git 忽略的数据，继续时不要误删：

- `outputs/context_analysis_20260609_034248/01_overview.json`
- `outputs/context_analysis_20260609_034248/02_input_buckets.csv`

已使用的 H200 数据：

- `data/H200/32B/1.csv`
- `data/H200/32B-FP8/1.csv`
- `data/H200/72B/2.csv`
- `data/H200/72B-FP8/2.csv`

## 剩余任务

### Task 7：增加 CLI 并生成 xlsx 初版

建议继续要求：

- 为 `scripts/build_inference_token_factory_report.py` 增加 CLI 入口。
- 默认输出：`inference-report/推理token工厂汇报大纲_汇报版_v2.0.xlsx`。
- CLI 运行后创建父目录并保存 workbook。
- 增加测试覆盖输出文件存在、sheet 顺序、关键 chart 保存后仍存在。
- 生成 xlsx 文件后提交脚本和 xlsx 初版。

建议验证：

```bash
python3 -m pytest tests/test_inference_token_factory_report.py -q
python3 scripts/build_inference_token_factory_report.py
```

### Task 8：最终验证与 xlsx 检查

必须使用 `xlsx` skill 做最终工作簿检查，重点确认：

- workbook 可以被 openpyxl 正常打开。
- sheet 顺序与目录页一致。
- 图表保存后仍存在。
- 已有数据结论、待补测状态、PD/MOE 资源方案没有丢失。
- 输出文件路径清晰，必要时说明 worktree 路径和是否需要合入主工作区。

建议同时运行：

```bash
python3 -m pytest tests/test_inference_token_factory_report.py -q
python3 scripts/build_inference_token_factory_report.py
python3 - <<'PY'
from pathlib import Path
from openpyxl import load_workbook
p = Path("inference-report/推理token工厂汇报大纲_汇报版_v2.0.xlsx")
wb = load_workbook(p)
print(p)
print(wb.sheetnames)
print(len(wb["01_背景与目标"]._charts), len(wb["04_模型量化"]._charts))
PY
```

## 注意事项

- 用户明确要求在 worktree 中进行，后续继续使用 `/tmp/Resource_Planning_Tool_worktrees/inference-token-factory-report`。
- 用户要求最终生成 `.xlsx`，并考虑使用 `xlsx` skill。
- 不要把 Continuous Batching 作为独立测试项；它只应作为框架已原生支持、已具备项说明。
- MOE 专家并行需要后续验证测试，资源申请按 GLM5.1 或 GLM5.2 FP8 部署口径，不要写成“不作为验证项”。
- PD 分离验证要考虑同构与异构组合：H200 做 PD、H200 做 P、H20 做 D。
- 当前完成的是 Excel 汇报大纲初版生成器，还没有最终 xlsx 文件生成提交。
