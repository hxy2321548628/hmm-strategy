# HMM Strategy：HMM 择时研究与组合回测

将期货特征、隐马尔可夫模型（HMM）、指数择时和成分股回测组织成可安装的 Python 包。配置使用 Pydantic v2，业务选项使用枚举；同一应用服务支持命令行、Notebook 和自动化实验。

## 快速开始

需要 Python 3.11+ 和 uv。先按 [数据规范](docs/data-and-backtesting.md) 准备本地行情并修改 `configs/default.toml` 中的路径；公开仓库不包含原始行情数据。

在项目根目录执行：

```bash
uv sync --locked --extra portfolio --extra plots
uv run benke validate --config configs/default.toml
uv run benke run --config configs/default.toml
```

`default.toml` 提供沪深 300 / IF 本地数据路径示例，运行结果写入独立的 `outputs/run-时间-随机标识/` 目录。路径相对 **配置文件所在目录** 解析；不要求固定工作目录。

```bash
# 多因子、PCA、BIC 状态数选择
uv run benke run --config configs/multi_factor.toml
# 股票组合模拟：等权 / 指数权重由 backtest.weighting 控制
uv run --extra portfolio benke portfolio --config configs/default.toml
# 参数网格，任何一组失败都会记录错误并返回非零退出码
uv run benke search --config configs/default.toml
# 配置字段、类型、默认值和范围的机器可读文档
uv run benke schema
# 测试与静态检查
uv run --extra portfolio pytest -q
uv run ruff check src tests main.py
uv run ruff format --check src tests main.py
```

也可通过 `uv run python -m benke ...` 或 `uv run python main.py ...` 调用同一入口。`validate` 校验配置、行情及首个训练窗口特征；不会训练 HMM，也不验证组合文件。

## 目录

```text
src/benke/
  config/         Pydantic 模型、枚举、配置加载
  data/           文件仓储、字段规范、日期对齐
  features/       因子公式及训练期拟合的变换管线
  models/         HMM 拟合、BIC 选择、逐日状态过滤
  strategies/     状态收益与持仓信号
  backtesting/    指数近似回测、Backtrader 组合、指标
  services/       研究流程与参数搜索
  reporting/      CSV、JSON、可选图表
  cli.py          命令行装配入口
configs/          单因子和多因子示例
tests/           配置、因果性、窗口、成本、组合等回归测试
docs/            架构、配置、数据规范、迁移与验证记录
data/            本地原始数据，不纳入版本控制
outputs/         新实验结果（不纳入版本控制）
```

## 配置与结果

环境变量覆盖 TOML，TOML 覆盖配置类默认值。嵌套字段用双下划线，所有环境变量以 `BENKE_` 开头：

```bash
BENKE_HMM__N_STATES=3 BENKE_REPORT__PLOT=true \
  uv run --extra plots benke run --config configs/default.toml
```

每次运行保存 `config.json`（最终配置）、`daily.csv`（信号、实际持仓、收益、成本）、`windows.csv`（每次训练的区间与模型信息）、`metrics.json`。启用绘图后另存 `equity.png`。组合命令增加 `portfolio_daily.csv`、`orders.csv`、`portfolio_metrics.json`。

指数择时是“t 日收盘信号影响 t+1 日收盘到收盘收益”的研究近似；组合回测在下一交易日开盘成交，两者不应被当作相同收益口径。修复了旧流程中的未来信息使用和窗口遗漏，因此新结果不保证复现旧论文中的数值。参数网格结果属于研究比较，不能据同一测试集的最高收益宣称样本外效果。

详细说明：

- [架构与扩展](docs/architecture.md)
- [配置参考](docs/configuration.md)
- [数据规范与回测约定](docs/data-and-backtesting.md)
- [旧代码迁移](docs/migration.md)
- [验证记录](docs/validation.md)
