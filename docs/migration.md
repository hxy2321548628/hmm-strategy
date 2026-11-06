# 旧代码迁移

## 文件处理

旧脚本、历史 Notebook、教学示例、实验图表和迁移备份已清理；项目只维护新包、配置、测试和文档。原始行情保留在本地 `data/`，不纳入 Git 或公开仓库。生成结果、构建产物和缓存也不纳入版本控制。

以下 API 对照和行为变更用于解释重构后的接口与研究口径，不依赖旧源码归档。

## API 对照

| 旧入口 / 逻辑 | 新位置 |
|---|---|
| scripV1～scripV6，通配符导入 | src/benke 中明确职责模块 |
| run / pca_run | ResearchService.run / run_frames，PCA 由配置启用 |
| feaFunc、eval 特征名 | FeatureKind + FeatureConfig + BUILDERS 注册表 |
| ma5、lr5、lr15、std5 等 | 通用特征 kind + window；不再为每个窗口复制函数 |
| ma5_boxcox、lr5_boxcox 等 | transform = box_cox / yeo_johnson |
| lr5_boxcox_settle | kind = log_return，price = settlement，window = 5 |
| get_short_signal_base_r | signal.threshold_mode = fixed |
| get_short_signal(r='benchmark') | signal.threshold_mode = benchmark |
| get_best_n_components | hmm.selection = bic，候选集可配置 |
| get_cumnet | backtesting/timing.evaluate_signals |
| get_index_weight_trade_info / get_index_comp / stock_price | FileMarketRepository |
| My_average_strategy / My_index_weight_strategy | portfolio 命令 + backtest.weighting |
| grid_search_r、chose_train_test、roll_info | search 命令 |
| feature_test / multi_test_run | 多份配置表达不同因子或组合，统一 run / search |
| cerebro_plot | reporting.export 与独立 metrics / daily 输出 |

## 有意改变的结果口径

1. 期货和指数按日期对齐，不再按末尾行数强行拼接。
2. 每个测试窗口覆盖完整 `test_size`，最后不足窗口的尾部保留；不再遗漏边界日。
3. Box-Cox 参数、标准化与 PCA 仅在训练集拟合；不在测试集重新估计尺度。
4. PCA 使用达到阈值的最少维数，不截成零维；保留原始载荷精度。
5. 测试期改为因果前向过滤，不使用整段测试序列的未来观测解码历史状态。
6. 自动状态数选择改为训练集 BIC。旧自定义“绝对年化收益 / 非年化波动率”加权指标没有保留为默认选择标准；固定状态数仍可配置。
7. 默认收益因子使用 Yeo-Johnson，可处理正负值；Box-Cox 可显式配置，但遇到训练域外值会报错。
8. high_low 默认是原始价差。旧版本中 `hml` 有时隐式做 Box-Cox，现在由 transform 明确指定。
9. 成交成本、初始回撤、实际换仓次数有明确口径；未观测状态不盲目开仓。
10. 组合持仓权重可每日更新，移除吞异常和零价补齐；成交失败保留订单状态，不打印成成功成交。

这些修改用于消除数据泄漏和执行错误，不意味着策略收益提高。新结果不应与旧图表做数值等价断言；需要论文复现实验时，先固定数据区间和配置，并说明方法变化。

## 迁移 Notebook

```python
from benke.config.settings import AppConfig, load_settings
from benke.services.research import ResearchService

base = load_settings('../../configs/default.toml')  # 按 Notebook 实际位置调整
payload = base.model_dump(mode='python')
payload['features'] = [
    {'name': 'ma', 'kind': 'moving_average', 'window': 5},
    {'name': 'lr', 'kind': 'log_return', 'window': 15},
]
payload['preprocessing']['pca_variance'] = 0.8
config = AppConfig.model_validate(payload)
result = ResearchService(config).run()
result.daily[['benchmark_equity', 'strategy_equity']].plot()
```

不再通过 `sys.path.append` 注入脚本目录。Notebook 内核应使用安装了当前项目的环境。
