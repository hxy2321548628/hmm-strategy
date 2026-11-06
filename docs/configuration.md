# 配置参考

完整字段和约束由 `uv run benke schema` 输出 JSON Schema。配置模型定义在 `src/benke/config/settings.py`，枚举定义在 `config/enums.py`。拼写错误的字段直接报错；不接受 NaN 或无穷大的配置数值。

| 配置组 | 主要字段 | 规则 |
|---|---|---|
| data | derivatives, benchmark | 必填，CSV 或本地可信 Pickle |
| data | weights, constituents, stocks | 组合回测所需文件；等权可用 constituents，否则使用 weights |
| data | start, end | 可选日期，包含端点，开始不能晚于结束 |
| data | allow_pickle | 默认 false；仅项目内已信任旧数据显式开启 |
| features | name, kind, window, slow_window, price, transform | 非空，name 唯一；窗口正整数，慢窗口大于快窗口 |
| preprocessing | standardize | 默认 true |
| preprocessing | pca_variance | 不填写表示关闭；取值 (0,1]，1 保留全部维度 |
| preprocessing | boxcox_margin, numerical_epsilon | Box-Cox 训练平移余量和常量检测精度 |
| hmm | selection | fixed 或 bic；bic 仅根据训练数据选择 |
| hmm | n_states, candidates | 固定数量或候选集；训练样本必须足够 |
| hmm | covariance | diag / full / tied / spherical |
| hmm | max_iterations, tolerance, min_covariance, random_seed | 迭代上限、收敛阈值、初始化协方差下限、种子 |
| window | train_size, test_size | 默认 520、30；训练窗口须覆盖特征预热及状态数 |
| signal | threshold_mode | fixed / benchmark |
| signal | cumulative_threshold | 默认 1.08，固定累计净值阈值，非百分比收益 |
| backtest | initial_cash, commission, slippage | 默认 10000000、0.00015、0.0001 |
| backtest | trading_days, annual_risk_free_rate | 默认 252、0，用于指标计算 |
| backtest | weighting, cash_reserve, lot_size | index / equal，默认现金缓冲 0.05、手数 100 |
| report | directory, plot, figure_size, dpi | 输出路径、是否绘图、尺寸、分辨率 |
| search | thresholds, train_sizes, test_sizes, pca_variances, max_workers | 取笛卡尔积；默认串行，避免无意过量占用 CPU |
| log_level | DEBUG / INFO / WARNING / ERROR | 默认 INFO |

## 特征枚举

| kind | 计算含义 | 使用参数 |
|---|---|---|
| moving_average | 移动平均 | window, price |
| moving_average_spread | 快均线减慢均线（旧代码的 macd_dif，并非标准 EMA MACD） | window, slow_window, price |
| log_return | 对数价格的 window 日差分 | window, price |
| volatility | 一日对数收益的滚动样本标准差 | window（至少 2）, price |
| high_low | 最高价减最低价 | high, low |
| candle_body | abs(close-open) × close/open，保留旧 cmo 的实际公式 | open, close |

`price` 为 close 或 settlement。`transform` 为 none、box_cox、yeo_johnson。所有变换只拟合训练集。Box-Cox 测试值若超出训练期确定的正数域会明确失败；不使用未来最小值重新平移。可能出现负值的收益特征建议配置 `yeo_johnson`。

PCA 选择达到累计解释方差阈值的最少维数，至少 1 维；测试集复用训练得到的中心、尺度和载荷。

## 环境变量与路径

优先级：环境变量 > TOML > 配置模型默认值。环境变量前缀为 `BENKE_`，嵌套分隔符为 `__`：

```bash
BENKE_WINDOW__TRAIN_SIZE=260 \
BENKE_HMM__SELECTION=bic \
BENKE_HMM__CANDIDATES='[3,4,5]' \
BENKE_REPORT__DIRECTORY=../outputs/debug \
uv run benke run --config configs/default.toml
```

列表字段使用 JSON；不隐式加载 `.env`。数据和报告的相对路径均相对于 TOML 所在目录，含环境变量提供的路径。`~` 会展开。默认参数只在配置层声明，具体数据集路径只在示例 TOML 中出现。

`Settings` 负责入口环境读取；解析并规范路径后返回不再读取环境的不可变 `AppConfig`。搜索任务修改配置字典后必须用 `AppConfig.model_validate` 再校验，不能用不验证的 `model_copy` 修改业务参数。

参考：[Pydantic Settings 官方说明](https://github.com/pydantic/pydantic-settings/blob/main/docs/index.md)。
