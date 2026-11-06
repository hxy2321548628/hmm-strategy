"""Optional Backtrader adapter: close-time decisions, next-open execution."""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from benke.backtesting.metrics import performance
from benke.config.enums import WeightingMode
from benke.config.settings import BacktestConfig
from benke.data.repository import dated_frame, validate_market
from benke.exceptions import DataError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PortfolioResult:
    daily: pd.DataFrame
    orders: pd.DataFrame
    metrics: dict[str, float]


def target_weights(
    signals: pd.Series, weights: pd.DataFrame, config: BacktestConfig
) -> dict[pd.Timestamp, dict[str, float]]:
    """Use only the latest snapshot known on the signal day, never a future snapshot."""
    snapshots = {date: group for date, group in weights.groupby(level=0)}
    dates = pd.DatetimeIndex(sorted(snapshots))
    targets = {}
    for date, signal in signals.items():
        if signal == 0:
            targets[date] = {}
            continue
        offset = dates.searchsorted(date, side="right") - 1
        if offset < 0:
            raise DataError(f"No constituent snapshot on or before {date}")
        group = snapshots[dates[offset]]
        group = group.loc[group.weight > 0]
        if group.empty:
            raise DataError(f"Empty portfolio snapshot at {date}")
        raw = (
            np.ones(len(group))
            if config.weighting == WeightingMode.EQUAL
            else group.weight.to_numpy()
        )
        normalized = raw / raw.sum() * (1 - config.cash_reserve)
        targets[date] = dict(zip(group.sec_code.astype(str), normalized, strict=True))
    return targets


def run_portfolio(
    signals: pd.Series, stocks: pd.DataFrame, weights: pd.DataFrame, config: BacktestConfig
) -> PortfolioResult:
    # Importing the core research package does not require Backtrader.
    try:
        import backtrader as bt
    except ImportError as exc:
        raise DataError("Install the portfolio extra: uv sync --extra portfolio") from exc
    if (
        signals.empty
        or not isinstance(signals.index, pd.DatetimeIndex)
        or not signals.index.is_unique
        or not signals.index.is_monotonic_increasing
        or signals.index.hasnans
        or not signals.isin([0, 1]).all()
    ):
        raise DataError("Portfolio signals must have sorted, unique dates and values 0/1")
    schedule = target_weights(signals, weights, config)
    required = sorted({code for target in schedule.values() for code in target})
    if not required:
        daily = pd.DataFrame(
            {"strategy_return": 0.0, "strategy_equity": 1.0, "value": config.initial_cash},
            index=signals.index,
        )
        return PortfolioResult(daily, pd.DataFrame(), performance(daily.strategy_return, config))
    stocks = dated_frame(stocks)
    if "sec_code" not in stocks:
        raise DataError("Stocks require sec_code")
    stocks["sec_code"] = stocks.sec_code.astype(str)
    available = set(stocks.sec_code)
    missing = set(required) - available
    if missing:
        raise DataError(f"Missing stock data: {sorted(missing)}")

    class TargetPortfolio(bt.Strategy):
        def __init__(self):
            self.pending = []
            self.events = []

        def prenext(self):
            self.next()

        def next(self):
            date = pd.Timestamp(self.datetime.date(0))
            targets = schedule.get(date)
            if targets is None:
                return
            for order in self.pending:
                if order.alive():
                    self.cancel(order)
            self.pending.clear()
            equity = self.broker.getvalue()
            changes = []
            for data in self.datas:
                if (
                    not len(data)
                    or pd.Timestamp(data.datetime.date(0)) != date
                    or not np.isfinite(data.close[0])
                ):
                    continue
                target = targets.get(data._name, 0.0)
                size = int(equity * target / data.close[0] / config.lot_size) * config.lot_size
                difference = size - self.getposition(data).size
                if difference:
                    changes.append((difference, data, size))
            # Sell orders enter the broker before buys; keep a configurable cash buffer.
            for _, data, size in sorted(changes, key=lambda item: item[0]):
                order = self.order_target_size(data=data, target=size)
                if order is not None:
                    self.pending.append(order)

        def notify_order(self, order):
            if order.status in (order.Submitted, order.Accepted):
                return
            self.events.append(
                {
                    "date": pd.Timestamp(self.datetime.date(0)),
                    "security": order.data._name,
                    "status": order.getstatusname(),
                    "size": order.executed.size,
                    "price": order.executed.price,
                    "commission": order.executed.comm,
                }
            )
            if order.status in (order.Margin, order.Rejected):
                logger.warning(
                    "Order %s for %s: %s", order.ref, order.data._name, order.getstatusname()
                )

    engine = bt.Cerebro(stdstats=False)
    engine.broker.setcash(config.initial_cash)
    engine.broker.setcommission(commission=config.commission)
    engine.broker.set_slippage_perc(config.slippage)
    stocks = stocks.loc[stocks.sec_code.isin(required)]
    groups = stocks.groupby("sec_code")
    calendar = signals.index
    for code in required:
        bars = validate_market(groups.get_group(code), ("open", "high", "low", "close"))
        if "volume" not in bars or not np.isfinite(bars.volume).all() or (bars.volume < 0).any():
            raise DataError(f"{code}: volume must be finite and non-negative")
        bars = bars.reindex(calendar)
        real_bar = bars.close.notna() & (bars.volume > 0)
        # Past closes are valuation marks only; missing sessions cannot execute an order.
        last_close = bars.close.ffill()
        for column in ("open", "high", "low", "close"):
            bars[column] = bars[column].fillna(last_close)
        bars["volume"] = bars.volume.where(real_bar, 0).fillna(0)
        bars["openinterest"] = 0
        # Keep leading NaNs to share one clock; never backfill pre-listing prices.
        engine.adddata(bt.feeds.PandasData(dataname=bars), name=code)

    def volume_filler(order, price, ago):
        return abs(order.executed.remsize) if order.data.volume[ago] > 0 else 0

    engine.broker.set_filler(volume_filler)
    engine.addstrategy(TargetPortfolio)
    engine.addanalyzer(bt.analyzers.TimeReturn, _name="returns")
    strategy = engine.run()[0]
    returns = pd.Series(strategy.analyzers.returns.get_analysis(), dtype=float)
    returns.index = pd.DatetimeIndex(returns.index, name="date")
    returns = returns.reindex(calendar)
    if returns.isna().any():
        raise DataError("Backtrader did not return every requested session")
    daily = returns.to_frame("strategy_return")
    daily["strategy_equity"] = (1 + returns).cumprod()
    daily["value"] = config.initial_cash * daily.strategy_equity
    orders = pd.DataFrame(strategy.events)
    metrics = performance(returns, config)
    metrics["completed_order_count"] = (
        int((orders.status == "Completed").sum()) if not orders.empty else 0
    )
    metrics["failed_order_count"] = (
        int(orders.status.isin(["Margin", "Rejected"]).sum()) if not orders.empty else 0
    )
    return PortfolioResult(daily, orders, metrics)
