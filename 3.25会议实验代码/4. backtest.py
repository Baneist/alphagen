from typing import Optional, TypeVar, Callable, Optional
import os
import pickle
import warnings
import pandas as pd
from qlib.backtest import backtest, executor as exec
from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.report.analysis_position import report_graph
from alphagen.data.expression import *

from alphagen_qlib.stock_data import StockData
from alphagen_generic.features import *
from alphagen_qlib.strategy import TopKSwapNStrategy


_T = TypeVar("_T")


def _create_parents(path: str) -> None:
    dir = os.path.dirname(path)
    if dir != "":
        os.makedirs(dir, exist_ok=True)


def write_all_text(path: str, text: str) -> None:
    _create_parents(path)
    with open(path, "w") as f:
        f.write(text)


def dump_pickle(path: str,
                factory: Callable[[], _T],
                invalidate_cache: bool = False) -> Optional[_T]:
    if invalidate_cache or not os.path.exists(path):
        _create_parents(path)
        obj = factory()
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        return obj


class BacktestResult(dict):
    sharpe: float
    annual_return: float
    max_drawdown: float
    information_ratio: float
    annual_excess_return: float
    excess_max_drawdown: float


class QlibBacktest:
    def __init__(
        self,
        benchmark: str = "SH000300",
        top_k: int = 30,
        n_drop: Optional[int] = None,
        deal: str = "close",
        open_cost: float = 0.0015,
        close_cost: float = 0.0015,
        min_cost: float = 5,
    ):
        self._benchmark = benchmark
        self._top_k = top_k
        self._n_drop = n_drop if n_drop is not None else top_k
        self._deal_price = deal
        self._open_cost = open_cost
        self._close_cost = close_cost
        self._min_cost = min_cost

    def run(
        self,
        prediction: pd.Series,
        output_prefix: Optional[str] = None,
        return_report: bool = True
    ) -> BacktestResult:
        prediction = prediction.sort_index()
        index: pd.MultiIndex = prediction.index.remove_unused_levels()  # type: ignore
        dates = index.levels[0]

        def backtest_impl(last: int = -1):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                strategy=TopKSwapNStrategy(
                    K=self._top_k,
                    n_swap=self._n_drop,
                    signal=prediction,
                    min_hold_days=1,
                    only_tradable=True,
                )
                executor=exec.SimulatorExecutor(
                    time_per_step="day",
                    generate_portfolio_metrics=True
                )
                return backtest(
                    strategy=strategy,
                    executor=executor,
                    start_time=dates[0],
                    end_time=dates[last],
                    account=100_000_000,
                    benchmark=self._benchmark,
                    exchange_kwargs={
                        "limit_threshold": 0.095,
                        "deal_price": self._deal_price,
                        "open_cost": self._open_cost,
                        "close_cost": self._close_cost,
                        "min_cost": self._min_cost,
                    }
                )[0]

        try:
            portfolio_metric = backtest_impl()
        except IndexError:
            print("Cannot backtest till the last day, trying again with one less day")
            portfolio_metric = backtest_impl(-2)

        report, _ = portfolio_metric["1day"]    # type: ignore
        result = self._analyze_report(report)
        graph = report_graph(report, show_notebook=False)[0]
        if output_prefix is not None:
            dump_pickle(output_prefix + "-report.pkl", lambda: report, True)
            dump_pickle(output_prefix + "-graph.pkl", lambda: graph, True)
            write_all_text(output_prefix + "-result.json", result)

        print(report)
        print(result)
        return report if return_report else result

    def _analyze_report(self, report: pd.DataFrame) -> BacktestResult:
        excess = risk_analysis(report["return"] - report["bench"] - report["cost"])["risk"]
        returns = risk_analysis(report["return"] - report["cost"])["risk"]

        def loc(series: pd.Series, field: str) -> float:
            return series.loc[field]    # type: ignore

        return BacktestResult(
            sharpe=loc(returns, "information_ratio"),
            annual_return=loc(returns, "annualized_return"),
            max_drawdown=loc(returns, "max_drawdown"),
            information_ratio=loc(excess, "information_ratio"),
            annual_excess_return=loc(excess, "annualized_return"),
            excess_max_drawdown=loc(excess, "max_drawdown"),
        )


if __name__ == "__main__":
    qlib_backtest = QlibBacktest()

    data = StockData(instrument='csi300',
                     start_time='2015-01-01',
                     end_time='2021-12-31')
    # expr = Mul(EMA(Sub(Delta(Mul(Log(open_),Constant(-30.0)),50),Constant(-0.01)),40),Mul(Div(Abs(EMA(low,50)),close),Constant(0.01)))
    # expr = Abs(Abs(Log(Sum(Mul(Mul(Constant(-5.0),Log(Div(Greater(Sub(Sub(vwap,Constant(-0.01)),Constant(1.0)),vwap),Constant(30.0)))),Constant(-0.5)),30))))
    # expr = Greater(Mul(Constant(-0.5),vwap),Mul(Sub(vwap,low),Constant(30.0)))
    # expr = Add(Div(Sub(Constant(-0.01),Sub(vwap,low)),Constant(-1.0)),vwap)
    report = []

    for expr in [
        Mul(EMA(Sub(Delta(Mul(Log(open_), Constant(-30.0)), 50), Constant(-0.01)), 40),Mul(Div(Abs(EMA(low, 50)), close), Constant(0.01))),
        Abs(Abs(Log(Sum(Mul(Mul(Constant(-5.0), Log(Div(Greater(Sub(Sub(vwap, Constant(-0.01)), Constant(1.0)), vwap),Constant(30.0)))), Constant(-0.5)), 30)))),
        Greater(Mul(Constant(-0.5), vwap), Mul(Sub(vwap, low), Constant(30.0))),
        Add(Div(Sub(Constant(-0.01), Sub(vwap, low)), Constant(-1.0)), vwap),
        EMA(Sub(Corr(open_, close, 20), Corr(close, volume, 10)), 40),
        Cov(Constant(-0.01), Cov(vwap, Sub(high, Sum(Var(Constant(-0.5), 10), 50)), 10), 30),
    ]:
        data_df = data.make_dataframe(expr.evaluate(data))
        report.append(qlib_backtest.run(data_df))
    def plt_show():
        import matplotlib.pyplot as plt
        for i in range(len(report)):
            report[i].index = pd.to_datetime(report[i].index)
        report.append(report[0])
        for i in range(1, 4):
            for j in range(len(report[-1]['account'])):
                report[-1]['account'][j] += report[i]['account'][j]
        for j in range(len(report[-1]['account'])):
            report[-1]['account'][j] /= 4
        # 绘制账户余额变化曲线
        plt.figure(figsize=(10, 6))
        colors = ['blue','blue','blue','blue', 'yellow', 'red', 'cyan', 'magenta', 'green', 'black', 'orange']
        labels = [f'alphagen-{i}' for i in range(1, 5)] + ['gplearn', 'dso', 'alphagen-avg']
        for i, account in enumerate(report):
            plt.plot(account.index, account['account'], color=colors[i % len(colors)], label=labels[i])#f'Alpha {i+1}')
        plt.xlabel('Date')
        plt.ylabel('Account Balance')
        plt.title('Account Balance Over Time')
        plt.legend()
        plt.grid(True)
        plt.show()

    plt_show()