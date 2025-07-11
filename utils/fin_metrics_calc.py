import numpy as np

"""
performance_metrics.py

A collection of portfolio performance metric functions for use in agentic AI workflows.
Each function accepts numeric arrays or lists and returns a scalar metric.

Functions:
  - sharpe_ratio(returns, risk_free_rate=0.0)
  - batting_average(port_returns, bench_returns)
  - capture_ratios(port_returns, bench_returns)
  - tracking_error(port_returns, bench_returns)
  - max_drawdown(returns)
"""

def sharpe_ratio(returns, risk_free_rate: float = 0.0) -> float:
    """
    Calculate the annualized Sharpe ratio of a return series.

    Args:
        returns (array-like): Sequence of periodic portfolio returns (e.g., daily decimal returns).
        risk_free_rate (float, optional): Periodic risk-free rate, svenv\Scripts\activate
ame frequency as returns. Defaults to 0.0.

    Returns:
        float: Sharpe ratio, computed as mean(excess returns) / sample standard deviation of excess returns.

    Raises:
        ValueError: If fewer than two return observations, or if standard deviation of excess returns is zero.
    """
    returns_arr = np.array(returns, dtype=float)
    n = returns_arr.size
    if n < 2:
        raise ValueError("At least two return observations are required to compute Sharpe ratio.")

    excess = returns_arr - risk_free_rate
    std_excess = np.std(excess, ddof=1)
    if std_excess == 0:
        raise ValueError("Standard deviation of excess returns is zero, Sharpe ratio is undefined.")

    return np.mean(excess) / std_excess


def batting_average(port_returns, bench_returns) -> float:
    """
    Compute the batting average: the fraction of periods in which the portfolio outperforms the benchmark.

    Args:
        port_returns (array-like): Sequence of periodic portfolio returns.
        bench_returns (array-like): Sequence of periodic benchmark returns; must be same length as port_returns.

    Returns:
        float: Proportion of periods where portfolio return > benchmark return.

    Raises:
        ValueError: If input lengths differ or arrays are empty.
    """
    port = np.array(port_returns, dtype=float)
    bench = np.array(bench_returns, dtype=float)
    if port.size == 0 or bench.size == 0:
        raise ValueError("Input return series must not be empty.")
    if port.size != bench.size:
        raise ValueError("Portfolio and benchmark returns must have the same length.")

    wins = np.sum(port > bench)
    return wins / port.size


def capture_ratios(port_returns, bench_returns) -> tuple:
    """
    Compute up- and down-market capture ratios relative to a benchmark.

    Args:
        port_returns (array-like): Sequence of periodic portfolio returns.
        bench_returns (array-like): Sequence of periodic benchmark returns; must be same length as port_returns.

    Returns:
        tuple:
            up_capture (float): Sum(port_returns when bench > 0) / Sum(bench_returns when bench > 0)
            down_capture (float): Sum(port_returns when bench < 0) / Sum(bench_returns when bench < 0)
            If no positive (or negative) benchmark periods exist, returns np.nan for that capture ratio.

    Raises:
        ValueError: If input lengths differ or arrays are empty.
    """
    p = np.array(port_returns, dtype=float)
    b = np.array(bench_returns, dtype=float)
    if p.size == 0 or b.size == 0:
        raise ValueError("Input return series must not be empty.")
    if p.size != b.size:
        raise ValueError("Portfolio and benchmark returns must have the same length.")

    # Up-market capture
    up_mask = b > 0
    if up_mask.any():
        upcap = p[up_mask].sum() / b[up_mask].sum()
    else:
        upcap = np.nan

    # Down-market capture
    down_mask = b < 0
    if down_mask.any():
        downcap = p[down_mask].sum() / b[down_mask].sum()
    else:
        downcap = np.nan

    return upcap, downcap


def tracking_error(port_returns, bench_returns) -> float:
    """
    Compute the tracking error: the sample standard deviation of the active return (portfolio minus benchmark).

    Args:
        port_returns (array-like): Sequence of periodic portfolio returns.
        bench_returns (array-like): Sequence of periodic benchmark returns; must be same length as port_returns.

    Returns:
        float: Sample standard deviation (ddof=1) of the return differences.

    Raises:
        ValueError: If input lengths differ or fewer than two observations.
    """
    p = np.array(port_returns, dtype=float)
    b = np.array(bench_returns, dtype=float)
    if p.size < 2 or b.size < 2:
        raise ValueError("At least two return observations are required to compute tracking error.")
    if p.size != b.size:
        raise ValueError("Portfolio and benchmark returns must have the same length.")

    diff = p - b
    return np.std(diff, ddof=1)


def max_drawdown(returns) -> float:
    """
    Calculate the maximum drawdown of a return series.

    Args:
        returns (array-like): Sequence of periodic returns.

    Returns:
        float: The most negative peak-to-trough decline, expressed as a negative decimal (e.g., -0.25 for 25%).

    Raises:
        ValueError: If input is empty.
    """
    returns_arr = np.array(returns, dtype=float)
    if returns_arr.size == 0:
        raise ValueError("Return series must not be empty.")

    # Compute cumulative wealth index
    wealth = np.cumprod(1 + returns_arr)
    # Compute running peak of the wealth index
    peak = np.maximum.accumulate(wealth)
    # Compute drawdowns
    drawdowns = (wealth - peak) / peak
    return float(drawdowns.min())


if __name__ == "__main__":
    # Sample data for testing
    daily_returns = [0.01, -0.005, 0.02, 0.015, -0.01]
    benchmark_returns = [0.008, -0.002, 0.018, 0.012, -0.015]
    rf_rate = 0.001  # daily risk-free rate

    print("Sharpe Ratio:", sharpe_ratio(daily_returns, rf_rate))
    print("Batting Average:", batting_average(daily_returns, benchmark_returns))

    upcap, downcap = capture_ratios(daily_returns, benchmark_returns)
    print(f"Up-Capture Ratio: {upcap:.4f}")
    print(f"Down-Capture Ratio: {downcap:.4f}")

    print("Tracking Error:", tracking_error(daily_returns, benchmark_returns))
    print("Max Drawdown:", max_drawdown(daily_returns))
