# Portfolio-Optimization-4-Criteria
This project implements four portfolio optimization strategies using Python to maximise financial performance according to different risk criteria. Perfect for quantitative analysis in finance. 

The script downloads historical prices for 50 equities, builds daily returns, trains four different portfolio optimizations on the first 70% of the data, tests them on the remaining 30%, plots cumulative returns and saves the portfolio weights and annualized expected returns to an Excel file and a PNG.

Step-by-step explanation

Imports and tickers

    The script imports the required Python libraries (yfinance for market data, pandas and numpy for data handling, scipy for optimization, matplotlib for plotting, and          stats functions).
    
    It defines a list of 50 tickers (large, diversified US stocks) and a start date (1-Jan-2000).

Data download and preparation

    It downloads historical daily closing prices for all tickers using yfinance.
    
    It keeps the closing prices, drops rows with missing values, and computes daily percentage returns (pct_change).
    
    It splits the returns into a training set (first 70%) and a test set (last 30%). The training set is used to fit/optimize portfolios and the test set is used to evaluate     performance out-of-sample.

Portfolio objective functions (the four criteria)

    The script defines four functions which receive a candidate vector of weights and compute a score that the optimizer minimizes. In simple terms, each function evaluates      a portfolio according to a different preference:

    MV_criterion: a mean-variance style objective that rewards higher expected return and penalizes variance (classic return vs. risk trade-off). The function returns the        negative of the criterion because the optimizer minimizes and we want to maximize the original objective.
    
    SK_criterion: an extended objective that includes skewness and kurtosis terms (it hence prefers portfolios with favorable tail properties in addition to mean and             variance).
    
    SR_criterion: negative Sharpe ratio (so minimizing it is equivalent to maximizing Sharpe), where Sharpe = (mean return) / (std dev).
    
    SOR_criterion: negative Sortino ratio (like Sharpe but penalizes only downside volatility).
    
    For every criterion the function:
    
    takes the candidate weights,
    
    computes portfolio daily return as the weighted sum of asset returns,
    
    computes mean, volatility and other moments (skewness/kurtosis when needed),
    
    computes the criterion and returns its negative (so the optimizer maximizes the intended metric).

Optimization setup and constraints

    The optimize function prepares the optimization problem: number of assets, initial guess (equal ones), bounds and constraint.
    
    Constraint: weights must sum to 1 (sum(x) - 1 == 0).
    
    Bounds: each weight is constrained between 0 and 1 (no short positions, no leverage).
    
    It runs the SLSQP solver to find the optimal weight vector for each criterion, using the training set returns.

Out-of-sample return calculation

    Using the optimized weights from each criterion, the script computes the portfolio returns on the test set (out-of-sample) by weighting daily returns and summing across      assets — this provides a fair evaluation of how the portfolios would have performed on unseen data.

Plotting results

    The script plots cumulative returns (in percent) over the test period for the four portfolios on the same chart, saves the chart as Cumulative_Return.png and displays         it. This chart visually compares out-of-sample performance.

Results tables and formatting

    It constructs df_weights, a DataFrame of tickers and the four sets of portfolio weights (converted to percentages).
    
    It computes the portfolios’ expected daily return, annualizes it (by compounding 252 trading days) and stores those values in df_returns (Expected Return - Annualized).
    
    Display options are set so numbers appear as percentages with two decimals.

Export to Excel
    
    Both DataFrames (df_weights and df_returns) are written to a single Excel file Results.xlsx with two separate sheets: “Weights” and “Returns”. This Excel file plus the       saved PNG are the deliverables.
