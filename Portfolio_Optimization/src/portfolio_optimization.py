import pandas as pd
import numpy as np
import yfinance as yf
import random as rnd
import scipy.optimize as opt
from conditional_var import cvar

### Step 1: Data extraction and transformation
sp500_data = pd.read_html("https://en.wikipedia.org/wiki/List_of_S&P_500_companies")[0]

# Replace all the . with -
sp500_data["Symbol"] =  sp500_data["Symbol"].str.replace(".", "-")
list_of_symbols = list(sp500_data["Symbol"].unique())

end_date = "2024-09-30"
start_date = "2019-09-30"
initial_stock_data = yf.download(tickers=list_of_symbols, start=start_date, end=end_date)
stock_data = initial_stock_data.stack(future_stack=True)
stock_data.index.names = ["Date", "Ticker"]

# We want to compute the dollar volume of each ticker and then take the 100 tickers with the highest volume
stock_data["Dollar_volume"] = stock_data["Adj Close"] * stock_data["Volume"] / 1e6
tickers_to_keep = stock_data.groupby(level=1)["Dollar_volume"].mean().sort_values(ascending=False).head(100).index.values
filtered_sample = stock_data.loc[(slice(None),tickers_to_keep), :]

# Exclude tickers with incomplete data
na_tickers = filtered_sample[filtered_sample["Adj Close"].isna()].index.get_level_values("Ticker").unique()
tickers_to_keep = list(set(tickers_to_keep) -  set(na_tickers))
filtered_sample = filtered_sample.loc[(slice(None),tickers_to_keep), :]

# Construct montly returns matrix
monthly_ret = np.log(filtered_sample.unstack("Ticker")["Adj Close"].resample("ME").last()).diff().iloc[1:]

# Split the data into in-sample and out-of-sample
cutoff_date = pd.to_datetime("2023-12-31")
in_sample = monthly_ret.loc[monthly_ret.index.values <= cutoff_date]
out_of_sample = monthly_ret.loc[monthly_ret.index.values > cutoff_date]

### Step 2: Strategy contrstuction
# Historical mCVAR optimization
random_tickers = rnd.sample(range(1, len(in_sample.columns)), 10) # Select 5 random tickets to create a postfolio from
portfolio_tickers = in_sample.iloc[:, random_tickers]

cvar_array = cvar(portfolio_tickers, alpha=0.05).iloc[0].to_numpy()

# Initialize parameters for optimization:
# weights is the array containing the weight of each ticker and l is the risk-aversion parameter of the portfolio owner.
weights = np.array(len(portfolio_tickers.columns) * [1./len(portfolio_tickers.columns)])
l = 3 # Risk aversion factor that varies according to the portfolio manager

def portfolio_returns(df, weights):
    result = np.array((df.mean() * weights) * 12)
    return result

def objective_function(weights):
    result = np.sum(- (l * portfolio_returns(portfolio_tickers, weights) + (1- l) * cvar_array)) # CVaR enters with a positive sign the equation since it at this point it represents return and not PnL in $ value
    return result

constraints_list = ({"type" : "eq", "fun" : lambda x: np.sum(x) - 1})

bounds = tuple((0,1) for x in range(len(portfolio_tickers.columns)))

mcvar_parameters = opt.minimize(objective_function,
                                weights,
                                method = "SLSQP",
                                bounds = bounds,
                                constraints = constraints_list)

### Step 3: Benchmark against SP500
spy_data = yf.download(tickers='SPY',start = "2023-12-01", end = end_date).stack(future_stack = True)
spy_data.index.names = ["Date", "Ticker"]
spy_return = np.log(spy_data.unstack("Ticker")["Adj Close"].resample("ME").last()).diff().iloc[1:]

oot_tickers = out_of_sample.iloc[:, random_tickers]
oot_portfolio_return = (oot_tickers * mcvar_parameters.x).sum(axis = 1)
 
comparative_df = pd.DataFrame({'Portfolio':np.array(oot_portfolio_return),
                               'SPY':np.array(spy_return),})
comparative_df = comparative_df.set_index(pd.date_range("2024-01-31", end_date, periods=(len(comparative_df))))
comparative_df = np.exp(np.log1p(comparative_df).cumsum()) - 1 
