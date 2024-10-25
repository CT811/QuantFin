import pandas as pd
import numpy as np

def cvar(df, alpha):
    """Calculates the conditional VaR for a given df of returns and a given alpha"""

    result = {}

    for col in df.columns:
        dataset = df[col].sort_values(ascending=False)
        n = len(dataset)
        cvar_index = int((1 - alpha) * n)
        cvar = np.mean(dataset[cvar_index:])
        result[col] = [cvar]
    
    result = pd.DataFrame.from_dict(result)

    return result