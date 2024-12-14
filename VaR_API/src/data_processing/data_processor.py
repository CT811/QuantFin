import pandas as pd

def DATA_PROCESS(response):
    """Turns a dictionary of type 'data' from the API call into the desired inputs for the VaR calculation."""
     
    try:
        data = response["value"]
        df = pd.json_normalize(data["Time Series (Daily)"], sep = '.', max_level=0).transpose()
        
    except Exception as e:
        print("Data are not in the expected format")
        return(None)
    
    rows_data = []
    for index, row in df.iterrows():
        nested_data = pd.json_normalize(row)
        rows_data.append(nested_data)
        
    processed_df = pd.concat(rows_data)
    processed_df.index = df.index
    
    sigma = processed_df["4. close"].astype(float).pct_change().std(ddof=0) # This is daily vol.
    s0 = processed_df["4. close"].astype(float).iloc[0]
    
    result = {}
    result["sigma"] = sigma
    result["s0"] = s0
    
    return(result)
