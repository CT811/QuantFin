import requests
import dotenv
import os
from pathlib import Path

def GET_DATA(symbol):
    """Makes a request to vantage API and gets daily data for the symbol provided. 
        You will have to create .env file in the root directory and store you API key as API_KEY."""
    
    env_path = Path(__file__).resolve().parents[2]/".env"
    dotenv.load_dotenv(env_path)
    api_key = os.environ.get("API_KEY")
    
    result = {} 
    try:    
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=full&apikey={api_key}"
        req = requests.get(url)
        data = req.json()
        result["type"] = "data"
        result["value"] = data
    except Exception as e:
        result["type"] = "error"
        result["value"] = e
    
    return(result)

