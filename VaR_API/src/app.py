from flask import Flask, request, jsonify
from api_extract import *
from data_processing import *

var_api = Flask(__name__)

@var_api.route("/get-var/<symbol>")
def GET_VAR(symbol):
    """Gets the request for which symbol we need to calculate the VaR, calculates the VaR and returns the appropriate respnse.
    
    Parameters
    ----------
    symbol : string
        The symbol/ticker for which we want to calculate VaR.
    alpha (optional) : float
        The confidence interval for which the VaR is calculated (defaults to 0.95).
    days (optional) : integer
        The number of days for which the VaR is calculated (defaults to 1)
    
    Returns
    -------
    json
        The result in json structure."""
    
    days = request.args.get("days")
    if not days:
        days = 1 #default is one-day VaR
    else:
        days = int(days)
     
    alpha = request.args.get("alpha")
    if not alpha:
        alpha = 0.95 #default confidence interval
    else :
        alpha = float(alpha) 
  
    try:
        response = a_vantage.GET_DATA(symbol)
    except Exception as e:
        return jsonify(e), 400
    
    data = data_processor.DATA_PROCESS(response)
    
    if not data:
        message = "Ooops something went wrong with the requested data. Please check if the symbol is a valid ticker."
        return jsonify(message), 400
    else:
        sigma = data["sigma"]
        s0 = data["s0"]
        var = mc_var.MC_VAR(alpha, sigma, days, s0)
        
        final_payload = {
            "symbol":symbol,
            "VaR" : var,
        }
        
        return jsonify(final_payload), 200

if __name__ == "__main__":
    var_api.run(debug=True)