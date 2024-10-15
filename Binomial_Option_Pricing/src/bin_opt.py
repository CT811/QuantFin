import pyinputplus as pyip
import numpy as np

# Pricing function
def get_option_price(input_parameters):
    S = input_parameters["Stock Price"]
    K = input_parameters["Strike Price"]
    r = input_parameters["Risk-free rate in decimal"]
    sigma = input_parameters["Annual volatility in decimal"]
    y = input_parameters["Dividend Yield in decimal"]
    N = input_parameters["Number of time steps"]
    T = input_parameters["Maturity of the option in years"]
    option_type = input_parameters["Type of option (type 1 for Call and 2 for Put)"]
    
    dt = T/N # Since in time step 0 dt is 0
    u = np.exp(sigma * np.sqrt(dt))
    d = np.exp(-sigma * np.sqrt(dt))
    p = (np.exp((r - y) * dt) - d) / (u - d)
    disc = np.exp(-r * dt)
    
    # Vector of underlying asset vales
    S_vector = np.zeros(N+1)

    # Initialize asset and option values at maturity
    S_vector = S * d ** (np.arange(N,-1,-1)) * u ** (np.arange(0,N+1,1))
    
    if option_type == 1:
        O_vector = np.maximum(S_vector - K , np.zeros(N+1)) # Call option 
    else:
        O_vector = np.maximum(K - S_vector , np.zeros(N+1)) # Put option
 

    # step backwards through tree
    for i in range(N,0,-1):
        O_vector = disc * (p * O_vector[1:i+1] + (1-p) * O_vector[0:i])

    return(O_vector[0])

if __name__ == '__main__':
    # Input parameters
    input_parameters = {"Stock Price": 0,
        "Strike Price" : 0,
        "Risk-free rate in decimal": 0,
        "Annual volatility in decimal" : 0,
        "Dividend Yield in decimal": 0,
        "Number of time steps": 0,
        "Maturity of the option in years": 0,
        "Type of option (type 1 for Call and 2 for Put)": 1}

    # User-eneterd parameters with validation
    for tag in input_parameters:
        input_parameters[tag] = pyip.inputNum(f"Please provide the {tag} .", min = 0)
        if ("decimal" in tag):
            while (input_parameters[tag] > 1):
                print("Please provide a value between 0 and 1.")
                pyip.inputNum(f"Please provide the {tag} .", min = 0)

    print(f"Your option price is {get_option_price(input_parameters)}.")



