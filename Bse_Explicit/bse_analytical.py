import numpy as np
import scipy as sp


def bse_analytical(sigma, r, T, K, S, option_type="call"):
  d1 = (np.log(S/K) + (r + 0.5 * sigma ** 2) * T) / (sigma*np.sqrt(T))
  d2 = d1 - sigma*np.sqrt(T)

  if option_type == "call":
    price = S * sp.stats.norm.cdf(d1) - K * np.exp(-r * T) * sp.stats.norm.cdf(d2)
  elif option_type == "put":
    price = K * np.exp(-r * T) * sp.stat.norm.cdf(-d2) - S * sp.stats.norm.cdf(-d1)
  print(price)
  return price

def imp_vol( r, T, K, S, market_price, option_type="call" ):
  def objective_function(sigma):
    model_price = bse_analytical(sigma, r, T, K, S, "call")
    return model_price - market_price
  implied_volatility = sp.optimize.brentq(objective_function,1e-5,5.0)
  print(f"Implied Vol {iv:.2%}")
  return implied_volatility