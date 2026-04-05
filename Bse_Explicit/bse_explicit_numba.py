import numpy as np
from numba import jit


@jit
def set_boundary_condition_bottom(MeshGrid):
  for i in range(MeshGrid.shape[0]):
    MeshGrid[i,0] = 0
  return MeshGrid

@jit
def number_of_days_asset_level(time_lvl, asset_level):
  return np.zeros((time_lvl + 1,asset_level + 1), dtype=np.float64)

@jit
def set_terminal_condition(MeshGrid, spatial_step, K, Max):
  for i in range(MeshGrid.shape[1]):
    MeshGrid[Max, i] = max(spatial_step * i - K, 0)
  return MeshGrid

@jit
def set_right_boundary_condition(MeshGrid, K, Smax,t, T, r):
  for k in range(MeshGrid.shape[0]):
    MeshGrid[k, -1] = Smax - K * np.exp(-r * (T - t[k]))
  return MeshGrid

@jit
def bse_exp_call_numba(sigma, risk_free_rate, time_to_exp, K, N, spatial_step, S0):
  #start = time.time()
  #Choosing a Smax for the right boundary condition
  Smax = max(3 * K, 2.5 * S0)

  #Choosing the stock price step
  M = int(Smax/spatial_step)
  delta_t = time_to_exp/N #time_step

  t = np.linspace(0, time_to_exp, N + 1) #Time axis N+1 points from t=0 to t=T
  S = np.linspace(0, Smax, M + 1) # Stock price axis M+1 points from S=0 S=Smax

  # Setting up the boundary and terminal conditions in our grid
  Grid = number_of_days_asset_level(N, M)
  Grid = set_boundary_condition_bottom(Grid)
  Grid = set_terminal_condition(Grid,spatial_step, K, N)
  Grid = set_right_boundary_condition(Grid, K, Smax, t, time_to_exp, risk_free_rate )

  A = np.zeros(M+1, dtype=np.float64)
  B = np.zeros(M+1,dtype=np.float64)
  C = np.zeros(M+1, dtype=np.float64)

  #Calculate A_ki B_ki C_ki
  for j in range(1,M):
        A[j] = 0.5 * delta_t * (sigma ** 2 * j ** 2 - risk_free_rate * j)
        B[j] = 1 - delta_t * (sigma ** 2 * j ** 2 + risk_free_rate)
        C[j] = 0.5 * delta_t * (sigma ** 2 * j ** 2 + risk_free_rate * j)

  for k in range(N-1, -1, -1):
      for i in range(1, M):
          Grid[k, i] = (
              A[i] * Grid[k+1, i-1] +
              B[i] * Grid[k+1, i] +
              C[i] * Grid[k+1, i+1]
          )
  return Grid, t,S


