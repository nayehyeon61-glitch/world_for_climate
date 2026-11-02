"""Geometric stochastic climate world model.

raw ERA5 (GraphCast task) -> shell/patch decomposition -> Shell-GNN encoder
(Lie-structured latent on S^2) -> DIAMOND-style latent world model
(z_t -> z_{t+1}) -> Shell-GNN decoder -> x_{t+1}, evaluated with the GraphCast
protocol (latitude-weighted RMSE, ACC).
"""

__version__ = "0.1.0"
