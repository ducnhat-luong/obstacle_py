import numpy as np
import pandas as pd

nx = 60
ny = 60
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
xv, yv = np.meshgrid(x, y, indexing='ij')
shape = xv.shape
occ = np.zeros(shape)

## convert your array into a dataframe
df = pd.DataFrame (occ)
filepath = 'map.xlsx'
df.to_excel(filepath, index=False)