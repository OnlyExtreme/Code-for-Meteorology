import numpy as np
import pandas as pd
from scipy.signal import lfilter
import matplotlib.pyplot as plt
import LPfilter


def load_nino_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    data_list = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) > 1 and parts[0].isdigit():
            data_list.extend([float(x) for x in parts[1:]])
    
    nino = np.array(data_list)
    nino = nino[nino > -10]
    return nino

nino = load_nino_data("Filters\\nino34long.txt")

h, Hr = LPfilter.solve_lp_filter(65, 13, smooth=True)
fnin = lfilter(h, 1, nino)

delay = (len(h) - 1) // 2

plt.figure(figsize=(15, 5))
plt.plot(nino[delay:-delay], label='Original Nino 3.4', color = 'lightgray')
plt.plot(fnin[2*delay:], label='Low-pass Filtered', color='red', linewidth = 1.5)
plt.title('ENSO Index Filtering')
plt.legend()
plt.show()