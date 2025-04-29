#%%
import numpy as np

data = np.load('ckpt/ckpt_epoch1.npz')
print(data.files)

print(data['b0'])