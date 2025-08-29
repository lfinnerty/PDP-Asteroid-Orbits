from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np 
import glob

fnames = glob.glob('*.fits')
for file in fnames:
	hdul = fits.open(file)
	print(file)
	data = hdul[0].data
	plt.imshow(data, vmin=np.nanpercentile(data,5),vmax=np.nanpercentile(data,95),cmap='gray')
	plt.show()
