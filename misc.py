import glob
from astropy.io import fits
import astroalign as aa
from astropy.wcs import WCS
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import sep
from joblib import Parallel, delayed
import time


def align(frames):
    return aa.register(frames[0], frames[1])[0]

files = sorted(glob.glob("C:/Users/Luca/Documents/Rigel/Test_Data/*"))[0]
images = sorted(glob.glob(files + "/*"))
data = [fits.getdata(f) for f in images]
reference_image = np.array(data[0])
target_image = np.array(data[1])
test_data = data[0:]



group0 = [[target_image, reference_image], [target_image, reference_image]]
out = []
for g in group0:
    out.append(align(g))



out_p = Parallel(n_jobs=2, return_as= 'generator')(delayed(align)(g) for g in group0)

out_parallel = [im for im in out_p]


diffs = []
for i in range(0, len(group0)):
    diffs.append(out[i] - np.array(out_parallel[i]))

for diff in diffs:
    print(diff)



"""image0 = images[0]
image1 = images[1]

data0 = fits.getdata(image0)
data1 = fits.getdata(image1)

print(data0.shape, data1.shape)

frames = [[data0, data1]]

aligned_serial = align([data0, data1])

#joblib:
aligned_parallel = np.array(Parallel(n_jobs=1)(delayed(align)(i) for i in frames)[0])



print(aligned_parallel.shape, aligned_serial.shape)
diff = aligned_serial - aligned_parallel

print(diff, diff.shape)
plt.imshow(diff, norm=LogNorm())
plt.show()"""




