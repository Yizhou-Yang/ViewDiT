from PIL import Image
import numpy as np, glob
files = sorted(glob.glob('/root/viewdit/data/v3_validation_data/JPEGImages/bear/*.jpg'))
a = np.asarray(Image.open(files[0]).convert('RGB').resize((120, 80)), dtype=float)
r, g, b = a[..., 0].mean(), a[..., 1].mean(), a[..., 2].mean()
print('frame0 mean RGB=%.2f %.2f %.2f' % (r, g, b))
print('brown(r>g>b):', r > g > b)
m = np.asarray(Image.open(files[len(files)//2]).convert('RGB').resize((120, 80)), dtype=float)
print('mid frame RGB=%.2f %.2f %.2f' % (m[...,0].mean(), m[...,1].mean(), m[...,2].mean()))