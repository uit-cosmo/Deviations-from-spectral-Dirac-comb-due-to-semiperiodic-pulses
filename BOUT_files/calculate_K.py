from xbout import open_boutdataset
import numpy as np

ds = open_boutdataset("public_input_data/BOUT.dmp.*.nc").squeeze()

ds = ds.isel(x=slice(2,-2))
ds = ds.assign_coords({"x": ds['z'].values})

ds['phi_0'] = ds['phi'].integrate('z')
ds['tmp'] = ds['phi'] - ds['phi_0']
ds['to_be_integrated'] = 0.5*(ds['tmp'].differentiate('x')**2 + ds['tmp'].differentiate('z')**2 )
ds['K'] = ds['to_be_integrated'].integrate(('x', 'z'))

K = ds['K'].values
time = ds['t'].values
np.save('K_1.6e-3_data', K)
np.save('time_1.6e-3_data', time)
