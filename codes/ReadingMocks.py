import numpy as np
import matplotlib.pyplot as plt

# !ls ../data/mocks/

# +
# the (true) physical parameters of each of the simulated UFs

data = np.loadtxt('../data/mocks/library.survey.UFGX_LIB99.dat')
# -

data.shape

# +
# data with Gaia errors. This tar file contains one file per simulated UF, named obs_ID.dat
#   where ID is the identifier used in the library log referenced above. 
#   For each UF, it contains data for stars observable by Gaia with end-of-mission errors (for the nominal mission time of 5yr)

obs = np.loadtxt('../data/mocks/obs.lib99/UFGX_TEST99_lib/obs_990002.dat')

obs_clean = np.loadtxt('../data/mocks/true.lib99/UFGX_TEST99_lib/true_990002.dat.gz')

# -

obs.shape

obs_clean.shape

plt.scate(obs[:,26])

# +
fig,ax = plt.subplots(1,2)

#ax[0].scatter(obs[:,2],obs[:,3])
#ax[1].scatter(obs[:,5],obs[:,6])

ax[0].scatter(obs_clean[:,1],obs_clean[:,2], c = 'red', marker = '+')
ax[1].scatter(obs_clean[:,5],obs_clean[:,6], c = 'red', marker = '+')
# -


