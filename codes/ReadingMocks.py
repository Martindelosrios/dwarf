import numpy as np
import matplotlib.pyplot as plt

# !ls ../data/mocks/

# +
# the (true) physical parameters of each of the simulated UFs
#ID    - UFGX ID
#Lv    - Total integrated luminosity of UFGX
#M/L   - Mass to light ratio, in solar units
#sigma - Velocity dispersion of Plummer model
#rh    - Half-light radius of Plummer model
#Rhel  - Heliocentric distance of center of mass
#l     - Galactic longitude of center of mass
#b     - Galactic latitude
#Vgal  - Velocity modulus of center of mass. Reference frame at rest, in GC
#Phi   - Azimuthal angle of center of mass velocity vector
#Theta - Latitude angle of center of mass velocity vector
#stMass - Total stellar mass
#Nobs  - Number of observable stars in UFGX
#Theta  - Apparent size in l-b plane 
#dmu    - Apparent size in mul-mub plane
#mucm   - Apparent center-of-mass proper motion
#Note: Lv recomputed with new function 02/2015
# unit:         Lsun           M/Lsun      km/s       kpc       kpc   deg   deg   km/s deg   deg   Msolar             deg   mas/yr   mas/yr   mas/yr   mas/yr
#    ID           Lv              M/L    sigmav        rh      dist     l     b   Vgal phiv thetav   stMass   Nobs    theta      dmu     mucm    mulcm    mubcm

data = np.loadtxt('../data/mocks/library.survey.UFGX_LIB99.dat')
# -

data.shape

i = 1
Id = int(data[i,0])

# +
# data with Gaia errors. This tar file contains one file per simulated UF, named obs_ID.dat
#   where ID is the identifier used in the library log referenced above. 
#   For each UF, it contains data for stars observable by Gaia with end-of-mission errors (for the nominal mission time of 5yr)

# VIobs_mag Vobs_mag gl_rad gb_rad xpi_muas xmul_muas xmub_muas RV_km/s el_muas eb_muas epi_muas epml_muas epmb_muas unknown0 eRV_km/s unknown1 unknown2 unknown3 unknown4 id logg_dex elogg_dex  Gmag_mag eGmag_mag GbGr_mag eGbGr_mag flagb sep 

obs = np.loadtxt('../data/mocks/obs.lib99/UFGX_TEST99_lib/obs_' + str(Id) + '.dat')

# FB alpha(deg) delta(deg) mualpha(mas/yr) mudelta(mas/yr) distance(pc) radVelocity(km/s) MeanAbsV G Gbp Grp intVMinusI teff logG feh age Av 

obs_clean = np.loadtxt('../data/mocks/true.lib99/UFGX_TEST99_lib/true_' + str(Id) + '.dat.gz')

# -

obs.shape

obs_clean.shape

# +
fig,ax = plt.subplots(1,2)

ax[0].scatter(obs[:,2], obs[:,3])
ax[1].scatter(obs[:,5], obs[:,6])

ax[0].set_xlabel(r'$\alpha$ [rad]')
ax[0].set_ylabel(r'$\beta$ [rad]')

ax[1].set_xlabel(r'$\mu_{\alpha}$ [muas/year]')
ax[1].set_ylabel(r'$\mu_{\beta}$ [muas/year]')
# +
fig,ax = plt.subplots(1,2)

ax[0].scatter(obs_clean[:,1] * (np.pi/2) / 180, obs_clean[:,2] * (np.pi/2) / 180, c = 'red', marker = '+')
ax[1].scatter(obs_clean[:,3], obs_clean[:,4], c = 'red', marker = '+')

#ax[0].scatter(data[1,6] * (np.pi/2) / 180, data[1,7] * (np.pi/2) / 180, color = 'magenta')

ax[0].set_xlabel(r'$\alpha$ [rad]')
ax[0].set_ylabel(r'$\beta$ [rad]')

ax[1].set_xlabel(r'$\mu_{\alpha}$ [mas/year]')
ax[1].set_ylabel(r'$\mu_{\beta}$ [mas/year]')
# -


