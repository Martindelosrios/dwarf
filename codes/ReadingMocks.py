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

igal = 95
data = np.loadtxt('../data/mocks/library.survey.UFGX_LIB' + str(igal) + '.dat')

data_labels = ['ID','$Log_{10} ( L_{v} [L_{\odot}])$','$Log_{10} ( M/L [M_{\odot}/L_{\odot}])$','$\sigma_{v} [km/s]$',
               'rh [kPc]','dist [kPc]','l [$^{\circ}$]','b$[^{\circ}]$','$V_{gal} [km/s]$', '$\phi_{v} [^{\circ}$]',
               '$\\theta_{v} [^{\circ}]$','stMass [$M_{\odot}$]','Nobs','$\\theta [^{\circ}]$','dmu [mas/yr]','$\mu_{cm} [mas/yr]$','$\mu_{l,cm} [mas/yr]$','$\mu_{b,cm} [mas/yr]$']
               
# -

data.shape

# +
fig, ax = plt.subplots(6,3, figsize = (10,10))
plt.subplots_adjust(wspace=0.3, hspace=0.7)

for j in range(6):
    for i in range(3):
        var = i + j*3
        if var in [1,2]:
            ax[j,i].hist(np.log10(data[:,var]))
        else:
            ax[j,i].hist(data[:,var])
        ax[j,i].set_xlabel(data_labels[var])

plt.savefig('../graph/TrueGalsProps_' + str(igal) + '.pdf')
# -

i = 320
Id = int(data[i,0])

# +
# data with Gaia errors. This tar file contains one file per simulated UF, named obs_ID.dat
#   where ID is the identifier used in the library log referenced above. 
#   For each UF, it contains data for stars observable by Gaia with end-of-mission errors (for the nominal mission time of 5yr)

# VIobs_mag Vobs_mag gl_rad gb_rad xpi_muas xmul_muas xmub_muas RV_km/s el_muas eb_muas epi_muas epml_muas epmb_muas unknown0 eRV_km/s unknown1 unknown2 unknown3 unknown4 id logg_dex elogg_dex  Gmag_mag eGmag_mag GbGr_mag eGbGr_mag flagb sep 

obs = np.loadtxt('../data/mocks/obs.lib' + str(igal) + '/UFGX_TEST' + str(igal) + '_lib/obs_' + str(Id) + '.dat')

# FB alpha(deg) delta(deg) mualpha(mas/yr) mudelta(mas/yr) distance(pc) radVelocity(km/s) MeanAbsV G Gbp Grp intVMinusI teff logG feh age Av 

obs_clean = np.loadtxt('../data/mocks/true.lib' + str(igal) + '/UFGX_TEST' + str(igal) + '_lib/true_' + str(Id) + '.dat.gz')

# -

obs.shape

obs_clean.shape

# +
fig,ax = plt.subplots(1,2)

ax[0].scatter(obs[:,2] * 180/np.pi, obs[:,3] * 180/np.pi)
ax[1].scatter(obs[:,5]/1e3, obs[:,6]/1e3)

ax[0].scatter(data[1,6], data[1,7], color = 'magenta', marker = '+')
ax[1].scatter(data[1,16], data[1,17], color = 'magenta', marker = '+')

ax[0].scatter(obs_clean[:,1]-180, obs_clean[:,2]-30, c = 'red', marker = '+')
ax[1].scatter(obs_clean[:,3], obs_clean[:,4], c = 'red', marker = '+')

ax[0].set_xlabel(r'$\alpha [^{\circ}]$')
ax[0].set_ylabel(r'$\beta [^{\circ}]$')

ax[1].set_xlabel(r'$\mu_{\alpha}$ [mas/year]')
ax[1].set_ylabel(r'$\mu_{\beta}$ [mas/year]')
# +
fig,ax = plt.subplots(1,2)

ax[0].scatter(obs_clean[:,1], obs_clean[:,2], c = 'red', marker = '+')
ax[1].scatter(obs_clean[:,3], obs_clean[:,4], c = 'red', marker = '+')

#ax[0].scatter(data[1,6], data[1,7], color = 'magenta')

ax[0].set_xlabel(r'$\alpha [^{\circ}]$')
ax[0].set_ylabel(r'$\beta [^{\circ}]$')

ax[1].set_xlabel(r'$\mu_{\alpha}$ [mas/year]')
ax[1].set_ylabel(r'$\mu_{\beta}$ [mas/year]')
# -


