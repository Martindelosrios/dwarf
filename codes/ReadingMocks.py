import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.vizier import Vizier
import sys
import scipy

# ## Let's read the simulated dwarf

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

ax[0].scatter(data[i,6], data[i,7], color = 'magenta', marker = '+')
ax[1].scatter(data[i,16], data[i,17], color = 'magenta', marker = '+')

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
# ## Let's read the simulated backgroung
#
# * Downloaded from https://vizier.cds.unistra.fr/viz-bin/VizieR-3?-source=VI/137/gum_mw&-out.max=50&-out.form=HTML%20Table&-out.add=_r&-out.add=_RAJ,_DEJ&-sort=_r&-oc.form=sexa
# * Data From https://arxiv.org/abs/1202.0132
# * In demeters paper they use data only from 20° < |b| < 90°

# +
glon_center = 162.5
glat_center = -42.5
gcenter = SkyCoord(frame="galactic", l = glon_center, b = glat_center, unit=(u.deg, u.deg))

glon_ref = 12.5
glat_ref = -42.5
gref = SkyCoord(frame="galactic", l = glon_ref, b = glat_ref, unit=(u.deg, u.deg))

size = 5 # degree
row_limit = -1 #5000

# +
ra_center = gcenter.transform_to('icrs').ra.value
dec_center = gcenter.transform_to('icrs').dec.value

v = Vizier(columns=['RAICRS','DEICRS','pmRA','pmDE'],
           column_filters={"Host":"1"}, row_limit = row_limit)

#v.ROW_LIMIT = 100

bkg = v.query_region(SkyCoord(ra = ra_center, dec=dec_center, unit=(u.deg, u.deg),frame='icrs'),
                        width= str(size) + "d",
                        catalog=["VI/137/gum_mw"])

equatorial_coords = SkyCoord( ra  = bkg[0]['RAICRS'], 
                              dec = bkg[0]['DEICRS'], 
                              pm_ra_cosdec =  bkg[0]['pmRA'], 
                              pm_dec = bkg[0]['pmDE'], 
                              frame = 'icrs')

# Transform to Galactic coordinates
galactic_coords = equatorial_coords.transform_to('galactic')
glon_full = galactic_coords.l.value
glat_full = galactic_coords.b.value

# Access proper motion in Galactic coordinates
pmlon_full = galactic_coords.pm_l_cosb.to(u.mas/u.yr).value  # Proper motion in Galactic longitude (l)
pmlat_full = galactic_coords.pm_b.to(u.mas/u.yr).value       # Proper motion in Galactic latitude (b)

full = np.vstack((glon_full, glat_full, pmlon_full, pmlat_full)).T
# -

full

# +
ra_ref = gref.transform_to('icrs').ra.value
dec_ref = gref.transform_to('icrs').dec.value

v = Vizier(columns=['RAICRS','DEICRS','pmRA','pmDE'],
           column_filters={"Host":"1"}, row_limit = row_limit)

#v.ROW_LIMIT = 100

ref = v.query_region(SkyCoord(ra = ra_ref, dec=dec_ref, unit=(u.deg, u.deg),frame='icrs'),
                        width= str(size) + "d",
                        catalog=["VI/137/gum_mw"])

equatorial_coords = SkyCoord( ra  = ref[0]['RAICRS'], 
                              dec = ref[0]['DEICRS'], 
                              pm_ra_cosdec =  ref[0]['pmRA'], 
                              pm_dec = ref[0]['pmDE'], 
                              frame = 'icrs')

# Transform to Galactic coordinates
galactic_coords = equatorial_coords.transform_to('galactic')
glon_ref = galactic_coords.l.value - glon_ref + glon_center # To center around the same latitud
glat_ref = galactic_coords.b.value

# Access proper motion in Galactic coordinates
pmlon_ref = galactic_coords.pm_l_cosb.to(u.mas/u.yr).value  # Proper motion in Galactic longitude (l)
pmlat_ref = galactic_coords.pm_b.to(u.mas/u.yr).value       # Proper motion in Galactic latitude (b)

ref = np.vstack((glon_ref, glat_ref, pmlon_ref, pmlat_ref)).T

# +
fig,ax = plt.subplots(1,2)

ax[0].scatter(full[:,0], full[:,1])
ax[1].scatter(full[:,2], full[:,3])

ax[0].scatter(ref[:,0], ref[:,1], c = 'red')
ax[1].scatter(ref[:,2], ref[:,3], c = 'red')

ax[0].set_xlabel('$l$ [°]')
ax[0].set_ylabel('$b$ [°]')

ax[1].set_xlabel('$\mu_{l}$ [mas/yr]')
ax[1].set_ylabel('$\mu_{b}$ [mas/yr]')
# -
# ## Let's inpaint a dwarf galaxy

# +
np.random.shuffle(full)
full = full[:500]

np.random.shuffle(ref)
ref = ref[:500]
# -

glon_dw = (obs[:,2] * 180/np.pi - data[i,6]) + glon_center # Just to center the dwarf galaxy
glat_dw = (obs[:,3] * 180/np.pi - data[i,7]) + glat_center
pmlon_dw = obs[:,5] 
pmlat_dw = obs[:,6] 
dw_data = np.vstack((glon_dw, glat_dw, pmlon_dw, pmlat_dw)).T

full = np.vstack((full, dw_data))

full.shape

# +
fig,ax = plt.subplots(1,2)

ax[0].scatter(full[:,0], full[:,1])
ax[0].scatter(dw_data[:,0], dw_data[:,1], color = 'magenta', marker = '+')

ax[1].scatter(full[:,2], full[:,3])
ax[1].scatter(dw_data[:,2], dw_data[:,3], color = 'magenta', marker = '+')
# -

# ## Let's try demeter

module_path = 'demeter/'
sys.path.append(module_path)
import demeter
import torch
import torch.nn as nn
import torch.nn.functional as F

# +
#test function to load gaia data -- reset median_ra and median_dec to
#test values around edges of survey 
#set parameters 
grid_size = 96
edge_size = 6
front = int(edge_size)
back  = int(grid_size-edge_size)

#center of the sky field in RA and DEC 
median_ra  = ra_center
median_dec = dec_center


#width of the field in gnomonic coordinates 
gnomonic_width = 0.0175

#width of the proper motion field 
pmfield_size = 500.0

# +
lmin = glon_center - 2.5
lmax = glon_center + 2.5
bmin = glat_center - 2.5
bmax = glat_center + 2.5

field_image, edges = np.histogramdd(full,\
                            bins=(np.linspace(lmin, lmax, grid_size +1),
                                  np.linspace(bmin, bmax, grid_size +1),
                                  np.linspace(-200, 200, grid_size +1),
                                  np.linspace(-200, 200, grid_size +1) ) )

random_image, edges = np.histogramdd(ref,\
                            bins=(np.linspace(lmin, lmax, grid_size +1),
                                  np.linspace(bmin, bmax, grid_size +1),
                                  np.linspace(-200, 200, grid_size +1),
                                  np.linspace(-200, 200, grid_size +1) ) )
# -

plt.imshow(field_image.sum(axis=(2,3)))

# +
smooth_field = scipy.ndimage.gaussian_filter(field_image,3.0)

plt.imshow(smooth_field.sum(axis=(2,3))/np.max(smooth_field.sum(axis=(2,3))))
plt.clim([0,1])
plt.show()

# +
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(device)

wv = demeter.WaveNet(grid_size, grid_size, J=4, wavelets = ['bior5.5'])
out = wv(torch.Tensor(field_image).to(device)).to('cpu').numpy()
rand_out = wv(torch.Tensor(random_image).to(device)).to('cpu').numpy()
# -

out = scipy.ndimage.gaussian_filter(out,3.0)
rand_out = scipy.ndimage.gaussian_filter(rand_out,3.0)

# +
#remove outer edges of image 
out1 = np.zeros((grid_size, grid_size, grid_size, grid_size))
out1[front:back,front:back,front:back,front:back] = 1
out1 = out*out1

plt.imshow(out1.sum(axis=(2,3))/np.max(out1.sum(axis=(2,3))))
plt.colorbar()
plt.clim([0,1])
plt.show()
# -

#standardize the real output 
out = (out-np.mean(rand_out))/np.std(rand_out)

# +
#apply threshold 
edge_size = 6
front = int(edge_size)
back = int(grid_size-edge_size)
nstars = len(ref)

#apply significance correction 
out = out / (-0.6*np.log10(nstars)+4.)

#applying thresholding 
mask = out < 5.0 
#out[mask] = 0

#remove outer edges of image
out1 = np.zeros((grid_size, grid_size, grid_size, grid_size))
out1[front:back,front:back,front:back,front:back] = 1
out1 = out*out1
# -

plt.imshow(out1.sum(axis=(2,3)))
plt.colorbar()
plt.show()

#find remaining hotspots 
blobs, significance = demeter.find_blobs(out1, threshold=9)

print(blobs)
print(significance)

# +
fig, ax = plt.subplots(figsize=(16,16))

#plot background stars 
plt.scatter(ref[:,0],ref[:,1], alpha=0.5, color='grey', s=100, marker='*')

#plot injected stars 
plt.scatter(dw_data[:,0], dw_data[:,1], alpha=1., color='salmon',s=800, marker='*')

#plot returned stars
for cluster in clusters:
    if(len(cluster)>=5):
        plt.scatter(cluster['ra'], cluster['dec'], color='midnightblue',s=300, marker='*')

plt.show()

mask = np.isin(cluster['ra'], dwarf['ra'])

if(len(cluster[mask]==len(dwarf))):
    print('all injected stars recovered successfully!')
