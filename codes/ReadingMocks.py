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
glon_center = 150.5
glat_center = -40
gcenter = SkyCoord(frame="galactic", l = glon_center, b = glat_center, unit=(u.deg, u.deg))

glon_ref = 12.5
glat_ref = glat_center
gref = SkyCoord(frame="galactic", l = glon_ref, b = glat_ref, unit=(u.deg, u.deg))

size = 2 # degree
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

full.shape

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
pmlon_dw = obs[:,5] / 10
pmlat_dw = obs[:,6] / 10
dw_data = np.vstack((glon_dw, glat_dw, pmlon_dw, pmlat_dw)).T

noDwarf = np.copy(full)
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
out[mask] = 0

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

stars = []
for blob in blobs:
    x_pix = x * grid_size/(2*gnomonic_width)+grid_size/2.
    y_pix = y * grid_size/(2*gnomonic_width)+grid_size/2.
    pmra_pix = data['pmra'] * grid_size/(2*pmfield_size)+grid_size/2.
    pmdec_pix = data['pmdec'] * grid_size/(2*pmfield_size)+grid_size/2.

    cut_idx1 = (x_pix-blob[0])**2/(blob[4])**2+(y_pix-blob[1])**2/(blob[5])**2<1.0
    cut_idx2 = (pmra_pix-blob[2])**2/(blob[6])**2+(pmdec_pix-blob[3])**2/(blob[7])**2<1.0
    stars.append(data[cut_idx1 & cut_idx2])
return stars

#find clusters in original dataset 
clusters = demeter.find_clusters(full, full[:,0], full[:,1], gnomonic_width, pmfield_size, grid_size, blobs)

# +
fig, ax = plt.subplots(figsize=(16,16))

#plot background stars 
ax.scatter(ref[:,0],ref[:,1], alpha=0.5, color='grey', s=100, marker='*')

#plot injected stars 
ax.scatter(dw_data[:,0], dw_data[:,1], alpha=1., color='salmon',s=800, marker='*')

# +
#plot returned stars
for cluster in clusters:
    if(len(cluster)>=5):
        plt.scatter(cluster['ra'], cluster['dec'], color='midnightblue',s=300, marker='*')

plt.show()

mask = np.isin(cluster['ra'], dwarf['ra'])

if(len(cluster[mask]==len(dwarf))):
    print('all injected stars recovered successfully!')
# -

# ## Let's try EE

# Directory setup for custom modules
module_path = 'EagleEye/eagleeye'
sys.path.append(module_path)
import EagleEye
import From_data_to_binary


# +
# Function to compute the Upsilon (𝛶) values from binary sequences
def compute_upsilon_values(binary_sequences, neighbor_range, num_cores):
    """Calculates the Upsilon values for anomaly detection using p-values across a range of neighbors."""
    # Create a PValueCalculatorParallel instance to access both pval_array_dict and smallest_pval_info
    p_value_calculator = EagleEye.PValueCalculatorParallel(binary_sequences, kstar_range=neighbor_range, num_cores=num_cores)
    
    # Calculate Upsilon values (𝛶) as the negative log of minimum p-values
    p_value_data = p_value_calculator.smallest_pval_info
    upsilon_values = -np.log(np.array(p_value_data['min_pval']))
    kstar_values = np.array(p_value_data['kstar_min_pval'])
    
    # Return both the calculator instance and the computed Upsilon values and k-star values
    return p_value_calculator, upsilon_values, kstar_values

# Function to retrieve Upsilon values for specific indices
def extract_upsilon_values(pval_info, indices, neighbor_range):
    """Extracts Upsilon values for given indices over a specified neighbor range."""
    return [[-np.log(pval_info.pval_array_dict[k][index, 0]) for k in neighbor_range] for index in indices]

# Function to find indices of interest based on Upsilon and k-star thresholds
def find_indices_by_threshold(upsilon_values, kstar_values, upsilon_thresh, kstar_thresh, condition='>'):
    """Finds indices where Upsilon and k-star values meet specified thresholds."""
    if condition == '>':
        indices = np.where((upsilon_values > upsilon_thresh) & (kstar_values < kstar_thresh))[0]
    else:
        indices = np.where((upsilon_values < upsilon_thresh) & (kstar_values > kstar_thresh))[0]
    return indices


# +
# Function to plot the Upsilon sequences
def plot_nlpval(ax, nlpval1, nlpval2, nlpval3, label1, label2, label3,crit_v):
    ax.plot(range(4, len(nlpval1) + 4), nlpval1, color='limegreen', linewidth=2, label=label1)
    ax.plot(range(4, len(nlpval2) + 4), nlpval2, color='darkorange', linewidth=2, label=label2)
    ax.plot(range(4, len(nlpval3) + 4), nlpval3, color='magenta', linewidth=2, label=label2)
    # Apply logarithmic scale if necessary
    ax.set_yscale('log')
    # Highlight max values
    max_idx1 = np.argmax(nlpval1) + 4
    max_idx2 = np.argmax(nlpval2) + 4
    max_idx3 = np.argmax(nlpval3) + 4

    # ax.axvline(max_idx1, color='darkcyan', linestyle='--', linewidth=1.5, label=f'{label1} Kstar')
    # ax.axvline(max_idx2, color='red', linestyle='--', linewidth=1.5, label=f'{label2} Kstar')

    ax.axhline(y=crit_v, color='red', linestyle='--', linewidth=1.5, label='Crit. value')
    # ax.text(max_idx1, max(nlpval1), 'Kstar', color='darkcyan', verticalalignment='bottom', horizontalalignment='right', fontsize=17, fontweight='bold')
    # ax.text(max_idx2, max(nlpval2), 'Kstar', color='red', verticalalignment='bottom', horizontalalignment='right', fontsize=17, fontweight='bold')

    arrowprops_settings = dict(facecolor='darkcyan', edgecolor='darkcyan', shrink=0.05, width=2, headwidth=10, headlength=10)
    ax.annotate('', xy=(max_idx1, max(nlpval1)), xytext=(max_idx1, max(nlpval1) * 1.5),  # Slight adjustment for log scale
                arrowprops=arrowprops_settings)

    arrowprops_settings_red = dict(facecolor='chocolate', edgecolor='chocolate', shrink=0.05, width=2, headwidth=10, headlength=10)
    ax.annotate('', xy=(max_idx2, max(nlpval2)), xytext=(max_idx2, max(nlpval2) * 1.5),  # Slight adjustment for log scale
                arrowprops=arrowprops_settings_red)
    
    arrowprops_settings_mag = dict(facecolor='darkmagenta', edgecolor='darkmagenta', shrink=0.05, width=2, headwidth=10, headlength=10)
    ax.annotate('', xy=(max_idx3, max(nlpval3)), xytext=(max_idx3, max(nlpval3) * 1.5),  # Slight adjustment for log scale
                arrowprops=arrowprops_settings_mag)

    # Add 'Crit. value' text above the critical value line
    ax.text(395, crit_v * 1.1, 'Crit. line', color='red', fontsize=12, fontweight='bold',
            verticalalignment='bottom', horizontalalignment='right')


    ax.set_ylabel(r'$\Upsilon_i$', rotation=90)
    ax.set_xlim(0, 400-1)
    # ax.set_ylim(-5, max( [20, max(nlpval1), max(nlpval2)] )+10)
    ax.set_xlabel('K-nearest neighbors')
    # ax.legend(loc='upper right')



# +
# Generate 100,000 Bernoulli sequences to determine the critical Upsilon threshold
num_sequences = 1000            # Number of Bernoulli sequences
K_M = 40                        # Length of each sequence
NUM_CORES = 1
NEIGHBOR_RANGE = range(4, K_M)
critical_quantile = 0.9999        # Quantile to calculate critical Upsilon threshold

# Generate a Bernoulli matrix with sequences of 0s and 1s
bernoulli_sequences = np.random.binomial(n=1, p=0.5, size=(num_sequences, K_M))

# Compute Upsilon values and optimal k-star values for Bernoulli sequences
p_value_data, upsilon_values_bernoulli, optimal_kstar_bernoulli = compute_upsilon_values(bernoulli_sequences, neighbor_range=NEIGHBOR_RANGE, num_cores=NUM_CORES)

# Calculate the critical Upsilon threshold at the specified quantile
critical_upsilon_threshold = np.quantile(upsilon_values_bernoulli, critical_quantile)

# -

critical_upsilon_threshold

# +
fig,ax = plt.subplots(1,2)

ax[0].scatter(ref[:,0], ref[:,1], c = 'black', marker = '+')
ax[0].scatter(full[:,0], full[:,1], c = 'red', marker = '+')
ax[0].scatter(dw_data[:,0], dw_data[:,1], c = 'blue', marker = '.')

# #%ax[0].set_xlim(-0.012, -0.006)
# #%ax[0].set_ylim(-0.012, -0.006)

ax[1].scatter(ref[:,2], ref[:,3], c = 'black', marker = '+')
ax[1].scatter(full[:,2], full[:,3], c = 'red', marker = '+')
ax[1].scatter(dw_data[:,2], dw_data[:,3], c = 'blue', marker = '.')
#ax[1].set_xlim(-3.5, -1.9)
#ax[1].set_ylim(-3.5, -1.9)
# -

# Generate and process binary sequences with anomalies
binary_sequences_anomaly, _ = From_data_to_binary.create_binary_array_cdist(
    full, ref, num_neighbors = K_M, num_cores = NUM_CORES
)
anomaly_pval_info, upsilon_values_anomaly, kstar_values_anomaly = compute_upsilon_values(
    binary_sequences_anomaly, NEIGHBOR_RANGE, NUM_CORES
)

# +
fig,ax = plt.subplots(1,2)

ax[0].hist(upsilon_values_anomaly)
ax[0].axvline(x = critical_upsilon_threshold, c = 'red', ls  = '--')
ax[0].set_xlabel('$\\Upsilon^{dwarf}$')

ax[1].hist(kstar_values_anomaly)
ax[1].set_xlabel('$K_{*}^{dwarf}$')

# +
# Define indices of interest for anomaly data
positive_indices = find_indices_by_threshold(upsilon_values_anomaly, kstar_values_anomaly, 
                                             upsilon_thresh = 6, kstar_thresh = 30)
negative_indices = find_indices_by_threshold(upsilon_values_anomaly, kstar_values_anomaly, 
                                             upsilon_thresh = 3, kstar_thresh = 20, condition = '<')

# Collect Upsilon values for selected indices in anomaly data
upsilon_values_anomaly_selected = extract_upsilon_values(anomaly_pval_info, [positive_indices[0], 359, negative_indices[17]], NEIGHBOR_RANGE)
# -

# Generate and process binary sequences without anomalies
binary_sequences_no_anomaly, _ = From_data_to_binary.create_binary_array_cdist(
    noDwarf, ref, num_neighbors = K_M, num_cores = NUM_CORES
)
no_anomaly_pval_info, upsilon_values_no_anomaly, kstar_values_no_anomaly = compute_upsilon_values(
    binary_sequences_no_anomaly, NEIGHBOR_RANGE, NUM_CORES
)

plt.scatter(upsilon_values_anomaly, kstar_values_anomaly, label = 'Full')
#plt.scatter(upsilon_values_no_anomaly, kstar_values_no_anomaly, marker = '+', label = 'No Dwarf')
plt.scatter(upsilon_values_anomaly[-120:], kstar_values_anomaly[-120:], facecolors='none', edgecolors='black', marker = 'o', label = 'Dwarf')
plt.ylabel('$K_{*}$')
plt.xlabel('$\\Upsilon$')
plt.legend()

# +
fig,ax = plt.subplots(1,2)

ax[0].hist(upsilon_values_no_anomaly, label = 'Ref', histtype = 'step')
ax[0].hist(upsilon_values_anomaly, label = 'Dwarf', histtype = 'step')
ax[0].axvline(x = critical_upsilon_threshold, c = 'red', ls  = '--')
ax[0].set_xlabel('$\\Upsilon$')
ax[0].legend()

ax[1].hist(kstar_values_no_anomaly, histtype = 'step')
ax[1].hist(kstar_values_anomaly, histtype = 'step')
ax[1].set_xlabel('$K_{*}$')

# +
fig,ax = plt.subplots(1,2)
th = 27#critical_upsilon_threshold
ax[0].scatter(full[upsilon_values_anomaly > th,0], full[upsilon_values_anomaly > th,1], 
              c = upsilon_values_anomaly[upsilon_values_anomaly > th], marker = 'o', vmin = 0, vmax = np.max(upsilon_values_anomaly))
#aux = Ellipse((np.mean(ra_inj), np.mean(dec_inj)), np.std(ra_inj), np.std(dec_inj), edgecolor='black',
#              facecolor='none', linewidth=2)
#ax[0].add_patch(aux)
#ax[0].scatter(xx_dwarf, yy_dwarf, facecolors='none', edgecolors='black', marker = 'o')
ax[0].scatter(dw_data[:,0], dw_data[:,1], marker = '.', c = 'black', s = 10)

ax[0].set_xlim(glon_center - 1, glon_center + 1)
ax[0].set_ylim(glat_center - 1, glat_center + 1)

#im1 = ax[1].scatter(ref[:,0], ref[:,1], c = upsilon_values_no_anomaly, marker = 'o', vmin = 0, vmax = 20)
#plt.colorbar(im1)
#ax[1].set_xlim(-0.012, -0.006)
#ax[1].set_ylim(-0.012, -0.006)
# -


