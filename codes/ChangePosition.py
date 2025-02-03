import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.vizier import Vizier
import sys
import scipy
from time import time
import healpy as hp
from itertools import product
from tqdm import tqdm

# Directory setup for custom modules
module_path = 'EagleEye/eagleeye'
sys.path.append(module_path)
import EagleEye
import From_data_to_binary


# # Some custom functions

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


# -

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


def LoadBkg(obs, glon_center, glat_center, eps_lon = 0, eps_lat = 0, row_limit = None, size = 2):
    
    gcenter = SkyCoord(frame="galactic", l = glon_center, b = glat_center, unit=(u.deg, u.deg))
    ra_center = gcenter.transform_to('icrs').ra.value
    dec_center = gcenter.transform_to('icrs').dec.value
    
    v = Vizier(columns=['RAICRS','DEICRS','pmRA','pmDE'],
               column_filters={"Host":"1"}, row_limit = -1)
    
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

    

    glon_ref = glon_center + 5
    glat_ref = glat_center
    gref = SkyCoord(frame="galactic", l = glon_ref, b = glat_ref, unit=(u.deg, u.deg))
    
    ra_ref = gref.transform_to('icrs').ra.value
    dec_ref = gref.transform_to('icrs').dec.value
    
    v = Vizier(columns=['RAICRS','DEICRS','pmRA','pmDE'],
               column_filters={"Host":"1"}, row_limit = -1)
    
    ref = v.query_region(SkyCoord(ra = ra_ref, dec=dec_ref, unit=(u.deg, u.deg),frame='icrs'),
                            width= str(size+2) + "d",
                            catalog=["VI/137/gum_mw"])
    
    equatorial_coords = SkyCoord( ra  = ref[0]['RAICRS'], 
                                  dec = ref[0]['DEICRS'], 
                                  pm_ra_cosdec =  ref[0]['pmRA'], 
                                  pm_dec = ref[0]['pmDE'], 
                                  frame = 'icrs')
    
    # Transform to Galactic coordinates
    galactic_coords = equatorial_coords.transform_to('galactic')
    glon_ref = galactic_coords.l.value - glon_ref + glon_center # To center around the same longitud
    glat_ref = galactic_coords.b.value
    
    # Access proper motion in Galactic coordinates
    pmlon_ref = galactic_coords.pm_l_cosb.to(u.mas/u.yr).value  # Proper motion in Galactic longitude (l)
    pmlat_ref = galactic_coords.pm_b.to(u.mas/u.yr).value       # Proper motion in Galactic latitude (b)
    
    ref = np.vstack((glon_ref, glat_ref, pmlon_ref, pmlat_ref)).T
    
    glon_dw = (obs[:,2] * 180/np.pi - 90) + glon_center + eps_lon # Just to center the dwarf galaxy
    glat_dw = (obs[:,3] * 180/np.pi - 30) + glat_center + eps_lat
    pmlon_dw = obs[:,5] / 1e3
    pmlat_dw = obs[:,6] / 1e3
    dw_data = np.vstack((glon_dw, glat_dw, pmlon_dw, pmlat_dw)).T

    if row_limit is not None:
        np.random.shuffle(full)
        full = full[:row_limit]
        
        np.random.shuffle(ref)
        ref = ref[:row_limit]
    
    full = np.vstack((full, dw_data))
    return full, ref, dw_data


# +
def purity(upsilon, ind, up_th = 20):
    '''
    Function to compute the purity of the cluster found by EE.

    Parameters
    ----------

    upsilon: (np.array) upsilon values of all the analized stars.
    ind: (np.array int) indices of the stars that trully belong to the dwarf galaxy.
    up_th: (float) upsilon threshold. default = 20

    Returns
    -------

    Purity defined as # dwarf stars (upsilon >= up_th) / # stars (upsilon >= up_th)
    '''
    return len(np.where(upsilon[ind] >= up_th)[0]) / len(np.where(upsilon >= up_th)[0])

def FN(upsilon, ind, up_th = 20):
    '''
    Function to compute the dwarf stars missed by EE.

    Parameters
    ----------

    upsilon: (np.array) upsilon values of all the analized stars.
    ind: (np.array int) indices of the stars that trully belong to the dwarf galaxy.
    up_th: (float) upsilon threshold. default = 20

    Returns
    -------

    FN defined as # dwarf stars (upsilon < up_th) / # dwarf stars
    '''
    return len(np.where(upsilon[ind] < up_th)[0]) / len(ind)

def Delta(full, upsilon, ind, up_th = 20):
    '''
    Function to compute the distances (angular and peculiar velocity) between the center of the cluster found by EE
        and the dwarf galaxy.

    Parameters
    ----------

    full: (np.array) data of the analized stars.    
    upsilon: (np.array) upsilon values of all the analized stars.
    ind: (np.array int) indices of the stars that trully belong to the dwarf galaxy.
    up_th: (float) upsilon threshold. default = 20

    Returns
    -------

    Distances defined as 
        ang = np.sqrt( (l_{real} - <l>)^{2} + (b_{real} - <b>)^{2} )
        vpec = np.sqrt( (mul_{real} - <mul>)^{2} + (mub_{real} - <mub>)^{2} )
    '''
    l_real = np.mean(full[ind,0])
    b_real = np.mean(full[ind,1])
    mul_real = np.mean(full[ind,2])
    mub_real = np.mean(full[ind,3])

    ind_aux = np.where(upsilon >= up_th)[0]
    l_cluster = np.mean(full[ind_aux, 0])
    b_cluster = np.mean(full[ind_aux, 1])
    mul_cluster = np.mean(full[ind_aux, 2])
    mub_cluster = np.mean(full[ind_aux, 3])
    
    return np.sqrt( (l_real - l_cluster)**2 + (b_real - b_cluster)**2 ), \
           np.sqrt( (mul_real - mul_cluster)**2 + (mub_real - mub_cluster)**2 )
# +
color_cl = 'red'
color_full = 'darkcyan'
color_dw = 'coral'
color_ref = 'black'

labels = ['$l$ [°]','$b$ [°]','$\mu_{l}$ [mas/yr]','$\mu_{b}$ [mas/yr]']



def plotDwarf(ax, dw_data, full = None, ref = None, indices = None, nbins = 20):
    bins = [np.linspace(np.mean(dw_data[:,0]) - 2, np.mean(dw_data[:,0]) + 2, nbins),
            np.linspace(np.mean(dw_data[:,1]) - 2, np.mean(dw_data[:,1]) + 2, nbins),
            np.linspace(np.mean(dw_data[:,2]) - 3, np.mean(dw_data[:,2]) + 3, nbins),
            np.linspace(np.mean(dw_data[:,3]) - 3, np.mean(dw_data[:,3]) + 3, nbins)]

    lims = [(np.mean(dw_data[:,0]) - 2, np.mean(dw_data[:,0]) + 2),
            (np.mean(dw_data[:,1]) - 2, np.mean(dw_data[:,1]) + 2),
            (np.mean(dw_data[:,2]) - 3, np.mean(dw_data[:,2]) + 3),
            (np.mean(dw_data[:,3]) - 3, np.mean(dw_data[:,3]) + 3)]

    if indices is not None: 
        cluster = full[indices,:] 
    else: 
        cluster = None
    
    for i in range(4):
        for j in range(4):
            if i < j: ax[i,j].set_axis_off()
            if i > j:
                #ax[i,j].set_xlim(lims[j])
                ax[i,j].set_ylim(lims[i])
                if cluster is not None: ax[i,j].scatter(cluster[:,j], cluster[:,i], edgecolor = color_cl, label = 'Cl', 
                                                        marker = 'o', s = 20, facecolor = 'none')
                if ref is not None: ax[i,j].scatter(ref[:,j], ref[:,i], color = color_ref, label = 'Ref', marker = '.', s = 5)
                if full is not None: ax[i,j].scatter(full[:,j], full[:,i], color = color_full, label = 'Bkg', marker = '.', s = 5)
                ax[i,j].scatter(dw_data[:,j], dw_data[:,i], color = color_dw, label = 'Dwarf', marker = '+')
            if i == j:
                if ref is not None: ax[i,j].hist(ref[:,i], histtype = 'step', density = True, bins = bins[i], color = color_ref)
                if full is not None: ax[i,j].hist(full[:,i], histtype = 'step', density = True, bins = bins[i], color = color_full)
                ax[i,j].hist(dw_data[:,i], histtype = 'step', density = True, bins = bins[i], color = color_dw)
                if cluster is not None: ax[i,j].hist(cluster[:,i], histtype = 'step', density = True, bins = bins[i], color = color_cl)
            if i == 3: 
                ax[i,j].set_xlabel(labels[j])
            #else:
             #   ax[i,j].set_xticks([])
            if j == 0: 
                ax[i,j].set_ylabel(labels[i])
            else: 
                ax[i,j].set_yticks([])
        
    ax[0,0].set_xlim(lims[0])
    ax[3,1].set_xlim(lims[1])
    ax[3,2].set_xlim(lims[2])
    ax[3,3].set_xlim(lims[3])
    
    ax[1,0].legend(loc = 'upper right', bbox_to_anchor = (2,2))
    return ax


# -

# # Let's run EE on a dwarf at different positions

# +
igal = 96
data = np.loadtxt('../data/mocks/library.survey.UFGX_LIB' + str(igal) + '.dat')

i = np.random.randint(len(data))
Id = int(data[i,0])
ang_size = data[i,13]
vpec_size = data[i,14]

obs = np.loadtxt('../data/mocks/obs.lib' + str(igal) + '/UFGX_TEST' + str(igal) + '_lib/obs_' + str(Id) + '.dat')
print(len(obs))

# +
# Generate 100,000 Bernoulli sequences to determine the critical Upsilon threshold
num_sequences = 1000            # Number of Bernoulli sequences
K_M = 100                        # Length of each sequence
NUM_CORES = 40
NEIGHBOR_RANGE = range(4, K_M)
critical_quantile = 0.9999        # Quantile to calculate critical Upsilon threshold

# Generate a Bernoulli matrix with sequences of 0s and 1s
bernoulli_sequences = np.random.binomial(n=1, p=0.5, size=(num_sequences, K_M))

# Compute Upsilon values and optimal k-star values for Bernoulli sequences
p_value_data, upsilon_values_bernoulli, optimal_kstar_bernoulli = compute_upsilon_values(bernoulli_sequences, neighbor_range=NEIGHBOR_RANGE, num_cores=NUM_CORES)

# Calculate the critical Upsilon threshold at the specified quantile
critical_upsilon_threshold = np.quantile(upsilon_values_bernoulli, critical_quantile)


# +
latitudes = np.array([20,40,60])
longitudes = np.array([10, 60, 120])

positions = list(product(longitudes, latitudes))

print(positions)
# -


NPIX = hp.nside2npix(3)

glat, glon = np.degrees(hp.pix2ang(3, np.arange(NPIX)))

glat = 90 - glat

hp.pix2ang(3, np.arange(NPIX))

len(np.where(np.abs(glat) > 20)[0])

# +
pur      = []
falseNeg = []
dang     = []
dvpec    = []
bkg      = []
# #%glat     = []
# #%glon     = []

indices = []
upsilons = []
# #%for pos in positions:
#%    start = time()
#%    full, ref, dw_data = LoadBkg(obs, glon_center = pos[0], glat_center = pos[1],
#%                                 eps_lat = 0, eps_lon = 0, row_limit = None, size = 2)

for i in tqdm(np.where(np.abs(glat) > 20)[0]):
    start = time()
    full, ref, dw_data = LoadBkg(obs, glon_center = glon[i], glat_center = glat[i],
                                 eps_lat = 0, eps_lon = 0, row_limit = None, size = 2)

    ind = np.arange(len(full))[-len(dw_data):]
    # Generate and process binary sequences with anomalies
    binary_sequences_anomaly, _ = From_data_to_binary.create_binary_array_cdist(
        full, ref, num_neighbors = K_M, num_cores = NUM_CORES
    )
    anomaly_pval_info, upsilon, kstar_values_anomaly = compute_upsilon_values(
        binary_sequences_anomaly, NEIGHBOR_RANGE, NUM_CORES
    )

    upsilons.append(upsilon)
    # #%glon.append(pos[0])
    # #%glat.append(pos[1])
    up_th = np.quantile(upsilon, 0.8)
    indices.append(np.where(upsilon >= up_th)[0])
    pur.append(purity(upsilon, ind, up_th = up_th))
    falseNeg.append(FN(upsilon, ind, up_th = up_th))
    aux1, aux2 = Delta(full, upsilon, ind, up_th = up_th)
    dang.append(aux1)
    dvpec.append(aux2)
    bkg.append((len(full) - len(dw_data)))
    stop = time()
    #print('It takes {:.2f} hs to analyze a dwarf with {} stars at l = {:.2f} and b = {:.2f} with {} bkg stars'.format((stop-start)/3600, len(dw_data), pos[0], pos[1], len(full) - len(dw_data)))
    print('It takes {:.2f} hs to analyze a dwarf with {} stars at l = {:.2f} and b = {:.2f} with {} bkg stars'.format((stop-start)/3600, len(dw_data), glon[i], glat[i], len(full) - len(dw_data)))
# -

pur      = np.asarray(pur)
falseNeg = np.asarray(falseNeg)
dang     = np.asarray(dang)
dvpec    = np.asarray(dvpec)
bkg      = np.asarray(bkg)
glon     = np.asarray(glon)
glat     = np.asarray(glat)


# +
def transform_func(x):
    return nstars / x  # Ejemplo de transformación: duplicar valores

def inverse_transform_func(x):
    return x / nstars  # Inversa de la transformación


# +
nstars = len(dw_data)

fig,ax = plt.subplots(2,2, figsize = (6,6), sharex = True)
plt.subplots_adjust(hspace = 0., wspace = 0.4)

ax[0,0].text(0.9,1.2,'Gal {} ; {} stars'.format(Id, len(dw_data)), transform = ax[0,0].transAxes)

ax[0,0].scatter(len(dw_data) / bkg, pur)
#ax[0,0].set_xlabel('S/N')
ax[0,0].set_ylabel('Purity')
ax00 = ax[0,0].secondary_xaxis('top', functions=(transform_func, inverse_transform_func))
ax00.set_xlabel('# Bkg')

ax[0,1].scatter(len(dw_data) / bkg, falseNeg)
#ax[0,0].set_xlabel('S/N')
ax[0,1].set_ylabel('FN')
ax01 = ax[0,1].secondary_xaxis('top', functions=(transform_func, inverse_transform_func))
ax01.set_xlabel('# Bkg')

ax[1,0].scatter(len(dw_data) / bkg, dang)
ax[1,0].set_xlabel('S/N')
ax[1,0].set_ylabel('$\Delta_{ang}$')

ax[1,1].scatter(len(dw_data) / bkg, dvpec)
ax[1,1].set_xlabel('S/N')
ax[1,1].set_ylabel('$\Delta_{\mu}$')

#plt.savefig('../graph/gal_{}_ChangePositionStats.pdf'.format(Id), bbox_inches='tight')


# +
fig,ax = plt.subplots(2,2, figsize = (6,6), sharex = True)
plt.subplots_adjust(hspace = 0., wspace = 0.4)

im00 = ax[0,0].scatter(glat, pur, c = glon, cmap = 'viridis', vmin = 0, vmax = 180)
#ax[0,0].set_xlabel('$b [°]$')
cbar = plt.colorbar(im00, ax = ax[0,:], orientation = 'horizontal', pad = 0.1, location = 'top', shrink = 0.5)
cbar.set_label('$l$ [°]')
ax[0,0].set_ylabel('Purity')

ax[0,1].scatter(glat, falseNeg, c = glon, cmap = 'viridis', vmin = 0, vmax = 180)
#ax[0,1].set_xlabel('$b [°]$')
ax[0,1].set_ylabel('Missing stars Ratio')

ax[1,0].scatter(glat, dang, c = glon, cmap = 'viridis', vmin = 0, vmax = 180)
ax[1,0].set_xlabel('$b [°]$')
ax[1,0].set_ylabel('$\Delta_{ang}$')

im11 = ax[1,1].scatter(glat, dvpec, c = glon, cmap = 'viridis', vmin = 0, vmax = 180)
ax[1,1].set_xlabel('$b [°]$')
ax[1,1].set_ylabel('$\Delta_{\mu}$')

#plt.savefig('../graph/gal_{}_ChangePositionStats_Lat.pdf'.format(Id), bbox_inches='tight')

# +
grid = np.meshgrid(longitudes, latitudes)
# Crear el mapa en Mollweide
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection='mollweide')
sc = ax.scatter(glon * np.pi/180, glat * np.pi/180, c=pur, cmap='viridis', s=40)
# Personalización
plt.colorbar(sc, label='Purity')
ax.grid(True)

#plt.savefig('../graph/gal_{}_Map.pdf'.format(Id), bbox_inches='tight')

# +
fig, ax  = plt.subplots(4,4, figsize = (10,10), sharex = 'col')

plt.subplots_adjust(hspace = 0.2, wspace = 0.2)

ax = plotDwarf(ax, dw_data, full, ref, indices[8])

#plt.savefig('../graph/gal_{}_4d.pdf'.format(Id), bbox_inches='tight')
# -
data = np.zeros(NPIX)  # Array 1D de ejemplo
data[np.where(np.abs(glat) > 20)[0]] = 1

# Graficar el mapa en coordenadas Mollweide
hp.mollview(data, title="Mapa de Healpix", unit="Amplitud", cmap="viridis")
plt.show()


pixel_index = hp.ang2pix(2, glon * np.pi/180, glat * np.pi/180, nest=False)

data[pixel_index] = pur

data[pixel_index]

1



