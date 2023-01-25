from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
import astropy.units as u
import astropy.utils as utils
from astropy.nddata import Cutout2D
import numpy as np
from operator import itemgetter, attrgetter
from scipy.optimize import curve_fit    

center_sky_x = 0.
center_sky_y = 0.
delta_sky_x = 3./2.
delta_sky_y = 3./2.
start_pix_e = 0
end_pix_e = 50

# Our function to fit is going to be a sum of two-dimensional Gaussians
def gaussian(x, y, x0, y0, sigma_core, A_core, sigma_tail, a_tail):
    core_func = A_core * np.exp(-((x-x0)**2+(y-y0)**2)/(2*sigma_core*sigma_core))                 /(2*np.pi*sigma_core*sigma_core)
    tail_func = A_core* a_tail *                 np.exp(-((x-x0)**2+(y-y0)**2)/(2*(sigma_tail+sigma_core)*(sigma_tail+sigma_core)))                 /(2*np.pi*(sigma_tail+sigma_core)*(sigma_tail+sigma_core))
    return core_func + tail_func

def king_function(x,y,x0,y0,sigma,gamma,A):
    return A * 1/(2*np.pi*sigma**2)*(1.-1./gamma)*pow(1.+((x-x0)**2+(y-y0)**2)/(2.*gamma*sigma**2),-gamma)
                                       
# https://scipython.com/blog/non-linear-least-squares-fitting-of-a-two-dimensional-data/
# This is the callable that is passed to curve_fit. M is a (2,N) array
# where N is the total number of data points in Z, which will be ravelled
# to one dimension.
def _gaussian(M, *args):
    x, y = M
    arr = np.zeros(x.shape)
    for i in range(len(args)//6):
        arr += gaussian(x, y, *args[i*6:i*6+6])
    return arr

def _king_function(M, *args):
    x, y = M
    arr = np.zeros(x.shape)
    for i in range(len(args)//5):
        arr += king_function(x, y, *args[i*5:i*5+5])
    return arr

def simulation_dict():
    #layout = 'mult3_zd20_f4'
    layout = 'mult3_zd20_m2'
    exposure_hour = '2'
    data_folder = '/content/drive/MyDrive/Colab_Notebooks_Extended_Ana/cta_sim_data/output_tehanu'

    bkg_folder_name = '%s/CTAsim_gps_%s_%shr_bkg'%(data_folder,layout,exposure_hour)
    
    src_extension = []
    src_extension += ['r0p005']
    src_extension += ['r0p010']
    src_extension += ['r0p015']
    src_extension += ['r0p020']
    src_extension += ['r0p025']
    src_extension += ['r0p030']
    src_extension += ['r0p035']
    src_extension += ['r0p040']
    src_folder_name = []
    for rb in range(0,len(src_extension)):
        src_folder_name += ['%s/CTAsim_gps_%s_%shr_pwn_%s'%(data_folder,layout,exposure_hour,src_extension[rb])]
        
    folder_dict = {}
    folder_dict['bkg'] = bkg_folder_name
    for rb in range(0,len(src_extension)):
        folder_dict[src_extension[rb]] = src_folder_name[rb]

    return folder_dict

def load_image(folder_name):

    filename = '%s/cntcube_emin=0.10_emax=10.00_side=3.00.fits'%(folder_name)
    hdu = fits.open(filename)[0]
    wcs = WCS(hdu.header)
    image_data = hdu.data

    pixs_start = wcs.all_world2pix(center_sky_x+delta_sky_x,center_sky_y-delta_sky_y,start_pix_e,1)
    pixs_end = wcs.all_world2pix(center_sky_x-delta_sky_x,center_sky_y+delta_sky_y,end_pix_e,1)
    start_pix_x = int(pixs_start[0])
    start_pix_y = int(pixs_start[1])
    end_pix_x = int(pixs_end[0])
    end_pix_y = int(pixs_end[1])
    
    nbins_x = end_pix_x - start_pix_x
    nbins_y = end_pix_y - start_pix_y
    
    image_data_reduced = np.full((image_data[start_pix_e, :, :].shape),0.)
    for pix in range(start_pix_e,end_pix_e):
        image_data_reduced += image_data[pix, :, :]

    return wcs, image_data_reduced

def plot_king_function(src_wcs,src_image_data_residual,par_x0=0.,par_y0=0.,par_sigma=0.05,par_gamma=2.0,par_A=1.):

    pixs_start = src_wcs.all_world2pix(center_sky_x+delta_sky_x,center_sky_y-delta_sky_y,start_pix_e,1)
    pixs_end = src_wcs.all_world2pix(center_sky_x-delta_sky_x,center_sky_y+delta_sky_y,end_pix_e,1)
    start_pix_x = int(pixs_start[0])
    start_pix_y = int(pixs_start[1])
    end_pix_x = int(pixs_end[0])
    end_pix_y = int(pixs_end[1])
    nbins_x = end_pix_x - start_pix_x
    nbins_y = end_pix_y - start_pix_y
    
    lon_min, lat_min, energy = src_wcs.all_pix2world(0, 0, 0, 1)
    if lon_min>180.:
        lon_min = lon_min-360.
    lon_max, lat_max, energy = src_wcs.all_pix2world(nbins_x-1, nbins_y-1, 0, 1)
    if lon_max>180.:
        lon_max = lon_max-360.
    x_axis = np.linspace(lon_max,lon_min,nbins_x)
    y_axis = np.linspace(lat_min,lat_max,nbins_y)
    X_grid, Y_grid = np.meshgrid(x_axis, y_axis)
    # We need to ravel the meshgrids of X, Y points to a pair of 1-D arrays.
    XY_stack = np.vstack((X_grid.ravel(), Y_grid.ravel()))
    
    fit_2d_point = np.zeros(src_image_data_residual.shape)
    fit_2d_point += king_function(X_grid, Y_grid, par_x0,par_y0,par_sigma,par_gamma,par_A)
        
    n_bins_r = 10
    radial_range = 0.25
    radial_axis = []
    point_in_slice = []
    bins_in_slice = []
    for rb in range(0,n_bins_r):
        radial_axis += [float(rb)*radial_range/float(n_bins_r)]
        point_in_slice += [0.]
        bins_in_slice += [0.]
        for binx in range(0,nbins_x):
            for biny in range(0,nbins_y):
                lon, lat, energy = src_wcs.all_pix2world(binx, biny, 0, 1)
                point_cnt = fit_2d_point[biny,binx]
                delta_x = abs(lon-center_sky_x)
                if delta_x>180.:
                    delta_x = 360.-delta_x
                delta_y = abs(lat-center_sky_y)
                dist_to_pwn = pow(pow(delta_x,2)+pow(delta_y,2),0.5)
                if dist_to_pwn>=radial_axis[rb] and dist_to_pwn<radial_axis[rb]+radial_range/float(n_bins_r):
                    point_in_slice[rb] += (point_cnt)
                    bins_in_slice[rb] += 1.
        if bins_in_slice[rb]>0.:
            point_in_slice[rb] = point_in_slice[rb]/(bins_in_slice[rb])
    
    return radial_axis, point_in_slice


def fit_king_function_to_data(src_wcs,src_image_data_residual,bkg_image_data_reduced,initial_prms,bound_upper_prms,bound_lower_prms):

    pixs_start = src_wcs.all_world2pix(center_sky_x+delta_sky_x,center_sky_y-delta_sky_y,start_pix_e,1)
    pixs_end = src_wcs.all_world2pix(center_sky_x-delta_sky_x,center_sky_y+delta_sky_y,end_pix_e,1)
    start_pix_x = int(pixs_start[0])
    start_pix_y = int(pixs_start[1])
    end_pix_x = int(pixs_end[0])
    end_pix_y = int(pixs_end[1])
    nbins_x = end_pix_x - start_pix_x
    nbins_y = end_pix_y - start_pix_y
    
    # Flatten the initial guess parameter list.
    p0 = [p for prms in initial_prms for p in prms]
    p0_lower = [p for prms in bound_lower_prms for p in prms]
    p0_upper = [p for prms in bound_upper_prms for p in prms]
    
    lon_min, lat_min, energy = src_wcs.all_pix2world(0, 0, 0, 1)
    if lon_min>180.:
        lon_min = lon_min-360.
    lon_max, lat_max, energy = src_wcs.all_pix2world(nbins_x-1, nbins_y-1, 0, 1)
    if lon_max>180.:
        lon_max = lon_max-360.
    x_axis = np.linspace(lon_max,lon_min,nbins_x)
    y_axis = np.linspace(lat_min,lat_max,nbins_y)
    X_grid, Y_grid = np.meshgrid(x_axis, y_axis)
    # We need to ravel the meshgrids of X, Y points to a pair of 1-D arrays.
    XY_stack = np.vstack((X_grid.ravel(), Y_grid.ravel()))
    popt, pcov = curve_fit(_king_function, XY_stack, src_image_data_residual.ravel(), p0, bounds=(p0_lower,p0_upper))
    
    fit_A = popt[4]
    fit_sigma = popt[2]
    fit_gamma = popt[3]
    #print ('fit sigma = %s'%(popt[2]))
    #print ('fit gamma = %s'%(popt[3]))
    
    fit_2d_point = np.zeros(src_image_data_residual.shape)
    for i in range(len(popt)//5):
        fit_2d_point += king_function(X_grid, Y_grid, *popt[i*5:i*5+5])
        
    n_bins_r = 10
    radial_range = 0.25
    radial_axis = []
    evts_in_slice = []
    point_in_slice = []
    errs_in_slice = []
    bins_in_slice = []
    for rb in range(0,n_bins_r):
        radial_axis += [float(rb)*radial_range/float(n_bins_r)]
        evts_in_slice += [0.]
        point_in_slice += [0.]
        errs_in_slice += [0.]
        bins_in_slice += [0.]
        for binx in range(0,nbins_x):
            for biny in range(0,nbins_y):
                lon, lat, energy = src_wcs.all_pix2world(binx, biny, 0, 1)
                point_cnt = fit_2d_point[biny,binx]
                src_cnt = src_image_data_residual[biny,binx]
                bkg_cnt = bkg_image_data_reduced[biny,binx]
                delta_x = abs(lon-center_sky_x)
                if delta_x>180.:
                    delta_x = 360.-delta_x
                delta_y = abs(lat-center_sky_y)
                dist_to_pwn = pow(pow(delta_x,2)+pow(delta_y,2),0.5)
                if dist_to_pwn>=radial_axis[rb] and dist_to_pwn<radial_axis[rb]+radial_range/float(n_bins_r):
                    point_in_slice[rb] += (point_cnt)
                    evts_in_slice[rb] += (src_cnt)
                    errs_in_slice[rb] += (bkg_cnt)
                    bins_in_slice[rb] += 1.
        if bins_in_slice[rb]>0.:
            point_in_slice[rb] = point_in_slice[rb]/(bins_in_slice[rb])
            evts_in_slice[rb] = evts_in_slice[rb]/(bins_in_slice[rb])
            errs_in_slice[rb] = pow(errs_in_slice[rb],0.5)/(bins_in_slice[rb])
            
    chi2_integral_radius = 0.20
    chi2_fit = 0.
    fit_2d_point = np.zeros(src_image_data_residual.shape)
    for i in range(len(popt)//5):
        fit_2d_point += king_function(X_grid, Y_grid, *popt[i*5:i*5+5])

    for binx in range(0,nbins_x):
        for biny in range(0,nbins_y):
            lon, lat, energy = src_wcs.all_pix2world(binx, biny, 0, 1)
            fit_cnt = fit_2d_point[biny,binx]
            src_cnt = src_image_data_residual[biny,binx]
            bkg_cnt = bkg_image_data_reduced[biny,binx]
            delta_x = abs(lon-center_sky_x)
            if delta_x>180.:
                delta_x = 360.-delta_x
            delta_y = abs(lat-center_sky_y)
            dist_to_pwn = pow(pow(delta_x,2)+pow(delta_y,2),0.5)
            if dist_to_pwn>chi2_integral_radius: continue
            if bkg_cnt==0.: continue
            chi2_fit += pow(src_cnt-fit_cnt,2)/bkg_cnt
    
    return fit_sigma, fit_gamma, fit_A, chi2_fit, radial_axis, point_in_slice, evts_in_slice, errs_in_slice

def load_psf_param(src_wcs,src_image_data_residual,bkg_image_data_reduced):

    init_gal_l = 0.
    init_gal_b = 0.
    init_sigma = 0.1
    init_gamma = 1.5
    init_norm = 100.
    
    # set initial values and bounds
    initial_prms = []
    bound_upper_prms = []
    bound_lower_prms = []
    lon = init_gal_l
    lat = init_gal_b
    if lon>180.:
        lon = lon-360.
    initial_prms += [(-lon,lat,init_sigma,init_gamma,init_norm)]
    bound_lower_prms += [(-lon-0.5,lat-0.5,0.,1.,0.)]
    bound_upper_prms += [(-lon+0.5,lat+0.5,init_sigma+3.0,1e10,1e10)]
    
    point_sigma, point_gamma, point_A, point_chi2, radial_axis, point_in_slice, evts_in_slice, errs_in_slice = \
    fit_king_function_to_data(src_wcs,src_image_data_residual,bkg_image_data_reduced,initial_prms,bound_upper_prms,bound_lower_prms)

    return point_sigma, point_gamma

def extended_src_fit(src_wcs,src_image_data_residual,bkg_image_data_reduced, init_sigma, init_gamma):

    init_gal_l = 0.
    init_gal_b = 0.
    init_norm = 100.  
    
    # set initial values and bounds
    initial_prms = []
    bound_upper_prms = []
    bound_lower_prms = []
    lon = init_gal_l
    lat = init_gal_b
    if lon>180.:
        lon = lon-360.
    initial_prms += [(-lon,lat,init_sigma,init_gamma,init_norm)]
    bound_lower_prms += [(-lon-0.5,lat-0.5,0.,1.,0.)]
    bound_upper_prms += [(-lon+0.5,lat+0.5,init_sigma+3.0,1e10,1e10)]
    
    point_sigma, point_gamma, point_A, point_chi2, radial_axis, point_in_slice, evts_in_slice, errs_in_slice = \
    fit_king_function_to_data(src_wcs,src_image_data_residual,bkg_image_data_reduced,initial_prms,bound_upper_prms,bound_lower_prms)

    return radial_axis, point_in_slice

def point_src_fit(src_wcs,src_image_data_residual,bkg_image_data_reduced, init_sigma, init_gamma):

    init_gal_l = 0.
    init_gal_b = 0.
    init_norm = 100.  
    
    # set initial values and bounds
    initial_prms = []
    bound_upper_prms = []
    bound_lower_prms = []
    lon = init_gal_l
    lat = init_gal_b
    if lon>180.:
        lon = lon-360.
    initial_prms += [(-lon,lat,init_sigma,init_gamma,init_norm)]
    bound_lower_prms += [(-lon-0.5,lat-0.5,init_sigma-1e-10,init_gamma-1e-10,0.)]
    bound_upper_prms += [(-lon+0.5,lat+0.5,init_sigma+1e-10,init_gamma+1e-10,1e10)]
    
    point_sigma, point_gamma, point_A, point_chi2, radial_axis, point_in_slice, evts_in_slice, errs_in_slice = \
    fit_king_function_to_data(src_wcs,src_image_data_residual,bkg_image_data_reduced,initial_prms,bound_upper_prms,bound_lower_prms)

    return radial_axis, point_in_slice

def get_normalization(data,fit_func):
    
    data_norm = 0.
    fit_norm = 0.
    for x in range(0,len(data)):
        data_norm += data[x]
        fit_norm += fit_func[x]
    scale = data_norm/fit_norm
    
    return scale

def get_radial_profile(src_wcs,src_image_data_residual,bkg_image_data_reduced):
    
    pixs_start = src_wcs.all_world2pix(center_sky_x+delta_sky_x,center_sky_y-delta_sky_y,start_pix_e,1)
    pixs_end = src_wcs.all_world2pix(center_sky_x-delta_sky_x,center_sky_y+delta_sky_y,end_pix_e,1)
    start_pix_x = int(pixs_start[0])
    start_pix_y = int(pixs_start[1])
    end_pix_x = int(pixs_end[0])
    end_pix_y = int(pixs_end[1])
    nbins_x = end_pix_x - start_pix_x
    nbins_y = end_pix_y - start_pix_y
    
    n_bins_r = 10
    radial_range = 0.25
    radial_axis = []
    evts_in_slice = []
    errs_in_slice = []
    bins_in_slice = []
    for rb in range(0,n_bins_r):
        radial_axis += [float(rb)*radial_range/float(n_bins_r)]
        evts_in_slice += [0.]
        errs_in_slice += [0.]
        bins_in_slice += [0.]
        for binx in range(0,nbins_x):
            for biny in range(0,nbins_y):
                lon, lat, energy = src_wcs.all_pix2world(binx, biny, 0, 1)
                src_cnt = src_image_data_residual[biny,binx]
                bkg_cnt = bkg_image_data_reduced[biny,binx]
                delta_x = abs(lon-center_sky_x)
                if delta_x>180.:
                    delta_x = 360.-delta_x
                delta_y = abs(lat-center_sky_y)
                dist_to_pwn = pow(pow(delta_x,2)+pow(delta_y,2),0.5)
                if dist_to_pwn>=radial_axis[rb] and dist_to_pwn<radial_axis[rb]+radial_range/float(n_bins_r):
                    evts_in_slice[rb] += (src_cnt)
                    errs_in_slice[rb] += (bkg_cnt)
                    bins_in_slice[rb] += 1.
        if bins_in_slice[rb]>0.:
            evts_in_slice[rb] = evts_in_slice[rb]/(bins_in_slice[rb])
            errs_in_slice[rb] = pow(errs_in_slice[rb],0.5)/(bins_in_slice[rb])
            
    return radial_axis, evts_in_slice, errs_in_slice