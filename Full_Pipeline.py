#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage.color import rgb2ycbcr as r2y
from skimage.feature import local_binary_pattern as LBP
from sklearn.preprocessing import MinMaxScaler as MMS
from skimage.filters import apply_hysteresis_threshold as ht
from skimage.filters import gaussian
from skimage.morphology import remove_small_objects as rso
import time
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


##Check Salt and Pepper Levels
#if there is sufficient salt and pepper noise in image, then return True boolean value
def check_salt_n_pep(mask, noise_thresh = 0.15):
    salt_n_pep = False
    
    mask_size = mask.shape
    
    mask_pad = np.zeros((mask_size[0]+2,mask_size[1]+2))
    mask_pad[1:mask_size[0]+1,1:mask_size[1]+1] = mask.copy()

    salt_noise_cnt = 0

    for i in range(mask_size[0]):
        for j in range(mask_size[1]):
            xor_arr = np.ndarray.flatten(mask_pad[i:i+3,j:j+3])
            xor_elem = np.ones((len(xor_arr)))*mask_pad[i+1,j+1]
            ##Use Logical XOR to check neighborhood if pixels are connected or not (if noise or not)
            if(np.uint8(np.sum(np.logical_xor(xor_arr,xor_elem))) > 0):
                salt_noise_cnt += 1

    ##Check amount of salt and pepper noise to size of image
    img_tot_size = mask_size[0]*mask_size[1]
    salt_pep_noise_ratio = salt_noise_cnt/img_tot_size
    if(salt_pep_noise_ratio > noise_thresh):
        salt_n_pep = True
    else:
        salt_n_pep = False
        
    return salt_n_pep


# In[3]:


def apply_thresholds(img_gray, bins_regions, save_imgs = False, cd_save = './', plots = True, check_salt_n_pep_test = False):

    th, mask = cv2.threshold(img_gray.copy(), 0, 255, cv2.THRESH_OTSU)
    mask = 1-np.double(mask)
    mask = np.uint8(mask%2)
    
    regions = np.digitize(img_gray.copy(), bins = bins_regions).astype('uint8')
        
    if(check_salt_n_pep_test == True):
        if(check_salt_n_pep(mask) == True):
            print("Applying Median Filter due to Salt and Pepper Noise")
            print("If the application of this filter is not desired, mark input value of 'check_salt_n_pep_test' to be False")
            regions = cv2.medianBlur(regions.copy(),7)
            img_gray = cv2.medianBlur(img_gray,7)

    if(plots == True):
        if(save_imgs == True):
            cd_dir_regions = cd_save + 'clusts_region.png'
            cd_dir_hists = cd_save + '_hist_.png'
        
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20,3.5))

        ax[0].imshow(img_rgb)
        ax[1].hist(img_gray.ravel(), bins=255)
        ax[1].set_title('Histogram')
        for thresh in bins_regions:
            ax[1].axvline(thresh, color='r')
        mean_img = np.mean(img_gray.ravel())
        ax[1].axvline(mean_img, color='g')
        
        if(save_imgs == True):
            extent = ax[1].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(cd_dir_hists, bbox_inches = extent.expanded(1.2,1.2))
        
        ax[2].imshow(regions, cmap = 'jet')
        ax[2].set_axis_off()
    
        if(save_imgs == True):
            extent = ax[2].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(cd_dir_regions, bbox_inches = extent.expanded(1.0,1.0))
        
    return(regions)


# In[23]:


def get_mask_kmeans(img_rgb, num_clusters, process = 'intensity', wt_pen = 5, neighborhood = 8, salt_n_pep = True, radius = 1, METHOD = 'uniform'):
    centeronis = prep_data_fit_to_kmeans(img_rgb, num_clusters = num_clusters, process = process, wt_pen = wt_pen, neighborhood = neighborhood, salt_n_pep = salt_n_pep, radius = radius, METHOD = METHOD)
    print('Centeronis = ', centeronis)
    img_gray = cv2.cvtColor(img_rgb.copy(), cv2.COLOR_RGB2GRAY)
    regions = apply_thresholds(img_gray, centeronis, save_imgs = False, plots = False)
    hist = cv2.calcHist([img_gray],[0],None,[256],[0,256])
    ##Take the average of the grayscale histogram, in-order-to use as a threshold for grabbing clusters.
    ##We have the cluster centers, but now we must do something with them, so this allows us to grab the thresholds
    ##that are below the average of the histogram. I wanted to make it dynamic to each image, b/c the histograms
    ##of the images are different and may be skewed to one side or the other in terms of pixel intensities.
    mean_hist = 0
    for k in range(len(hist)):
        mean_hist = mean_hist + ((k+1)*hist[k])
    mean_hist = (mean_hist/(img_size[0]*img_size[1])) - 1
    
    regions_mask = regions.copy()
    idx = np.where(centeronis <= mean_hist)
    
    if not idx[0].all():
        #There are thresholds below the mean
        regions_mask[regions_mask <= max(idx[0])] = 0
    else:
        #There are not thresholds below the mean. Taking first cluster to mask
        regions_mask[regions_mask <= centeronis[0]] = 0
        
    regions_mask[regions_mask != 0] = 1
    mask = 1-regions_mask
    ##perform morphological OPEN in-order-to try and clean up the mask and get rid of pesky noise in our shadow mask
    SE = np.ones((5,5)).astype('uint8')
    mask_morph = cv2.morphologyEx(mask.copy(), cv2.MORPH_OPEN, SE).astype('uint8')

    return mask_morph


# In[34]:


def prep_data_fit_to_kmeans(img_rgb, num_clusters, process = 'intensity', wt_pen = 5, neighborhood = 8, salt_n_pep = True, radius = 1, METHOD = 'uniform'):
    from sklearn.cluster import KMeans
    ##returns centers of kmeans prediction (bins of histogram [thresholds])
    
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    if(salt_n_pep == True):
        th, mask = cv2.threshold(img_gray.copy(), 0, 255, cv2.THRESH_OTSU)
        mask = 1-np.double(mask)
        mask = np.uint8(mask%2)
        if(check_salt_n_pep(mask) == True):
            print("Applying Median Filter due to Salt and Pepper Noise")
            print("If the application of this filter is not desired, mark input value of 'check_salt_n_pep_test' to be False")
            img_gray = cv2.medianBlur(img_gray,7)
            img_rgb = cv2.medianBlur(img_rgb,7)
    
    #topographical data analysis
    #look at images with saturated pixel regions (very white)
    feat_vects = get_feature_vectors(img_rgb, process = process, neighborhood = neighborhood, radius = radius, METHOD = METHOD)
    kmeans = KMeans(n_clusters = num_clusters, n_jobs = -1)
    kmeans.fit(feat_vects)
    
    centers = kmeans.cluster_centers_*255
    centers_mean = np.zeros((centers.shape[0],1))
    for i in range(centers.shape[0]):
        centers_mean[i,0] = np.mean(centers[i,:].copy())
        
    #if thresholds dip below zero from wt_pen, then saturate at zero
    centers_mean[centers_mean < wt_pen] = wt_pen
    centers_ret = centers_mean[:,0] - wt_pen
    centers_ret.sort()
    
    return(centers_ret)


# In[19]:


class fvs:
    def __init__(self, img_rgb, neighborhood = 8, radius = 1, METHOD = 'uniform'):
        self.img_rgb = img_rgb
        self.img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        self.neighborhood = neighborhood
        self.radius = radius
        self.METHOD = METHOD
    
    def _intensity_(self):
        fvs = np.zeros((self.img_gray.shape[0]*self.img_gray.shape[1],self.neighborhood)).astype('uint8')
        img_gray_pad = np.pad(self.img_gray.copy(), 1, mode = 'edge')
        cnt = 0
        for i in range(1,self.img_gray.shape[1],1):
            for j in range(1,self.img_gray.shape[0],1):
                if(self.neighborhood == 8):
                    fvs[cnt,0] = img_gray_pad[j-1,i-1]
                    fvs[cnt,1] = img_gray_pad[j-1,i]
                    fvs[cnt,2] = img_gray_pad[j-1,i+1]
                    fvs[cnt,3] = img_gray_pad[j,i-1]
                    fvs[cnt,4] = img_gray_pad[j,i+1]
                    fvs[cnt,5] = img_gray_pad[j+1,i-1]
                    fvs[cnt,6] = img_gray_pad[j+1,i]
                    fvs[cnt,7] = img_gray_pad[j+1,i+1]
                elif(self.neighborhood == 4):
                    fvs[cnt,0] = img_gray_pad[j,i-1]
                    fvs[cnt,1] = img_gray_pad[j,i+1]
                    fvs[cnt,2] = img_gray_pad[j-1,i]
                    fvs[cnt,3] = img_gray_pad[j+1,i]
                cnt = cnt + 1

        fvs = MMS().fit_transform(fvs)
        ##returns vector that is neighborhood wide of data length
        return fvs
    
    def _gradient_(self):
        data = self.img_gray.ravel().copy()
        fvs = np.zeros((len(data),self.neighborhood)).astype('uint8')
        img_gray_pad = np.pad(self.img_gray.copy(), 1, mode = 'edge')
        cnt = 0
        for i in range(1,self.img_gray.shape[1],1):
            for j in range(1,self.img_gray.shape[0],1):
                if(self.neighborhood == 8):
                    fvs[cnt,0] = data[cnt] - img_gray_pad[j-1,i-1]
                    fvs[cnt,1] = data[cnt] - img_gray_pad[j-1,i]
                    fvs[cnt,2] = data[cnt] - img_gray_pad[j-1,i+1]
                    fvs[cnt,3] = data[cnt] - img_gray_pad[j,i-1]
                    fvs[cnt,4] = data[cnt] - img_gray_pad[j,i+1]
                    fvs[cnt,5] = data[cnt] - img_gray_pad[j+1,i-1]
                    fvs[cnt,6] = data[cnt] - img_gray_pad[j+1,i]
                    fvs[cnt,7] = data[cnt] - img_gray_pad[j+1,i+1]
                elif(self.neighborhood == 4):
                    fvs[cnt,0] = data[cnt] - img_gray_pad[j,i-1]
                    fvs[cnt,1] = data[cnt] - img_gray_pad[j,i+1]
                    fvs[cnt,2] = data[cnt] - img_gray_pad[j-1,i]
                    fvs[cnt,3] = data[cnt] - img_gray_pad[j+1,i]
                cnt = cnt + 1

        fvs = MMS().fit_transform(fvs)
        ##returns vector that is neighborhood wide of data length
        return fvs
    
    def _texture_(self):
        n_pts = 8*self.radius
        lbp_img = LBP(self.img_gray.copy(), n_pts, self.radius, self.METHOD)
        fvs = np.expand_dims(lbp_img.ravel(),1)

        fvs = MMS().fit_transform(fvs)
        ##returns vector that is the length of img_gray.shape[0]*img_gray.shape[1]
        return fvs
    
    def _ycbcr_mask(self):
        img_ycrcb = r2y(self.img_rgb)
        y_channel = img_ycrcb[:,:,0]
        mean = np.mean(y_channel.copy())
        std = np.std(y_channel.copy())
        y_channel[y_channel < (mean - std)] = 0
        y_channel[y_channel != 0] = 1
        y_channel[y_channel == 0] = 255
        y_channel[y_channel == 1] = 0

        cr_channel = img_ycrcb[:,:,1]
        mean = np.mean(cr_channel.copy())
        std = np.std(cr_channel.copy())
        cr_channel[cr_channel < (mean - std)] = 0
        cr_channel[cr_channel != 0] = 1
        cr_channel[cr_channel == 0] = 255
        cr_channel[cr_channel == 1] = 0

        y_cr_or = np.logical_and(y_channel, np.uint8(1 - cr_channel)).astype('uint8')
        y_cr_or[y_cr_or != 0] = 255

        cb_channel = img_ycrcb[:,:,2]
        mean = np.mean(cb_channel.copy())
        std = np.std(cb_channel.copy())
        cb_channel[cb_channel < (mean - std)] = 0
        cb_channel[cb_channel != 0] = 1
        cb_channel[cb_channel == 0] = 255
        cb_channel[cb_channel == 1] = 0

        cb_y_cr_or = np.logical_or(y_cr_or, cb_channel).astype('uint8')
        cb_y_cr_or[cb_y_cr_or != 0] = 1
        SE = np.ones((3,3)).astype('uint8')
        final_mask = cv2.morphologyEx(cb_y_cr_or.copy(), cv2.MORPH_OPEN, SE).astype('uint8')
        mask_on_img = np.multiply(final_mask, self.img_gray)
        fvs = np.expand_dims(mask_on_img.ravel(),1)
        
        fvs = MMS().fit_transform(fvs)
        ##returns vector the length of img_gray.shape[0]*img_gray.shape[1]
        return fvs


# In[32]:


def get_feature_vectors(img_rgb, process = 'intensity', neighborhood = 8, radius = 1, METHOD = 'uniform'):
    from skimage.color import rgb2ycbcr as r2y
    from skimage.feature import local_binary_pattern as LBP
    import cv2
    
    img_gray = cv2.cvtColor(img_rgb.copy(), cv2.COLOR_RGB2GRAY)
    feats = np.expand_dims(img_gray.ravel().copy(),1)
    feats = MMS().fit_transform(feats)
    get_features = fvs(img_rgb, neighborhood = neighborhood, radius = radius, METHOD = METHOD)
    if('intensity' in process):
        temp = get_features._intensity_()
        feats = np.concatenate((feats,temp), axis = 1)
    if('gradient' in process):
        temp = get_features._gradient_()
        feats = np.concatenate((feats,temp), axis = 1)
    if('texture' in process):
        temp = get_features._texture_()
        feats = np.concatenate((feats,temp), axis = 1)
    if('cb_channel' in process):
        temp = get_features._ycbcr_mask()
        feats = np.concatenate((feats,temp), axis = 1)
        
    return feats


# In[21]:


'''No multiprocessing'''
class spurious_responses():
    SE_3 = np.ones((3,3)).astype('uint8')
    SE_5 = np.ones((5,5)).astype('uint8')
    
    def __init__(self, mask, th = 0.05):
        self.mask = mask.astype('uint8')
        self.mask_pad = np.pad(mask, 1, mode = 'edge').astype('uint8')
        self.th = th
        
    def _fsp_(self):
        convergence = False

        while(convergence == False):
            tot_cnt = 0
            ##count number of occurrences of masked shadow in image ('ON' pixel)
            tot_masked_pixels = np.ndarray.tolist(self.mask.ravel()).count(1)
            for i in range(1,self.mask.shape[1]+1,1):
                for j in range(1,self.mask.shape[0]+1,1):
                    tot_cnt = tot_cnt + spurious_responses.filter_spurious_responses(self, i, j)
            inter_tot_masked_pixels = np.ndarray.tolist(self.mask.ravel()).count(1)
            
            ##get back any information that may have been lost from the main masked shadow pixels
            self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_CLOSE, self.SE_3).astype('uint8')
            if(tot_cnt/tot_masked_pixels <= self.th):
                convergence = True
            else:
                self.mask_pad = np.pad(self.mask, 1, mode = 'edge')
              
        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_CLOSE, self.SE_5).astype('uint8')
        return self

    def filter_spurious_responses(self, i, j):
        window = 0
        window = self.mask_pad[j-1,i-1]
        window = window + self.mask_pad[j-1,i]
        window = window + self.mask_pad[j-1,i+1]
        window = window + self.mask_pad[j,i-1]
        window = window + self.mask_pad[j,i+1]
        window = window + self.mask_pad[j+1,i-1]
        window = window + self.mask_pad[j+1,i]
        window = window + self.mask_pad[j+1,i+1]
        if(window < 5):
            '''filtering out mask_pad while we go is too destructive and erodes the mask too quickly'''
            #self.mask_pad[j,i] = 0
            ##if self.mask[j,i] = 0, return 0; if = 1, return 1
            ##multiply by one to get number from boolean
            ##due to padding, must offset mask indexing by padding number of 1
            ret_val = np.logical_and(1,self.mask[j-1,i-1].copy())*1
            self.mask[j-1,i-1] = 0
            return ret_val
        else:
            return 0


# In[8]:


def ycbcr_shadow_correction(img_rgb, mask):
    from skimage.color import rgb2ycbcr as r2y
    import cv2
    import numpy as np
    '''
    Inputted image should be RGB
    
    shadow_core is a mask of the image where the shadow regions are
    a 1 and non-shadow regions are a 0.
    
    lit_core will be taken care of within the function
    '''
    shadow_core = mask.copy()
    lit_core = np.uint8(1-mask.copy())
    
    skim_ycrcb = r2y(img_rgb)
    skim_ycbcr = skim_ycrcb[:,:,(0,2,1)].astype('uint8')
    ycrcb = skim_ycrcb

    shadow_core_sum = sum(np.double(sum(np.double(shadow_core))))
    lit_core_sum = sum(np.double(sum(np.double(lit_core))))

    ##Calculate average illumination, blue-chrominance, red-chrominance for the shadow regions
    shadowavg_y = sum(np.double(sum(np.double(ycrcb[:,:,0]*np.uint8(shadow_core)))))/shadow_core_sum
    shadowavg_cb = sum(np.double(sum(np.double(ycrcb[:,:,1]*np.uint8(shadow_core)))))/shadow_core_sum
    shadowavg_cr = sum(np.double(sum(np.double(ycrcb[:,:,2]*np.uint8(shadow_core)))))/shadow_core_sum

    ##Calculate average illumination, blue-chrominance, red-chrominance for the lit regions (not shadows)
    litavg_y = sum(np.double(sum(np.double(ycrcb[:,:,0]*np.uint8(lit_core)))))/lit_core_sum
    litavg_cb = sum(np.double(sum(np.double(ycrcb[:,:,1]*np.uint8(lit_core)))))/lit_core_sum
    litavg_cr = sum(np.double(sum(np.double(ycrcb[:,:,2]*np.uint8(lit_core)))))/lit_core_sum

    ##Calculate the difference between the lit region averages and the shadow region averages
    diff_y = litavg_y - shadowavg_y;
    diff_cb = litavg_cb - shadowavg_cb;
    diff_cr = litavg_cr - shadowavg_cr;

    ##Calculate the ratios between the lit region averages and the shadow region averages
    ratio_y = litavg_y/shadowavg_y
    ratio_cb = litavg_cb/shadowavg_cb
    ratio_cr = litavg_cr/shadowavg_cr

    res_ycrcb = ycrcb.copy()

    ##Y-channel
    Y_interm_mult = np.multiply(np.double(shadow_core),diff_y)
    #np.around needed to round up b/c MATLAB does
    Y_intermediate = np.around(np.add(ycrcb[:,:,0],Y_interm_mult))
    res_ycrcb[:,:,0] = Y_intermediate

    ##Cb-channel
    Cb_interm_mults = np.around(ratio_cb*np.multiply(np.double(shadow_core),np.double(ycrcb[:,:,1])))
    Cb_interm_ycbcr_mask = np.around(np.multiply(ycrcb[:,:,1],lit_core))
    Cb_intermediate = np.add(Cb_interm_ycbcr_mask,Cb_interm_mults)
    res_ycrcb[:,:,1] = np.squeeze(Cb_intermediate)

    ##Cr-channel
    Cr_interm_mults = np.around(ratio_cr*np.multiply(np.double(shadow_core),np.double(ycrcb[:,:,2])))
    Cr_interm_ycbcr_mask = np.around(np.multiply(ycrcb[:,:,2],lit_core))
    Cr_intermediate = np.add(Cr_interm_ycbcr_mask,Cr_interm_mults)
    res_ycrcb[:,:,2] = np.squeeze(Cr_intermediate)

    res_ycrcb = res_ycrcb

    res_ycrcb = res_ycrcb[:,:,(0,2,1)]
    result = cv2.cvtColor(np.uint8(res_ycrcb), cv2.COLOR_YCrCb2BGR)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    return(result)


# In[39]:


def theta_color_correction(shadow_corrected_img, mask, save_ccimg_deg = False, cd_save = './ccimg_deg_', SNR_return = False, save_SNR = False, cd_SNR = './SNR_img'):
    import cv2
    from scipy.signal import find_peaks as fp
    from functools import reduce
    import numpy as np
    import math
    '''
    Color Correction Using Degrees:

    I_i^(shadow free) = ((r + 1)/(k_i*r + 1))I_i

    r = L_d/L_e; L_d = intensity of non-shadow pixels
                 L_e = intensity of shadow pixels
             
    k_i = t_i*cos(theta_i); t_i = 1 if object in sunshine region
                                  and 0 if object in shadow region
                                  (attenuation factor of direct light)
                            theta_i = angle between direct lighting
                                  direction and the surface normal
                                  
    I_i = pixel at i^th value

    mask already gives us if object is in a sunshine region,
    where t_i = 1, and if object is in a shadow region, where
    t_i = 0.
    Then iterate through varying values of theta to see how these
    affect the outcome of the color correction to the image.

    Need to find L_d and L_e, which seem to be global variables.
    This doesn't seem like an especially smart move as there will
    be many different intensities throughout the image, especially
    with a very busy image.


    Purpose of k_i is that if point is in a direct sunshine region, no angle between direct light and ambient light, then
    that pixels value will not change as k_i will equal 1 due to being in a sunshine region and having a theta_i = 0. If
    a pixel is in a shadow region, then pixel intensity should be brought up slightly to match sunshine region.
    '''
    
    ##shadow_corrected_image should come from ycbcr_color_correction function
    
    ##mask is needed for SNR calculation
    
    ##save_ccimg_deg is optional to save all of the output pictures
    
    ##cd_save is where you want to save the 181 output pictures, as this function
    ##runs through all the angles from 0 to 180 to test on further correcting the colors of the image
    
    ##SNR_return will allow a return of the SNR calculated for each image
    
    
    ##Convert output image to HSI space to get purely intensity values of shadow to non-shadow pixels
    result_HSI = cv2.cvtColor(shadow_corrected_img.copy(), cv2.COLOR_RGB2HSV)
    
    ##for SNR
    sf_SNR = []

    mask_rep = np.repeat(mask[:,:,np.newaxis],3,axis=2)
    
    ##range of degrees from 0 to 180
    deg_range = 181
    for i in range(deg_range):
        ##intensity of shadow pixels
        L_e_I = np.sum(np.multiply(result_HSI[:,:,2],mask.copy()))/(mask == 1).sum()
        ##intensity of non-shadow pixels
        L_d_I = np.sum(np.multiply(result_HSI[:,:,2],np.logical_not(mask.copy())))/(mask == 0).sum()
        ##ratio of non-shadow pixel intensities to shadow pixel intensities
        r_I = L_d_I/L_e_I
    
        k_i = mask.copy()*math.cos(math.radians(i))
        k_i[k_i == 0] = 1
        result_shadow_free_I = np.uint8(((r_I+1)/(k_i*r_I + 1))*result_HSI[:,:,2].copy())
        result_HSI[:,:,2] = result_shadow_free_I.copy()
        result_sf = cv2.cvtColor(result_HSI.copy(), cv2.COLOR_HSV2RGB)
        result_sf = cv2.cvtColor(result_sf, cv2.COLOR_BGR2RGB)
        
        ##note that any pixels within the shadow region that are intrinsically
        ##zero will be discarded using this methodology, but this was the only way
        ##I figured to calculate SNR, specifically in the mask areas
        result_sf_mask_area_mult = np.multiply(result_sf.copy(), mask_rep)
        result_sf_mask_area_flat = result_sf_mask_area_mult.flat
        result_sf_mask_area_fl8 = result_sf_mask_area_flat[result_sf_mask_area_flat != 0]
        
        sf_mean = np.mean(result_sf_mask_area_fl8.copy())
        sf_std = np.std(result_sf_mask_area_fl8.copy())
        ratio = sf_mean/sf_std
        sf_SNR.append(ratio)
    
        if save_ccimg_deg == True:
            ccimg_deg = cd_save + str(i) + '.jpg'
            cv2.imwrite(ccimg_deg,result_sf)
    
    ##finds mean of sf_SNR and divides by max value of sf_SNR to get unique threshold
    thresh = (reduce(lambda x, y: x+y, sf_SNR)/181)/(np.amax(sf_SNR))
    SNR_peaks = fp(sf_SNR, height = np.amax(sf_SNR) - thresh, distance = 10)      
    
    degrees = np.linspace(0,180,181)
    plt.plot(degrees, sf_SNR)
    plt.xlabel('degrees')
    plt.ylabel('SNR')
    plt.title('SNR vs degrees of ambient light to reflected light')
    
    for i in range(SNR_peaks[0].shape[0]):
        y_max = sf_SNR[SNR_peaks[0][i]]
        x_max = SNR_peaks[0][i]
        text_max = "x={:.3f}".format(x_max)
        plt.annotate(text_max, xy=(x_max,y_max))
        
    y_min = np.amin(sf_SNR)
    x_min = sf_SNR.index(min(sf_SNR))
    text_min = "x={:.3f}".format(x_min)
    plt.annotate(text_min, xy=(x_min,y_min))
    
    if save_SNR == True:
        plt.savefig(cd_SNR + '.png', bbox_inches = 'tight')
    
    if SNR_return == True:
        return sf_SNR


# In[11]:


img_name = '92-1.png'
img_name_no_file_class = img_name.replace(".jpg","")
img_name_no_file_class = img_name_no_file_class.replace(".png","")
img_cd = 'C:/Users/Chris/Documents/Assistant_Research/Research_Papers/Shadows/Python/Code_2/Untitled Folder/' +img_name
img = cv2.imread(img_cd)
img_rgb = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
img_size = img.shape

plt.imshow(img_rgb)


# In[14]:


'''
mask_name = '92-1.png'
mask_name_no_file_class = img_name.replace(".jpg","")
mask_name_no_file_class = img_name_no_file_class.replace(".png","")
mask_cd = 'C:/Users/Chris/Documents/Assistant_Research/Research_Papers/Shadows/4_16to4_22/Pics/' +mask_name_no_file_class+ '/gradient/8_mask_no_mp.jpg'
mask_import = cv2.imread(mask_cd)
mask_imp = mask_import[:,:,0]
##to get all 'ON' pixels to have a value of 1, as when saving mask it alters the 'ON' pixels values for some reason
mask_imp[mask_imp < 255] = 0
mask_imp[mask_imp > 0] = 1
plt.imshow(mask_imp, cmap = 'gray')
'''
pass


# In[35]:


cluster_size = 8
process = 'gradient'
salt_n_pep = True
METHOD = 'uniform'

img_rgb_copy = img_rgb.copy()
mask = get_mask_kmeans(img_rgb, num_clusters = cluster_size, process = process, salt_n_pep = salt_n_pep, METHOD = METHOD)

#stopping threshold for convergence of filtering spurious responses from image
#may need to be adjusted for larger images, haven't really tested it on very large images.
th = 0.05
sr = spurious_responses(mask, th)
sr_mask_filt = sr._fsp_()
mask_filt = sr_mask_filt.mask

final_output = ycbcr_shadow_correction(img_rgb_copy, mask_filt)
plt.imshow(final_output)


# In[40]:


#wherever you want to save the output images to, there will be 181, so know that
cd_save = 'C:/Users/Chris/Documents/Assistant_Research/Research_Papers/Shadows/4_23to4_29/Pics/' +img_name_no_file_class+ '/ccimg_deg_'
#returns SNR plot for all 181 images (1 plot)
cd_SNR = 'C:/Users/Chris/Documents/Assistant_Research/Research_Papers/Shadows/4_23to4_29/Pics/' +img_name_no_file_class+ '/SNR_img'
theta_color_correction(final_output, mask_filt, save_ccimg_deg = True, cd_save = cd_save, SNR_return = True, save_SNR = True, cd_SNR = cd_SNR)


# In[ ]:




