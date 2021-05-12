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


# In[4]:


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


# In[5]:


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


# In[6]:


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


# In[7]:


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


# In[8]:


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


# In[17]:


img_name = '119-11.png'
img_name_no_file_class = img_name.replace(".jpg","")
img_name_no_file_class = img_name_no_file_class.replace(".png","")
img_cd = 'C:/Users/Chris/Documents/Assistant_Research/Research_Papers/Shadows/Python/Code_2/Untitled Folder/' +img_name
img = cv2.imread(img_cd)
img_rgb = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
img_size = img.shape

plt.imshow(img_rgb)


# In[18]:


process = ['cb_channel','cb_channel_and_gradient','cb_channel_and_gradient_and_texture','cb_channel_and_intensity','cb_channel_and_intensity_and_gradient','cb_channel_and_intensity_and_gradient_and_texture','cb_channel_and_intensity_and_texture','cb_channel_and_texture','gradient','gradient_and_texture','intensity','intensity_and_gradient','intensity_and_gradient_and_texture','intensity_and_texture','texture']
cluster_size = np.arange(3,10)
salt_n_pep = True
METHOD = 'uniform'

for i in range(len(process)):
    print('Process = ', process[i])
    cd_save_mask = 'C:/Users/Chris/Documents/Assistant_Research/Research_Papers/Shadows/4_30to5_7/Pics/' +img_name_no_file_class+ '/Test_Masks/' +process[i]+ '/'
    for j in range(len(cluster_size)):
        mask = get_mask_kmeans(img_rgb, num_clusters = cluster_size[j], process = process[i], salt_n_pep = salt_n_pep, METHOD = METHOD)
        mask_save = mask.copy()*255
        cd_save_mask_clust = cd_save_mask + str(cluster_size[j])+ '_mask.jpg'
        print(cd_save_mask_clust)
        cv2.imwrite(cd_save_mask_clust, mask_save)


# In[ ]:




