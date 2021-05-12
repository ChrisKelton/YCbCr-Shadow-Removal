#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2ycbcr as r2y
from scipy.signal import convolve2d as conv2
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


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


# In[3]:


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


# In[33]:


def double_otsu(img_rgb, plots = False):
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    th, mask = cv2.threshold(img_gray.copy(), 0, 255, cv2.THRESH_OTSU)
    mask = 1-np.double(mask)
    mask = np.uint8(mask%2)

    if(check_salt_n_pep(mask) == True):
        img_rgb = cv2.medianBlur(img_rgb,7)
        img_gray_filt = cv2.medianBlur(img_gray.copy(),7)
        th, mask = cv2.threshold(img_gray_filt, 0, 255, cv2.THRESH_OTSU)
        mask = 1-np.double(mask)
        mask = np.uint8(mask%2)

    ##Beginning of Second Iteration Of Otsu's Threshold
    mask = 1 - mask
    mask_sub = mask*127
    ##Want to lessen intensity of pixels that are biasing the mask due to their high intensity values
    img_hsi = cv2.cvtColor(img_rgb.copy(), cv2.COLOR_RGB2HSV)
    img_darkened_hsi = np.subtract(img_hsi[:,:,2].copy(),mask_sub)
    img_hsi[:,:,2] = img_darkened_hsi
    img_rgb_darkened = cv2.cvtColor(img_hsi, cv2.COLOR_HSV2RGB)
    gray_darkened = cv2.cvtColor(img_rgb_darkened, cv2.COLOR_RGB2GRAY)

    ##take Otsu threshold of newly darkened image
    th_2, mask_2 = cv2.threshold(gray_darkened, 0, 255, cv2.THRESH_OTSU)
    mask_2 = 1 - np.double(mask_2)
    mask_2 = np.uint8(mask_2%2)

    ##create final mask by using XOR between both masks
    mask_xor = np.logical_xor(mask_sub,mask_2)

    ##structuring element for morphological close operation
    SE = np.uint8(np.ones((7,7)))
    mask = mask_xor.astype('uint8')
    mask_morph = cv2.morphologyEx(mask, cv2.MORPH_OPEN, SE).astype('uint8')

    if(plots == True):
        ##Final mask output from double Otsu Threshold
        fig,ax = plt.subplots(nrows=1, ncols=2, figsize=(15,10))
        ax[0].hist(gray_darkened.ravel(), bins=255)
        ax[0].axvline(th_2 ,color = 'r')

        ax[1].imshow(mask_morph, cmap = 'gray')
        #extent = ax[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        #fig.savefig('92_1_hist_double_otsu.png', bbox_inches = extent.expanded(1.2,1.2))
        #plt.imshow(mask_morph,cmap = 'gray')
    
    return(mask_morph)


# In[41]:


img_name = 'IMG_1140.jpg'
img_name_no_file_class = img_name.replace(".jpg","")
img_name_no_file_class = img_name_no_file_class.replace(".png","")
img_cd = 'C:/Users/Chris/Documents/Assistant_Research/Research_Papers/Shadows/Python/Code_2/Untitled Folder/' +img_name
img = cv2.imread(img_cd)
img_rgb = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
img_size = img.shape

cd_save_mask = 'C:/Users/Chris/Documents/Assistant_Research/Research_Papers/Shadows/4_30to5_7/Pics/' +img_name_no_file_class+ '/Beginning_Implementation/mask_'
cd_save_result = 'C:/Users/Chris/Documents/Assistant_Research/Research_Papers/Shadows/4_30to5_7/Pics/' +img_name_no_file_class+ '/Beginning_Implementation/result_'

plt.imshow(img_rgb)


# # Result Without Salt and Pepper Noise Filter

# In[36]:


SE = np.ones((5,5)).astype('uint8')
th, mask_0 = cv2.threshold(img_gray.copy(), 0, 255, cv2.THRESH_OTSU)
mask_0 = 1-np.double(mask_0)
mask_0 = np.uint8(mask_0%2)
mask_0 = cv2.morphologyEx(mask_0, cv2.MORPH_CLOSE, SE)

mask_0_save = mask_0.copy()*255
cv2.imwrite(cd_save_mask + 'single_otsu_no_snp.jpg', mask_0_save)

result_no_snp_filt = ycbcr_shadow_correction(img_rgb, mask_0)
result_no_snp_filt_save = cv2.cvtColor(result_no_snp_filt, cv2.COLOR_BGR2RGB)
cv2.imwrite(cd_save_result + 'single_otsu_no_snp.jpg', result_no_snp_filt_save)
plt.imshow(result_no_snp_filt)


# # Result With Salt and Pepper Noise Filter

# In[37]:


SE = np.ones((5,5)).astype('uint8')
th, mask_1 = cv2.threshold(img_gray.copy(), 0, 255, cv2.THRESH_OTSU)
mask_1 = 1-np.double(mask_1)
mask_1 = np.uint8(mask_1%2)

if(check_salt_n_pep(mask_1) == True):
    print("Salt and Pepper Filter Applied")
    img_gray_filt = cv2.medianBlur(img_gray.copy(),7)
    th, mask_1 = cv2.threshold(img_gray_filt, 0, 255, cv2.THRESH_OTSU)
    mask_1 = 1-np.double(mask_1)
    mask_1 = np.uint8(mask_1%2)

mask_1 = cv2.morphologyEx(mask_1, cv2.MORPH_CLOSE, SE)
mask_1_save = mask_1.copy()*255
cv2.imwrite(cd_save_mask + 'single_otsu_snp.jpg', mask_1_save)
    
result_snp_filt = ycbcr_shadow_correction(img_rgb, mask_1)
result_snp_filt_save = cv2.cvtColor(result_snp_filt, cv2.COLOR_BGR2RGB)
cv2.imwrite(cd_save_result + 'single_otsu_snp.jpg', result_snp_filt_save)

plt.imshow(result_snp_filt)


# # Result With Double Otsu Threshold

# In[42]:


SE = np.ones((5,5)).astype('uint8')
mask_double_otsu = double_otsu(img_rgb)
mask_double_otsu = cv2.morphologyEx(mask_double_otsu, cv2.MORPH_CLOSE, SE)
mask_double_otsu_save = mask_double_otsu.copy()*255
cv2.imwrite(cd_save_mask + 'double_otsu.jpg', mask_double_otsu_save)

result_double_otsu = ycbcr_shadow_correction(img_rgb, mask_double_otsu)
result_double_otsu_save = cv2.cvtColor(result_double_otsu, cv2.COLOR_BGR2RGB)
cv2.imwrite(cd_save_result + 'double_otsu.jpg', result_double_otsu_save)

plt.imshow(result_double_otsu)


# In[ ]:




