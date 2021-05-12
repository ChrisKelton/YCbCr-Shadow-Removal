#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
from multiprocessing.pool import Pool
import queue
import time
from worker import preprocess_fsp
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


'''No multiprocessing'''
class spurious_responses():
    SE = np.ones((3,3)).astype('uint8')
    
    def __init__(self, mask, th = 0.05):
        self.mask = mask.astype('uint8')
        self.mask_pad = np.pad(mask, 1, mode = 'edge').astype('uint8')
        self.th = th
        #self.tot_masked_pixels = np.ndarray.tolist(mask.ravel()).count(1)
        
    def _fsp_(self):
        convergence = False

        while(convergence != True):
            tot_cnt = 0
            ##count number of occurrences of masked shadow in image ('ON' pixel)
            tot_masked_pixels = np.ndarray.tolist(self.mask.ravel()).count(1)
            for i in range(1,self.mask.shape[1],1):
                for j in range(1,self.mask.shape[0],1):
                    tot_cnt = tot_cnt + spurious_responses.filter_spurious_responses(self, i, j)
            print('tot_masked_pixels = ', tot_masked_pixels)
            print('tot_cnt = ', tot_cnt)
            inter_tot_masked_pixels = np.ndarray.tolist(self.mask.ravel()).count(1)
            print('inter_tot_masked_pixels = ', inter_tot_masked_pixels)
            #print('Ratio = ', tot_cnt/tot_masked_pixels)
            ##get back any information that may have been lost from the main masked shadow pixels
            self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_CLOSE, self.SE).astype('uint8')
            if(tot_cnt/tot_masked_pixels <= self.th):
                convergence = True
            else:
                self.mask_pad = np.pad(self.mask, 1, mode = 'edge')
              
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
        if(window < 4):
            '''filtering out mask_pad while we go is too destructive and erodes the mask too quickly'''
            #self.mask_pad[j,i] = 0
            ##if self.mask[j,i] = 0, return 0; if = 1, return 1
            ##multiply by one to get number from boolean
            ret_val = np.logical_and(1,self.mask[j-1,i-1].copy())*1
            self.mask[j-1,i-1] = 0
            return ret_val
        else:
            return 0


# In[3]:


mask_cd = 'C:/Users/Chris/Documents/Assistant_Research/Research_Papers/Shadows/3_26to4_1/Pics/92-1/taking_mean_of_centers/gradient/8_mask.jpg'
#mask_cd = 'C:/Users/Chris/Documents/Assistant_Research/Research_Papers/Shadows/3_26to4_1/Pics/IMG_1138/intensity/3_mask.jpg'
mask_import = cv2.imread(mask_cd)
mask_imp = mask_import[:,:,0]
##to get all 'ON' pixels to have a value of 1, as when saving mask it alters the 'ON' pixels values for some reason
mask_imp[mask_imp < 255] = 0
mask_imp[mask_imp > 0] = 1
plt.imshow(mask_imp, cmap = 'gray')


# # Not Multiprocessing

# In[4]:


th = 0.05
start = time.time()
sr = spurious_responses(mask_imp, th)
new_mask = sr._fsp_()
end = time.time()
print("Time Elapsed = ", end - start)
plt.imshow(new_mask.mask, cmap = 'gray')


# # Multiprocessing

# In[35]:


##need for multiprocessing; otherwise, bad stuff seems to happen
if __name__=='__main__':
    th = 0.05
    start = time.time()
    new_mask = preprocess_fsp(mask_imp, th)
    end = time.time()
    print("Time Elapsed = ", end - start)
    plt.imshow(new_mask, cmap = 'gray')


# In[ ]:




