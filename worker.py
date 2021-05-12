#!/usr/bin/env python
# coding: utf-8

# In[7]:


'''For use with multiprocessing'''
def master_fsp(key):
    import numpy as np
    ##nodeInd is list of indices corresponding to coordinates of the mask image
    mask, mask_pad, nodeInd, idx = key
    ##add one to indices, b/c mask is padded by one
    ##add whatever # that corresponds to the padding size
    i = nodeInd[2*idx]+1
    j = nodeInd[(2*idx)+1]+1
    
    window = 0
    window = mask_pad[j-1,i-1]
    window = window + mask_pad[j-1,i]
    window = window + mask_pad[j-1,i+1]
    window = window + mask_pad[j,i-1]
    window = window + mask_pad[j,i+1]
    window = window + mask_pad[j+1,i-1]
    window = window + mask_pad[j+1,i]
    window = window + mask_pad[j+1,i+1]
    if(window < 4):
        ##if mask[j,i] = 0, return 0; if = 1, return 1
        ##index is offset by one due to padding
        ret_val = np.logical_and(1,mask[j-1,i-1])
        return ret_val
    else:
        return 0
    
def preprocess_fsp(mask, th = 0.05):
    import numpy as np
    XX, YY = np.meshgrid(np.arange(mask.shape[1]), np.arange(mask.shape[0]))
    ##get flattened array of (x,y) coordinates of each pixel, x & y are
    ##alternating with x starting at the 0th element
    indices = [None]*(len(XX.ravel())+len(YY.ravel()))
    indices[::2] = XX.ravel()
    indices[1::2] = YY.ravel()
    ##pad image to retain full image size
    mask_pad = np.pad(mask, 1, mode = 'edge').astype('uint8')
    ret_mask = process_fsp(mask, mask_pad, indices, th)
    return ret_mask

def process_fsp(mask, mask_pad, indices, th):
    import cv2
    import numpy as np
    from multiprocessing import Pool
    ##structuring element for morphological operations
    SE = np.ones((3,3)).astype('uint8')
    convergence = False
    tot_cnt = 0
    pool = Pool()
    mask_shape = mask.shape
    ##iteratively filter mask until ratio of modified pixels to total 'ON' pixels reaches below threshold
    while(convergence == False):
        ##get total number of 'ON' pixels
        tot_masked_pixels = np.ndarray.tolist(mask.ravel()).count(1)
        ##multiply by 1 to convert from boolean answers to 1 or 0
        ##get total sum of modified pixels
        mod_mask = np.multiply(np.array(pool.map(master_fsp, [(mask, mask_pad, indices, v) for v in range(0,int(len(indices)/2))])),1)
        tot_cnt = np.sum(mod_mask.copy())
        ##invert output to multiply against current mask to retain pixels we didn't modify
        ##and filter out modified pixels, just doing logical and
        mod_mask = 1 - mod_mask
        mask = np.multiply(mask,np.reshape(mod_mask,(mask_shape[0],mask_shape[1]))).astype('uint8')
        ##to help retain the integrity of the true mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, SE).astype('uint8')
        if(tot_cnt/tot_masked_pixels <= th):
            convergence = True
            pool.close()
        else:
            ##not converged iterate through image again
            tot_cnt = 0
            mask_pad = np.pad(mask_pad, 1, mode = 'edge')
            
    return mask

