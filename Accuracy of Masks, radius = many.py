#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
from sklearn.cluster import KMeans
from skimage.color import rgb2ycbcr as r2y
from skimage.feature import local_binary_pattern as LBP
from skimage.metrics import structural_similarity as ssim
import time
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


gt_mask_name = '92-1_gt_mask.jpg'
gt_mask_name_no_file_class = gt_mask_name.replace(".jpg","")
gt_mask_name_no_file_class = gt_mask_name_no_file_class.replace(".png","")
gt_mask_cd = 'C:/Users/Chris/Documents/Assistant_Research/Research_Papers/Shadows/3_19to3_25/Pics/GroundTruthMasks/' + gt_mask_name
gt_mask = cv2.imread(gt_mask_cd)
gt_mask = gt_mask[:,:,0]
plt.imshow(gt_mask, cmap = 'gray')


# In[15]:


img_name = gt_mask_name_no_file_class.replace("_gt_mask","")
process = ['cb_channel', 'cb_channel_and_texture', 'cb_channel_and_intensity', 'cb_channel_and_gradient', 'cb_channel_and_intensity_and_texture', 'cb_channel_and_intensity_and_gradient', 'cb_channel_and_texture_and_gradient', 'cb_channel_and_intensity_and_texture_and_gradient', 'intensity_and_gradient', 'intensity', 'gradient', 'texture', 'intensity_and_texture', 'texture_and_gradient', 'intensity_and_texture_and_gradient']
radius = [1,2,3,4,5]
method = ['nri_uniform','uniform','var']
clusts = np.arange(3,10)
center_pixel = [False]
##contains all ssim values
ssim_mask_all = np.zeros((len(clusts),len(method),len(radius),len(process)))
for i in range(len(process)):
    print('process = ', process[i])
    ##texture has different parameters than the other feature vectors, so it is treated specially :)
    if('texture' in process[i]):
        ssim_mask = np.zeros((len(clusts), len(method), len(radius)))
        for m in range(len(radius)):
            for n in range(len(method)):
                for j in range(len(clusts)):
                    mask_cd = 'C:/Users/Chris/Documents/Assistant_Research/Research_Papers/Shadows/3_12to3_18/Pics/KMeans/center_pixel=' +str(center_pixel[0])+ '/' + img_name + '/Pre_Process_Median_Filter/neighborhood=8/' + process[i] + '/radius=' + str(radius[m]) + '/method=' + method[n] + '/' + str(clusts[j]) + '_mask.jpg'
                    test_mask = cv2.imread(mask_cd)
                    test_mask = test_mask[:,:,0]
                    test_mask[test_mask > 127] = 255
                    test_mask[test_mask < 128] = 0
                    ssim_val = ssim(gt_mask, test_mask, data_range = test_mask.max() - test_mask.min())
                    if(math.isnan(ssim_val) == True):
                        ssim_mask[j,n,m] = 0
                    else:
                        ssim_mask[j,n,m] = ssim_val.copy()
                    
                ssim_mask_all[:,n,m,i] = ssim_mask[:,n,m].copy()
                ##plot clusts against each other
                ssim_mask_cd = 'C:/Users/Chris/Documents/Assistant_Research/Research_Papers/Shadows/3_19to3_25/ssim_vals/all_radius_at_once/center_pixel=' +str(center_pixel[0])+ '/' + img_name + '/' + process[i] + '/radius=' + str(radius[m]) + '/method=' + method[n] + '/ssim_mask'
                np.save(ssim_mask_cd + '.npy', ssim_mask[:,n])
                plt.figure(figsize = (10,5))
                plt.plot(clusts, ssim_mask[:,n,m])
                plt.xlabel('clusters')
                plt.ylabel('ssim value')
                plt.title('ssim values of ' +process[i]+ ' radius = ' +str(radius[m])+ ' method = ' +method[n]+ ' vs clusters')
                plt.savefig(ssim_mask_cd + '.png', bbox_inches = 'tight')
                
            ##plot all methods against each other
            ssim_mask_cd = 'C:/Users/Chris/Documents/Assistant_Research/Research_Papers/Shadows/3_19to3_25/ssim_vals/all_radius_at_once/center_pixel=' +str(center_pixel[0])+ '/' + img_name + '/' + process[i] + '/radius=' + str(radius[m]) + '/methods/ssim_mask'
            np.save(ssim_mask_cd + '.npy', ssim_mask)
            plt.figure(figsize = (10,5))
            p1, = plt.plot(clusts, ssim_mask[:,0,m], label = method[0])
            p2, = plt.plot(clusts, ssim_mask[:,1,m], label = method[1])
            p3, = plt.plot(clusts, ssim_mask[:,2,m], label = method[2])
            plt.legend(handles=[p1, p2, p3], title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xlabel('clusters')
            plt.ylabel('ssim value')
            plt.title('ssim values of ' +process[i]+ ' radius = ' +str(radius[m])+ ' all methods vs clusters')
            plt.savefig(ssim_mask_cd + '.png', bbox_inches = 'tight')
        
        ##plot max radius methods against each other
        ssim_mask_cd = 'C:/Users/Chris/Documents/Assistant_Research/Research_Papers/Shadows/3_19to3_25/ssim_vals/all_radius_at_once/center_pixel=' +str(center_pixel[0])+ '/' +img_name+ '/' +process[i] + '/radii/ssim_mask'
        np.save(ssim_mask_cd + '.npy', ssim_mask)
        plt.figure(figsize = (10,5))
        labels = []
        for k in range(len(radius)):
            maxes = np.zeros((len(method), 1))
            for p in range(len(method)):
                maxes[p,0] = np.max(ssim_mask[:,p,k].copy())
            row, col = np.where(maxes == np.amax(maxes.copy()))
            print('row = ', row)
            if not row[0].all():
                #there is a max
                plt.plot(clusts, ssim_mask[:,row[0],k])
            else:
                plt.plot(clusts, np.zeros((len(clusts),1)))
            labels.append(process[k]+ ' radius = ' +str(radius[k])+ ' method = ' +method[row[0]])
        plt.legend(labels, title = 'Legend', bbox_to_anchor=(1.05,1), loc = 'upper left')
        plt.xlabel('clusters')
        plt.ylabel('ssim value')
        plt.title('ssim values of ' +process[i]+ ' all radii vs clusters')
        plt.savefig(ssim_mask_cd + '.png', bbox_inches = 'tight')
            
    else:
        ##all processes that don't involve texture
        ssim_mask = np.zeros((len(clusts),1))
        for j in range(len(clusts)):
            mask_cd = 'C:/Users/Chris/Documents/Assistant_Research/Research_Papers/Shadows/3_12to3_18/Pics/KMeans/center_pixel=' +str(center_pixel[0])+ '/' + img_name + '/Pre_Process_Median_Filter/neighborhood=8/' + process[i] + '/' + str(clusts[j]) + '_mask.jpg'
            test_mask = cv2.imread(mask_cd)
            test_mask = test_mask[:,:,0]
            test_mask[test_mask > 127] = 255
            test_mask[test_mask < 128] = 0
            ssim_val = ssim(gt_mask, test_mask, data_range = test_mask.max() - test_mask.min())
            if(math.isnan(ssim_val) == True):
                ssim_mask[j,0] = 0
            else:
                ssim_mask[j,0] = ssim_val.copy()

        ssim_mask_all[:,0,0,i] = ssim_mask[:,0].copy()
        ##plot clusts against each other
        ssim_mask_cd = 'C:/Users/Chris/Documents/Assistant_Research/Research_Papers/Shadows/3_19to3_25/ssim_vals/all_radius_at_once/center_pixel=' +str(center_pixel[0])+ '/' + img_name + '/' + process[i] + '/ssim_mask'
        np.save(ssim_mask_cd + '.npy', ssim_mask)
        plt.figure(figsize = (10,5))
        plt.plot(clusts, ssim_mask)
        plt.xlabel('clusters')
        plt.ylabel('ssim value')
        plt.title('ssim values of ' +process[i]+ ' vs clusters')
        plt.savefig(ssim_mask_cd + '.png', bbox_inches = 'tight')


# In[16]:


##plot every different combination of feature vectors against each other
ssim_mask_all_cd = 'C:/Users/Chris/Documents/Assistant_Research/Research_Papers/Shadows/3_19to3_25/ssim_vals/all_radius_at_once/center_pixel=' +str(center_pixel[0])+ '/' +img_name+ '/all_processes/ssim_mask'
np.save(ssim_mask_all_cd + '.npy', ssim_mask_all)
plt.figure(figsize = (10,5))
labels = []
for i in range(len(process)):
    if('texture' in process[i]):
        for m in range(len(radius)):
            for j in range(len(method)):
                plt.plot(clusts, ssim_mask_all[:,j,m,i])
                if(np.max(ssim_mask_all[:,j,m,i]) == 0):
                    print('ZEROS in ' +process[i]+ ' method = ' +method[j])
                labels.append(process[i]+ ' radius = ' +str(radius[0])+ ' method = ' +method[j])
    else:
        plt.plot(clusts,ssim_mask_all[:,0,0,i])
        if(np.max(ssim_mask_all[:,0,0,i]) == 0):
            print('ZEROS in ' +process[i])
        labels.append(process[i])

plt.legend(labels, title='Legend', bbox_to_anchor = (1.05,1), loc = 'upper left')
plt.xlabel('clusters')
plt.ylabel('ssim value')
plt.title('ssim values of all process vs clusters')
plt.savefig(ssim_mask_all_cd + '.png', bbox_inches = 'tight')


# In[17]:


ssim_mask_all_cd = 'C:/Users/Chris/Documents/Assistant_Research/Research_Papers/Shadows/3_19to3_25/ssim_vals/all_radius_at_once/center_pixel=' +str(center_pixel[0])+ '/' +img_name+ '/all_processes/best_ssim_masks'
mean_clusters = np.zeros((len(method),len(radius),len(process)))
max_methods = np.zeros((len(radius),len(process)))
##len(radius) size will hold the idx of the best method
##len(process) size will hold the idx of the best radius
max_methods_idx = np.zeros((len(radius),len(process)))
max_radii = np.zeros((len(process),1))
max_radii_idx = np.zeros((len(process),1))
for i in range(len(process)):
    #print('process = ', process[i])
    if('texture' in process[i]):
        for j in range(len(radius)):
            #print('radius = ', radius[j])
            for m in range(len(method)):
                #print('method = ', method[m])
                mean_clusters[m,j,i] = np.mean(ssim_mask_all[:,m,j,i])
            max_methods[j,i] = np.max(mean_clusters[:,j,i])
            ##maintain index of max(mean(cluster))
            if(len(np.squeeze(mean_clusters[:,:,i]).shape) == 1):
                temp = mean_clusters[:,0]
            else:
                temp = np.squeeze(mean_clusters[:,:,i])
            idx_r, idx_c = np.where(temp == np.max(mean_clusters[:,j,i]))
            max_methods_idx[j,i] = idx_r[0]
        max_radii[i,0] = np.max(max_methods[:,i])
        ##maintain index of max(radii) -> max(max(mean(cluster)))
        idx_r, idx_c = np.where(max_methods == np.max(max_methods[:,i]))
        max_radii_idx[i,0] = idx_r[0]
    else:
        means_clusts = np.mean(ssim_mask_all[:,0,0,i])
        max_radii[i,0] = np.amax(means_clusts)
##max_radii holds all the max means of the clusters and where they are stored, if a texture, are in the idx arrays
top_num = 5
best_fvs = np.zeros((len(clusts),top_num))
labels = []
plt.figsize = ((10,5))
for i in range(top_num):
    idx_r, idx_c = np.where(max_radii == np.max(max_radii))
    max_radii[idx_r[0],0] = 0
    if('texture' in process[idx_r[0]]):
        radii_idx = max_radii_idx[idx_r[0],0].astype('int')
        method_idx = max_methods_idx[radii_idx,idx_r[0]].astype('int')
        best_fvs[:,i] = np.squeeze(ssim_mask_all[:,method_idx,radii_idx,idx_r[0]])
        plt.plot(clusts, best_fvs[:,i])
        labels.append(process[idx_r[0]]+ ' radius = ' +str(radius[radii_idx])+ ' method = ' +method[method_idx])
    else:
        best_fvs[:,i] = np.squeeze(ssim_mask_all[:,0,0,idx_r[0]])
        plt.plot(clusts, best_fvs[:,i])
        labels.append(process[idx_r[0]])
        
np.save(ssim_mask_all_cd + '.npy', best_fvs)
plt.legend(labels, title='Legend', bbox_to_anchor = (1.05,1), loc = 'upper left')
plt.xlabel('clusters')
plt.ylabel('ssim value')
plt.title('Best SSIM Value Feature Vectors')
plt.savefig(ssim_mask_all_cd + '.png', bbox_inches = 'tight')


# In[ ]:




