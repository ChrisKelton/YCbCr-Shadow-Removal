#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[8]:


img_name = gt_mask_name_no_file_class.replace("_gt_mask","")
print(img_name)
process = ['cb_channel','cb_channel_and_gradient','cb_channel_and_gradient_and_texture','cb_channel_and_intensity','cb_channel_and_intensity_and_gradient','cb_channel_and_intensity_and_gradient_and_texture','cb_channel_and_intensity_and_texture','cb_channel_and_texture','gradient','gradient_and_texture','intensity','intensity_and_gradient','intensity_and_gradient_and_texture','intensity_and_texture','texture']
radius = [5]
method = ['nri_uniform','uniform','var']
center_pixel = [False]
clusts = np.arange(3,10)
ssim_mask_all = np.zeros((len(clusts),len(method),len(process)))
for i in range(len(process)):
    #print('process = ', process[i])
    '''
    if('texture' in process[i]):
        for m in range(len(radius)):
            #print('radius = ', radius[m])
            ssim_mask = np.zeros((len(clusts), len(method)))
            for n in range(len(method)):
                #print('method = ', method[n])
                for j in range(len(clusts)):
                    #print('cluster = ', clusts[j])
                    #mask_cd = 'C:/Users/Chris/Documents/Assistant_Research/Research_Papers/Shadows/3_26to4_1/Pics/' +img_name+ '/' + process[i] + '/radius=' + str(radius[m]) + '/method=' + method[n] + '/' + str(clusts[j]) + '_mask.jpg'
                    mask_cd = 'C:/Users/Chris/Documents/Assistant_Research/Research_Papers/Shadows/3_26to4_1/Pics/' +img_name+ '/' +process[i]+ '/' +str(clusts[j])+ '_mask.jpg'
                    test_mask = cv2.imread(mask_cd)
                    test_mask = test_mask[:,:,0]
                    test_mask[test_mask > 127] = 255
                    test_mask[test_mask < 128] = 0
                    ssim_val = ssim(gt_mask, test_mask, data_range = test_mask.max() - test_mask.min())
                    if(math.isnan(ssim_val) == True):
                        ssim_mask[j,n] = 0
                    else:
                        ssim_mask[j,n] = ssim_val.copy()
                    
                ssim_mask_all[:,n,i] = ssim_mask[:,n]
                ##plot clusts against each other
                ssim_mask_cd = 'C:/Users/Chris/Documents/Assistant_Research/Research_Papers/Shadows/3_19to3_25/ssim_vals/one_radius_at_a_time/center_pixel=' +str(center_pixel[0])+ '/' + img_name + '/radius=' + str(radius[m]) + '/' + process[i] + '/method=' + method[n] + '/ssim_mask'
                np.save(ssim_mask_cd + '.npy', ssim_mask[:,n])
                plt.figure(figsize = (10,5))
                plt.plot(clusts, ssim_mask[:,n])
                plt.xlabel('clusters')
                plt.ylabel('ssim value')
                plt.title('ssim values of ' +process[i]+ ' radius = ' +str(radius[0])+ ' method = ' +method[n]+ ' vs clusters')
                plt.savefig(ssim_mask_cd + '.png', bbox_inches = 'tight')
                
            ##plot all methods against each other
            ssim_mask_cd = 'C:/Users/Chris/Documents/Assistant_Research/Research_Papers/Shadows/3_19to3_25/ssim_vals/one_radius_at_a_time/center_pixel=' +str(center_pixel[0])+ '/' + img_name + '/radius=' + str(radius[0]) + '/' + process[i] + '/methods/ssim_mask'
            np.save(ssim_mask_cd + '.npy', ssim_mask)
            plt.figure(figsize = (10,5))
            p1, = plt.plot(clusts, ssim_mask[:,0], label = method[0])
            p2, = plt.plot(clusts, ssim_mask[:,1], label = method[1])
            p3, = plt.plot(clusts, ssim_mask[:,2], label = method[2])
            plt.legend(handles=[p1, p2, p3], title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xlabel('clusters')
            plt.ylabel('ssim value')
            plt.title('ssim values of ' +process[i]+ ' radius = ' +str(radius[0])+ ' all methods vs clusters')
            plt.savefig(ssim_mask_cd + '.png', bbox_inches = 'tight')
    else:'''
    ssim_mask = np.zeros((len(clusts),1))
    for j in range(len(clusts)):
        #mask_cd = 'C:/Users/Chris/Documents/Assistant_Research/Research_Papers/Shadows/3_12to3_18/Pics/KMeans/center_pixel=' +str(center_pixel[0])+ '/' + img_name + '/Pre_Process_Median_Filter/neighborhood=8/' + process[i] + '/' + str(clusts[j]) + '_mask.jpg'
        mask_cd = 'C:/Users/Chris/Documents/Assistant_Research/Research_Papers/Shadows/3_26to4_1/Pics/' +img_name+ '/' +process[i]+ '/' +str(clusts[j])+ '_mask.jpg'
        #print(mask_cd)
        test_mask = cv2.imread(mask_cd)
        test_mask = test_mask[:,:,0]
        test_mask[test_mask > 127] = 255
        test_mask[test_mask < 128] = 0
        ssim_val = ssim(gt_mask, test_mask, data_range = test_mask.max() - test_mask.min())
        if(math.isnan(ssim_val) == True):
            ssim_mask[j,0] = 0
        else:
            ssim_mask[j,0] = ssim_val.copy()

    ssim_mask_all[:,0,i] = ssim_mask[:,0].copy()
    ##plot clusts against each other
    #ssim_mask_cd = 'C:/Users/Chris/Documents/Assistant_Research/Research_Papers/Shadows/3_19to3_25/ssim_vals/one_radius_at_a_time/center_pixel=' +str(center_pixel[0])+ '/' + img_name + '/radius=' + str(radius[0]) + '/' + process[i] + '/ssim_mask'
    ssim_mask_cd = 'C:/Users/Chris/Documents/Assistant_Research/Research_Papers/Shadows/3_26to4_1/ssim_vals/' +img_name+ '/' +process[i]+ '/ssim_mask' 
    np.save(ssim_mask_cd + '.npy', ssim_mask)
    plt.figure(figsize = (10,5))
    plt.plot(clusts, ssim_mask)
    plt.xlabel('clusters')
    plt.ylabel('ssim value')
    plt.title('ssim values of ' +process[i]+ ' vs clusters')
    plt.savefig(ssim_mask_cd + '.png', bbox_inches = 'tight')


# In[13]:


#ssim_mask_all_cd = 'C:/Users/Chris/Documents/Assistant_Research/Research_Papers/Shadows/3_19to3_25/ssim_vals/one_radius_at_a_time/center_pixel=' +str(center_pixel[0])+ '/' +img_name+ '/radius=' + str(radius[0]) + '/all_processes/ssim_mask'
ssim_mask_all_cd = 'C:/Users/Chris/Documents/Assistant_Research/Research_Papers/Shadows/3_26to4_1/ssim_vals/' +img_name+ '/all_processes/ssim_mask'
np.save(ssim_mask_all_cd + '.npy', ssim_mask_all)
plt.figure(figsize = (10,5))
labels = []
for i in range(len(process)):
    #if('texture' in process[i]):
    #    for j in range(len(method)):
    #        plt.plot(clusts, ssim_mask_all[:,j,i])
    #        labels.append(process[i]+ ' radius = ' +str(radius[0])+ ' method = ' +method[j])
    #else:
    plt.plot(clusts,ssim_mask_all[:,0,i])
    labels.append(process[i])

plt.legend(labels, title='Legend', bbox_to_anchor = (1.05,1), loc = 'upper left')
plt.xlabel('clusters')
plt.ylabel('ssim value')
plt.title('ssim values of all process vs clusters')
plt.savefig(ssim_mask_all_cd + '.png', bbox_inches = 'tight')


# In[16]:


means = np.zeros((len(process),len(method)))
for i in range(len(process)):
    #if('texture' in process):
    #    for j in range(len(method)):
    #        means[i,j] = np.mean(ssim_mask_all[:,j,i].copy())
    #else:
    means[i,0] = np.mean(ssim_mask_all[:,0,i].copy())
        
idx_r, idx_c = np.where(means == np.amax(means.copy()))

print(idx_r, idx_c)
for i in range(len(idx_r)):
    #ssim_mask_all_cd = 'C:/Users/Chris/Documents/Assistant_Research/Research_Papers/Shadows/3_19to3_25/ssim_vals/one_radius_at_a_time/center_pixel=' +str(center_pixel[0])+ '/' +img_name+ '/radius=' + str(radius[0]) + '/all_processes/best_ssim_mask_' +process[idx_r[i]]
    ssim_mask_all_cd = 'C:/Users/Chris/Documents/Assistant_Research/Research_Papers/Shadows/3_26to4_1/ssim_vals/' +img_name+ '/all_processes/best_ssim_mask_' +process[idx_r[i]]
    np.save(ssim_mask_all_cd + '.npy', ssim_mask_all[:, idx_c[i], idx_r[i]])
    plt.figure(figsize = (10,5))
    plt.plot(clusts, ssim_mask_all[:, idx_c[i], idx_r[i]])
    plt.xlabel('clusters')
    plt.ylabel('ssim value')
    #if('texture' in process[idx_r[i]]):
    #    plt.title('ssim values of best process (' +process[idx_r[i]]+ ' radius = ' +str(radius[0])+ ' method = ' +method[idx_c[i]]+ ') vs clusters')
    #else:
    plt.title('ssim values of best process (' +process[idx_r[i]]+ ') vs clusters')
    plt.savefig(ssim_mask_all_cd + '.png', bbox_inches = 'tight')


# In[18]:


#ssim_mask_all_cd = 'C:/Users/Chris/Documents/Assistant_Research/Research_Papers/Shadows/3_19to3_25/ssim_vals/one_radius_at_a_time/center_pixel=' +str(center_pixel[0])+ '/' +img_name+ '/radius=' + str(radius[0]) + '/all_processes/best_ssim_masks'
ssim_mask_all_cd = 'C:/Users/Chris/Documents/Assistant_Research/Research_Papers/Shadows/3_26to4_1/ssim_vals/' +img_name+ '/all_processes/best_ssim_masks'
means = np.zeros((len(process),1))
max_method_idx = np.zeros((len(process),1))
for i in 
range(len(process)):
    #if('texture' in process):
    #    mean_inter = np.zeros((len(method),1))
    #    for j in range(len(method)):
    #        mean_inter[j,0] = np.mean(ssim_mask_all[:,j,i].copy())
    #    max_methods = np.max(mean_inter)
    #    idx_r, idx_c = np.where(mean_inter == np.max(mean_inter))
    #    max_method_idx[i,0] = idx_r[0]
    #    means[i,0] = max_methods
    #else:
    means[i,0] = np.mean(ssim_mask_all[:,0,i].copy())
        
top_num = 5
best_fvs = np.zeros((len(clusts),top_num))
plt.figure(figsize = (10,5))
labels = []
for i in range(top_num):
    idx_r, idx_c = np.where(means == np.amax(means.copy()))
    means[idx_r[0],0] = 0
    #if('texture' in process[idx_r[0]]):
    #    method_idx = max_method_idx[idx_r[0],0].astype('int')
    #    best_fvs[:,i] = np.squeeze(ssim_mask_all[:,method_idx,idx_r[0]])
    #    plt.plot(clusts, best_fvs[:,i])
    #    labels.append(process[idx_r[0]]+ ' radius = ' +str(radius[0])+ ' method = ' +method[method_idx])
    #else:
    best_fvs[:,i] = np.squeeze(ssim_mask_all[:,0,idx_r[0]])
    plt.plot(clusts, best_fvs[:,i])
    labels.append(process[idx_r[0]])

np.save(ssim_mask_all_cd + '.npy', best_fvs)
plt.legend(labels, title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left')    
plt.xlabel('clusters')
plt.ylabel('ssim value')
plt.title('best processes vs clusters')
plt.savefig(ssim_mask_all_cd + '.png', bbox_inches = 'tight')


# In[ ]:




