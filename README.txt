***************************README**************************
'Full_Pipeline.py' contains all of the below functions to use in conjunction in an attempt to remove and correct the shadows of an image.

'AA_TestMasks.py' contains a script to create masks for all of the available feature vectors.

'Beginning Implementation.py' contains the first three methods that were employed very early on in testing. I've included this just for completeness sake.

I've also included 'Accuracy of Masks, radius = 1.py' and 'Accuracy of Masks, radius = many.py', which compares the structural similarity between the mask and the ground truth mask of the image and outputs which features were used in the best mask. I got the ground truth masks from subtracting the ground truth images from the shadow images, so these files are purely for testing and wouldn't be deployed in the full pipelining.

function: check_salt_n_pep(mask, noise_thresh = 0.15)
	mask = shadow mask, where an 'ON' pixel correlates to where a shadow is believed to be
	noise_thresh = percent of pixels in image that are deemed salt and pepper noise worthy of applying median filter to image to rid of noise

	This function works by utilizing an XOR window around each pixel to determine if there is a spurious enough reaction around it to deem there being noise in the image.


function: apply_thresholds(img_gray, bins_regions, save_imgs = False, cd_save = './', plots = True, check_salt_n_pep_test = False)
	img_gray = grayscale image
	bins_regions = centers (found from KMeans implementation) to segment image in order to find shadow regions
	save_imgs = boolean value to save images and plots or not, won't have any affect if plots = False
	cd_save = where to save images to
	plots = boolean value to display images & plots or not

	Digitizes image according to centers in order to better visualize the segmented regions of the image and returns this digitized image.


function: get_mask_kmeans(img_rgb, num_clusters, process = 'intensity', wt_pen = 5, neighborhood = 8, salt_n_pep = True, radius = 1, METHOD = 'uniform')
	img_rgb = RGB image
	num_clusters = number of clusters to apply to KMeans on image. I've found that between 6 and 8 return the best centers.
	process = what features to include when taking the KMeans
		-list of process:
1. 'intensity'
2. 'intensity_and_gradient'
3. 'intensity_and_texture'
4. 'intensity_and_cb_channel'
5. 'intensity_and_gradient_and_texture'
6. 'intensity_and_gradient_and_cb_channel'
7. 'intensity_and_gradient_and_texture_and_cb_channel'
8. 'gradient'
9. 'gradient_and_texture'
10. 'gradient_and_cb_channel'
11. 'gradient_and_texture_and_cb_channel'
12. 'texture*'
13. 'texture_and_cb_channel'
14. 'cb_channel**'

*texture uses local binary pattern algorithm

**cb_channel applies a ycbcr mask by a method of taking the means and standard deviations of each channel and filtering out all pixels less than mean-std for each channel and multiplying this to the grayscale image. This is taken from the 'Shadow Detection and Removal Based on YCbCr Color Space' research paper, which can be found here: https://www.researchgate.net/publication/263662695_Shadow_Detection_and_Removal_Based_on_YCbCr_Color_Space. This paper is where I got the YCbCr shadow removal algorithm from. I've also included the paper in this directory.


function: prep_data_fit_to_kmeans(img_rgb, num_clusters, process = 'intensity', wt_pen = 5, neighborhood = 8, salt_n_pep = True, radius = 1, METHOD = 'uniform')
	img_rgb = RGB image
	num_clusters = number of clusters to input to KMeans
	process = whatever process you want that was previously listed above
	wt_pen = weight penalty to apply to centers returned from KMeans, this was found experimentally to be best around 5; otherwise, more of the image is deemed shadow than true.
	neighborhood = window size to apply to each pixel, currently only a neighborhood size of eight is supported, but one could implement higher orders of neighborhoods or lower orders
	salt_n_pep = boolean value to check for salt and pepper noises or not
	radius* = radius value to input to local binary pattern algorithm (only matters when using texture feature)
	METHOD** = method type to input to local binary pattern algorithm (only matters when using texture feature)

*was experimenting with different sized radii but had to rehaul code as I was making a mistake implementing the center values, so I have no real experimental data for different sized radius. Not sure if it's worth the effor of experimenting with different sized radii or not, but could be a quick thing to look at. An additional possible test, not something that would probably add a lot, if any, to results was a thought to append different radii to the feature vectors; e.g., radius = 3, then have 3 texture features correlating to radius = 1, radius = 2, and radius = 3.

**same as above, different methods are listed in the skimage.feature.local_binary_pattern function on their website.


class fvs:
	class of the different feature vector functions to make it easier to read.
	Example of use:

get_features = fvs(img_rgb, neighborhood = neighborhood, radius = radius, METHOD = METHOD)
temp = get_features._intensity_()



function: get_feature_vectors(img_rgb, process = 'intensity', neighborhood = 8, radius = 1, METHOD = 'uniform')
	img_rgb = RGB image
	process = type of feature vectors to use
	neighborhood = window to apply to spatial feature vectors
	radius = radius to input to local binary pattern
	METHOD = method to input to local binary pattern


class spurious_responses:
	Used to filter out spurious responses picked up during mask creation as KMeans implementation doesn't return a perfect mask. Helps filter out spurious shadow pixels that will needlessly distort the mask. This implementation does not utilize multiprocessing.
	Example of use:
#stopping threshold for convergence of filtering spurious responses from image
#may need to be adjusted for larger images, haven't really tested it on very large images.
th = 0.05
sr = spurious_responses(mask, th)
sr_mask_filt = sr._fsp_()
mask_filt = sr_mask_filt.mask


function: ycbcr_shadow_correction(img_rgb, mask)
	img_rgb = RGB image
	mask = mask obtained from utilizing above functions
	
	-This is the last step in the pipeline before getting out a potentially shadow free image as all preceeding steps revolve around grabbing an accurate mask of the shadows.
	- This will return a color corrected image, where the shadows should be much better color matched to its surroundings.
	- This was obtained from that paper I linked previously, which contains a much more in depth look at what they're doing. There are also a bunch more papers out there that have to do with shadow detection and removal to look at.


function: theta_color_correction(shadow_corrected_img, mask, save_ccimg_deg = False, cd_save = './ccimg_deg_', SNR_return = False, save_SNR = False, cd_SNR = './SNR_img')
	shadow_corrected_img = image that has had its shadow color corrected in an attempt to remove the shadow. Output from ycbcr_shadow_correction
	mask = mask used to color correct image in ycbcr_shadow_correction
	save_ccimg_deg = boolean value to save all the different degrees of theta color corrected images, there will be 181
	cd_save = where to save the theta color corrected images
	SNR_return = boolena value to return signal to noise ratio plot or not
	save_SNR = boolean value to save the SNR or not
	cd_SNR = where to save the SNR plot

	-This iterates through possible degrees at which the reflected light (coming from the surface) might be in correlation to the incident light (direct light, from camera or sun). I don't believe it takes into account ambient light, but it seems to get really into the weeds when you start trying to use degrees of reflection between different light sources to try and correct images.
	- This comes from the research paper I mentioned above as well. They don't do a great job of explaining it, but it gives you a pretty good overview. It's essentially providing a scale of values from 1 (sunshine region) to 0 (umbra of shadow) in order to further correct the color corrected shadow regions of the image.
	- I put a more in depth discussion in the function itself in Python.



I'll also include a multiprocessing method of the spurious_responses function, which doesn't work in Jupyter Notebook, it works in Spyder and probably most other Python IDEs, but I never got it to be quicker than the non multiprocessing implementation, which is the exact opposite purpose of implementing multiprocessing. That file should also show you how to use it if you want to look at it and maybe play around with it to get it working correctly.
	- The multiprocessing implementation is in 'worker.py' and how to use it is in 'Remove Spurious Responses Comparison.py'



This is also a link to my google drive containing pictures from the research papers I had been looking at and the pictures I've been using are in there as well: https://drive.google.com/drive/folders/1wxa-JFLylqI4q1M5M8RZ6RkMNGLAn8bP?usp=sharing
You should probably take pictures with the source of the shadow in the image as well though, as most of these images do not contain the source.