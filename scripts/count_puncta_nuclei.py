# count puncta and nuclei in a folder of images.
# report number, intensity, and area (+ relevant summary statistics) of all puncta in each image.

import os
import numpy as np
from skimage import measure, morphology
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from skimage import exposure
from skimage.segmentation import find_boundaries
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import disk, white_tophat
from skimage.morphology import erosion, disk
from skimage.morphology import dilation
import scipy.stats as stats
import imageio.v2 as imageio
import logging
from skimage.measure import label, regionprops
from skimage.segmentation import relabel_sequential
from skimage.util import img_as_bool


# initiate logfile
def initiate_log(iflog, logfile):
    if not iflog:
        return
    logging.basicConfig(
    filename=logfile,
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)

# print msg if iflog = False, log if iflog = True
def log_print(iflog, msg):
    if iflog:
        logging.info(msg)
    else:
        print(msg)

def show_image(image, title="none", cmap='gray'):
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()
    plt.close()

def well_from_filename(filename):
    # return well name from filename (Incell format). strip of all internal and trailing/leading whitespaces.
    return ''.join(filename[0:6].split())

# test well_from_filename_function
# print(well_from_filename("B - 02(fld 13 wv UV - DAPI).tif"))
# print(well_from_filename("C - 03(fld 14 wv UV - DAPI).tif"))
# print(well_from_filename("E - 03(fld 12 wv UV - DAPI).tif"))

def fld_from_filename(filename):
    return filename[11:13]

# test fld_from_filename_function
# print(fld_from_filename("B - 02(fld 13 wv UV - DAPI).tif"))
# print(fld_from_filename("C - 03(fld 14 wv UV - DAPI).tif"))
# print(fld_from_filename("E - 03(fld 12 wv UV - DAPI).tif"))

def remove_large_objects(mask, max_size, connectivity=2):
    """
    Remove connected components (objects) larger than or equal to max_size pixels.

    Parameters:
        mask (ndarray): Binary (boolean) mask.
        max_size (int): Maximum allowed object size (in pixels).
        connectivity (int): Pixel connectivity; 1=4-connectivity, 2=8-connectivity.

    Returns:
        ndarray: Mask with large objects removed.
    """
    labeled_mask = measure.label(mask, connectivity=connectivity)
    out_mask = mask.copy()
    for region in measure.regionprops(labeled_mask):
        if region.area >= max_size:
            out_mask[labeled_mask == region.label] = False
    return out_mask

# Example usage:
# cleaned_mask = remove_large_objects(mask, max_size=200, connectivity=2)

def remove_large_low_intensity_objects(mask, image, min_size, max_intensity, connectivity=2):
    """
    Remove connected components (objects) larger than or equal to min_size pixels
    AND with mean intensity less than or equal to max_intensity.

    Parameters:
        mask (ndarray): Binary (boolean) mask.
        image (ndarray): Grayscale image (same shape as mask).
        min_size (int): Minimum size threshold (objects >= this are considered).
        max_intensity (float): Maximum mean intensity threshold (objects <= this are removed).
        connectivity (int): Pixel connectivity; 1=4-connectivity, 2=8-connectivity.

    Returns:
        ndarray: Mask with specified objects removed.
    """
    labeled_mask = measure.label(mask, connectivity=connectivity)
    out_mask = mask.copy()
    for region in measure.regionprops(labeled_mask, intensity_image=image):
        if region.area >= min_size and region.mean_intensity <= max_intensity:
            out_mask[labeled_mask == region.label] = False
    return out_mask

def imagej_style_watershed(mask, min_distance=3, erode=0):

    # Step 1: erode mask to enhance separation of touching objects.
    if erode > 0:
        mask = erosion(mask, disk(erode))

    # Step 2: Compute the distance transform. helps define basins/local maxima.
    distance = ndi.distance_transform_edt(mask)

    # Step 3: Find local maxima (object centers)
    coordinates = peak_local_max(distance, labels=mask, min_distance=min_distance)
    markers = np.zeros_like(distance, dtype=int)
    for i, (r, c) in enumerate(coordinates, 1):
        markers[r, c] = i

    # Step 4: Apply watershed
    labels = watershed(-distance, markers, mask=mask, watershed_line=True)

    # Step 5: Convert to binary mask
    separated_mask = labels > 0
    # show_image(separated_mask, title="separated mask after watershed")

    return separated_mask

# Usage:
# mask = ... # your binary mask (numpy array, dtype=bool or 0/1)
# separated_mask = imagej_style_watershed(mask)

# test watershed function:
# generate an initial image with two overlapping circles
# x, y = np.indices((80, 80))
# x1, y1, x2, y2 = 28, 28, 44, 52
# r1, r2 = 16, 20
# mask_circle1 = (x - x1) ** 2 + (y - y1) ** 2 < r1**2
# mask_circle2 = (x - x2) ** 2 + (y - y2) ** 2 < r2**2
# image = np.logical_or(mask_circle1, mask_circle2)
# show_image(image, title="Test Image with Two Overlapping Circles")

# apply watershed to the test image
# separated_mask = imagej_style_watershed(image)
# show_image(separated_mask, title="Separated Mask from Watershed")

def _kmeans_cluster(coords, k, random_state=0, max_iter=100):
    """
    Cluster (row, col) pixel coordinates into k groups.
    Prefers sklearn.KMeans; falls back to scipy.cluster.vq.kmeans2;
    final fallback is a simple axis-based split.
    """
    try:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=k, n_init=5, max_iter=max_iter, random_state=random_state)
        return km.fit_predict(coords)
    except Exception:
        try:
            from scipy.cluster.vq import kmeans2
            _, labels = kmeans2(coords.astype(float), k, minit='points', iter=max_iter)
            return labels
        except Exception:
            # Last resort: slice along longer axis into k contiguous chunks
            y, x = coords[:, 0], coords[:, 1]
            order = np.argsort(x) if (x.max()-x.min()) >= (y.max()-y.min()) else np.argsort(y)
            labels = np.zeros(len(coords), dtype=int)
            edges = np.linspace(0, len(coords), k+1, dtype=int)
            for i in range(k):
                labels[order[edges[i]:edges[i+1]]] = i
            return labels
        
def separate_large_objects(binary_mask, factor=3.0, rounding="floor",
                           carve_boundaries=True, boundary_thickness="thick",
                           return_labels=False, random_state=0, est_nuc_size=6300):
    """
    Split overly-large connected components by clustering their pixels into
    ~median-sized chunks.

    Parameters
    ----------
    binary_mask : 2D bool/0-1 array
    factor : float
        Split components with area > factor * median_area.
    rounding : {'floor','round','ceil'}
        How to convert area/median to number of pieces.
    carve_boundaries : bool
        If True, carve 1-px (or 'thick') zero lines between pieces so a
        *boolean* mask can still be labeled into separate objects later.
    boundary_thickness : {'thick','inner','outer','subpixel'}
        Passed to skimage.segmentation.find_boundaries when carving.
    return_labels : bool
        If True, also return the labeled segmentation and median_area.
    random_state : int

    Returns
    -------
    mask_out : 2D bool
    labels_out (optional) : 2D int (compact labels, 0 is background)
    median_area (optional) : float
    """
    mask = img_as_bool(binary_mask)
    lbl = label(mask, connectivity=1)
    props = regionprops(lbl)
    areas = np.array([p.area for p in props], dtype=float)

    # return array of 0s if there are no obj areas
    if areas.size == 0:
        return (mask if not return_labels else (mask, np.zeros_like(lbl, dtype=int), 0.0))

    median_area = float(est_nuc_size)

    mask_out = np.zeros_like(mask, dtype=bool)
    labels_out = np.zeros_like(lbl, dtype=int)
    next_id = 1

    for p in props:
        coords = p.coords  # (row, col)
        area = float(p.area)

        if area > factor * median_area:
            # decide number of pieces
            ratio = area / median_area
            if rounding == "ceil":
                k = int(np.ceil(ratio))
            elif rounding == "round":
                k = max(2, int(np.round(ratio)))
            else:
                k = max(2, int(np.floor(ratio)))

            k = min(k, len(coords))
            cl = _kmeans_cluster(coords, k, random_state=random_state)

            # Build a mini label map in the bbox to carve boundaries cleanly
            minr, minc, maxr, maxc = p.bbox
            h, w = maxr - minr, maxc - minc
            mini = np.zeros((h, w), dtype=int)

            for ci in range(k):
                part = coords[cl == ci]
                rr, cc = part[:, 0] - minr, part[:, 1] - minc
                mini[rr, cc] = ci + 1  # local labels 1..k

            if carve_boundaries:
                cuts = find_boundaries(mini, mode=boundary_thickness)
                mini_mask = mini > 0
                mini_mask[cuts] = False  # carve zero-width lines between parts
            else:
                mini_mask = mini > 0

            # write boolean mask
            rr, cc = np.nonzero(mini_mask)
            mask_out[minr + rr, minc + cc] = True

            # write integer labels (each cluster gets a global id)
            for ci in range(1, k + 1):
                part_rr, part_cc = np.nonzero(mini == ci)
                labels_out[minr + part_rr, minc + part_cc] = next_id
                next_id += 1
        else:
            # keep as one
            mask_out[tuple(coords.T)] = True
            labels_out[tuple(coords.T)] = next_id
            next_id += 1

    labels_out, _, _ = relabel_sequential(labels_out)

    if return_labels:
        return mask_out, labels_out, median_area
    else:
        return mask_out

def count_puncta(image, imagename, folder_path_output, params, iflog=False):

    original_image = image.copy()  # Save a copy of the original image before processing

    # show images at each step so user can see what is happening.
    # show_image(image, title="Original Image")

    # rolling ball background subtraction
    # Use a disk-shaped structuring element with radius 20 for 2D images
    selem = disk(params['rolling_ball_disk'])
    # Use white_tophat to subtract background (sliding paraboloid approximation)
    image = white_tophat(image, selem)
    # show_image(image, title="After Rolling Ball Background Subtraction")
    log_print(iflog, f"img: {imagename}: Successfully completed background subtraction.")

    # Apply min/max thresholding for 8-bit images
    mask = image >= params['threshold'][0]
    image = np.where(mask, 255, 0).astype(np.uint8)
    mask = image > 0
    # show_image(mask, title="Mask after thresholding")
    log_print(iflog, f"img: {imagename}: Successfully completed thresholding.")

    # Apply watershed to separate touching puncta    
    mask = imagej_style_watershed(mask, min_distance=params['watershed'][0], erode=params['watershed'][1])
    # show_image(mask, title="Mask after Watershed Separation")
    log_print(iflog, f"img: {imagename}: Successfully completed watershed separation.")


    # Remove small objects (optional, can set min_size as needed)
    mask = morphology.remove_small_objects(mask, min_size=params['remove_small'])
    # show_image(mask, title="Mask after removing small objects")

    # remove very large objects (optional, can set max_size as needed)
    mask = remove_large_objects(mask, max_size=params['remove_large'][0], connectivity=params['remove_large'][1])
    # show_image(mask, title="Mask after removing large objects")

    # remove medium-sized objects that have a large area but low intensity--optional
    # remove_large_low_intensity_objects(mask, image, min_size=30, max_intensity=15, connectivity=2)
    # show_image(mask, title="Removed large low intensity objects")
    log_print(iflog, f"img: {imagename}: Successfully removed all irrelevant objects.")


    # show outline of final mask on original image
    boundaries = find_boundaries(mask, mode='outer')
    # Prepare overlay: convert grayscale to RGB if needed
    overlay = np.stack([original_image]*3, axis=-1) if original_image.ndim == 2 else original_image.copy()
    overlay = exposure.rescale_intensity(overlay, out_range=(0, 255)).astype(np.uint8)
    overlay[boundaries] = [255, 255, 0]  # yellow
    # show_image(overlay, title="Mask Outlines (Yellow) on original image", cmap=None)
    
    
    # save overlay image to assess nuclei counting afterwards.
    # Create the overlay_images directory if it doesn't exist
    overlay_dir = os.path.join(folder_path_output, "overlay_images")
    os.makedirs(overlay_dir, exist_ok=True)
    save_imgname = os.path.join(overlay_dir, imagename + '_overlay.png')
    imageio.imwrite(save_imgname, overlay)
    log_print(iflog, f"img: {imagename}: Successfully completed saving of overlay image.")


    # Label connected components
    labeled = measure.label(mask)

    # Measure region properties using the processed image for shape, but use original for intensity
    props = measure.regionprops(labeled)

    # Collect area and raw integrated density for each particle
    data = []
    for prop in props:
        area = prop.area
        # Use the mask for this region to sum over the original image
        coords = prop.coords
        raw_int_density = int(np.sum(original_image[coords[:, 0], coords[:, 1]]))
        data.append({'area': area, 'rawintdens': raw_int_density})

    log_print(iflog, f"img: {imagename}: returning all puncta properties as DF. Total puncta counted: {len(data)}.")

    # Return as a DataFrame
    return pd.DataFrame(data)

def count_nuclei(image, imagename, folder_path_output, params, iflog=False):
    """ Count nuclei in an image using a similar approach to count_puncta, but tailored to the larger,
    less bright nuclei."""

    original_image = image.copy()  # Save a copy of the original image before processing
    # show_image(original_image, title="original image")

    # Use a disk-shaped structuring element with radius 30 for 2D images
    selem = disk(params['rolling_ball_disk'])
    # Use white_tophat to subtract background (sliding paraboloid approximation)
    image = white_tophat(image, selem)
    # show_image(image, title="After Rolling Ball Background Subtraction")
    log_print(iflog, f"img: {imagename}: Successfully completed background subtraction.")

    # Apply thresholding
    mask = (image >= params['threshold'][0]) & (image <= params['threshold'][1])
    # show_image(mask, title="Nuclei Mask")
    log_print(iflog, f"img: {imagename}: Successfully completed thresholding.")

    # erode mask
    mask = erosion(mask, disk(params['erode+dilate']))
    # show_image(mask, title="eroded mask")
    log_print(iflog, f"img: {imagename}: Successfully completed erosion.")

    # dilate mask
    mask = dilation(mask, disk(params['erode+dilate']))
    # show_image(mask, title="dilated mask")
    log_print(iflog, f"img: {imagename}: Successfully completed dilation.")

    # fill small holes in mask
    mask = morphology.remove_small_holes(mask, area_threshold=params['fill_holes_max_size'])
    # show_image(mask, title="Filled Holes in Nuclei Mask")
    log_print(iflog, f"img: {imagename}: Successfully completed hole filling.")

    # Apply watershed to separate touching nuclei
    mask = imagej_style_watershed(mask, min_distance=params['watershed'][0], erode=params['watershed'][1])
    # show_image(mask, title="Separated Mask from Watershed")
    log_print(iflog, f"img: {imagename}: Successfully completed watershed separation.")

    # remove small objects in mask
    mask = morphology.remove_small_objects(mask, min_size=params['remove_small_obj'])
    # show_image(mask, title="Removed Small Objects in Nuclei Mask")
    log_print(iflog, f"img: {imagename}: Successfully completed removal of small objects.")

    # remove very large objects (optional, can set max_size as needed: 40x mag: avg nuc size = 10k, 20x mag ~ 5kpix
    # filters out when there is a lot of background haze that is close to threshold value)
    # mask = remove_large_objects(mask, max_size=100000, connectivity=2)
    # show_image(mask, title="Mask after removing large objects")
    # log_print(iflog, f"img: {imagename}: Successfully completed removal of very large objects.")

    # separate large objects that are > 3x size of median nucleus
    mask = separate_large_objects(mask, factor=params['kmeans_factor'], est_nuc_size=params['est_nuc_size'])  # returns boolean mask with cuts
    # show_image(mask, title="Mask after separating large objects into multiple nuclei")
    log_print(iflog, f"img: {imagename}: Successfully separated very large objects.")

    # show outline of final mask on original image
    boundaries = find_boundaries(mask, mode='outer')
    # Prepare overlay: convert grayscale to RGB if needed
    overlay = np.stack([original_image]*3, axis=-1) if original_image.ndim == 2 else original_image.copy()
    overlay = exposure.rescale_intensity(overlay, out_range=(0, 255)).astype(np.uint8)
    overlay[boundaries] = [255, 255, 0]  # yellow
    # show_image(overlay, title="Mask Outlines (Yellow) on original image", cmap=None)
    # save overlay image to assess nuclei counting afterwards.
    overlay_dir = os.path.join(folder_path_output, "overlay_images")
    os.makedirs(overlay_dir, exist_ok=True)
    save_imgname = os.path.join(overlay_dir, imagename + '_overlay.png')
    imageio.imwrite(save_imgname, overlay)
    log_print(iflog, f"img: {imagename}: Successfully completed overlay saving.")

    # Label connected components
    labeled = measure.label(mask)

    # Measure region properties using the processed image for shape, but use original for intensity
    props = measure.regionprops(labeled)

    # Collect area and raw integrated density for each nucleus
    data = []
    for prop in props:
        area = prop.area
        # Use the mask for this region to sum over the original image
        coords = prop.coords
        raw_int_density = int(np.sum(original_image[coords[:, 0], coords[:, 1]]))
        data.append({'area': area, 'rawintdens': raw_int_density}) # area in pixels, rawintdens is sum pixel intensity in nucleus area

    log_print(iflog, f"img: {imagename}: returning all nuclei properties as DF. Total nuclei counted: {len(data)}.")

    # Return as a DataFrame--contains each nucleus, its area, and its rawintdens
    return pd.DataFrame(data)

def open_and_convert_image(image_path, max_val=65535):
    # open .png or .tif image and convert to grayscale if needed

    if image_path.endswith('.tif'):
        image = Image.open(image_path)
        # show_image(image, title=f"Original imported Image: {filename}")
        image = np.array(image)

    elif image_path.endswith('.png'):
        image = plt.imread(image_path)
    
    else:
        return None
    
    # Ensure the image is in 8-bit format (0-255) without stretching actual min/max
    if image.dtype != np.uint8:
        # If the image is 16-bit, map 0->0 and max_val->255 without stretching actual min/max
        if image.dtype == np.uint16:
            image = exposure.rescale_intensity(image, in_range=(0, max_val), out_range=(0, 255)).astype(np.uint8)
        else:
            image = exposure.rescale_intensity(image, out_range=(0, 255)).astype(np.uint8)
    
    return image # if the file is not a supported image format

def calc_mean_intens(df, area='area', intens='rawintdens'):
    n = len(df['area'])
    temp_ls = []
    for i in range(n):
        temp_ls.append(df[intens][i] / df[area][i])
    
    return temp_ls

def ensure_equal_list_lengths(d):
    """
    Ensure all values in the dictionary are lists of the same length
    as the list at the first key. Raise ValueError if any differ.
    """
    if not d:
        return  # nothing to check

    keys = list(d.keys())
    reference_len = len(d[keys[0]])
    mismatched = [k for k in keys if len(d[k]) != reference_len]

    if mismatched:
        raise ValueError(
            f"The following keys have list lengths different from '{keys[0]}' (length {reference_len}): "
            f"{[(k, len(d[k])) for k in mismatched]}" )

def calc_n_print_sum_stats(data, label="test", iflog=False):

    # calculate + return/print the following values for list "data":
    # mean (float), stdev (float), median (float), 95CI (float,float)

    if len(data) == 0:
        ssum = 0
        mean = 0
        stdev = 0
        med = 0
        ci95 = (float('nan'), float('nan'))
    elif len(data) == 1: 
        ssum = data.sum()
        mean = data.mean()
        stdev = 0
        med = data.median()
        ci95 = (med, med)
    elif len(data) > 1 and np.std(data, ddof=1) == 0:
        ssum = data.sum()
        mean = data.mean()
        med = data.median()
        stdev = 0
        ci95 = (med, med)
    else: # len (data > 1 and stdev > 0)
        ssum = data.sum()
        mean = data.mean()
        stdev = data.std()
        med = data.median()
        ci95 = stats.t.interval(0.95, df=len(data)-1, loc=mean, scale=stdev)

    # print values
    log_print(iflog, f"Summary stats for {label} values: sum = {ssum}, mean = {mean}; stdev = {stdev}, median = {med}, 95%CI = {ci95}")

    return ssum, mean, stdev, med, ci95
    
# return a dictionary with key = metrics, val = value for this puncta image. This will be turned into a 1-line df that is appended to a .csv
def calc_summary_stats(df, imagename, area='area', rawintdens='rawintdens', meanint='meanintens', iflog=False):

    # make sure passed df (puncta data for this image) is actually a numpy dataframe
    if not isinstance(df, pd.DataFrame):
        raise TypeError("The input must be a Pandas DataFrame.")

    # check to make sure all passed column names are in df. if not, throw error
    required_cols = [area, rawintdens, meanint]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")
        
    # check to make sure the return_dict already has all the necessary keys
    keys = ['image_name', 'n_obj', 'pix-intens_avg', 'area_sum', 'area_mean', 'area_stdev', 'area_med', 'area_95CI', 
            'rawintdens_sum', 'rawintdens_mean', 'rawintdens_stdev', 'rawintdens_med', 'rawintdens_95CI',
            'meanintens_sum', 'meanintens_mean', 'meanintens_stdev', 'meanintens_med', 'meanintens_95CI']
    
    # calculate each summary stat and add to new dict under proper key

    return_dict = {}

    a_data = df[area].dropna()
    r_data = df[rawintdens].dropna()
    return_dict[keys[0]] = imagename
    log_print(iflog, f"----------------------\n Summary statistics for image: {imagename}")
    return_dict[keys[1]] = (len(a_data)) # n_obj
    log_print(iflog, f"Total number of objects: {len(a_data)}")
    pix_avg = r_data.sum() / a_data.sum() # pix-intens_avg
    return_dict[keys[2]] = (pix_avg)
    log_print(iflog, f"Average object pixel intensity: {pix_avg}")

    # calculations for object areas
    a_sum, a_mean, a_stdev, a_med, a_ci95 = calc_n_print_sum_stats(a_data, label="area", iflog=iflog)
    return_dict[keys[3]] = a_sum # area_sum
    return_dict[keys[4]] = a_mean # area mean
    return_dict[keys[5]] = a_stdev # area stdev
    return_dict[keys[6]] = a_med # area_med
    return_dict[keys[7]] = a_ci95# area_95CI
    
    # calculations for rawintdens
    r_sum, r_mean, r_stdev, r_med, r_ci95 = calc_n_print_sum_stats(r_data, label="raw integrated density", iflog=iflog)
    return_dict[keys[8]] = r_sum # rawintdens_sum
    return_dict[keys[9]] = r_mean # rawintdens_mean
    return_dict[keys[10]] = r_stdev # rawintdens_stdev
    return_dict[keys[11]] = r_med # rawintdens_med
    return_dict[keys[12]] = r_ci95 # rawintdens_95CI

    # calculations for mean intens
    m_sum, m_mean, m_stdev, m_med, m_ci95 = calc_n_print_sum_stats(df[meanint].dropna(), label="mean object intensity", iflog=iflog)
    return_dict[keys[13]] = m_sum # meanintens_sum
    return_dict[keys[14]] = m_mean # meanintens_mean
    return_dict[keys[15]] = m_stdev # meanintens_stdev
    return_dict[keys[16]] = m_med # meanintens_med
    return_dict[keys[17]] = m_ci95 # meanintens_95CI

    # passed reference of return_dict should be changed automatically; no need to return
    return return_dict

# return true if all strings in the list of ls_matches are contained in search_string
def match_strings(search_string, ls_matches):
    return all(s in search_string for s in ls_matches)

# test match_strings
# s = "My name is Heide and I like to code."
# print(match_strings(s, ["Heide", "I ", "code"])) # true
# print(match_strings(s, ["Heide", "you", "code"])) # false
# print(match_strings(s, ["Heide", "like to", "code", "I", "is"])) # true
# print(match_strings(s, ["bad"])) # false
# print(match_strings(s, [""])) #true

# s2 = "B - 02(fld 04 wv UV - DAPI).tif"
# print(match_strings(s2, ["B - 02", "fld 04"])) # true
# print(match_strings(s2, ["C - 02", "fld 04"])) # false
# print(match_strings(s2, ["B - 02", "fld 14"])) # false
# print(match_strings(s2, ["C - 02", "fld 12"])) # false

# return the appropriate treatment label using the well (contained in the search_string/image title)
def match_treat(treatments, search_string):

    for key in treatments.keys():
        if any(s in search_string for s in treatments[key]):
            return key
        
    return ""

# extract well, field information from an imagename. Incell format, ex: "B - 02(fld 04 wv Blue - FITC).tif"
def extract_well_field(imgname):

    well = imgname[0:6]
    fld = imgname[7:13]

    return well, fld

# test extracting well/field information from imagename:
# s = "B - 02(fld 04 wv Blue - FITC).tif"
# t = "B - 02(fld 04 wv UV - DAPI).tif"
# u = "C - 04(fld 16 wv UV - DAPI).tif"
# v = "F - 03(fld 10 wv Green - dsRed).tif"
# print(extract_well_field(s))
# print(extract_well_field(t))
# print(extract_well_field(u))
# print(extract_well_field(v))

# for a given well, determine the associated treatment
def get_treatment(well, treatments):
    """
    Given a well name, return the associated treatment from the treatments dictionary.
    If the well is not found in the treatments dictionary, return None.
    """
    for treatment, wells in treatments.items():
        if well in wells:
            return treatment
    return None  # or raise an error if preferred

# for a given imagename and a given all_punct/nuc df, return a df that has only the object attributes of the relevant image.
def get_img_alldata(imgname, df):
    keys = df.columns.tolist()
    d = {}
    for key in keys:
        if key != "image_name":
            d[key] = []

    for i in range(len(df[keys[0]])):
        if imgname in df["image_name"][i]:
            for key in d.keys():
                d[key].append(df[key][i])
    
    return pd.DataFrame(d)

def check_filename_fortrt(filename, treatments):

    # check if the passed filename contains any of the treatments. Note that treatments is a dictionary with key = treatment, val = list of wells
    for wells in treatments.values():
        for well in wells:
            if well in filename:
                return True
    return False

# main processing function--process all images in folder and quantify puncta + nuclei summary stats per field, per treatment
def process_images(folder_path, folder_path_output, nuc_params, punct_params, treatments=['none'], iflog=False):

    # used passed values for puncta vs. nuclei images, only open the images that have that in the name

    # make csv of all puncta + their values + assc image name in one sheet of an excel file (all_puncta)
    # for each puncta image, add one row of summary stats to a different csv file (puncta_summary)
    # make csv of al nuclei and their data + assc image name in another sheet of same excel file (all nuclei)
    # for each nuclei image, add one row of summary stats to a different csv file (nuclei_summary)

    # NOTE: if user wants an image to be reprocessed, they MUST delete from puncta_summary/nuc_summary AND all_puncta/all_nuc (or just all_puncta). If not deleted from all_puncta/nuc, then summary stats will just be calculated from values that are already in that image.
    # for images to be re-processed, overlay images will be overwritten.
    # read in puncta_summary and nuclei_summary .csvs as dfs in the folder_output directory to determine if that image has already been processed at the beginning of each loop.
    puncta_summary_path = os.path.join(folder_path_output, 'puncta_summary.csv')
    nuclei_summary_path = os.path.join(folder_path_output, 'nuclei_summary.csv')
    all_puncta_path = os.path.join(folder_path_output, 'all_puncta.csv')
    all_nuclei_path = os.path.join(folder_path_output, 'all_nuclei.csv')

    # open all four .csvs as dfs to check whether values have been deleted from puncta-summary/nuclei-summary before continuing
    if os.path.exists(puncta_summary_path):
        puncta_summary_df = pd.read_csv(puncta_summary_path)
    else:
        puncta_summary_df = pd.DataFrame()
        puncta_summary_df["image_name"] = []

    if os.path.exists(nuclei_summary_path):
        nuclei_summary_df = pd.read_csv(nuclei_summary_path)
    else:
        nuclei_summary_df = pd.DataFrame()
        nuclei_summary_df["image_name"] = []

    if os.path.exists(all_puncta_path):
        all_puncta_df = pd.read_csv(all_puncta_path)
    else:
        all_puncta_df = pd.DataFrame()
        all_puncta_df["image_name"] = []

    if os.path.exists(all_nuclei_path):
        all_nuclei_df = pd.read_csv(all_nuclei_path)
    else:
        all_nuclei_df = pd.DataFrame()
        all_nuclei_df["image_name"] = []


    for filename in os.listdir(folder_path):

        # check if this is a relevant image (either puncta or nuclei) to be processed. if not, skip
        if not punct_params['channel'] in filename and not nuc_params['channel'] in filename:
            continue

        # check if it is listed in the any of treatments. If not, it's not a relevant image--skip
        if check_filename_fortrt(filename, treatments) == False:
            continue

        # check if the image is in the puncta/nuc_summary. If so, image has already been processed--skip.
        if filename in puncta_summary_df["image_name"].values:
            # Image has already been processed
            log_print(iflog, f"Skipping {filename}: already processed.")
            continue

        if filename in nuclei_summary_df["image_name"].values:
            # Image has already been processed
            log_print(iflog, f"Skipping {filename}: already processed.")
            continue
        
        ## these functions were taken out in later iterations with large numbers of objects since all_puncta and all_nuc.csvs would exceed maximum csv size allowance. if this is wanted, can remove quotes.
        '''# check if the image is both the all_puncta/nuc AND puncta/nuc_summary. If so, image has already been processed--skip.
        if filename in puncta_summary_df["image_name"].values and filename in all_puncta_df["image_name"].values:
            # Image has already been processed
            log_print(iflog, f"Skipping {filename}: already processed.")
            continue

        if filename in nuclei_summary_df["image_name"].values and filename in all_nuclei_df["image_name"].values:
            # Image has already been processed
            log_print(iflog, f"Skipping {filename}: already processed.")
            continue'''
        
        '''# check if imagename is in all_puncta/nuc but NOT punct/nuc_summary. If so, 
        # calculate summary stats from the all_punct/nuc .csv and then continue w/o reprocessing image.
        if filename not in puncta_summary_df["image_name"].values and filename in all_puncta_df["image_name"].values:
            log_print(iflog, f"Image {filename} has already been processed, but not in summary .csv. Calcuting PUNCTA summary stats from all_puncta and adding to puncta_summary.csv. Image will NOT be reprocessed.")
            # calculate puncta_summary values from values that are in all_puncta
            df_img_puncta = get_img_alldata(filename, all_puncta_df)
            summary_row = calc_summary_stats(df_img_puncta, filename, iflog=iflog)
            csv_path2 = os.path.join(folder_path_output, 'puncta_summary.csv')
            pd.DataFrame([summary_row]).to_csv(csv_path2, mode='a', header=not os.path.exists(csv_path2), index=False)
            log_print(iflog, f"Finished calculating summary stats for puncta image {filename}. Summary stats added to .csv")
            continue

        if filename not in nuclei_summary_df["image_name"].values and filename in all_nuclei_df["image_name"].values:
            log_print(iflog, f"Image {filename} has already been processed, but not in summary .csv. Calcuting NUCLEI summary stats from all_nuclei and adding to nuclei_summary.csv. Image will NOT be reprocessed.")
            # calculate nuclei_summary values from values that are in all_puncta
            df_img_nuc = get_img_alldata(filename, all_nuclei_df)
            n_obj = df_img_nuc['area'].count()
            pix_avg = df_img_nuc['rawintdens'].sum() / df_img_nuc['area'].sum()
            log_print(iflog, f"Summary stats for nuclei image {filename}:")
            log_print(iflog, f"total nuclei: {n_obj}, average pixel intensity: {pix_avg}")

            # calculate summary stats for this image and append to nuclei summary .csv
            summary_row = {'image_name': filename,
                            'n_obj': n_obj,
                            'pix-intens_avg': pix_avg}

            csv_path2 = os.path.join(folder_path_output, 'nuclei_summary.csv')
            pd.DataFrame([summary_row]).to_csv(csv_path2, mode='a', header=not os.path.exists(csv_path2), index=False)

            log_print(iflog, f"Finished calculating summary stats for nuclei image {filename}. Summary stats added to .csv")
            continue


        # check if imagename is in puncta/nuc_summary but NOT all_punct/nuc. If so, delete value from summary .csv and process as normal.
        if filename in puncta_summary_df["image_name"].values and filename not in all_puncta_df["image_name"].values:
            log_print(iflog, f"Image {filename} found in puncta_summary but not in all_puncta. Removing from summary and reprocessing image.")
            df = puncta_summary_df.drop(puncta_summary_df[puncta_summary_df["image_name"] == filename].index)
            df.to_csv(os.path.join(folder_path_output, 'puncta_summary.csv'), mode='w', header=True, index=False)

        if filename in nuclei_summary_df["image_name"].values and filename not in all_nuclei_df["image_name"].values:
            log_print(iflog, f"Image {filename} found in nuclei_summary but not in all_nuclei. Removing from summary and reprocessing image.")
            df = nuclei_summary_df.drop(nuclei_summary_df[nuclei_summary_df["image_name"] == filename].index)
            df.to_csv(os.path.join(folder_path_output, 'nuclei_summary.csv'), mode='w', header=True, index=False) '''
        
        image_path = os.path.join(folder_path, filename)
        if punct_params['channel'] in filename:
            image = open_and_convert_image(image_path, punct_params['max_img_val'])
        else:
            image = open_and_convert_image(image_path, nuc_params['max_img_val'])


        # Check if the image is valid
        if image is None:
            log_print(iflog, f"Skipping {filename}: unsupported format or not an image.")
            continue

        log_print(iflog, f"Processing image: {filename}")

        # Check if the image is a puncta image or a nuclei image based on the filename. quantify as such.
        # create concatenated dfs, one for the puncta images and one for the nuclei images that contain
        # all information about all objects in every image
        if punct_params['channel'] in filename:
            log_print(iflog, f"Processing as puncta image.")

            # Count puncta in the image
            puncta_data = count_puncta(image, filename, folder_path_output, params=punct_params, iflog=iflog)
            # corner case: puncta data is empty; all vals should be 0
            
            # if image has 0 puncta
            if puncta_data.empty:
                pd_dict = {'area': 0, 'rawintdens': 0, 'image_name': filename, 'meanintens': 0 }
                puncta_data = pd.DataFrame([pd_dict])
                log_print(iflog, f"# obj in area, rawintdens, meanintens, image_name: 0, 0, 0, 0")
                summary_row = {'image_name': filename, 'n_obj': 0, 'pix-intens_avg': 0, 'area_sum': 0, 'area_mean': 0, 'area_stdev': np.nan, 'area_med': 0, 'area_95CI': (np.nan,np.nan),
                               'rawintdens_sum': 0, 'rawintdens_mean': 0, 'rawintdens_stdev': np.nan, 'rawintdens_med': 0, 'rawintdens_95CI': (np.nan,np.nan),
                               'meanintens_sum': 0, 'meanintens_mean': 0, 'meanintens_stdev': np.nan, 'meanintens_med': 0, 'meanintens_95CI': (np.nan,np.nan)}
            # if image has > 0 puncta
            else:
                puncta_data['image_name'] = filename

                # add column to df that denotes the average intensity of each puncta: rawint/area
                puncta_data['meanintens'] = calc_mean_intens(puncta_data)

                log_print(iflog, f"# obj in area, rawintdens, meanintens, image_name: {len(puncta_data['area'])}, {len(puncta_data['rawintdens'])}, {len(puncta_data['meanintens'])}, {len(puncta_data['image_name'])})")

                # calculate summary stats and make a 1-line df, and append to csv.
                summary_row = calc_summary_stats(puncta_data, filename, iflog=iflog)
            
            # append info to all_puncta --> again, removed in later iterations with large numbers of objects since all_nuclei.csv exceeded maxmum csv size allowance
            # csv_path = os.path.join(folder_path_output, 'all_puncta.csv') 
            # puncta_data.to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)

            # append info to puncta_summary
            csv_path2 = os.path.join(folder_path_output, 'puncta_summary.csv')
            pd.DataFrame([summary_row]).to_csv(csv_path2, mode='a', header=not os.path.exists(csv_path2), index=False)
            log_print(iflog, f"Finished processing puncta image {filename}. Summary stats added to .csvs")

        elif nuc_params['channel'] in filename:
            log_print(iflog, f"Processing as nuclei image.")
            # Count nuclei in the image
            nuclei_data = count_nuclei(image, filename, folder_path_output, params=nuc_params, iflog=iflog)

            # if image has 0 nuclei
            if nuclei_data.empty:
                nd_dict = {'area': 0, 'rawintdens': 0, 'image_name': filename}
                nuclei_data = pd.DataFrame([nd_dict])
                log_print(iflog, f"# obj in area, rawintdens, image_name: 0, 0, 0")
                log_print(iflog, f"Summary stats for nuclei image {filename}:")
                log_print(iflog, f"total nuclei: 0, average pixel intensity: 0")

                # set summary stats for img. w/ 0 nuclei
                summary_row = {'image_name': filename, 'n_obj': 0, 'pix-intens_avg': 0}
            
            # if image has > 0 nuclei
            else:
                nuclei_data['image_name'] = filename

                # calculate nuclei summary stats for this image (note: only need n_obj, filename, and avg pix intens)
                n_obj = nuclei_data['area'].count()
                pix_avg = nuclei_data['rawintdens'].sum() / nuclei_data['area'].sum()

                # report stats on image
                log_print(iflog, f"# obj in area, rawintdens, image_name: {len(nuclei_data['area'])}, {len(nuclei_data['rawintdens'])}, {len(nuclei_data['image_name'])})")
                log_print(iflog, f"Summary stats for nuclei image {filename}:")
                log_print(iflog, f"total nuclei: {n_obj}, average pixel intensity: {pix_avg}")

                # calculate summary stats for this image
                summary_row = {'image_name': filename,
                               'n_obj': n_obj,
                               'pix-intens_avg': pix_avg}

            # add to all_nuclei.csv --> again, removed in later iterations with large numbers of objects since all_nuclei.csv exceeded maxmum csv size allowance
            # csv_path = os.path.join(folder_path_output, 'all_nuclei.csv')
            # nuclei_data.to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)

            # append summary row to nuclei_summary.csv
            csv_path2 = os.path.join(folder_path_output, 'nuclei_summary.csv')
            pd.DataFrame([summary_row]).to_csv(csv_path2, mode='a', header=not os.path.exists(csv_path2), index=False)

            log_print(iflog, f"Finished processing nuclei image {filename}. Summary stats added to .csvs.")


    # END: all image processing
    log_print(iflog, f"All images processed. Data stored in {folder_path_output} in files all_puncta.csv, puncta_summary.csv, all_nuclei.csv, and nuclei_summary.csv.")

# takes the dictionary outputs for all objects from process_images and outputs to excel file + summarizes, organizes, normalizes
# data, and compile by treatment into final excel sheet that is compatible with graphpad prism.
def output_to_excel(folder_path_output, excel_title, treatments, iflog=False):

    # read in .csvs from folder_path_output to get dfs for: df_punct_total, df_nuc_total, df_punct, df_nuc.
    # csv_path1 = os.path.join(folder_path_output, 'all_puncta.csv') --> again, removed in later iterations with large numbers of objects since all_nuclei.csv exceeded maxmum csv size allowance
    # csv_path2 = os.path.join(folder_path_output, 'all_nuclei.csv') --> again, removed in later iterations with large numbers of objects since all_nuclei.csv exceeded maxmum csv size allowance
    csv_path3 = os.path.join(folder_path_output, 'puncta_summary.csv')
    csv_path4 = os.path.join(folder_path_output, 'nuclei_summary.csv')

    # df_punct_total = pd.read_csv(csv_path1)
    # df_nuc_total = pd.read_csv(csv_path2)
    df_punct = pd.read_csv(csv_path3)
    df_nuc = pd.read_csv(csv_path4)

    log_print(iflog, f"First four dfs created from read-in .csvs: all_puncta, all_nuclei, puncta_summary, nuclei_summary.")
    log_print(iflog, f"First two dfs created from read-in .csvs: puncta_summary, nuclei_summary.")


    # BEGIN: assess summary stats per field for puncta/nuclei + output compiled information for matched field/well

    # for both options, we'd like to have matching field-well information for puncta-nuclei
    # create a dictionary where the keys are the names of all the puncta images, and the values are the names of the matching nuclei image
    dict_punct_nuc_match = {}
    for puncta_img in df_punct['image_name']:
        # extract well, field information from the puncta image name
        well, fld = extract_well_field(puncta_img)
        # create a value for the matching nuclei image
        nuc_img = f"{well}({fld} wv UV - DAPI).tif"
        # add to the dictionary
        dict_punct_nuc_match[puncta_img] = nuc_img

    log_print(iflog, f"Beginning option 1 processing: output mean/sd/n for each treatment + each metric into 1 sheet of excel file {excel_title}.")

    # OPTION 1: assess statistics of each treatment via mean, sd, n (not stacked replicate values)
    # for option 1: we first need to make an intermediary excel sheet where all information for each matching puncta-nuclei image (each well/field) is
    # concatenated together (ie: one row = 1 well/field). Each row will have the following columns:
    # well, field, n_puncta, n_nuclei, punct_per_nuc, mean_puncta_area, med_puncta_area, mean_puncta_rawintdens, med_puncta_rawintdens, mean_puncta_meanintens, med_puncta_meanintens
    # make dict to hold all data that can easily be converted to pandas df
    dict_wellfield_summary = {'well': [], 'field': [], 'n_puncta': [], 'n_nuclei': [], 'punct_per_nuc': [], 
                              'sumpunctarea_pernuc': [], 'sumrawintdens_pernuc': [],
                              'mean_puncta_area': [], 'med_puncta_area': [], 'mean_puncta_rawintdens': [], 
                              'med_puncta_rawintdens': [], 'mean_puncta_meanintens': [], 'med_puncta_meanintens': []}
    for i in range(len(df_punct['image_name'])):

        # extract well, field information from the puncta image name, append to dictionary list
        puncta_img = df_punct['image_name'][i]
        well, fld = extract_well_field(puncta_img)

        # get the matching nuclei image name + the matching index of that nuclei image in the df_nuc_summary
        nuc_img = dict_punct_nuc_match[puncta_img]
        # Find the index of the matching nuclei image in df_nuc
        nuc_index_list = df_nuc.index[df_nuc['image_name'] == nuc_img].tolist()
        if not nuc_index_list:
            log_print(iflog, f"Warning: No matching nuclei image found for {puncta_img}. Skipping this entry.")
            continue
        nuc_index = nuc_index_list[0]

        # get the number of puncta and nuclei for this well/field
        n_puncta = df_punct['n_obj'][i]
        n_nuclei = df_nuc['n_obj'][nuc_index] # NOTE: in the wellfield summary, include 0punct/0 nuc images. When concatenating, filter out 0nuc images.
        # calculate punct_per_nuc
        punct_per_nuc = n_puncta / n_nuclei if n_nuclei > 0 else -1 # avoid division by zero nuclei

        # calculate normalized sum(puncta area) / n_nuclei and normalized sum(puncta rawintdens) / n_nuclei
        sumpunctarea_pernuc = df_punct['area_sum'][i] / n_nuclei if n_nuclei > 0 else -1
        sumrawintdens_pernuc = df_punct['rawintdens_sum'][i] / n_nuclei if n_nuclei > 0 else -1

        # all other values are not normalized (puncta values only) and can be added directly from df_punct

        # append relevant values to the dictionary
        dict_wellfield_summary['well'].append(well)
        dict_wellfield_summary['field'].append(fld)
        dict_wellfield_summary['n_puncta'].append(n_puncta)
        dict_wellfield_summary['n_nuclei'].append(n_nuclei)
        dict_wellfield_summary['punct_per_nuc'].append(punct_per_nuc)
        dict_wellfield_summary['sumpunctarea_pernuc'].append(sumpunctarea_pernuc)
        dict_wellfield_summary['sumrawintdens_pernuc'].append(sumrawintdens_pernuc)
    
    dict_wellfield_summary['mean_puncta_area'] = df_punct['area_mean'].tolist()
    dict_wellfield_summary['med_puncta_area'] = df_punct['area_med'].tolist()
    dict_wellfield_summary['mean_puncta_rawintdens'] = df_punct['rawintdens_mean'].tolist()
    dict_wellfield_summary['med_puncta_rawintdens'] = df_punct['rawintdens_med'].tolist()
    dict_wellfield_summary['mean_puncta_meanintens'] = df_punct['meanintens_mean'].tolist()
    dict_wellfield_summary['med_puncta_meanintens'] = df_punct['meanintens_med'].tolist()

    # convert the dictionary to a pandas DataFrame and output to excel sheet
    ensure_equal_list_lengths(dict_wellfield_summary)  # ensure all lists are of equal length
    df_wellfield_summary = pd.DataFrame(dict_wellfield_summary)

    log_print(iflog, f"Well-field summary df created.")

    # continuing with option 1, now we need to separate wells by treatment, then calculate mean, stddev, n for all fields in that treatment
    # the output should be as follows: 
    # row 1 = headers (treatement1mean-treatment1sd-treatmen1n-treatment2mean-treatment2sd-treatment2n...)
    # row 2 = metric#1 (punct_per_nuc): mean/stdev/n of metric1 for treatment1, then mean/stdev/n of metric1 for treatment2, etc.
    # row 3 = metric#2 (mean_puncta_area): mean/stdev/n of metric2 for treatment1, then mean/stdev/n of metric2 for treatment2, etc.

    # create a dictionary to hold the treatment summary stats that can be easily converted to the correctly formated pandas df --> excel sheet
    # populate summary dictionary with keys, using the treatment names (+mean, +stdev, +n) as the keys, and empty lists as values
    treat_summary_stats = {}
    for key in treatments.keys():
        treat_summary_stats[key + '_mean'] = []
        treat_summary_stats[key + '_stdev'] = []
        treat_summary_stats[key + '_n'] = []

    # will need a temporary dictionary to hold the values for each treatment for each metric.
    # the keys will be the treatment names, and the values will be a 2D list where the first dimension is the metric IN ORDER:
    # 0 = punct_per_nuc, 1 = mean_puncta_area, 2 = sumpunctarea_pernuc, 3 = sumrawintdens_pernuc, 4 = med_puncta_area, 5 = mean_puncta_rawintdens, 6 = med_puncta_rawintdens, 7 = mean_puncta_meanintens, 8 = med_puncta_meanintens

    # list of all the metrics of interest (ie, the number of sheets to create)
    metrics = ['n_nuclei', 'punct_per_nuc', 'sumpunctarea_pernuc', 'sumrawintdens_pernuc', 'mean_puncta_area', 'med_puncta_area',
               'mean_puncta_rawintdens', 'med_puncta_rawintdens', 'mean_puncta_meanintens', 'med_puncta_meanintens']

    # and the second dimension is all the values for that metric for that treatment
    treat_metrics_temp = {key: [[] for _ in range(len(metrics))] for key in treatments.keys()}  

    # now, go through the df_wellfield_summary and calculate the mean, stdev, n for each treatment
    for i in range(len(df_wellfield_summary['well'])):
        
        # find treatment associated with this well
        well = df_wellfield_summary['well'][i]
        trt = get_treatment(well, treatments)
        if trt is None:
            continue # skip if no treatment found for this well

        # if there are fewer than 3 nuclei, skip this field
        if df_wellfield_summary['n_nuclei'][i] < 3:
            continue

        # populate treat_metrics_temp
        treat_metrics_temp[trt][0].append(df_wellfield_summary[metrics[0]][i]) # n_nuclei
        treat_metrics_temp[trt][1].append(df_wellfield_summary[metrics[1]][i])  # punct_per_nuc
        treat_metrics_temp[trt][2].append(df_wellfield_summary[metrics[2]][i])  # sumpunctarea_pernuc
        treat_metrics_temp[trt][3].append(df_wellfield_summary[metrics[3]][i])  # sumrawintdens_pernuc
        treat_metrics_temp[trt][4].append(df_wellfield_summary[metrics[4]][i])  # mean_puncta_area
        treat_metrics_temp[trt][5].append(df_wellfield_summary[metrics[5]][i])  # med_puncta_area
        treat_metrics_temp[trt][6].append(df_wellfield_summary[metrics[6]][i])  # mean_puncta_rawintdens
        treat_metrics_temp[trt][7].append(df_wellfield_summary[metrics[7]][i])  # med_puncta_rawintdens
        treat_metrics_temp[trt][8].append(df_wellfield_summary[metrics[8]][i])  # mean_puncta_meanintens
        treat_metrics_temp[trt][9].append(df_wellfield_summary[metrics[9]][i])  # med_puncta_meanintens

    # now calculate the mean, stdev, n for each treatment for each metric
    treat_summary_stats['metric'] = metrics
    for trt, metric_lists in treat_metrics_temp.items():
        treat_summary_stats[trt + '_mean'].extend([np.mean(metric) for metric in metric_lists])  # mean for each metric
        treat_summary_stats[trt + '_stdev'].extend([np.std(metric, ddof=1) for metric in metric_lists])  # stdev for each metric
        treat_summary_stats[trt + '_n'].extend([len(metric) for metric in metric_lists]) # n for each metric

    # convert the treat_summary_stats dictionary to a pandas DataFrame
    treat_summary_df = pd.DataFrame(treat_summary_stats)
    # output ALL DFs to excel sheet
    save_excel = os.path.join(folder_path_output, excel_title)
    with pd.ExcelWriter(save_excel, engine='openpyxl') as writer:
        # df_punct_total.to_excel(writer, sheet_name='all_puncta', index=False)
        # df_nuc_total.to_excel(writer, sheet_name='all_nuclei', index=False)
        df_punct.to_excel(writer, sheet_name='puncta_summary', index=False)
        df_nuc.to_excel(writer, sheet_name='nuclei_summary', index=False)
        df_wellfield_summary.to_excel(writer, sheet_name='well_field_summary', index=False)
        treat_summary_df.to_excel(writer, sheet_name='treat_summary_stats', index=False)
    
    log_print(iflog, f"All dfs written to {excel_title}, including the last one-- 'treat_summary_stats'.")

    # OPTION 2: output treatment stats as STACKED REPLICATES (1 replicate = 1 field). this way, can get box/violin plot in prism.
    # will need a separate dicitonary/df/sheet for EACH metric, where columns are treatments, and each row is a replicate (ie, a field).
    # make this a separate excel sheet, and concatenate _stacked_reps onto the name of excel_title

    new_excel_title = excel_title.replace('.xlsx', '_stacked_reps.xlsx')
    log_print(iflog, f"Beginning option 2 processing: output stacked replicates (avg value for each field) for each treatment. Each metric output onto separate sheet of excel file {new_excel_title}.")
    
    # create a dictionary for each metric, to easily create a pands_df later
    dicts = []
    for i in range(len(metrics)):
        dicts.append({})

    # each dictionary should have treatments as keys, and empty lists as values
    for d in dicts:
        for key in treatments.keys():
            d[key] = []
    
    # now, go through the df_wellfield_summary and populate the dictionaries for each metric
    for i in range(len(df_wellfield_summary['well'])):

        # find treatment associated with this well
        well = df_wellfield_summary['well'][i]
        trt = get_treatment(well, treatments)
        if trt is None:
            continue

        # if there are fewer than 3 nuclei, skip this field
        if df_wellfield_summary['n_nuclei'][i] < 3:
            continue

        # append the value for each metric to the appropriate dictionary
        dicts[0][trt].append(df_wellfield_summary[metrics[0]][i])  # n_nuclei
        dicts[1][trt].append(df_wellfield_summary[metrics[1]][i])  # punct_per_nuc
        dicts[2][trt].append(df_wellfield_summary[metrics[2]][i])  # sumpunctarea_pernuc
        dicts[3][trt].append(df_wellfield_summary[metrics[3]][i])  # sumrawintdens_pernuc
        dicts[4][trt].append(df_wellfield_summary[metrics[4]][i])  # mean_puncta_area
        dicts[5][trt].append(df_wellfield_summary[metrics[5]][i])  # med_puncta_area
        dicts[6][trt].append(df_wellfield_summary[metrics[6]][i])  # mean_puncta_rawintdens
        dicts[7][trt].append(df_wellfield_summary[metrics[7]][i])  # med_puncta_rawintdens
        dicts[8][trt].append(df_wellfield_summary[metrics[8]][i])  # mean_puncta_meanintens
        dicts[9][trt].append(df_wellfield_summary[metrics[9]][i])  # med_puncta_meanintens

    # since some reps may have been omitted (too few nuclei, etc), check if each dictionary has lists of different lengths.
    # is the same length before trying to convert to pandas df. For any lists that are shorter than the max length,
    # append NaN values until they are the same length
    max_length = max(len(d[key]) for d in dicts for key in d)
    for d in dicts:
        for key in d:
            while len(d[key]) < max_length:
                d[key].append(np.nan)

    # now, convert each dictionary to a pandas DataFrame and output to separate excel sheet
    dfs = []
    for d in dicts:
        df = pd.DataFrame(d)
        dfs.append(df)
    
    save_excel2 = os.path.join(folder_path_output, new_excel_title)
    with pd.ExcelWriter(save_excel2, engine='openpyxl') as writer:
        for i in range(len(dfs)):
            df = dfs[i]
            df.to_excel(writer, sheet_name=metrics[i], index=False)


    log_print(iflog, f"Stacked replicate sheets written to {new_excel_title}.")

    # END OPTION 2


# MAIN EXECUTION CODE BLOCK

# define nuclei parameters ### these parameters are changed by users with new iterations of the script
nuc_params = {
    'rolling_ball_disk': 30, 'est_nuc_size': 1000, 'threshold': [7,255], 'erode+dilate': 3, 'fill_holes_max_size': 50, 
    'watershed': [20, 3], 'kmeans_factor': 1.7, 'channel': 'DAPI', 'max_img_val': 45000
}
nuc_params['remove_small_obj'] = nuc_params['est_nuc_size'] // 4

# define puncta parameters ### these parameters are changed by users with new iterations of the script
punct_params = {
    'threshold': [47, 255], 'rolling_ball_disk': 10, 'remove_small': 0, 'remove_large': [600, 2],
    'watershed': [2, 0], 'channel': 'FITC', 'max_img_val': 65535
}

# FOR FULL RUNNING:
iflog = True
folder_path = "./input_images"  # Replace with your folder path where images are stored
folder_path_output = "./output_images"  # Replace with your desired output folder path
excel_title = 'fullrun_output_quantifications.xlsx'
logfile = os.path.join(folder_path_output, "LOG_FILE.txt")
# define wells + labels associated w/ treatments of interest
treatments = {'1uM B': ['E - 05', 'F - 06']}  # define what treatments are associated with what wells on plate. if want to process all images, can pass empty dict {}

initiate_log(iflog, logfile)
log_print(iflog, "-----------------------------------------------------------------------------")
log_print(iflog, "BEGIN: NEW INITIATION OF IMAGE PROCESSING.")
log_print(iflog, "Image processing specifications:\n\n")
log_print(iflog, f"treatments+their associated wells:")
for val in treatments.keys():
    log_print(iflog, f"  {val}: {treatments[val]}")
log_print(iflog, f"puncta images: puncta threshold = {punct_params['threshold']}, processing instructions: \n 8bit conversion from 0-{punct_params['max_img_val']} -> 0-255 \n- Rolling ball background subtraction, disk={punct_params['rolling_ball_disk']}"
          + f"\n- binary threshold at {punct_params['threshold']} \n- watershed separation, min_dist={punct_params['watershed'][0]}, erode={punct_params['watershed'][1]} \n- remove small objects <={punct_params['remove_small']}"
          + f"\n- remove large objects, {punct_params['remove_large'][0]}pix with connectivity {punct_params['remove_large'][1]}")
log_print(iflog, f"nuclei images: nuclei threshold = {nuc_params['threshold']}, estimated nuclei size = {nuc_params['est_nuc_size']}, processing instructions: \n 8bit conversion from 0-{nuc_params['max_img_val']} -> 0-255 \n- Rolling ball background subtraction, disk={nuc_params['rolling_ball_disk']} \n- binary threshold at {nuc_params['threshold']}"
          + f"\n- erode and dilate {nuc_params['erode+dilate']} \n fill small holes <= {nuc_params['fill_holes_max_size']}pix \n- watershed separation, min_dist={nuc_params['watershed'][0]}, erode={nuc_params['watershed'][1]} \n- remove small objects <={nuc_params['remove_small_obj']}"
          + f"\n- separate large objects that are >={nuc_params['kmeans_factor']}x the size of a median nucleus via k-means into n = int(size/median_size) objects")
log_print(iflog, f"Processing images from {folder_path} to {folder_path_output}.")
# note that if you want to process all images, you can pass nothing for "treatments" and it will not skip any images in folder path.
process_images(folder_path, folder_path_output, nuc_params, punct_params, treatments=treatments, iflog=iflog)
output_to_excel(folder_path_output, excel_title, treatments, iflog=iflog)