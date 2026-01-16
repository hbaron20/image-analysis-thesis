# mask and quant: calculate the signal of tau (FITC) within each cell (ex, MAP2, phalloidin, etc)
# mask is created from cell channel (MAP2/phalloidin) and applied to tau channel (FITC) to get signal
# normalized to the total quantified area per cell.
# also do a nuclei count so that the area per #cells, and total rawintdens per #cells can be calculated

# values output per field:
# - average rawintdens per pixel in cell+ area
# - total rawintdens for all pixels in cell+ area
# - total cell area (in pixels)
# - total nuclei count
# - sum cell area in pixels / # nuclei 
# - sum rawintdens / # nuclei

# this is the python/batch translation of the FIJI macro "maskandquant_04152025.ijm", with the additional nuclei count.

import os
import logging
import imageio.v2 as imageio
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage import exposure
from skimage import measure, morphology
from skimage.segmentation import find_boundaries
import imageio.v2 as imageio
from skimage.morphology import erosion, dilation, disk
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage as ndi
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

# extract well, field information from an imagename. Incell format, ex: "B - 02(fld 04 wv Blue - FITC).tif"
def extract_well_field(imgname):

    well = imgname[0:6]
    fld = imgname[7:13]

    return well, fld

# for a given imagename and a given all_tau/nuc df, return a df that has only the object attributes of the relevant image.
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
        # If the image is 16-bit, map 0->0 and max_val->255 without stretching actual min/max differently between images.
        if image.dtype == np.uint16:
            image = exposure.rescale_intensity(image, in_range=(0, max_val), out_range=(0, 255)).astype(np.uint8)
        else:
            image = exposure.rescale_intensity(image, out_range=(0, 255)).astype(np.uint8)
    
    return image # if the file is not a supported image format

def find_tau_image(cell_filename, tau_channel, all_files):

    # for the passed cell image filename, find the associated (same well, field) tau image filename in the list of all_files
    # if not found, return None
    well_field_prefix = cell_filename.split('wv')[0]  # get the part before "wv": This is well-field info
    tau_filename = None
    for f in all_files:
        if tau_channel in f and well_field_prefix in f:
            tau_filename = f
            break

    return tau_filename  # will be None if no associated tau image is found

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
        
def separate_large_objects(binary_mask, factor=5.0, rounding="floor",
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

    # return array as is if the median is less than 50% of the estimated nuclei area
    median_area = float(np.median(areas))
    if median_area <= (0.5 * est_nuc_size):
        return (mask if not return_labels else (mask, lbl, median_area))
    
    # no k-means separation if there are fewer than 6 nuclei to get medians from
    if areas.size < 6:
        return (mask if not return_labels else (mask, lbl, median_area))

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

def check_filename_fortrt(filename, treatments):

    # check if the passed filename contains any of the treatments. Note that treatments is a dictionary with key = treatment, val = list of wells
    for wells in treatments.values():
        for well in wells:
            if well in filename:
                return True
    return False

def mask_and_quant_tau(cell_image, tau_image, cell_filename, tau_filename, folder_path_output, cell_params, iflog=False):

    orig_cell_image = cell_image.copy()  # Save a copy of the original image before processing
    orig_tau_image = tau_image.copy()

    # show images at each step so user can see what is happening.
    # show_image(orig_cell_image, title="Original Cell Image")

    # Apply min/max thresholding for 8-bit images
    mask = cell_image >= cell_params['threshold'][0]
    image = np.where(mask, 255, 0).astype(np.uint8)
    mask = image > 0
    # show_image(mask, title="Cell mask after thresholding")
    log_print(iflog, f"img: {cell_filename}: Successfully completed thresholding.")

    # Remove small objects (optional, can set min_size as needed)
    mask = morphology.remove_small_objects(mask, min_size=cell_params['remove_small'])
    # show_image(mask, title="Cell mask after removing small objects")

    # fill small holes in mask
    mask = morphology.remove_small_holes(mask, area_threshold=cell_params['fill_holes_max_size'])
    # show_image(mask, title="Cell mask after filling small holes")
    log_print(iflog, f"img: {cell_filename}: Successfully completed hole filling.")

    # apply mask created from cell_image to tau_image
    if tau_image.shape != mask.shape:
        raise ValueError("tau_image and mask must have the same shape")
    tau_image = np.where(mask, tau_image, 0).astype(np.uint8)
    log_print(iflog, f"img: {tau_filename}: Successfully applied cell mask to tau image.")
    
    # show outline of final mask on both images (cell, tau)
    boundaries = find_boundaries(mask, mode='outer')
    
    # Prepare overlay: convert grayscale to RGB if needed
    overlay_cell = np.stack([orig_cell_image]*3, axis=-1) if orig_cell_image.ndim == 2 else orig_cell_image.copy()
    overlay_cell = exposure.rescale_intensity(overlay_cell, out_range=(0, 255)).astype(np.uint8)
    overlay_cell[boundaries] = [255, 255, 0]  # yellow
    # show_image(overlay_cell, title=f"Mask Outlines (Yellow) on original {cell_params['channel']} image", cmap=None)

    overlay_tau = np.stack([orig_tau_image]*3, axis=-1) if orig_tau_image.ndim == 2 else orig_tau_image.copy()
    overlay_tau = exposure.rescale_intensity(overlay_tau, out_range=(0, 255)).astype(np.uint8)
    overlay_tau[boundaries] = [255, 255, 0]  # yellow
    # show_image(overlay_tau, title=f"Mask Outlines (Yellow) on original {tau_params['channel']} image", cmap=None)
    
    # save overlay image to assess nuclei counting afterwards.
    # Create the overlay_images directory if it doesn't exist
    overlay_dir = os.path.join(folder_path_output, "overlay_images")
    os.makedirs(overlay_dir, exist_ok=True)
    
    # cell image
    save_imgname = os.path.join(overlay_dir, cell_filename + '_overlay.png')
    imageio.imwrite(save_imgname, overlay_cell)
    log_print(iflog, f"img: {cell_filename}: Successfully completed saving of cell overlay image.")

    # tau image
    save_imgname = os.path.join(overlay_dir, tau_filename + '_overlay.png')
    imageio.imwrite(save_imgname, overlay_tau)
    log_print(iflog, f"img: {tau_filename}: Successfully completed saving of tau overlay image.")


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
        raw_int_density = int(np.sum(orig_tau_image[coords[:, 0], coords[:, 1]]))
        data.append({'area': area, 'rawintdens': raw_int_density})

    log_print(iflog, f"img: {tau_filename}: returning all quantified properties as DF. Total obj counted: {len(data)}.")
    # Return as a DataFrame
    return pd.DataFrame(data)

def count_nuclei(image, imagename, folder_path_output, params, iflog=False):
    """ Count nuclei in an image using a similar approach to count_puncta, but tailored to the larger,
    less bright nuclei."""

    original_image = image.copy()  # Save a copy of the original image before processing
    # show_image(original_image, title="original image")

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

    # separate large objects that are > 4x size of median nucleus
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

# main processing function--process all images in folder and quantify tau signal + nuclei summary stats per field, per treatment
def process_images(folder_path, folder_path_output, nuc_params, tau_params, cell_params, iflog=False, pix=0.02640625, pic_area=332.8*332.8):
    
    # used passed values for tau, nuclei, and cell images, only open the images that have that in the name

    # make csv of all tau + their values + assc image name in one sheet of an excel file (all_tau) **area is in pix
    # for each tau image, add one row of summary stats to a different csv file (tau_summary) **area is in sq.um
    # make csv of all nuclei and their data + assc image name in another sheet of same excel file (all nuclei) ***area is in pix
    # for each nuclei image, add one row of summary stats to a different csv file (nuclei_summary) **area is in sq.um
    # cell images (ex MAP2, phalloidin) are only used to create the mask--no data is saved/quantified from these images.


    # NOTE: if user wants an image to be reprocessed, they MUST delete from tau_summary/nuclei_summary AND all_tau/all_nuclei (or just all_tau). If not deleted from all_tau/nuclei, then summary stats will just be calculated from values that are already in that image.
    # for images to be re-processed, overlay images will be overwritten.
    # read in tau_summary and nuclei_summary .csvs as dfs in the folder_output directory to determine if that image has already been processed at the beginning of each loop.
    tau_summary_path = os.path.join(folder_path_output, 'tau_summary.csv')
    nuclei_summary_path = os.path.join(folder_path_output, 'nuclei_summary.csv')
    all_tau_path = os.path.join(folder_path_output, 'all_tau.csv')
    all_nuclei_path = os.path.join(folder_path_output, 'all_nuclei.csv')

    # open all four .csvs as dfs to check whether values have been deleted from tau-summary/nuclei-summary before continuing
    if os.path.exists(tau_summary_path):
        tau_summary_df = pd.read_csv(tau_summary_path)
    else:
        tau_summary_df = pd.DataFrame()
        tau_summary_df["image_name"] = []

    if os.path.exists(nuclei_summary_path):
        nuclei_summary_df = pd.read_csv(nuclei_summary_path)
    else:
        nuclei_summary_df = pd.DataFrame()
        nuclei_summary_df["image_name"] = []

    if os.path.exists(all_tau_path):
        all_tau_df = pd.read_csv(all_tau_path)
    else:
        all_tau_df = pd.DataFrame()
        all_tau_df["image_name"] = []

    if os.path.exists(all_nuclei_path):
        all_nuclei_df = pd.read_csv(all_nuclei_path)
    else:
        all_nuclei_df = pd.DataFrame()
        all_nuclei_df["image_name"] = []

    all_files = os.listdir(folder_path)
    for filename in all_files:
        # check if this is a relevant image (either cell, nuclei, or tau) to be processed. If not,skip
        if not cell_params['channel'] in filename and not nuc_params['channel'] in filename and not tau_params['channel'] in filename:
            continue

        # check if it is listed in the any of treatments. If not, it's not a relevant image--skip
        if check_filename_fortrt(filename, treatments) == False:
            continue

        # check if the image is both the all_tau/nuc AND tau/nuc_summary. If so, image has already been processed--skip.
        if filename in tau_summary_df["image_name"].values and filename in all_tau_df["image_name"].values:
            # Image has already been processed
            log_print(iflog, f"Skipping {filename}: already processed.")
            continue

        if filename in nuclei_summary_df["image_name"].values and filename in all_nuclei_df["image_name"].values:
            # Image has already been processed
            log_print(iflog, f"Skipping {filename}: already processed.")
            continue
        
        # check if imagename is in all_tau/nuc but NOT tau/nuc_summary. If so,
        # calculate summary stats from the all_tau/nuc .csv and then continue w/o reprocessing image.
        if filename not in tau_summary_df["image_name"].values and filename in all_tau_df["image_name"].values:
            log_print(iflog, f"Image {filename} has already been processed, but summary values are missing in summary .csv. Calcuting TAU summary stats from all_tau and adding to tau_summary.csv. Image will NOT be reprocessed.")
            # calculate tau_summary values from values that are in all_tau
            df_img_tau = get_img_alldata(filename, all_tau_df)
            sumrawint = df_img_tau['rawintdens'].sum()
            sumarea = df_img_tau['area'].sum() # in pixels
            sumarea = sumarea * pix  # convert to sq.um
            avg_rawint = sumrawint / sumarea if sumarea > 0 else 0 # this average is norm by area in sq.um
            pct_coverage = sumarea / (pic_area) * 100 # pic_area needs to be in sq.um
            summary_row = {'image_name': filename,
                            'sum_cell+_rawintdens': sumrawint,
                            'sum_cell+_area': sumarea,
                            'avg_cell+_rawintdens': avg_rawint,
                            'pct_img_coverage': pct_coverage}
            # append all info to summary.csv
            csv_path2 = os.path.join(folder_path_output, 'tau_summary.csv')
            pd.DataFrame([summary_row]).to_csv(csv_path2, mode='a', header=not os.path.exists(csv_path2), index=False)
            log_print(iflog, f"Finished calculating summary stats for tau image {filename}. Summary stats added to .csv")
            continue

        if filename not in nuclei_summary_df["image_name"].values and filename in all_nuclei_df["image_name"].values:
            log_print(iflog, f"Image {filename} has already been processed, but not in summary .csv. Calcuting NUCLEI summary stats from all_nuclei and adding to nuclei_summary.csv. Image will NOT be reprocessed.")
            # calculate nuclei_summary values from values that are in all_nuclei
            df_img_nuc = get_img_alldata(filename, all_nuclei_df)
            n_nuc = df_img_nuc['area'].count() # n_nuclei
            avg_nuc_area = df_img_nuc['area'].sum() / n_nuc if n_nuc > 0 else 0  # in pixels
            avg_nuc_area = avg_nuc_area * pix  # convert to sq.um
            pix_avg = df_img_nuc['rawintdens'].sum() / df_img_nuc['area'].sum() # norm to area in sq.um

            log_print(iflog, f"Summary stats for nuclei image {filename}:")
            log_print(iflog, f"total nuclei: {n_nuc}, average nuc area: {avg_nuc_area}, avg nuc intensity: {pix_avg}")

            # calculate summary stats for this image and append to nuclei summary.csv
            summary_row = {'image_name': filename,
                            'n_nuclei': n_nuc,
                            'avg_nuc_area': avg_nuc_area, # in sq.um
                            'avg_nuc_intens': pix_avg}

            csv_path2 = os.path.join(folder_path_output, 'nuclei_summary.csv')
            pd.DataFrame([summary_row]).to_csv(csv_path2, mode='a', header=not os.path.exists(csv_path2), index=False)

            log_print(iflog, f"Finished calculating summary stats for nuclei image {filename}. Summary stats added to .csv")
            continue


        # check if imagename is in puncta/nuc_summary but NOT all_punct/nuc. If so, delete value from summary .csv and process as normal.
        if filename in tau_summary_df["image_name"].values and filename not in all_tau_df["image_name"].values:
            log_print(iflog, f"Image {filename} found in tau_summary but not in all_tau. Removing from summary and reprocessing image.")
            df = tau_summary_df.drop(tau_summary_df[tau_summary_df["image_name"] == filename].index)
            df.to_csv(os.path.join(folder_path_output, 'tau_summary.csv'), mode='w', header=True, index=False)

        if filename in nuclei_summary_df["image_name"].values and filename not in all_nuclei_df["image_name"].values:
            log_print(iflog, f"Image {filename} found in nuclei_summary but not in all_nuclei. Removing from summary and reprocessing image.")
            df = nuclei_summary_df.drop(nuclei_summary_df[nuclei_summary_df["image_name"] == filename].index)
            df.to_csv(os.path.join(folder_path_output, 'nuclei_summary.csv'), mode='w', header=True, index=False)


        log_print(iflog, f"Processing image: {filename}")

        # Check if the image is a quant (here, referred to as "tau") image or a nuclei image based on the filename. quantify as such.
        # create concatenated dfs, one for the puncta images and one for the nuclei images that contain
        # all information about all objects in every image
        if cell_params['channel'] in filename:
            
            log_print(iflog, f"Processing as cell image.")

            image_path = os.path.join(folder_path, filename)
            cell_image = open_and_convert_image(image_path, max_val=cell_params['max_img_val'])
            # Check if the image is valid
            if cell_image is None:
                log_print(iflog, f"Skipping {filename}: unsupported format or not an image.")
                continue

            # find filename of associated tau image and open
            tau_filename = find_tau_image(filename, tau_params['channel'], all_files)
            image_path2 = os.path.join(folder_path, tau_filename)
            tau_image = open_and_convert_image(image_path2, max_val=tau_params['max_img_val'])
            if tau_image is None:
                log_print(iflog, f"Skipping {filename}: unsupported format or not an image.")
                continue
            log_print(iflog, f"Assc tau image found: {tau_filename}.")

            
            # get tau data from cell image, tau image.
            tau_data = mask_and_quant_tau(cell_image, tau_image, filename, tau_filename, folder_path_output, cell_params, iflog=iflog)

            # if image has 0 tau objects
            if tau_data.empty:
                pd_dict = {'area': 0, 'rawintdens': 0, 'image_name': filename}
                tau_data = pd.DataFrame([pd_dict])
                log_print(iflog, f"# obj in area, rawintdens, image_name: 0, 0, 0")
                summary_row = {'image_name': filename,
                            'sum_cell+_rawintdens': 0,
                            'sum_cell+_area': 0,
                            'avg_cell+_rawintdens': 0,
                            'pct_img_coverage': 0}
            # if image has > 0 tau objects
            else:
                tau_data['image_name'] = filename

                log_print(iflog, f"# obj in area, rawintdens, image_name: {len(tau_data['area'])}, {len(tau_data['rawintdens'])}, {len(tau_data['image_name'])})")

                sumrawint = tau_data['rawintdens'].sum()
                sumarea = tau_data['area'].sum() # in pixels
                sumarea = sumarea * pix  # convert to sq.um
                avg_rawint = sumrawint / sumarea if sumarea > 0 else 0 # this average is norm by area in sq.um
                pct_coverage = sumarea / (pic_area) * 100

                summary_row = {'image_name': filename,
                            'sum_cell+_rawintdens': sumrawint,
                            'sum_cell+_area': sumarea,
                            'avg_cell+_rawintdens': avg_rawint,
                            'pct_img_coverage': pct_coverage}

            # append info to all_tau
            csv_path = os.path.join(folder_path_output, 'all_tau.csv')
            tau_data.to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)

            # append info to tau_summary
            csv_path2 = os.path.join(folder_path_output, 'tau_summary.csv')
            pd.DataFrame([summary_row]).to_csv(csv_path2, mode='a', header=not os.path.exists(csv_path2), index=False)
            log_print(iflog, f"Finished processing tau image {filename}. Summary stats added to .csvs")

        elif nuc_params['channel'] in filename:

            log_print(iflog, f"Processing as nuclei image.")

            # open image

            image_path = os.path.join(folder_path, filename)
            image = open_and_convert_image(image_path, max_val=nuc_params['max_img_val'])
            # Check if the image is valid
            if image is None:
                log_print(iflog, f"Skipping {filename}: unsupported format or not an image.")
                continue

            # Count nuclei in the image
            nuclei_data = count_nuclei(image, filename, folder_path_output, nuc_params, iflog=iflog)
            
            # if image has 0 nuclei
            if nuclei_data.empty:
                nd_dict = {'area': 0, 'rawintdens': 0, 'image_name': filename}
                nuclei_data = pd.DataFrame([nd_dict])
                log_print(iflog, f"# obj in area, rawintdens, image_name: 0, 0, 0")
                log_print(iflog, f"Summary stats for nuclei image {filename}:")
                log_print(iflog, f"total nuclei: 0, average pixel intensity: 0")

                summary_row = {'image_name': filename,
                            'n_nuclei': 0,
                            'avg_nuc_area': 0, 
                            'avg_nuc_intens': 0}
            
            # if image has > 0 nuclei
            else:
                nuclei_data['image_name'] = filename

                # calculate nuclei summary stats for this image
                n_nuc = nuclei_data['area'].count() # n_nuclei
                avg_nuc_area = nuclei_data['area'].sum() / n_nuc if n_nuc > 0 else 0  # in pixels
                avg_nuc_area = avg_nuc_area * pix  # convert to sq.um
                pix_avg = nuclei_data['rawintdens'].sum() / nuclei_data['area'].sum() # norm to area in sq.um   

                # report stats on image
                log_print(iflog, f"# obj in area, rawintdens, image_name: {len(nuclei_data['area'])}, {len(nuclei_data['rawintdens'])}, {len(nuclei_data['image_name'])})")
                log_print(iflog, f"Summary stats for nuclei image {filename}:")
                log_print(iflog, f"total nuclei: {n_nuc}, average pixel intensity: {pix_avg}")

                # calculate summary stats for this image
                summary_row = {'image_name': filename,
                            'n_nuclei': n_nuc,
                            'avg_nuc_area': avg_nuc_area, # in sq.um
                            'avg_nuc_intens': pix_avg}
               

            # add to all_nuclei.csv
            csv_path = os.path.join(folder_path_output, 'all_nuclei.csv')
            nuclei_data.to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)

            # append summary row to nuclei_summary.csv
            csv_path2 = os.path.join(folder_path_output, 'nuclei_summary.csv')
            pd.DataFrame([summary_row]).to_csv(csv_path2, mode='a', header=not os.path.exists(csv_path2), index=False)

            log_print(iflog, f"Finished processing nuclei image {filename}. Summary stats added to .csvs.")


    # END: all image processing
    log_print(iflog, f"All images processed. Data stored in {folder_path_output} in files all_tau.csv, tau_summary.csv, all_nuclei.csv, and nuclei_summary.csv.")

# takes the dictionary outputs for all objects from process_images and outputs to excel file + summarizes, organizes, normalizes
# data, and compile by treatment into final excel sheet that is compatible with graphpad prism.
def output_to_excel(folder_path_output, excel_title, treatments, iflog=False):

    # read in .csvs from folder_path_output to get dfs for: df_punct_total, df_nuc_total, df_punct, df_nuc.
    csv_path1 = os.path.join(folder_path_output, 'all_tau.csv')
    csv_path2 = os.path.join(folder_path_output, 'all_nuclei.csv')
    csv_path3 = os.path.join(folder_path_output, 'tau_summary.csv')
    csv_path4 = os.path.join(folder_path_output, 'nuclei_summary.csv')

    df_tau_total = pd.read_csv(csv_path1)
    df_nuc_total = pd.read_csv(csv_path2)
    df_tau = pd.read_csv(csv_path3)
    df_nuc = pd.read_csv(csv_path4)

    log_print(iflog, f"First four dfs created from read-in .csvs: all_tau, all_nuclei, tau_summary, nuclei_summary.")

    # BEGIN: assess summary stats per field for tau/nuclei + output compiled information for matched field/well

    # for both options, we'd like to have matching field-well information for tau-nuclei
    # create a dictionary where the keys are the names of all the tau images, and the values are the names of the matching nuclei image
    dict_tau_nuc_match = {}
    for tau_img in df_tau['image_name']:
        # extract well, field information from the tau image name
        well, fld = extract_well_field(tau_img)
        # create a value for the matching nuclei image
        nuc_img = f"{well}({fld} wv UV - DAPI).tif"
        # add to the dictionary
        dict_tau_nuc_match[tau_img] = nuc_img

    log_print(iflog, f"Beginning option 1 processing: output mean/sd/n for each treatment + each metric into 1 sheet of excel file {excel_title}.")

    # OPTION 1: assess statistics of each treatment via mean, sd, n (not stacked replicate values)
    # for option 1: we first need to make an intermediary excel sheet where all information for each matching tau-nuclei image (each well/field) is
    # concatenated together (ie: one row = 1 well/field). 
    dict_wellfield_summary = {'well': [], 'field': [], 'n_nuclei': [], 'sum_cell+_rawintdens': [], 
                              'sum_cell+_area': [], 'avg_cell+_rawintdens': [], 'pct_img_coverage': [], 
                              'avg_nuc_area': [], 'avg_nuc_intens': [], 'sum_cell+_rawintdens_pernuc': [], 'sum_cell+_area_pernuc': []}
    for i in range(len(df_tau['image_name'])):

        # extract well, field information from the tau image name, append to dictionary list
        tau_img = df_tau['image_name'][i]
        well, fld = extract_well_field(tau_img)

        # get the matching nuclei image name + the matching index of that nuclei image in the df_nuc_summary
        nuc_img = dict_tau_nuc_match[tau_img]
        # Find the index of the matching nuclei image in df_nuc
        nuc_index_list = df_nuc.index[df_nuc['image_name'] == nuc_img].tolist()
        if not nuc_index_list:
            log_print(iflog, f"Warning: No matching nuclei image found for {tau_img}. Skipping this entry.")
            continue
        nuc_index = nuc_index_list[0]

        # get the number of tau obj and nuclei for this well/field
        n_nuclei = df_nuc['n_nuclei'][nuc_index] # NOTE: in the wellfield summary, include 0punct/0 nuc images. When concatenating, filter out 0nuc images.
    
        # calculate normalized sum(tau area) / n_nuclei and normalized sum(tau rawintdens) / n_nuclei
        sumarea_pernuc = df_tau['sum_cell+_area'][i] / n_nuclei if n_nuclei > 0 else -1
        sumrawintdens_pernuc = df_tau['sum_cell+_rawintdens'][i] / n_nuclei if n_nuclei > 0 else -1

        # all other values are not normalized (puncta values only) and can be added directly from df_punct

        # append relevant values to the dictionary
        dict_wellfield_summary['well'].append(well)
        dict_wellfield_summary['field'].append(fld)
        dict_wellfield_summary['n_nuclei'].append(n_nuclei)
        dict_wellfield_summary['sum_cell+_rawintdens'].append(df_tau['sum_cell+_rawintdens'][i])
        dict_wellfield_summary['sum_cell+_area'].append(df_tau['sum_cell+_area'][i])
        dict_wellfield_summary['avg_cell+_rawintdens'].append(df_tau['avg_cell+_rawintdens'][i])
        dict_wellfield_summary['pct_img_coverage'].append(df_tau['pct_img_coverage'][i])
        dict_wellfield_summary['avg_nuc_area'].append(df_nuc['avg_nuc_area'][nuc_index])
        dict_wellfield_summary['avg_nuc_intens'].append(df_nuc['avg_nuc_intens'][nuc_index])
        dict_wellfield_summary['sum_cell+_rawintdens_pernuc'].append(sumrawintdens_pernuc)
        dict_wellfield_summary['sum_cell+_area_pernuc'].append(sumarea_pernuc)
        

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
    # the keys will be the treatment names, and the values will be a 2D list where the first dimension is the metric IN ORDER

    # 0 = n_nuclei, 1 = sum_cell+_rawintdens, 2 = sum_cell+_area... 8 = sum_cell+_area_pernuc

    # list of all the metrics of interest (ie, the number of sheets to create)
    metrics = ['n_nuclei', 'sum_cell+_rawintdens', 'sum_cell+_area', 'avg_cell+_rawintdens', 'pct_img_coverage', 
                'avg_nuc_area', 'avg_nuc_intens', 'sum_cell+_rawintdens_pernuc', 'sum_cell+_area_pernuc']

    # and the second dimension is all the values for that metric for that treatment
    treat_metrics_temp = {key: [[] for _ in range(len(metrics))] for key in treatments.keys()}  

    # now, go through the df_wellfield_summary and calculate the mean, stdev, n for each treatment
    for i in range(len(df_wellfield_summary['well'])):
        
        # find treatment associated with this well
        well = df_wellfield_summary['well'][i]
        trt = get_treatment(well, treatments)
        if trt is None:
            continue # skip if no treatment found for this well

        # if there are < 3 nuclei, skip this field
        if df_wellfield_summary['n_nuclei'][i] < 3:
            continue

        # populate treat_metrics_temp
        treat_metrics_temp[trt][0].append(df_wellfield_summary[metrics[0]][i]) # n_nuclei
        treat_metrics_temp[trt][1].append(df_wellfield_summary[metrics[1]][i]) # sum_cell+_rawintdens
        treat_metrics_temp[trt][2].append(df_wellfield_summary[metrics[2]][i]) # sum_cell+_area
        treat_metrics_temp[trt][3].append(df_wellfield_summary[metrics[3]][i]) # avg_cell+_rawintdens
        treat_metrics_temp[trt][4].append(df_wellfield_summary[metrics[4]][i]) # pct_img_coverage
        treat_metrics_temp[trt][5].append(df_wellfield_summary[metrics[5]][i]) # avg_nuc_area
        treat_metrics_temp[trt][6].append(df_wellfield_summary[metrics[6]][i]) # avg_nuc_intens
        treat_metrics_temp[trt][7].append(df_wellfield_summary[metrics[7]][i]) # sum_cell+_rawintdens_pernuc
        treat_metrics_temp[trt][8].append(df_wellfield_summary[metrics[8]][i]) # sum_cell+_area_pernuc

    # now calculate the mean, stdev, n for each treatment for each metric
    treat_summary_stats['metric'] = metrics
    for trt, metric_lists in treat_metrics_temp.items():
        for metric in metric_lists:
            if len(metric) > 0:
                treat_summary_stats[trt + '_mean'].append(np.mean(metric))
                treat_summary_stats[trt + '_stdev'].append(np.std(metric, ddof=1))
                treat_summary_stats[trt + '_n'].append(len(metric))
            else:
                treat_summary_stats[trt + '_mean'].append(0)
                treat_summary_stats[trt + '_stdev'].append(0)
                treat_summary_stats[trt + '_n'].append(0)

    # convert the treat_summary_stats dictionary to a pandas DataFrame
    treat_summary_df = pd.DataFrame(treat_summary_stats)
    # output ALL DFs to excel sheet
    save_excel = os.path.join(folder_path_output, excel_title)
    with pd.ExcelWriter(save_excel, engine='openpyxl') as writer:
        df_tau_total.to_excel(writer, sheet_name='all_tauobj', index=False)
        df_nuc_total.to_excel(writer, sheet_name='all_nuclei', index=False)
        df_tau.to_excel(writer, sheet_name='tau_summary', index=False)
        df_nuc.to_excel(writer, sheet_name='nuclei_summary', index=False)
        df_wellfield_summary.to_excel(writer, sheet_name='well_field_summary', index=False)
        treat_summary_df.to_excel(writer, sheet_name='treat_summary_stats', index=False)
    
    log_print(iflog, f"All dfs written to {excel_title}, including the last one-- 'treat_summary_stats'.")

    # OPTION 2: output treatment stats as STACKED REPLICATES (1 replicate = 1 field). this way, can get box/violin plot in prism.
    # will need a separate dicitonary/df/sheet for EACH metric, where columns are treatments, and each row is a replicate (ie, a field).
    # make this a separate excel sheet, and concatenate _stacked_reps onto the name of excel_title

    new_excel_title = excel_title.replace('.xlsx', '_stacked_reps.xlsx')
    log_print(iflog, f"Beginning option 2 processing: output stacked replicates (avg value for each field) for each treatment. Each metric output onto separate sheet of excel file {new_excel_title}.")
    
    # create a dictionary for each metric, to easily create a pandas_df later
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

        # if there are less than 3 nuclei, skip this field
        if df_wellfield_summary['n_nuclei'][i] < 3:
            continue

        # append the value for each metric to the appropriate dictionary
        dicts[0][trt].append(df_wellfield_summary[metrics[0]][i]) # n_nuclei
        dicts[1][trt].append(df_wellfield_summary[metrics[1]][i]) # sum_cell+_rawintdens
        dicts[2][trt].append(df_wellfield_summary[metrics[2]][i]) # sum_cell+_area
        dicts[3][trt].append(df_wellfield_summary[metrics[3]][i]) # avg_cell+_rawintdens
        dicts[4][trt].append(df_wellfield_summary[metrics[4]][i]) # pct_img_coverage
        dicts[5][trt].append(df_wellfield_summary[metrics[5]][i]) # avg_nuc_area
        dicts[6][trt].append(df_wellfield_summary[metrics[6]][i]) # avg_nuc_intens
        dicts[7][trt].append(df_wellfield_summary[metrics[7]][i]) # sum_cell+_rawintdens_pernuc
        dicts[8][trt].append(df_wellfield_summary[metrics[8]][i]) # sum_cell+_area_pernuc

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

### MAIN CODE BLOCK ###

# define nuclei parameters ## Changed by user with each run
nuc_params = {
    'est_nuc_size': 6300, 'threshold': [4,255], 'erode+dilate': 2, 'fill_holes_max_size': 400, 
    'watershed': [30, 3], 'kmeans_factor': 3.0, 'channel': 'DAPI', 'max_img_val': 40000
}
nuc_params['remove_small_obj'] = nuc_params['est_nuc_size'] // 4

# define tau parameters ## changed by user with each run
tau_params = {
    'channel': 'FITC', 'max_img_val': 65535
}

# define cell parameters
cell_params = {
    'threshold': [3, 255], 'remove_small': 800, 'channel': 'dsRed', 'fill_holes_max_size': 10000, 'max_img_val': 30000
}

# ALL AREA OUTPUTS FOR THIS SCRIPT ARE IN SQ.um, NOT PIXELS! easy to convert/rerun--by changing pixel value to 1.0
# sq.um per sq.pixel. This value will change for different magnification. Easy way to find out the value--open image in fiji and look at the top--
# it should say something like: 332.8x332.8um (2048x2048 pix). math=(332.8*332.8)/(2048*2048)=0.02640625 sq.um/sq.pix
# NOTE: pix is ONLY converted to um at the very end after all the image processing is done. All intermediate steps are done in pixels.
pix = 0.02640625  # for 40x images with 2048x2048 pix and 332.8x332.8um

# image area in sq.um. This is only used for calculating % image coverage.
pic_area = 332.8*332.8

folder_path = "./input_images" # replace with folder that contains raw input images to be processed
folder_path_output = "./output_images" # replace with folder that will contain output overlay images + .csv + excel files"
excel_title = "fullrun_output_quantifications.xlsx"
iflog = True
logfile = os.path.join(folder_path_output, "LOG_FILE.txt")
treatments = {'1: treament A': ['B - 02', 'C - 02']} # replace with dictionary of treatments and their associated wells, can pass empty dict {} if you want all wells to be processed


initiate_log(iflog, logfile)
log_print(iflog, "-----------------------------------------------------------------------------")
log_print(iflog, "BEGIN: NEW INITIATION OF IMAGE PROCESSING.")
log_print(iflog, f"treatments+their associated wells:")
for val in treatments.keys():
    log_print(iflog, f"  {val}: {treatments[val]}")
log_print(iflog, "Image processing specifications:\n\n")
log_print(iflog, f"cell images: processing instructions (creation of mask to use on tau images): "
          + f"\n- open img, set b/c to 0-{cell_params['max_img_val']}, convert to 8bit \n- binary threshold at {cell_params['threshold']} \n- remove small objects <={cell_params['remove_small']}pix"
          + f"\n- fill holes <{cell_params['fill_holes_max_size']}pix \n- open tau img, set b/c to 0-{tau_params['max_img_val']}, convert to 8bit\n- apply mask to tau_image and report tau signal/area within the masked area only")
log_print(iflog, f"nuclei images: nuclei threshold = {nuc_params['threshold']}, processing instructions: \n- erode, dilate: {nuc_params['erode+dilate']}pix"
          + f"\n- open nuc img, set b/c to 0-{nuc_params['max_img_val']}, convert to 8bit \n- binary threshold at {nuc_params['threshold']} \n fillsmall holes <= {nuc_params['fill_holes_max_size']}pix \n- watershed separation: min_dist={nuc_params['watershed'][0]}, erode={nuc_params['watershed'][1]}"
          + f"\n- separate large objects that are >={nuc_params['kmeans_factor']}x the size of a median nucleus via k-means into n = int(size/median_size) objects")
log_print(iflog, f"Processing images from {folder_path} to {folder_path_output}.")
process_images(folder_path, folder_path_output, nuc_params, tau_params, cell_params, iflog=iflog, pix=pix, pic_area=pic_area)
output_to_excel(folder_path_output, excel_title, treatments, iflog=iflog)

