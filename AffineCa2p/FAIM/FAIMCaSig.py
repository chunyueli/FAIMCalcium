import glob
import os.path
import cv2 as cv
import skimage.io
import numpy as np
import pandas as pd
import itertools as it
from skimage import measure
from copy import copy, deepcopy
from AffineCa2p.cellpose import models, utils
from AffineCa2p.FAIM.asift import affine_detect
from multiprocessing.pool import ThreadPool
from skimage.segmentation import find_boundaries
from AffineCa2p.FAIM.find_obj import init_feature, filter_matches, explore_match

def AlignIm(path_to_FOV, path_to_masks=[], preprocess=False, diameter=None, templateID=0 ,iterNum=100, method='sift'):
    """ perform fully affine invariant method on calcium imaging field-of-view (FOV) images.

    The function save the transformation matmatrices and the registered FOV images under the input folder.

    Parameters
    -------------
    path_to_FOV: str
        the path of the folder containing FOV images
    path_to_masks: str (default [ ])
        the path of the folder containing ROI masks. If the value is empty, the code will automatically extract ROI masks using cellpose. If the ROI masks have already obtained, provide the path to the folder can save time.
    preprocess: bool (default False)
        whether or not to apply contrast adjustment for the original FOV images
    diameter: int (default None)
        neuron diameter. If the default value is None, the diameter will be estimated by Cellpose from the original FOV images. Otherwise, the neuron will be detected based on the given diameter value.
    templateID: int (default 0)
        choose which FOV image as a template for alignment
    iterNum: int (default 100)
        the number of iterations for fully affine invariant method
    method: name of the fully affine invariant method (default ['sift'])
        name of the method:
        'akaze' corresponds to 'AAKAZE'
        'sift' corresponds to 'ASIFT'
        'surf' corresponds to 'ASURF'
        'brisk' corresponds to 'ABRISK'
        'orb' corresponds to 'AORB'

    Returns
    ----------------
    Tmatrices: list of the trnaformation matrices
    regImages: list of the registered FOV images
    regROIs: list of the registered ROIs masks

    """
    files=get_file_names(path_to_FOV)
    generate_summary(templateID, files)
    imgs=[]
    if preprocess==True:
        imgs = Image_enhance_contrast(files)
    else:
        imgs = [skimage.io.imread(f) for f in files]

    nimg = len(imgs)

    if path_to_masks == []:
        model = models.Cellpose(gpu=False, model_type='cyto')
        channels = []
        for idx in range(nimg):
            channels.append([0,0])

        if diameter==None:
            masks, flows, styles, diams = model.eval(imgs, diameter=None, channels=channels)
        else:
            masks, flows, styles, diams = model.eval(imgs, diameter=diameter, channels=channels)

        ROIs_mask = generate_ROIs_mask(masks, imgs)
    else:
        ROI_files=get_file_names(path_to_masks)
        ROIs_mask = [skimage.io.imread(f) for f in ROI_files]


    if not (os.path.exists(path_to_FOV+'/ROIs_mask/')):
        os.makedirs(path_to_FOV+'/ROIs_mask/')
    for i in range(len(files)):
        skimage.io.imsave(path_to_FOV+'/ROIs_mask/' + os.path.split(files[i])[-1], ROIs_mask[i])

    Template = imgs[templateID] # FOV_template
    Template = cv.normalize(Template, Template, 0, 255, cv.NORM_MINMAX)
    Template_ROI = ROIs_mask[templateID]

    Tmatrices=[]
    regImages=[]
    regROIs=[]

    if method=='akaze':
        print('A'+ method.upper() + ' is running')
        for j in range(len(imgs)):
            if j != templateID:
                print('registering ' + os.path.split(files[j])[-1])
                Regimage = imgs[j]
                Regimage = cv.normalize(Regimage, Regimage, 0, 255, cv.NORM_MINMAX)
                Regimage_ROI = ROIs_mask[j]
                T_matrix, regIm, regROI= Apply_affine_methods(Template, Template_ROI, Regimage, Regimage_ROI, iterNum, 'akaze')
                Tmatrices.append(T_matrix)
                regImages.append(regIm)
                regROIs.append(regROI)

        output_results(path_to_FOV, files, templateID, Template, Template_ROI, Tmatrices, regImages, regROIs, 'akaze')
        return Tmatrices, regImages, regROIs

    elif method=='sift':
        print('A'+ method.upper() + ' is running')
        for j in range(len(imgs)):
            if j != templateID:
                print('registering ' +  os.path.split(files[j])[-1])
                Regimage = imgs[j]
                Regimage = cv.normalize(Regimage, Regimage, 0, 255, cv.NORM_MINMAX)
                Regimage_ROI = ROIs_mask[j]
                T_matrix, regIm, regROI= Apply_affine_methods(Template, Template_ROI, Regimage, Regimage_ROI, iterNum, 'sift')
                Tmatrices.append(T_matrix)
                regImages.append(regIm)
                regROIs.append(regROI)

        output_results(path_to_FOV, files, templateID, Template, Template_ROI, Tmatrices, regImages, regROIs, 'sift')
        return Tmatrices, regImages, regROIs

    elif method=='surf':
        print('A'+ method.upper() + ' is running')
        for j in range(len(imgs)):
            if j != templateID:
                print('registering ' +  os.path.split(files[j])[-1])
                Regimage = imgs[j]
                Regimage = cv.normalize(Regimage, Regimage, 0, 255, cv.NORM_MINMAX)
                Regimage_ROI = ROIs_mask[j]
                T_matrix, regIm, regROI= Apply_affine_methods(Template, Template_ROI, Regimage, Regimage_ROI, iterNum, 'surf')
                Tmatrices.append(T_matrix)
                regImages.append(regIm)
                regROIs.append(regROI)

        output_results(path_to_FOV, files, templateID, Template, Template_ROI, Tmatrices, regImages, regROIs, 'surf')
        return Tmatrices, regImages, regROIs

    elif method=='brisk':
        print('A'+ method.upper() + ' is running')
        for j in range(len(imgs)):
            if j != templateID:
                print('registering ' +  os.path.split(files[j])[-1])
                Regimage = imgs[j]
                Regimage = cv.normalize(Regimage, Regimage, 0, 255, cv.NORM_MINMAX)
                Regimage_ROI = ROIs_mask[j]
                T_matrix, regIm, regROI= Apply_affine_methods(Template, Template_ROI, Regimage, Regimage_ROI, iterNum, 'brisk')
                Tmatrices.append(T_matrix)
                regImages.append(regIm)
                regROIs.append(regROI)

        output_results(path_to_FOV, files, templateID, Template, Template_ROI, Tmatrices, regImages, regROIs, 'brisk')
        return Tmatrices, regImages, regROIs

    elif method=='orb':
        print('A'+ method.upper() + ' is running')
        for j in range(len(imgs)):
            if j != templateID:
                print('registering '  + os.path.split(files[j])[-1])
                Regimage = imgs[j]
                Regimage = cv.normalize(Regimage, Regimage, 0, 255, cv.NORM_MINMAX)
                Regimage_ROI = ROIs_mask[j]
                T_matrix, regIm, regROI= Apply_affine_methods(Template, Template_ROI, Regimage, Regimage_ROI, iterNum, 'orb')
                Tmatrices.append(T_matrix)
                regImages.append(regIm)
                regROIs.append(regROI)

        output_results(path_to_FOV, files, templateID, Template, Template_ROI, Tmatrices, regImages, regROIs, 'orb')
        return Tmatrices, regImages, regROIs


def get_file_names(folder):
    image_names = []
    image_names.extend(glob.glob(folder + '/*.png'))
    image_names.extend(glob.glob(folder + '/*.jpg'))
    image_names.extend(glob.glob(folder + '/*.jpeg'))
    image_names.extend(glob.glob(folder + '/*.tif'))
    image_names.extend(glob.glob(folder + '/*.tiff'))
    if image_names==[]:
        print('Load image failed: please check the path')
    elif len(image_names)==1:
        print('Error: the folder needs to contain at least two images')
    else:
        return image_names


def generate_summary(ID, files):
    print('Template image:' + os.path.split(files[ID])[-1])
    regfiles=[]
    for j in range(len(files)):
        if j != ID:
            regfiles.append(os.path.split(files[j])[-1])
    print('Registered images:')
    print(regfiles)


def Image_enhance_contrast(image_names):
    images=[]
    for n in range(len(image_names)):
        img = skimage.io.imread(image_names[n])
        if np.ptp(img)>0:
            img = utils.normalize99(img)
            img = np.clip(img, 0, 1)
        img *= 255
        img = np.uint8(img)
        images.append(img)
    return images


def generate_ROIs_mask(masks, imgs):
    ROIs_mask=[]
    nimg = len(imgs)
    for idx in range(nimg):
        raw_mask= np.zeros((imgs[idx].shape[0], imgs[idx].shape[1]), np.uint8)
        maski = masks[idx]
        for n in range(int(maski.max())):
            ipix = (maski==n+1).nonzero()
            if len(ipix[0])>60:
                raw_mask[ipix[0],ipix[1]] = 255
        ROIs_mask.append(raw_mask)
    return ROIs_mask


def Apply_affine_methods(img2, img2_ROI, img1, img1_ROI, iterNum, method):

    w = np.size(img2,1)
    h = np.size(img2,0)

    feature_name = method + '-flann'
    detector, matcher = init_feature(feature_name)

    # Detect ROI counter on raw ROI
    Img1_contours = measure.find_contours(img1_ROI , 128)
    Img1_Cmax=0
    Img1_K=0
    for n, contour in enumerate(Img1_contours):
        if len(contour[:, 1])>Img1_Cmax:
            Img1_Cmax=len(contour[:, 1])
    for n, contour in enumerate(Img1_contours):
        if len(contour[:, 1])>(Img1_Cmax*(2/3)):
            Img1_K+=1

    # detect features
    r_err=w*h*255
    pool=ThreadPool(processes = cv.getNumberOfCPUs())
    kp1, desc1 = affine_detect(detector, img1, pool=pool)
    kp2, desc2 = affine_detect(detector, img2, pool=pool)

    # choose best H
    H=np.zeros((3,3))
    inliers = 0
    matched = 0

    for i in range(iterNum):
        ROI_temp=deepcopy(img2_ROI)
        raw_matches = matcher.knnMatch(desc1, trainDescriptors = desc2, k = 2) #2
        p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches)
        if len(p1) >= 4:
            temp_H, status = cv.findHomography(p1, p2, cv.RANSAC, 3.0, 150000)
            kp_pairs = [kpp for kpp, flag in zip(kp_pairs, status) if flag]
            temp_inliers=np.sum(status)
            temp_matched=len(status)
            img1_ROIwrap = cv.warpPerspective(img1_ROI, temp_H, (h, w))
            img1_ROIwrap[img1_ROIwrap<255] = 0

            # Detect ROI counter on registered ROI
            Img1wrap_contours = measure.find_contours(img1_ROIwrap , 128)

            Img1wrap_K=0
            for n, contour in enumerate(Img1wrap_contours):
                if len(contour[:, 1])>(Img1_Cmax*(2/3)) and len(contour[:, 1])<Img1_Cmax:
                    Img1wrap_K+=1

            if Img1wrap_K<(Img1_K/2):
                continue

            # L1-Norm
            temp_err =np.array(np.abs(ROI_temp-img1_ROIwrap))
            err=np.sum(temp_err)
            if err < r_err:
                r_err = err
                H = temp_H
                inliers = temp_inliers
                matched = temp_matched
                #cv.imwrite('C:/Users/Chuny/FAIM-master/A5/New folder/'+ str(i) + '.png', img1_ROIwrap)

        else:
            H, status = None, None
            print('%d matches found, not enough for homography estimation' % len(p1))

    if matched>0:
        img1_wrap = cv.warpPerspective(img1, H, (h, w))
        img1_ROI_wrap = cv.warpPerspective(img1_ROI, H, (h, w))
        T=H
    else:
        img1_wrap=np.zeros([h, w], np.uint8)
        img1_ROI_wrap = np.zeros([h, w], np.uint8)
        T= np.zeros([3, 3])

    return T, img1_wrap, img1_ROI_wrap


def output_results(path, files, ID, Template, Template_ROI, Tmatrices, regImages, regROIs, method):
    # save transformation matrix
    if not (os.path.exists(path+'/A' + method.upper() + '/')):
        os.makedirs(path+'/A' + method.upper() + '/')
    k=0
    for i in range(len(files)):
        if i!=ID:
            raw_data = {'Registered_file': [os.path.split(files[i])[1]],
                        'Template_file': [os.path.split(files[ID])[1]],
                        'Transformation_matrix':[Tmatrices[k]]}
            df = pd.DataFrame(raw_data, columns = ['Registered_file', 'Template_file', 'Transformation_matrix'])
            dfsave=path +'/A' + method.upper() + '/'+os.path.split(files[i])[1][:-4]+'.csv'
            df.to_csv(dfsave)

            output_Im=np.zeros([np.size(Template,1), np.size(Template,1), 3], np.uint8)
            outlines1 = np.zeros(Template_ROI.shape, np.bool)
            outlines1[find_boundaries(Template_ROI, mode='inner')] = 1
            outX1, outY1 = np.nonzero(outlines1)
            output_Im[outX1, outY1] = np.array([255, 0, 0])

            outlines2 = np.zeros(regROIs[k].shape, np.bool)
            outlines2[find_boundaries(regROIs[k], mode='inner')] = 1
            outX2, outY2 = np.nonzero(outlines2)
            output_Im[outX2, outY2] = np.array([255, 255, 22])

            img=cv.hconcat([cv.cvtColor(Template, cv.COLOR_GRAY2BGR),cv.cvtColor(regImages[k], cv.COLOR_GRAY2BGR), output_Im])
            skimage.io.imsave(path+'/A' + method.upper() + '/results_' + os.path.split(files[i])[1], img)
            k=k+1
