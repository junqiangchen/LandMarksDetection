from __future__ import print_function, division
import SimpleITK as sitk
import pandas as pd
import math
import numpy as np
import os
import matplotlib.pyplot as plt


def reduce_dimension(image, axis=None):
    dim = image.GetDimension()
    if axis is None:
        axis = dim - 1
    size = list(image.GetSize())
    assert size[axis] == 1, 'size in dimension to reduce must be 1'
    size[axis] = 0
    index = [0] * dim
    return sitk.Extract(image, size, index)


def resize_image_itk(itkimage, newSize, resamplemethod=sitk.sitkLinear):
    """
    image resize withe sitk resampleImageFilter
    :param itkimage:
    :param newSpacing:such as [1,1,1]
    :param resamplemethod:
    :return:
    """
    dim = itkimage.GetDimension()
    newSize = np.array(newSize, float)
    # originSpcaing = itkimage.GetSpacing()
    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()
    originSpcaing = itkimage.GetSpacing()
    factor = originSize / newSize
    newSpacing = factor * originSpcaing
    newSize = newSize.astype(np.int)
    resampler.SetReferenceImage(itkimage)
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetSize(newSize.tolist())
    resampler.SetTransform(sitk.Transform(dim, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)
    return itkimgResampled


def resizeimageandlandmarks(itkimage, shape, landmarkdata):
    itkimagesize = itkimage.GetSize()
    offsetx, offsety = shape[0] / itkimagesize[0], shape[1] / itkimagesize[1]
    # step 1 rezie image to shape size
    rezieitkimage = resize_image_itk(itkimage, newSize=shape)
    # step 2 resize landmark to shape size
    resizelandmarkdata = np.around(landmarkdata * np.array((offsetx, offsety)))
    return rezieitkimage, resizelandmarkdata


def normalize(slice, bottom=99, down=1):
    """
    normalize image with mean and std for regionnonzero,and clip the value into range
    :param slice:
    :param bottom:
    :param down:
    :return:
    """
    b = np.percentile(slice, bottom)
    t = np.percentile(slice, down)
    slice = np.clip(slice, t, b)

    image_nonzero = slice[np.nonzero(slice)]
    if np.std(slice) == 0 or np.std(image_nonzero) == 0:
        return slice
    else:
        tmp = (slice - np.mean(image_nonzero)) / np.std(image_nonzero)
        return tmp


def gen_image_mask(srcimg, seg_image, index, trainImage, trainMask):
    if os.path.exists(trainImage) is False:
        os.mkdir(trainImage)
    if os.path.exists(trainMask) is False:
        os.mkdir(trainMask)
    filepath = trainImage + "\\" + str(index) + ".npy"
    filepath2 = trainMask + "\\" + str(index) + ".npy"
    np.save(filepath, srcimg)
    np.save(filepath2, seg_image)


def onelandmarktoheatmap(srcimage, coords, sigma, sigma_scale_factor=1.0, size_sigma_factor=10, normalize_center=True):
    """
    Generates a numpy array of the landmark image for the specified point and parameters.
    :param srcimage:input src image
    :param coords:one landmark coords on src image([x], [x, y] or [x, y, z]) of the point.
    :param sigma:Sigma of Gaussian
    :param sigma_scale_factor:Every value of the gaussian is multiplied by this value.
    :param size_sigma_factor:the region size for which values are being calculated
    :param normalize_center:if true, the value on the center is set to scale_factor
                             otherwise, the default gaussian normalization factor is used
    :return:heatmapimage
    """
    # landmark holds the image
    srcimage = np.squeeze(srcimage)
    image_size = np.shape(srcimage)
    assert len(image_size) == len(coords), "image dim is not equal landmark coords dim"
    dim = len(coords)
    heatmap = np.zeros(image_size, dtype=np.float)
    # flip point is form [x, y, z]
    flipped_coords = coords
    region_start = (flipped_coords - sigma * size_sigma_factor / 2).astype(int)
    region_end = (flipped_coords + sigma * size_sigma_factor / 2).astype(int)
    # check the region start and region end size is in the image range
    region_start = np.maximum(0, region_start).astype(int)
    region_end = np.minimum(image_size, region_end).astype(int)
    # return zero landmark, if region is invalid, i.e., landmark is outside of image
    if np.any(region_start >= region_end):
        return heatmap
    region_size = (region_end - region_start).astype(int)
    sigma = sigma * sigma_scale_factor
    scale = 1.0
    if not normalize_center:
        scale /= math.pow(math.sqrt(2 * math.pi) * sigma, dim)
    if dim == 1:
        dx = np.meshgrid(range(region_size[0]))
        x_diff = dx + region_start[0] - flipped_coords[0]
        squared_distances = x_diff * x_diff
        cropped_heatmap = scale * np.exp(-squared_distances / (2 * math.pow(sigma, 2)))
        heatmap[region_start[0]:region_end[0]] = cropped_heatmap[:]
    if dim == 2:
        dy, dx = np.meshgrid(range(region_size[1]), range(region_size[0]))
        x_diff = dx + region_start[0] - flipped_coords[0]
        y_diff = dy + region_start[1] - flipped_coords[1]
        squared_distances = x_diff * x_diff + y_diff * y_diff
        cropped_heatmap = scale * np.exp(-squared_distances / (2 * math.pow(sigma, 2)))
        heatmap[region_start[0]:region_end[0], region_start[1]:region_end[1]] = cropped_heatmap[:, :]
    if dim == 3:
        dy, dx, dz = np.meshgrid(range(region_size[1]), range(region_size[0]), range(region_size[2]))
        x_diff = dx + region_start[0] - flipped_coords[0]
        y_diff = dy + region_start[1] - flipped_coords[1]
        z_diff = dz + region_start[2] - flipped_coords[2]
        squared_distances = x_diff * x_diff + y_diff * y_diff + z_diff * z_diff
        cropped_heatmap = scale * np.exp(-squared_distances / (2 * math.pow(sigma, 2)))
        heatmap[region_start[0]:region_end[0], region_start[1]:region_end[1],
        region_start[2]:region_end[2]] = cropped_heatmap[:, :, :]
    return heatmap


def LandmarkGeneratorHeatmap(srcimage, lanmarks, sigma=3.0):
    """
    Generates a numpy array landmark images for the specified points and parameters.
    :param srcimage:src image itk
    :param lanmarks:image landmarks array
    :param sigma:Sigma of Gaussian
    :return:heatmap
    """
    image_size = np.shape(srcimage)
    stack_axis = len(image_size)
    heatmap_list = []
    for landmark in lanmarks:
        heatmap_list.append(onelandmarktoheatmap(srcimage, landmark, sigma))
    heatmaps = np.stack(heatmap_list, axis=stack_axis)
    # convert (x,y,c)array to (y,x,c)
    # max_heat = np.sum(heatmaps, axis=2)
    # plt.figure("Image")
    # plt.imshow(srcimage)
    # plt.axis('on')
    # plt.title('image')
    # plt.show()
    # plt.figure("heatmap")
    # plt.imshow(max_heat)
    # plt.axis('on')
    # plt.title('heatmap')
    # plt.show()
    return heatmaps


def LoadimageandLandmark(csv_file, num_landmarks, dim):
    """
    load image and landmark from csv file
    :param csv_file:landmark and image string
    like this Imagepath,mark1,mark2.....,the number is num_landmarks*dim
    :param num_landmarks:number of landmarks
    :param dim:lanmark dimension
    :return:image(numbersamples,),landmark(numbersamples,num_landmarks,dim)
    """
    csvdata = pd.read_csv(csv_file)
    imagedata = csvdata.iloc[:, 0].values
    lanmarkdata = csvdata.iloc[:, 1:].values
    assert num_landmarks * dim == np.shape(lanmarkdata)[1], 'csv landmarks is not equal to num_landmarks*dim'
    newlanmarkdata = np.reshape(lanmarkdata, (np.shape(lanmarkdata)[0], num_landmarks, dim))
    return imagedata, newlanmarkdata


def preparedata():
    path = "D:\Project\python\\boneproject\\bonelandmarkdetection\dataprocess\data\hand_xray_dataset\images\\"
    csv_file = "D:\Project\python\\boneproject\\bonelandmarkdetection\dataprocess\data\landmark.csv"
    trainImage = "E:\Data\Bone_CT\landmark\Image"
    trainMask = "E:\Data\Bone_CT\landmark\Mask"
    shape = (512, 512)
    # step 1 load the landmark and image from csv
    imagedata, landmarkdata = LoadimageandLandmark(csv_file, 37, 2)
    for indx in range(len(imagedata)):
        print(str(imagedata[indx]))
        filepath = path + str(imagedata[indx]) + ".mha"
        src_itkimage = sitk.ReadImage(filepath, sitk.sitkFloat32)
        # step 2 reduce dimension
        src_itkimage = reduce_dimension(src_itkimage)
        # step 3 resize image and landmark to fixed shape
        resize_itkimage, resize_landmarks = resizeimageandlandmarks(src_itkimage, shape, landmarkdata[indx])
        # step 4 generate landmarks heatmaps
        # the order is (y,x)
        image_array = sitk.GetArrayFromImage(resize_itkimage)
        image_array = np.transpose(image_array, (1, 0))
        heatmaps = LandmarkGeneratorHeatmap(image_array, resize_landmarks, sigma=5.0)
        # src_image = sitk.GetArrayFromImage(src_itkimage)
        # src_image = np.transpose(src_image, (1, 0))
        # heatmaps1 = LandmarkGeneratorHeatmap(src_image, landmarkdata[indx], sigma=5)
        # step 5 normalization the image to mean 0 std 1
        image_array = normalize(image_array)
        # step 6 save image and heatmaps image to file
        gen_image_mask(image_array, heatmaps, indx, trainImage, trainMask)


#preparedata()
