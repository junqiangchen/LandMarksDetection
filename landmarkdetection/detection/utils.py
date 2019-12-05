import numpy as np
import SimpleITK as sitk


def reduce_dimension(image, axis=None):
    dim = image.GetDimension()
    if axis is None:
        axis = dim - 1
    size = list(image.GetSize())
    if size[axis] != 1:
        return image
    size[axis] = 0
    index = [0] * dim
    return sitk.Extract(image, size, index)


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


def resize_image_itk(itkimage, newSize, resamplemethod=sitk.sitkLinear):
    """
    image resize withe sitk resampleImageFilter
    :param itkimage:input itk image
    :param newSize:such as [512,512]
    :param resamplemethod:resamplemethod such as linear
    :return:resize image
    """
    dim = itkimage.GetDimension()
    newSize = np.array(newSize, float)
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
