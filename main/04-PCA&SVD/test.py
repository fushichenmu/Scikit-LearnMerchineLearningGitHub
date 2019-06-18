###人脸识别数据中，对VK的运用
import numpy as np
import os
from skimage import io
from skimage import transform


def read_imagelist(filelist):
    # brief：从列表文件中，读取图像数据到矩阵文件中
    # param： filelist 图像列表文件
    # return ：4D 的矩阵

    fid = open(filelist)
    lines = fid.readlines()
    test_num = len(lines)
    fid.close()
    X = np.empty((test_num, 250,250))
    for index,line in enumerate(lines):
        word = line.split('\n')
        filename = 'E:/openData/faceData/lfw/lfw/lfw' + word[0]
        if not os.path.exists(filename):
            continue
        im1 = io.imread(filename, as_gray=True)
        X[index,:,:] = im1
    return X


def read_labels(labelfile):
    fin = open(labelfile)
    lines = fin.readlines()
    labels = np.empty((len(lines),))
    k = 0;
    for line in lines:
        labels[k] = int(line)
        k = k + 1;
    fin.close()
    return labels


# left = read_imagelist('E:/openData/faceData/lfw/LFW数据读取（两种方法）/filelist_left.list')


import os
os.environ["KERAS_BACKEND"] = "theano"
import numpy as np
from os import listdir
from os.path import join, isdir
import logging
logger = logging.getLogger(__name__)
try:
    import urllib.request as urllib  # for backwards compatibility
except ImportError:
    import urllib


def _load_imgs(file_paths, slice_, color, resize):
    """Internally used to load images"""

    # Try to import imread and imresize from PIL. We do this here to prevent
    # the whole sklearn.datasets module from depending on PIL.
    try:
        try:
            from scipy.misc import imread
        except ImportError:
            from scipy.misc.pilutil import imread
        from scipy.misc import imresize
    except ImportError:
        raise ImportError("The Python Imaging Library (PIL)"
                          " is required to load data from jpeg files")

    # compute the portion of the images to load to respect the slice_ parameter
    # given by the caller
    default_slice = (slice(0, 250), slice(0, 250))
    if slice_ is None:
        slice_ = default_slice
    else:
        slice_ = tuple(s or ds for s, ds in zip(slice_, default_slice))

    h_slice, w_slice = slice_
    h = (h_slice.stop - h_slice.start) // (h_slice.step or 1)
    w = (w_slice.stop - w_slice.start) // (w_slice.step or 1)

    if resize is not None:
        resize = float(resize)
        h = int(resize * h)
        w = int(resize * w)

    # allocate some contiguous memory to host the decoded image slices
    n_faces = len(file_paths)
    if not color:
        faces = np.zeros((n_faces, h, w), dtype=np.float32)
    else:
        faces = np.zeros((n_faces, h, w, 3), dtype=np.float32)

    # iterate over the collected file path to load the jpeg files as numpy
    # arrays
    for i, file_path in enumerate(file_paths):
        if i % 1000 == 0:
            logger.info("Loading face #%05d / %05d", i + 1, n_faces)

        # Checks if jpeg reading worked. Refer to issue #3594 for more
        # details.
        img = imread(file_path)
        if img.ndim is 0:
            raise RuntimeError("Failed to read the image file %s, "
                               "Please make sure that libjpeg is installed"
                               % file_path)

        face = np.asarray(img[slice_], dtype=np.float32)
        face /= 255.0  # scale uint8 coded colors to the [0.0, 1.0] floats
        if resize is not None:
            face = imresize(face, resize)
        if not color:
            # average the color channels to compute a gray levels
            # representation
            face = face.mean(axis=2)

        faces[i, ...] = face

    return faces
def _fetch_lfw_people(data_folder_path, slice_=None, color=False, resize=None,
                      min_faces_per_person=0):
    """Perform the actual data loading for the lfw people dataset

    This operation is meant to be cached by a joblib wrapper.
    """
    # scan the data folder content to retain people with more that
    # `min_faces_per_person` face pictures
    person_names, file_paths = [], []
    for person_name in sorted(listdir(data_folder_path)[0:100]):
        folder_path = join(data_folder_path, person_name)
        if not isdir(folder_path):
            continue
        paths = [join(folder_path, f) for f in listdir(folder_path)]
        n_pictures = len(paths)
        if n_pictures >= min_faces_per_person:
            person_name = person_name.replace('_', ' ')
            person_names.extend([person_name] * n_pictures)
            file_paths.extend(paths)

    n_faces = len(file_paths)
    if n_faces == 0:
        raise ValueError("min_faces_per_person=%d is too restrictive" %
                         min_faces_per_person)

    target_names = np.unique(person_names)
    target = np.searchsorted(target_names, person_names)

    faces = _load_imgs(file_paths, slice_, color, resize)

    # shuffle the faces with a deterministic RNG scheme to avoid having
    # all faces of the same person in a row, as it would break some
    # cross validation and learning algorithms such as SGD and online
    # k-means that make an IID assumption

    indices = np.arange(n_faces)
    np.random.RandomState(42).shuffle(indices)
    faces, target = faces[indices], target[indices]
    return faces, target, target_names
# myfaces,taret,target_names=_fetch_lfw_people('F:\lfw_funneled',color=True)
# print(myfaces.shape)



from PIL import Image
im = io.imread(r'E:\openData\faceData\lfw\lfw\lfw\Aaron_Eckhart\Aaron_Eckhart_0001.jpg',as_grey=True)
type(im)
