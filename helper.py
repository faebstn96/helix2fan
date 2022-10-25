from skimage.io import imread, imsave
import numpy as np
import pandas as pd
import json
import tifffile


def load_tiff_stack(file):
    '''

    :param file: Path object describing the location of the file
    :return: a numpy array of the volume
    '''
    if not (file.name.endswith('.tif') or file.name.endswith('.tiff')):
        raise FileNotFoundError('Input has to be tif.')
    else:
        volume = imread(file, plugin='tifffile')
        return volume


def load_tiff_stack_with_metadata(file):
    '''

    :param file: Path object describing the location of the file
    :return: a numpy array of the volume, a dict with the metadata
    '''
    if not (file.name.endswith('.tif') or file.name.endswith('.tiff')):
        raise FileNotFoundError('File has to be tif.')
    with tifffile.TiffFile(file) as tif:
        data = tif.asarray()
        metadata = tif.pages[0].tags["ImageDescription"].value
    metadata = metadata.replace("'", "\"")
    try:
        metadata = json.loads(metadata)
    except:
        print('The tiff file you try to open does not seem to have metadata attached.')
        metadata = None
    return data, metadata


def save_to_tiff_stack(array, file):
    '''

    :param array: Array to save
    :param file: Path object to save to
    '''
    if not file.parent.is_dir():
        file.parent.mkdir(parents=True, exist_ok=True)
    if not (file.name.endswith('.tif') or file.name.endswith('.tiff')):
        raise FileNotFoundError('File has to be tif.')
    else:
        imsave(file, array, plugin='tifffile', check_contrast=False)


def save_to_tiff_stack_with_metadata(array, file, metadata):
    '''

    :param array:
    :param file:
    :return:
    '''
    if not file.parent.is_dir():
        file.parent.mkdir(parents=True, exist_ok=True)
    if not (file.name.endswith('.tif') or file.name.endswith('.tiff')):
        raise FileNotFoundError('File has to be tif.')
    else:
        # metadata = json.dumps(metadata)
        # tifffile.imsave(file, array, description=metadata)
        tifffile.imwrite(file, shape=array.shape, dtype=array.dtype, metadata=metadata)

        # memory map numpy array to data in OME-TIFF file
        memmap_stack = tifffile.memmap(file)

        # write data to memory-mapped array
        for t in range(array.shape[0]):
            memmap_stack[t] = array[t, :, :]
            memmap_stack.flush()


def load_csv(file):
    '''

    :param file: Path object describing the location of the file
    :return: pandas dataframe containing the information from the file
    '''
    if not file.name.endswith('.csv'):
        raise FileNotFoundError('Input has to be a csv file.')
    else:
        data = pd.read_csv(file)
        return data


def save_to_json(params, file):
    '''

    :param params: parameter dict
    :param file: path to a file to save to
    :return:
    '''
    if not file.parent.is_dir():
        file.parent.mkdir(parents=True, exist_ok=True)
    with open(file, 'w') as f:
        json.dump(params, f, indent=2)


def load_from_json(file):
    '''

    :param file: path to a file to load
    :return: contents of the file
    '''
    with open(file) as f:
        data = json.load(f)
    return data
