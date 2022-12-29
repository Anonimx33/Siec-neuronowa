import numpy as np
import matplotlib.pyplot as plt
from mlxtend.data import loadlocal_mnist
PATH_TRAIN_IMAGES = "wsi/data/train-images-idx3-ubyte"
PATH_TRAIN_LABELS = "wsi/data/train-labels-idx1-ubyte"

PATH_TEST_IMAGES = "wsi/data/t10k-images-idx3-ubyte"
PATH_TEST_LABELS = "wsi/data/t10k-labels-idx1-ubyte"

COL_SIZE = 28

def load_datasets(set_name="train"):
    if set_name == "train":
        images, labels = loadlocal_mnist(
            images_path=PATH_TRAIN_IMAGES,
            labels_path=PATH_TRAIN_LABELS)
    else:
        images, labels = loadlocal_mnist(
            images_path=PATH_TEST_IMAGES,
            labels_path=PATH_TEST_LABELS)

    return images, labels

def show_image(array, label=None):
    img = np.reshape(array, (COL_SIZE, COL_SIZE))
    plt.title(f"Label: {label}")
    plt.imshow(img, cmap='gray')
    plt.show()
    
def show_image_with_index(index: int, set_name="train"):
    images, labels = load_datasets(set_name)
    img = images[index]
    lbl = labels[index]
    print(lbl)
    show_image(img, lbl)

def normalize_pixel_values(arr: np.ndarray, max_val=255.0):
    normalized_arr = arr.astype('float32')
    normalized_arr /= max_val
    return normalized_arr