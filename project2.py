import numpy as np
import cv2


FILENAME = 'lena_test_image.jpg'

def load_img(file_name: str, grey_scale: bool = True) -> np.ndarray:
    if grey_scale:
        image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE) #reads as grayscale, as (pix_h, pix_w), otherwise utilizes 3 channels
        image = np.expand_dims(image, axis=-1) #convert to 3d by adding one channel. image of size (pix_h, pix_w) -> (pix_h, pix_w, 1)
    else:
        image = cv2.imread(file_name)
    
    if image is None:
        raise OSError(f"Image '{file_name}' could not be read properly.\n")

    return image

def display_img(image: np.ndarray, window_name: str = 'Image') -> None:
    image = image = np.clip(image, 0, 255).astype(np.uint8)
    cv2.imshow(window_name, image)
    cv2.waitKey(0) #wait for user to press any key
    cv2.destroyAllWindows()