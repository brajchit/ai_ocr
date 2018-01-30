from PIL import Image
from resizeimage import resizeimage
import numpy as np

img = Image.open("../media/img011-00011.png")
img_resize = resizeimage.resize_contain(img, [28, 28])
gray = img_resize.convert('L')
bw = gray.point(lambda x: 255 if x<128 else 0, '1')
bw.save("../media/preprocessing_data/result_bw.png")
