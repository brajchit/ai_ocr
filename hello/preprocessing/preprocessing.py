from PIL import Image, ImageChops
from resizeimage import resizeimage
import numpy as np

def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

img = Image.open("../media/img011-00011.png")
gray = img.convert('L')
bw = gray.point(lambda x: 255 if x<128 else 0, '1')
cropp = trim(bw)
img_resize = resizeimage.resize_cover(cropp, [28, 28])
img_resize.save("../media/preprocessing_data/result_bw.png")

