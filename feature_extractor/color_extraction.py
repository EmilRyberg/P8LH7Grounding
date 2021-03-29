from PIL import Image, ImageChops

import math

color_labels = {
    "white" : [255, 255, 255],
    "black" : [0, 0, 0],
    "red" : [255, 0, 0],
    "blue" : [0, 255, 0],
    "green" : [0, 0, 255],
    "yellow" : [255, 255, 0],
    "cyan" : [0, 255, 255],
    "magenta" : [255, 0, 255],
    "orange" : [255, 127, 0]
}

def bbox_and_mask_to_color(self, img, mask):
    bbox = mask.getbbox()
    no_pixels = sum(mask.crop(bbox)
                    .point(lambda x: 255 if x else 0)
                    .point(bool)
                    .getdata())

    mask = mask.convert('RGB')

    itemImg = ImageChops.multiply(mask, img)
    pixels = list(itemImg.getdata())
    itemImg.show()

    rgb = [0, 0, 0]
    for pixel in pixels:
        rgb = list(map(add, rgb, list(pixel)))
        
    rgb = [rgb_val / no_pixels for rgb_val in rgb]
    
    out = ('nocolor', 1000)
    for color in color_labels:
        dist = math.sqrt((color_labels[color][0] - rgb[0])**2 + (color_labels[color][1] - rgb[1])**2 + (color_labels[color][2] - rgb[2])**2)
        if out[1] > dist:
            out = (color, dist)

    print(out)
    return out[0]