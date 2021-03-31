import cv2
import numpy as np


class ObjectInfo:
    def __init__(self):
        self.mask_full = None
        self.mask_cropped = None
        self.object_img_cutout_full = None
        self.object_img_cutout_cropped = None
        self.bbox_xxyy = None
        self.bbox_xywh = None


class FindObjects:
    def __init__(self, background_img=None, crop_widths=None):
        """
        :param background_img:
        :param crop_widths: top, bottom, left, right
        """
        if crop_widths is None:
            crop_widths = [0, 0, 0, 0]
        self.crop_widths = crop_widths
        self.scale = 1.0
        self.diff_threshold = 20
        self.size_threshold = 500*self.scale
        self.background_img = background_img

    def find_objects(self, image, debug=False):
        if self.background_img is None:
            raise Exception("background image must be set before running")
        background_img = self.background_img.copy()
        size = background_img.shape
        background_img = background_img[self.crop_widths[0]:size[0]-self.crop_widths[1], self.crop_widths[2]:size[1]-self.crop_widths[3]]
        background_img = cv2.copyMakeBorder(background_img, self.crop_widths[0], self.crop_widths[1], self.crop_widths[2], self.crop_widths[3], cv2.BORDER_CONSTANT)
        background_img = cv2.resize(background_img, (int(background_img.shape[1] * self.scale), int(background_img.shape[0] * self.scale)))
        current_size = (background_img.shape[1], background_img.shape[0])
        if debug:
            cv2.imshow("background", background_img)
        original_img = image.copy()
        original_img = original_img[self.crop_widths[0]:size[0] - self.crop_widths[1], self.crop_widths[2]:size[1] - self.crop_widths[3]]
        original_img = cv2.copyMakeBorder(original_img, self.crop_widths[0], self.crop_widths[1], self.crop_widths[2], self.crop_widths[3], cv2.BORDER_CONSTANT)
        original_img = cv2.resize(original_img, (int(original_img.shape[1] * self.scale), int(original_img.shape[0] * self.scale)))
        if debug:
            cv2.imshow("img", original_img)

        diff = cv2.subtract(background_img, original_img)

        diff_g = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        if debug: cv2.imshow("diff", diff)
        #cv2.imshow("diff_g", diff_g)

        _, thresholded_img = cv2.threshold(diff_g, self.diff_threshold, 255, cv2.THRESH_BINARY)
        if debug:
            cv2.imshow("thr", thresholded_img)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        thresholded_img = cv2.morphologyEx(thresholded_img, cv2.MORPH_CLOSE, kernel)
        thresholded_img = cv2.morphologyEx(thresholded_img, cv2.MORPH_OPEN, kernel)
        if debug:
            cv2.imshow("morph", thresholded_img)

        contours, _ = cv2.findContours(thresholded_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours_img = cv2.cvtColor(thresholded_img, cv2.COLOR_GRAY2BGR)
        contours_img = cv2.drawContours(contours_img, contours, -1, (0,255,0), 1)
        if debug:
            cv2.imshow("cont", contours_img)

        list_of_objects = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < self.size_threshold:
                continue
            empty_img = np.zeros(diff.shape, dtype=np.uint8)
            mask_full = cv2.drawContours(empty_img, contours, i, (255, 255, 255), -1)
            mask_full = cv2.cvtColor(mask_full, cv2.COLOR_BGR2GRAY)
            #cv2.imshow("mask", mask)
            object_cutout_full = cv2.bitwise_and(original_img, original_img, mask=mask_full)
            x, y, w, h = cv2.boundingRect(contour)
            mask_cropped = mask_full[y:y+h, x:x+w].copy()
            object_cutout_cropped = object_cutout_full[y:y+h, x:x+w].copy()
            if debug: cv2.imshow("object cutout cropped"+str(i), object_cutout_cropped)
            object_info = ObjectInfo()
            object_info.mask_full = mask_full
            object_info.mask_cropped = mask_cropped
            object_info.object_img_cutout_full = object_cutout_full
            object_info.object_img_cutout_cropped = object_cutout_cropped
            object_info.bbox_xywh = [x, y, w, h]
            object_info.bbox_xxyy = [x, x+w, y, y+h]
            list_of_objects.append(object_info)
        return list_of_objects


if __name__ == "__main__":
    background_img = cv2.imread("a.png")
    img = cv2.imread("b.png")
    object_finder = FindObjects(background_img, crop_widths=[50, 50, 200, 600])
    #object_finder.scale = 0.4
    result = object_finder.find_objects(img, debug=True)
    #result2 = object_finder.find_objects(img, debug=True)
    cv2.waitKey()