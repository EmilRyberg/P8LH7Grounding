import cv2
import numpy as np

class FindObjects:

    def __init__(self, background_img=None, crop_region=None):
        if crop_region is None:
            crop_region = [0, 0, 1080, 1920]
        self.crop_region = crop_region
        self.scale = 1.0
        self.diff_threshold = 20
        self.size_threshold = 500*self.scale
        self.background_img = background_img

    def find_objects(self, image, crop_to_bbox=False):
        if self.background_img is None:
            raise Exception("background image must be set before running")
        background_img = self.background_img[self.crop_region[0]:self.crop_region[2], self.crop_region[1]:self.crop_region[3]].copy()
        background_img = cv2.resize(background_img, (int(background_img.shape[1] * self.scale), int(background_img.shape[0] * self.scale)))
        current_size = (background_img.shape[1], background_img.shape[0])
        #cv2.imshow("background", background)
        original_img = image.copy()
        original_img = original_img[self.crop_region[0]:self.crop_region[2], self.crop_region[1]:self.crop_region[3]]
        original_img = cv2.resize(original_img, (int(original_img.shape[1] * self.scale), int(original_img.shape[0] * self.scale)))
        #cv2.imshow("img", original_img)

        diff = cv2.subtract(background_img, original_img)

        diff_g = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        #cv2.imshow("diff", diff)
        #cv2.imshow("diff_g", diff_g)

        _, thresholded_img = cv2.threshold(diff_g, self.diff_threshold, 255, cv2.THRESH_BINARY)
        #cv2.imshow("thr", thresholded_img)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        thresholded_img = cv2.morphologyEx(thresholded_img, cv2.MORPH_CLOSE, kernel)
        thresholded_img = cv2.morphologyEx(thresholded_img, cv2.MORPH_OPEN, kernel)
        #cv2.imshow("morph", thresholded_img)

        contours, _ = cv2.findContours(thresholded_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours_img = cv2.cvtColor(thresholded_img, cv2.COLOR_GRAY2BGR)
        contours_img = cv2.drawContours(contours_img, contours, -1, (0,255,0), 1)
        #cv2.imshow("cont", contours_img)

        list_of_masked_images = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < self.size_threshold:
                continue
            empty_img = np.zeros(diff.shape, dtype=np.uint8)
            mask = cv2.drawContours(empty_img, contours, i, (255, 255, 255), -1)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            #cv2.imshow("mask", mask)
            masked_img = cv2.bitwise_and(original_img, original_img, mask=mask)
            if crop_to_bbox:
                x, y, w, h = cv2.boundingRect(contour)
                masked_img = masked_img[y:y+h, x:x+w]
            #cv2.imshow("masked"+str(i), masked_img)
            list_of_masked_images.append(masked_img)
        return list_of_masked_images

if __name__ == "__main__":
    background_img = cv2.imread("a.png")
    img = cv2.imread("b.png")
    object_finder = FindObjects(img, [20, 180, 1050, 1350])
    result = object_finder.find_objects(img, crop_to_bbox=False)
    result2 = object_finder.find_objects(img, crop_to_bbox=True)
    cv2.waitKey()