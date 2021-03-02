import cv2
import numpy as np

class FindObjects:

    def __init__(self, background_img, crop_region=None, crop_to_bbox=False):
        if crop_region is None:
            crop_region = [0, 0, 1080, 1920]
        self.crop_region = crop_region
        self.scale = 1.0
        self.do_crop_to_contour = True
        self.background_img = background_img

    def find_objects(self, image):
        self.background_img = self.background_img[self.crop_region[0]:self.crop_region[2], self.crop_region[1]:self.crop_region[3]]
        self.background_img = cv2.resize(self.background_img, (int(self.background_img.shape[1] * self.scale), int(self.background_img.shape[0] * self.scale)))
        current_size = (self.background_img.shape[1], self.background_img.shape[0])
        #cv2.imshow("background", background)
        original_img = image
        original_img = original_img[self.crop_region[0]:self.crop_region[2], self.crop_region[1]:self.crop_region[3]]
        original_img = cv2.resize(original_img, (int(original_img.shape[1] * self.scale), int(original_img.shape[0] * self.scale)))
        #cv2.imshow("img", original_img)

        diff = cv2.subtract(self.background_img, original_img)

        diff_g = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        #cv2.imshow("diff", diff)
        #cv2.imshow("diff_g", diff_g)

        _, thresholded_img = cv2.threshold(diff_g, 20, 255, cv2.THRESH_BINARY)
        cv2.imshow("thr", thresholded_img)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        thresholded_img = cv2.morphologyEx(thresholded_img, cv2.MORPH_CLOSE, kernel)
        thresholded_img = cv2.morphologyEx(thresholded_img, cv2.MORPH_OPEN, kernel)
        cv2.imshow("morph", thresholded_img)

        contours, _ = cv2.findContours(thresholded_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours_img = cv2.cvtColor(thresholded_img, cv2.COLOR_GRAY2BGR)
        contours_img = cv2.drawContours(contours_img, contours, -1, (0,255,0), 1)
        #cv2.imshow("cont", contours_img)

        list_of_masked_images = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < 300*self.scale:
                continue
            empty_img = np.zeros(diff.shape, dtype=np.uint8)
            mask = cv2.drawContours(empty_img, contours, i, (255, 255, 255), -1)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            #cv2.imshow("mask", mask)
            masked_img = cv2.bitwise_and(original_img, original_img, mask=mask)
            if self.do_crop_to_contour:
                x, y, w, h = cv2.boundingRect(contour)
                masked_img = masked_img[y:y+h, x:x+w]
            cv2.imshow("masked"+str(i), masked_img)
            list_of_masked_images.append(masked_img)
        return list_of_masked_images

if __name__ == "__main__":
    background_img = cv2.imread("a.png")
    img = cv2.imread("b.png")
    object_finder = FindObjects(background_img, [20, 180, 1050, 1350], True)
    object_finder.find_objects(img)
    cv2.waitKey()