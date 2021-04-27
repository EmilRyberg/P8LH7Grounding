from find_objects.find_objects import FindObjects, ObjectInfo

class ObjectInfoWithFeatures(ObjectInfo):
    def __init__(self, object_info: ObjectInfo = None, features = None):
        super().__init__()
        if object_info is not None:
            self.object_img_cutout_cropped = object_info.object_img_cutout_cropped
            self.object_img_cutout_full = object_info.object_img_cutout_full
            self.mask_full = object_info.mask_full
            self.bbox_xxyy = object_info.bbox_xxyy
            self.bbox_xywh = object_info.bbox_xywh
            self.mask_cropped = object_info.mask_cropped
        if features is not None:
            self.features = features
        else:
            self.features = None