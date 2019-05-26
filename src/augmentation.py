from albumentations import *
import albumentations.augmentations.functional as F


def train_aug(image_size=224):
    return Compose([
        Resize(448, 448),
        RandomCrop(image_size, image_size),
        HorizontalFlip(),
        Normalize()
    ], p=1)


def valid_aug(image_size=224):
    return Compose([
        Resize(448, 448),
        CenterCrop(image_size, image_size),
        Normalize()
    ], p=1)




class FivesCrops(DualTransform):
    def __init__(self, width=320, height=320, crop_pos=0, always_apply=False, p=1.0):
        super(FivesCrops, self).__init__(always_apply, p)
        self.crop_pos = crop_pos
        self.width = width
        self.height = height

    def apply(self, img, **params):
        w, h = img.shape[:2]
        crop_w, crop_h = self.width, self.height

        if self.crop_pos == 0:
            return F.crop(img, 0, 0, crop_w, crop_h)
        elif self.crop_pos == 1:
            return F.crop(img, w - crop_w, 0, w, crop_h)
        elif self.crop_pos == 2:
            return F.crop(img, 0, h - crop_h, crop_w, h)
        elif self.crop_pos == 3:
            return F.crop(img, w - crop_w, h - crop_h, w, h)
        else:
            return F.center_crop(img, crop_h, crop_w)


def infer_aug_five_crops(pos, image_size=320, p=1.0):
    return Compose([
        Resize(448, 448, p=1),
        FivesCrops(image_size, image_size, pos, always_apply=True),
        Normalize(),
    ], p=p)


def infer_aug_five_crops_hflip(pos, image_size=320, p=1.0):
    return Compose([
        Resize(448, 448, p=1),
        FivesCrops(image_size, image_size, pos, always_apply=True),
        HorizontalFlip(p=1),
        Normalize(),
    ], p=p)


def test_tta(image_size=224, p=1.0):
    tta_simple = [
        Compose([
            Resize(image_size, image_size),
            Normalize()
        ], p=p),
        Compose([
            Resize(image_size, image_size),
            HorizontalFlip(p=1.0),
            Normalize()
        ], p=p),
    ]

    for i in range(5):
        tta_simple += [infer_aug_five_crops(i, image_size)]
        tta_simple += [infer_aug_five_crops_hflip(i, image_size)]

    return tta_simple
