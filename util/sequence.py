import imgaug as ia
from imgaug import augmenters as iaa


def get():
    def sometimes(aug): return iaa.Sometimes(0.5, aug)

    return iaa.Sequential(
        [
            
            iaa.Flipud(0.3),  # vertically flip 20% of all images
            # crop images by -5% to 10% of their height/width
            
            # execute 0 to 5 of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong
            iaa.SomeOf((0, 5),
                       [
                # convert images into their superpixel representation
                
                iaa.OneOf([
                    # blur images with a sigma between 0 and 3.0
                    iaa.GaussianBlur((0, 3.0)),
                    # blur image using local means with kernel sizes between 2 and 7
                    iaa.AverageBlur(k=(2, 5)),
                    # blur image using local medians with kernel sizes between 2 and 7
                    iaa.MedianBlur(k=(3, 5)),
                ]),
                iaa.Sharpen(alpha=(0, 1.0), lightness=(
                    0.75, 1.5)),  # sharpen images
                # add gaussian noise to images
                iaa.AdditiveGaussianNoise(loc=0, scale=(
                    0.0, 0.05*255), per_channel=0.5),
                # invert color channels
                iaa.Add((-10, 10), per_channel=0.5),
                # either change the brightness of the whole image (sometimes
                # per channel) or change the brightness of subareas
                # improve or worsen the contrast
                iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
                iaa.Grayscale(alpha=(0.0, 1.0)),
                iaa.ScaleX((0.5, 1.5)),
                iaa.ScaleY((0.5, 1.5)),
                iaa.Rotate((-180, 180)),
                iaa.TranslateX(px=(-20, 20)),
                iaa.TranslateY(px=(-20, 20))

            ],
                random_order=True
            )
        ],
        random_order=True
    )