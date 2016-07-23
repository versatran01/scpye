from sklearn.preprocessing import StandardScaler

from scpye.image_pipeline import ImagePipeline, FeatureUnion
from scpye.image_transformer import (ImageRotator, ImageCropper, ImageResizer,
                                     ImageSmoother, DarkRemover)
from scpye.feature_transformer import (CspaceTransformer, MaskLocator,
                                       PatchCreator)


def create_image_pipeline(ccw=-1, bbox=None, k=0.5):
    """
    Create an image pipeline to do image space transform
    Includes rotation, cropping, resize, smoothing
    :param ccw: rotation, counter-clockwise 90 degrees is -1
    :param bbox: bounding box of image
    :param k: scale of image
    :rtype: ImagePipeline
    """
    img_ppl = ImagePipeline([
        ('rotate_image', ImageRotator(ccw)),
        ('crop_image', ImageCropper(bbox)),
        ('resize_image', ImageResizer(k)),
        ('smooth_image', ImageSmoother())
    ])
    return img_ppl


def create_image_features(cspace=None, loc=True, patch=True):
    """
    Factory function for making a feature union
    :param cspace: features - colorspace
    :param loc: features - pixel location
    :param patch: features - patch around pixel
    :return: feature union
    :rtype: FeatureUnion
    """
    if cspace is None:
        cspace = ["hsv"]

    transformer_list = [(cs, CspaceTransformer(cs)) for cs in cspace]

    if loc:
        transformer_list.append(('mask_location', MaskLocator()))

    if patch:
        transformer_list.append(('create_patch', PatchCreator()))

    # Unfortunately, cannot do a parallel feature extraction
    return FeatureUnion(transformer_list)


def create_feature_pipeline(pmin=25, cspace=None, loc=True, patch=True):
    """
    Create a feature pipeline to generate features from
    :param pmin: minimum greyscale value
    :param cspace: list of color space transformation
    :param loc: use pixel location
    :param patch: extract patch around pixels
    :rtype: ImagePipeline
    """
    features = create_image_features(cspace, loc, patch)

    ftr_ppl = ImagePipeline([('remove_dark', DarkRemover(pmin)),
                             ('features', features),
                             ('scale', StandardScaler())])
    return ftr_ppl
