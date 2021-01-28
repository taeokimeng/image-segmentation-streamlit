import pixellib
from pixellib.semantic import semantic_segmentation
import cv2
import os

def do_semantic(img_path):
    absolute_path = os.path.abspath("model/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")
    segment_image = semantic_segmentation()
    # Load the xception model trained on pascal voc for segmenting objects.
    segment_image.load_pascalvoc_model(absolute_path)
    # segment_image.load_pascalvoc_model("pascal.h5")

    # Perform segmentation on an image
    segmap, segoverlay = segment_image.segmentAsPascalvoc(img_path, output_image_name="image1_new.jpg",
                                                          overlay=True)

    return segoverlay

# output, segmap = segment_image.segmentAsPascalvoc("sample1.jpg")
# cv2.imwrite("img.jpg", segoverlay)
# print(segoverlay.shape)