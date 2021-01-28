import pixellib
from pixellib.instance import instance_segmentation
import cv2
import os

def do_instance(img_path):
    absolute_path = os.path.abspath("model/mask_rcnn_coco.h5")
    segment_image = instance_segmentation(infer_speed="average")
    segment_image.load_model(absolute_path)
    # Perform instance segmentation
    segmask, output = segment_image.segmentImage(img_path, show_bboxes=True)

    # Bounding boxes
    # segment_image.segmentImage("sample2.jpg", output_image_name="image2_bb_new.jpg", show_bboxes=True)

    # Return the segmented image
    return output

# output = do_instance("../../image/load/meeting_room.jpg")
# cv2.imwrite("../../image/save/meeting_room_new.jpg", output)
# cv2.imwrite("../../model/mask_rcnn_coco.h5")
# print(output.shape)