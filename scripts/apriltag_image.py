#!/usr/bin/env python

from argparse import ArgumentParser
import os
import cv2
import apriltag

################################################################################

def apriltag_image(input_images=['/home/suman/projects/spad_dataset/data/test_single_tag_all.png'],
                   output_images=False,
                   display_images=True,
                   detection_window_name='AprilTag',
                   camera_params = (923.57, 923.94, 641.69, 365.36),
                   tag_size = 0.06
                  ):

    '''
    Detect AprilTags from static images.

    Args:   input_images [list(str)]: List of images to run detection algorithm on
            output_images [bool]: Boolean flag to save/not images annotated with detections
            display_images [bool]: Boolean flag to display/not images annotated with detections
            detection_window_name [str]: Title of displayed (output) tag detection window
    '''

    parser = ArgumentParser(description='Detect AprilTags from static images.')
    apriltag.add_arguments(parser)
    options = parser.parse_args()

    '''
    Set up a reasonable search path for the apriltag DLL.
    Either install the DLL in the appropriate system-wide
    location, or specify your own search paths as needed.
    '''

    detector = apriltag.Detector(options, searchpath=apriltag._get_dll_path())

    for image in input_images:

        img = cv2.imread(image)

        print('Reading {}...\n'.format(os.path.split(image)[1]))
        # K =  [923.565673828125, 0.0, 641.6881713867188, 0.0, 923.939208984375, 365.9615478515625, 0.0, 0.0, 1.0]

        result, overlay = apriltag.detect_tags(img,
                                               detector,
                                               camera_params=camera_params,
                                               tag_size=tag_size,
                                               vizualization=3,
                                               verbose=3,
                                               annotation=True
                                              )

        if output_images:
            output_path = '../media/output/'+str(os.path.split(image)[1])
            output_path = output_path.replace(str(os.path.splitext(image)[1]), '.jpg')
            cv2.imwrite(output_path, overlay)

        if display_images:
            cv2.imshow(detection_window_name, overlay)
            while cv2.waitKey(5) < 0:   # Press any key to load subsequent image
                pass

################################################################################

if __name__ == '__main__':
    apriltag_image()
