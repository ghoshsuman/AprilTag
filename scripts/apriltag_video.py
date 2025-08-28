#!/usr/bin/env python

from argparse import ArgumentParser
import os
import cv2
from . import apriltag

# print(apriltag.get_dll_path())

################################################################################

def apriltag_video(input_streams=['/home/suman/projects/spad_dataset/data/video.mp4'], # For default cam use -> [0]
                   output_stream=True,
                   display_stream=True,
                   detection_window_name='AprilTag',
                   camera_params=(923.57, 923.94, 641.69, 365.36),
                   tag_size=0.06
                  ):

    '''
    Detect AprilTags from video stream.

    Args:   input_streams [list(int/str)]: Camera index or movie name to run detection algorithm on
            output_stream [bool]: Boolean flag to save/not stream annotated with detections
            display_stream [bool]: Boolean flag to display/not stream annotated with detections
            detection_window_name [str]: Title of displayed (output) tag detection window
    '''

    parser = ArgumentParser(description='Detect AprilTags from video stream.')
    apriltag.add_arguments(parser)
    options = parser.parse_args()

    '''
    Set up a reasonable search path for the apriltag DLL.
    Either install the DLL in the appropriate system-wide
    location, or specify your own search paths as needed.
    '''

    detector = apriltag.Detector(options, searchpath=apriltag._get_dll_path())

    for stream in input_streams:

        video = cv2.VideoCapture(stream)

        output = None

        if output_stream:
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(video.get(cv2.CAP_PROP_FPS))
            codec = cv2.VideoWriter_fourcc(*'XVID')
            if type(stream) != int:
                output_path = 'data/output/'+str(os.path.split(stream)[1])
                output_path = output_path.replace(str(os.path.splitext(stream)[1]), '.mp4')
            else:
                output_path = 'data/output/'+'camera_'+str(stream)+'.mp4'
            output = cv2.VideoWriter(output_path, codec, fps, (width, height))

        result_video = []
        while(video.isOpened()):

            success, frame = video.read()
            if not success:
                break

            result, overlay = apriltag.detect_tags(frame,
                                                   detector,
                                                   camera_params=camera_params, #(3156.71852, 3129.52243, 359.097908, 239.736909),
                                                   tag_size=tag_size, #0.0762,
                                                   vizualization=3,
                                                   verbose=3,
                                                   annotation=True
                                                  )

            result_video.append(result)

            if output_stream:
                output.write(overlay)

            if display_stream:
                cv2.imshow(detection_window_name, overlay)
                if cv2.waitKey(1) & 0xFF == ord(' '): # Press space bar to terminate
                    break
    return result_video
################################################################################

if __name__ == '__main__':
    apriltag_video()
