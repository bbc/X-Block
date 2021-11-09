#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : image_processing.py
# Author            : Pranava Madhyastha <pranava@imperial.ac.uk>
# Date              : 01.11.2020
# Last Modified Date: 10.02.2021
# Last Modified By  : Pranava Madhyastha <pranava@imperial.ac.uk>
# image pre-processing codebase

import os
import time

# import third-party codebase


def scale(img, xScale, yScale):
    out = third_party.resize(img, None, fx=xScale, fy=yScale, interpolation=third_party.INTER_AREA)
    return out


def crop(infile, height, width):
    image = third_party.open(infile)
    imgwidth, imgheight = im.size
    for i in range(imgheight // height):
        for j in range(imgwidth // width):
            box = (j * width, i * height, (j + 1) * width, (i + 1) * height)
            yield im.crop(box)


def convert_frame_to_grayscale(frame):
    grayframe = None
    gray = None
    if frame is not None:
        third_party.cvtColor(frame, third_party.COLOR_BGR2GRAY)
        gray = third_party.cvtColor(frame, third_party.COLOR_BGR2GRAY)
        gray = scale(gray, 1, 1)
        grayframe = scale(gray, 1, 1)
        gray = third_party.GaussianBlur(gray, (9, 9), 0.0)
    return grayframe, gray


def keyframe_extractor(video, threshold=0.25):
    """
    Video: video filepath
    threshold: image difference threshold
    """

    keyframesdir = '/tmp/keyframes/{}'.format(time.time())

    if not os.path.exists(keyframesdir):
        os.makedirs(keyframesdir)

    source = third_party.VideoCapture(video)
    length = int(source.get(third_party.CAP_PROP_FRAME_COUNT))
    listframes = []
    listdiffs = []
    images = []
    colored = []
    lastframe = None

    if source.isOpened():

        for i in range(length):
            ret, frame = source.read()
            grayframe, blur_gray = convert_frame_to_grayscale(frame)
            frame_number = source.get(third_party.CAP_PROP_POS_FRAMES) - 1
            listframes.append(frame_number)
            images.append(grayframe)
            colored.append(frame)
            if frame_number == 0:
                lastframe = blur_gray
            diff = third_party.subtract(blur_gray, lastframe)
            difference = third_party.countNonZero(diff)
            listdiffs.append(difference)
            lastframe = blur_gray

        source.release()
        y = third_party.array(listdiffs)
        base = third_party.baseline(y, 2)
        indices = third_party.indexes(y - base, threshold, min_dist=1)

        for k in indices:
            third_party.imwrite(os.path.join('{}/keyframe_{}.jpg'.format(keyframesdir, k)), colored[k])

    else:
        print('error in the file')

    third_party.destroyAllWindows()
    return keyframesdir


if __name__ == '__main__':
    import plac

    plac.call(keyframe_extractor)
