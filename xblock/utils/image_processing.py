#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : image_processing.py
# Author            : Pranava Madhyastha <pranava@imperial.ac.uk>
# Date              : 01.11.2020
# Last Modified Date: 09.11.2021
# Last Modified By  : Pranava Madhyastha <pranava@imperial.ac.uk>
#
# Copyright (c) 2020, Imperial College, London
# All rights reserved.
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#   1. Redistributions of source code must retain the above copyright notice, this
#      list of conditions and the following disclaimer.
#   2. Redistributions in binary form must reproduce the above copyright notice,
#      this list of conditions and the following disclaimer in the documentation
#      and/or other materials provided with the distribution.
#   3. Neither the name of Imperial College nor the names of its contributors may
#      be used to endorse or promote products derived from this software without
#      specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR 
# TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# image pre-processing codebase

import os
import time
import cv2
import numpy as np
import peakutils
from PIL import Image


def scale(img, xScale, yScale):
    out = cv2.resize(img, None, fx=xScale, fy=yScale, interpolation=cv2.INTER_AREA)
    return out


def crop(infile, height, width):
    image = Image.open(infile)
    imgwidth, imgheight = im.size
    for i in range(imgheight // height):
        for j in range(imgwidth // width):
            box = (j * width, i * height, (j + 1) * width, (i + 1) * height)
            yield im.crop(box)


def convert_frame_to_grayscale(frame):
    grayframe = None
    gray = None
    if frame is not None:
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = scale(gray, 1, 1)
        grayframe = scale(gray, 1, 1)
        gray = cv2.GaussianBlur(gray, (9, 9), 0.0)
    return grayframe, gray


def keyframe_extractor(video, threshold=0.25):
    """
    Video: video filepath
    threshold: image difference threshold
    """

    keyframesdir = '/tmp/keyframes/{}'.format(time.time())

    if not os.path.exists(keyframesdir):
        os.makedirs(keyframesdir)

    source = cv2.VideoCapture(video)
    length = int(source.get(cv2.CAP_PROP_FRAME_COUNT))
    listframes = []
    listdiffs = []
    images = []
    colored = []
    lastframe = None

    if source.isOpened():

        for i in range(length):
            ret, frame = source.read()
            grayframe, blur_gray = convert_frame_to_grayscale(frame)
            frame_number = source.get(cv2.CAP_PROP_POS_FRAMES) - 1
            listframes.append(frame_number)
            images.append(grayframe)
            colored.append(frame)
            if frame_number == 0:
                lastframe = blur_gray
            diff = cv2.subtract(blur_gray, lastframe)
            difference = cv2.countNonZero(diff)
            listdiffs.append(difference)
            lastframe = blur_gray

        source.release()
        y = np.array(listdiffs)
        base = peakutils.baseline(y, 2)
        indices = peakutils.indexes(y - base, threshold, min_dist=1)

        for k in indices:
            cv2.imwrite(os.path.join('{}/keyframe_{}.jpg'.format(keyframesdir, k)), colored[k])

    else:
        print('error in the file')

    cv2.destroyAllWindows()
    return keyframesdir


if __name__ == '__main__':
    import plac

    plac.call(keyframe_extractor)
