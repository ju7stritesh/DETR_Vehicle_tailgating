#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module is for extracting Deep SORT feature embeddings given an image and
bounding boxes (in top-left-bottom-right format) corresponding to detections in
that image. 

Instantiate a `Features` object once with the settings you want and call the
instance every time you want to extract Deep SORT features. 
"""

import numpy as np
import cv2
import os

# Deep SORT tools
from generate_detections import create_box_encoder

class Features:
    def __init__(self, pad_percentage=0.0, gray_or_rgb='rgb'):
        """
        An instance of Features represents the model used to pre-process
        bounding-box-detected image patches and extract features of these
        patches. 
        
        @author: Kush
        """
        model_filename = os.getcwd() + '/model_data/mars-small128.pb' # Deep SORT model
        self.encoder = create_box_encoder(model_filename, pad_percentage, batch_size=1)
        self.gray_or_rgb = gray_or_rgb
    
    def __call__(self, image, tlbr_boxes):
        """
        Returns `len(tlbr_boxes)` feature vectors (2-D numpy arrays) extracted from
        `image` using the Deep SORT image encoder `self.encoder`.
        
        @author: Kush
        """
        if self.gray_or_rgb == 'gray':
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # encoder requires RGB array shape
            image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        
        # convert from tlbr to ltwh for the Deep SORT encoder
        ltwh_boxes = tlbr_boxes
        # for box in tlbr_boxes:
        #     t, l, b, r = box
        #     ltwh_boxes.append([l, t, r-l, b-t])
            
        features = self.encoder(image, np.array(ltwh_boxes))
        return(features)
