import os
import cv2
import time
import argparse
import torch
import warnings
import numpy as np

# from detector import build_detector
from deep_sort import build_tracker
from utils.draw import draw_boxes
from utils.parser import get_config
from utils.log import get_logger
from utils.io import write_results
from scipy import spatial
import math
from scipy.spatial import distance
from xlwt import Workbook
import json
from features_from_boxes import Features
from Detr import inference
import draw_roi

class VideoTracker(object):
    def __init__(self, cfg):
        self.logger = get_logger("root")

        self._features_from_boxes = Features(0.0, "rgb")


        use_cuda = True and torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)

        self.vdo = cv2.VideoCapture()
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)


    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def _xyxy_to_xywh(self, bbox_yxyx):
        (y1,x1,y2,x2) = bbox_yxyx
        w = abs(x2-x1)
        h = abs(y2-y1)
        return x1,y1,w,h



    def find_overlap(self, r1, r2):
        # region shape: [top left bottom right]
        r1_top, r1_left, r1_bottom, r1_right = r1
        r2_top, r2_left, r2_bottom, r2_right = r2
        left = max(r1_left, r2_left)
        right = min(r1_right, r2_right)
        bottom = min(r1_bottom, r2_bottom)
        top = max(r1_top, r2_top)
        if top < bottom and left < right:
            return float((bottom - top) * (right - left)) / ((r1_bottom - r1_top) * (r1_right - r1_left))
        return 0.0

    def find_center(self, bounding_box):
        y1, x1, y2, x2 = bounding_box
        center = [int(x1+abs(x2-x1)/2), int(y1+abs(y2-y1)/2)]
        return center

    def rmsValue(self, arr, n):
        square = 0

        #Calculate square
        for i in range(0,n):
            square += (arr[i]**2)

        #Calculate Mean
        mean = (square / (float)(n))

        #Calculate Root
        root = math.sqrt(mean)

        return root

    def first_feature(self, features, features_list, bounding_box_list, bbox_yxyx, centers, average_features):

        for t in range(len(features)):
            features_list.append(features[t])
            bounding_box_list.append(bbox_yxyx[t])
            y1, x1, y2, x2 = bounding_box_list[t]
            center = [abs(x2-x1), abs(y2-y1)]
            centers.append(center)
            average_features[t+1] = features[t]
        return features_list, bounding_box_list, centers, average_features

    def convert_bbox_yxyx_to_xywh(self, bbox_yxyx):
        bbox_xywh = []
        for box in bbox_yxyx:
            x,y,w,h = self._xyxy_to_xywh(box)
            bbox_xywh.append([x,y,w,h])
        bbox_xywh = np.array(bbox_xywh)
        return bbox_xywh

    def compare_features(self,old_feature, new_feature, cosine_similarity, new_box,old_box, dist):
        similarity = 1 - spatial.distance.cosine(old_feature, new_feature)
        cosine_similarity.append(similarity)
        center1 = self.find_center(list(new_box))
        center2 = self.find_center(list(old_box))
        dist.append((distance.euclidean(center1, center2)))
        return dist, cosine_similarity


    def run(self, path, similarity, max_dist_covered, min_dist):        #Similarity of vehicle, max distance a vehicle can cover else another vehicle, minimum distance for a moving vehicle less than which is the same vehicle
        features_list = []
        results = []
        idx_frame = 0
        bounding_box_list = []
        centers = []

        metadata = {}
        average_features = {}
        center_id = {}
        cam = cv2.VideoCapture(path)
        fps = cam.get(cv2.CAP_PROP_FPS)
        im_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        im_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        file = path.split('/')
        filename = file[1].split('.')
        save_video_path = os.path.join("VideoResults/", filename[0] + '.avi')

        # create video writer
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(save_video_path, fourcc, 1, (im_width, im_height))

        roi = [[0,0,50,50]]
        start_time = time.time()

        while True:
            idx_frame += 1
            ret, ori_im =  cam.read()
            if ret == False:
                break
            if idx_frame == 1:
                roi = draw_roi.draw_roi(ori_im)
            if idx_frame % math.ceil(fps) != 0:
                continue
            im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)

            if idx_frame == 30:
                cv2.imwrite("test.jpg", cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            bbox_yxyx, cls_conf, cls_ids, img = inference.predict(im)

            bbox_xywh = self.convert_bbox_yxyx_to_xywh(bbox_yxyx)

            features = self._features_from_boxes(im, bbox_xywh)


            if len(features_list) == 0:
                features_list, bounding_box_list,centers,average_features = self.first_feature(features, features_list,bounding_box_list, bbox_yxyx, centers, average_features)
                first_frame_features = len(features_list)

            dir = []
            new_features = []
            new_boxes = []
            new_centers = []
            list_index = []
        #
            final_bounding_box = []
            final_cosine = []
            distance_travelled = []


            for j in range(len(features)):  #New Features
                cosine_similarity = []
                dist = []
                new_vehcile = True

                #Old Features
                for i in range(len(features_list)):
                    dist, cosine_similarity = self.compare_features(features_list[i], features[j], cosine_similarity, bbox_yxyx[j], bounding_box_list[i], dist)

                index = cosine_similarity.index(max(cosine_similarity))
                if ((max(cosine_similarity) > similarity and dist[index] < max_dist_covered and (index+1) not in list_index)) or ((dist[index] < min_dist and (index+1) not in list_index and max(cosine_similarity) > 0.5)):
                    #Condition for Old ID or firt two New ID
                    final_cosine.append(cosine_similarity[index])
                    distance_travelled.append(int(dist[index]))
                    new_vehcile = False
                    features_list[index] = features[j]
                    bounding_box_list[index] = bbox_yxyx[j]
                    center = self.find_center(bbox_yxyx[j])
                    centers[index] = center
                    if len(center_id) > (first_frame_features-1):
                        if center[0] - center_id[index+1][0] < 0:
                            dir.append("goingLeft")
                        else:
                            dir.append("goingRight")
                    else:
                        dir.append("New ID")
                    center_id[index+1] = center

                    list_index.append(index+1)
                    average_features[index+1] = (np.array(average_features[index+1]) + features[j])/2

                #No match found means New Vehicle, update the tracker
                if new_vehcile == True:
                    final_cosine.append(max(cosine_similarity))
                    distance_travelled.append(int(max(dist)))
                    new_features.append(features[j])
                    new_number = len(features_list) + len(new_features)
                    list_index.append(new_number)
                    new_boxes.append(bbox_yxyx[j])
                    center = self.find_center(bbox_yxyx[j])
                    center_id[new_number] = center
                    dir.append("New Vehicle ID")
                    new_centers.append(center)
                    average_features[new_number] = features[j]
            for k in range(len(new_features)):
                features_list.append(new_features[k])
                bounding_box_list.append(new_boxes[k])
                centers.append(new_centers[k])
            outputs = [1]

            if len(outputs) > 0:
                bbox_tlwh = []
                identities = list_index
                final_bounding_box = []
                for g in range(len(list_index)):
                    final_bounding_box.append(bounding_box_list[list_index[g]-1])
                ori_im = draw_boxes(ori_im, final_bounding_box,final_cosine, distance_travelled,dir, identities)
                results.append((idx_frame - 1, bbox_tlwh, identities))
            cv2.rectangle(ori_im,(roi[0][1], roi[0][0]),(roi[0][3],roi[0][2]),(0,0,255),3)
            cv2.putText(ori_im,"ROI",(roi[0][1], roi[0][0]), cv2.FONT_HERSHEY_PLAIN, 3, [0,0,255], 3)

            cv2.imshow("test", ori_im)
            cv2.waitKey(1)

            # if self.args.save_path:
            writer.write(ori_im)

            metadata[idx_frame + 1] = [list_index, final_bounding_box, roi, str(dir)]

        for l in average_features:
            average_features[l] = str(average_features[l])

        with open('Metadata/' + 'results4.json', 'w') as json_file:
              json.dump(metadata, json_file)
        print ("Time",time.time() - start_time)
        return metadata


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("VIDEO_PATH", type=str, default="test.mp4")
    parser.add_argument("--config_detection", type=str, default="./configs/yolov3.yaml")
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    # parser.add_argument("--ignore_display", dest="display", action="store_false", default=True)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--frame_interval", type=int, default=30)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="./output/")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")
    return parser.parse_args()


if __name__ == "__main__":
    cfg = get_config()
    cfg.merge_from_file("./configs/deep_sort.yaml")
    path = "Video/tailgating.mp4"

    # with VideoTracker(cfg, video_path=path) as vdo_trk:
    #     metadata = vdo_trk.run()
    #     print (metadata)

    vdo_trk = VideoTracker(cfg)
    metadata = vdo_trk.run(path,0.7,820,144)
    # print (metadata)
