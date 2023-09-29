import os
import cv2
import time
import argparse
import torch
import warnings
import numpy as np

from detector import build_detector
from deep_sort import build_tracker
from utils.draw import draw_boxes
from utils.parser import get_config
from utils.log import get_logger
from utils.io import write_results
from scipy import spatial
import MaskRCNN.demo as demo
import math
from scipy.spatial import distance
from xlwt import Workbook
import json
from features_from_boxes import Features

wb = Workbook()
sheet1 = wb.add_sheet('Sheet 1')

sheet1.write(0,0, "Frame Number")
sheet1.write(0,1,"IDs")
sheet1.write(0,2,"Bounding Boxes")

class VideoTracker(object):
    def __init__(self, cfg, args, video_path):
        self.cfg = cfg
        self.args = args
        self.video_path = video_path
        self.logger = get_logger("root")
        self.demo = demo.demo()
        self._features_from_boxes = Features(0.0, "rgb")


        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)

        # if args.display:
        cv2.namedWindow("test", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("test", args.display_width, args.display_height)

        if args.cam != -1:
            print("Using webcam " + str(args.cam))
            self.vdo = cv2.VideoCapture(args.cam)
        else:
            self.vdo = cv2.VideoCapture()
        self.detector = build_detector(cfg, use_cuda=use_cuda)
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)
        self.class_names = self.detector.class_names

    def __enter__(self):
        if self.args.cam != -1:
            ret, frame = self.vdo.read()
            assert ret, "Error: Camera error"
            self.im_width = frame.shape[0]
            self.im_height = frame.shape[1]

        else:
            assert os.path.isfile(self.video_path), "Path error"
            self.vdo.open(self.video_path)
            self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
            assert self.vdo.isOpened()

        if self.args.save_path:
            os.makedirs(self.args.save_path, exist_ok=True)

            # path of saved video and results
            self.save_video_path = os.path.join(self.args.save_path, "results7.avi")
            self.save_results_path = os.path.join(self.args.save_path, "results7.txt")

            # create video writer
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            fps = self.vdo.get(cv2.CAP_PROP_FPS)
            print ("fps", fps)
            self.writer = cv2.VideoWriter(self.save_video_path, fourcc, 1, (self.im_width, self.im_height))

            # logging
            self.logger.info("Save results to {}".format(self.args.save_path))

        return self

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
        # print (r1_top, r2_top)
        left = max(r1_left, r2_left)
        right = min(r1_right, r2_right)
        bottom = min(r1_bottom, r2_bottom)
        top = max(r1_top, r2_top)
        # print (top, bottom, left, right)
        if top < bottom and left < right:
            return float((bottom - top) * (right - left)) / ((r1_bottom - r1_top) * (r1_right - r1_left))
        return 0.0

    def find_center(self, bounding_box):
        y1, x1, y2, x2 = bounding_box
        center = [int(x1+abs(x2-x1)/2), int(y1+abs(y2-y1)/2)]
        return center

    def rmsValue(self, arr, n):
        square = 0
        mean = 0.0
        root = 0.0

        #Calculate square
        for i in range(0,n):
            square += (arr[i]**2)

        #Calculate Mean
        mean = (square / (float)(n))

        #Calculate Root
        root = math.sqrt(mean)

        return root

    def run(self):
        features_list = []
        results = []
        idx_frame = 0
        bounding_box_list = []
        centers = []

        metadata = {}
        average_features = {}
        # final_cosine = 0
        while self.vdo.grab():
            idx_frame += 1
            # print (idx_frame)
            if idx_frame % self.args.frame_interval != 0:
                continue

            # print (self.args.frame_interval, idx_frame)
            start = time.time()
            _, ori_im = self.vdo.retrieve()
            im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)
            height, width = im.shape[:2]
            # print (idx_frame)

            # do detection
            bbox_yxyx, cls_conf, cls_ids =  self.demo.detect(self.demo, im)
            # bbox_xywh, cls_conf, cls_ids =  self.detector(im)

            bbox_xywh = []
            for box in bbox_yxyx:
                x,y,w,h = self._xyxy_to_xywh(box)
                bbox_xywh.append([x,y,w,h])
                crop_img = ori_im[y:y+h, x:x+w]
                cv2.imwrite("CroppedImages/frame_" + str(idx_frame) + '_' + str(box) + '.jpg', crop_img)
            bbox_xywh = np.array(bbox_xywh)
            # print (bbox_xywh, cls_conf, type(bbox_xywh))


            # select person class
            # mask = cls_ids == 0

            # bbox_xywh = bbox_xywh[mask]
            # bbox dilation just in case bbox too small, delete this line if using a better pedestrian detector
            # bbox_xywh[:, 3:] *= 1.2
            # cls_conf = cls_conf[mask]
            # if len(bbox_xywh) == 0:
            #     bbox_xywh = [[]]
            # do tracking
            features = self._features_from_boxes(im, bbox_xywh)
            # outputs, features = self.deepsort.update(bbox_xywh, cls_conf, im)
            # print ("New Features",len(features_array), len(features))
            # features = features_array
            if len(features_list) == 0:
                for t in range(len(features)):
                    features_list.append(features[t])
                    bounding_box_list.append(bbox_yxyx[t])
                    y1, x1, y2, x2 = bounding_box_list[t]
                    center = [abs(x2-x1), abs(y2-y1)]
                    centers.append(center)
                    average_features[t+1] = features[t]

            new_features = []
            new_boxes = []
            new_centers = []
            # print ("Features length",len(features_list), len(features))
            list_index = []

            final_bounding_box = []
            final_cosine = []
            for j in range(len(features)):  #New Features
                cosine_similarity = []
                dist1 = []
                dist = []
                new_person = True

                # print ("test",len(features_list), len(cosine_similarity))
                for i in range(len(features_list)):             #Old Features
                        similarity = 1 - spatial.distance.cosine(features_list[i], features[j])
                        # print (cosine_similarity, similarity)
                        cosine_similarity.append(similarity)
                        # overlap12.append(self.find_overlap(list(bbox_yxyx[j]), list(bounding_box_list[i])))
                        # overlap21.append(self.find_overlap(list(bounding_box_list[i]), list(bbox_yxyx[j])))
                        # score = cosine_similarity*.66 + (overlap12 + overlap21) *0.33
                        center1 = self.find_center(list(bbox_yxyx[j]))
                        center2 = self.find_center(list(bounding_box_list[i]))
                        dist.append((distance.euclidean(center1, center2)))
                        dist1.append((distance.euclidean(features_list[i], features[j])))
                        # print (cosine_similarity, i, similarity, dist1, dist)
                        # print (cosine_similarity, overlap12, overlap21, list(bbox_yxyx[j]), list(bounding_box_list[i]), len(features_list), len(bounding_box_list))
                        # print (dist)
                        # print (cosine_similarity)
                        # for k in range(len(previous_boxes)):
                        #
                        #     overlap = self.find_overlap(list(bbox_yxyx[j]), list(previous_boxes[k]))
                        #     relative_motion_score.append(overlap)
                        #     score = cosine_similarity*.66 + overlap*0.33
                        #     print (score, overlap, cosine_similarity, list(bbox_xywh[j]), list(previous_boxes[k]))
                        #     scores.append(score)
                        # # print (scores)
                        # max_scores.append(max(scores))
                        # indexes.append(i)
                index = cosine_similarity.index(max(cosine_similarity))
                if (max(cosine_similarity) > 0.5 and dist[index] < 1220 and (index+1) not in list_index) or (dist[index] < 144 and (index+1) not in list_index):
                    print (max(cosine_similarity),index, dist[index], cosine_similarity[index])
                    # input("test")
                    final_cosine.append(cosine_similarity[index])
                    new_person = False
                    features_list[index] = features[j]
                    bounding_box_list[index] = bbox_yxyx[j]
                    center = self.find_center(bbox_yxyx[j])
                    centers[index] = center
                    list_index.append(index+1)
                    average_features[index+1] = (np.array(average_features[index+1]) + features[j])/2
                    # break


                if new_person == True:
                    print ("New Person", final_cosine, dist)
                    # input("testnew")
                    final_cosine.append(max(cosine_similarity))
                    new_features.append(features[j])
                    new_number = len(features_list) + len(new_features)
                    list_index.append(new_number)
                    new_boxes.append(bbox_yxyx[j])
                    # print (new_number, list_index)
                    center = self.find_center(bbox_yxyx[j])
                    new_centers.append(center)
                    average_features[new_number] = features[j]
            # print (features)
            # draw boxes for visualization
            # print(list_index)
            # print (len(new_features))
            for k in range(len(new_features)):
                features_list.append(new_features[k])
                bounding_box_list.append(new_boxes[k])
                centers.append(new_centers[k])
            outputs = [1]

            if len(outputs) > 0:
                bbox_tlwh = []
                # bbox_xyxy = outputs[:, :4]
                # identities = outputs[:, -1]
                # print (identities, list_index, bbox_xyxy)
                # if len(identities) == len(list_index):
                identities = list_index
                final_bounding_box = []
                for g in range(len(list_index)):
                    final_bounding_box.append(bounding_box_list[list_index[g]-1])
                ori_im = draw_boxes(ori_im, final_bounding_box,final_cosine, identities)

                # for bb_xyxy in bbox_xyxy:
                #     bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(bb_xyxy))

                results.append((idx_frame - 1, bbox_tlwh, identities))

            end = time.time()
            # for l in range(len(bounding_box_list)):
            #     if (l+1) in (list_index):
            #         [y1,x1,y2,x2] = bounding_box_list[l]
            #         cv2.rectangle(ori_im,(x1, y1),(x2,y2),(255,255,255),3)
            #         cv2.putText(ori_im,str(l),(x1,y1+5), cv2.FONT_HERSHEY_PLAIN, 3, [0,0,255], 3)
            #         cv2.putText(ori_im,str("C"),tuple(centers[l]), cv2.FONT_HERSHEY_PLAIN, 3, [0,0,255], 3)
            # if self.args.display:
            cv2.imshow("test", ori_im)
            cv2.waitKey(10)

            if self.args.save_path:
                self.writer.write(ori_im)

            # save results
            write_results(self.save_results_path, results, 'mot')
            previous_boxes = bbox_yxyx
            sheet1.write(int(idx_frame/30) , 0, str(idx_frame))
            sheet1.write(int(idx_frame/30) , 1, str(list_index))
            sheet1.write(int(idx_frame/30) , 2, str(final_bounding_box))
            metadata[idx_frame + 1] = [str(list_index), str(final_bounding_box)]

        for l in average_features:
            average_features[l] = str(average_features[l])

        with open('Metadata/' + 'results3.json', 'w') as json_file:
              json.dump(metadata, json_file)
        with open('Metadata/' + 'average_features.json', 'w') as json_file:
              json.dump(average_features, json_file)
        wb.save('Results3.xls')
            # logging
            # self.logger.info("time: {:.03f}s, fps: {:.03f}, detection numbers: {}, tracking numbers: {}" \
            #                  .format(end - start, 1 / (end - start), bbox_xywh.shape[0], len(outputs)))


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
    args = parse_args()
    cfg = get_config()
    cfg.merge_from_file(args.config_detection)
    cfg.merge_from_file(args.config_deepsort)
    # path = "CLEAN-DIRTY/AREA3/CLEAN/Sub Waiting Lobby2 - Cam 53/5_14_2020 7_15_18 AM (UTC-04_00).mkv"
    path = "Video/tailgating1.mp4"

    with VideoTracker(cfg, args, video_path=path) as vdo_trk:
        vdo_trk.run()
