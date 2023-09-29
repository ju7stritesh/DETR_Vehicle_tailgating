from utils.parser import get_config
from detr_deep_sort_vehicle import VideoTracker
import cv2
import copy
import json
import os

class Tailgating():
    def __init__(self):
        self.cfg = get_config()
        self.cfg.merge_from_file("./configs/deep_sort.yaml")
        self.vdo_trk = VideoTracker(self.cfg)

        similarity = 0.7 # input("Similarity")
        self.similarity = float(similarity)
        max_dist_covered = 700 # input("Max Dist")  #
        self.max_dist_covered = int(max_dist_covered)
        min_dist = 144 # input("Minimum distance")
        self.min_dist = int(min_dist)
        self.img = cv2.imread("test.jpg")

        self.stop_second_threshold = 4
        self.ROI_OVERLAP_THRESHOLD = 0.8

    def find_overlap(self, r1, r2, id):
            # region shape: [top left bottom right]
            img1 = copy.deepcopy(self.img)
            r1_top, r1_left, r1_bottom, r1_right = r1
            r2_top, r2_left, r2_bottom, r2_right = r2
            cv2.rectangle(img1, (r1_left, r1_top), (r1_right, r1_bottom), (0, 255, 0), 2)
            cv2.rectangle(img1, (r2_left, r2_top), (r2_right, r2_bottom), (255, 0, 0), 2)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

            left = max(r1_left, r2_left)
            right = min(r1_right, r2_right)
            bottom = min(r1_bottom, r2_bottom)
            top = max(r1_top, r2_top)
            if top < bottom and left < right:
                val =  float((bottom - top) * (right - left)) / ((r1_bottom - r1_top) * (r1_right - r1_left))
                cv2.putText(img1,"ID:" + str(id),(r1_left, r1_top+40), cv2.FONT_HERSHEY_PLAIN, 3, [0,0,255], 3)
                cv2.putText(img1,str(round(val,2)),(150,50), cv2.FONT_HERSHEY_PLAIN, 3, [0,0,255], 3)
                cv2.imshow("test", img1)
                cv2.waitKey(1)
                return float((bottom - top) * (right - left)) / ((r1_bottom - r1_top) * (r1_right - r1_left))
            return 0.0

    def event(self, path):

        metadata = self.vdo_trk.run(path, self.similarity, self.max_dist_covered, self.min_dist)
        overlap_dict = {}
        count_id = {}

        for key in metadata:
            IDs = metadata[key][0]
            bboxes = metadata[key][1]
            roi = metadata[key][2][0]
            directions = metadata[key][3]
            for i in range(len(bboxes)):
                overlap = self.find_overlap(bboxes[i], roi, IDs[i])
                if IDs[i] in overlap_dict:
                    overlap_dict[IDs[i]].append(overlap)
                    if overlap > self.ROI_OVERLAP_THRESHOLD:
                        count_id[IDs[i]] += 1
                else:
                    overlap_dict[IDs[i]] = [overlap]
                    if overlap > self.ROI_OVERLAP_THRESHOLD:
                        count_id[IDs[i]] = 1
                    else:
                        count_id[IDs[i]] = 0
        return overlap_dict, count_id


if __name__ == "__main__":
    folder = "Video/"
    tailgat = Tailgating()
    for file in os.listdir("Video/"):
        Tailgating_alert = False
        path = folder + file
        overlap_dict, count_id = tailgat.event(path)
        for key in count_id:
            if count_id[key] >= tailgat.stop_second_threshold:
                print ("Vehicle ID " + str(key) + " entered the Region of Interest and stayed there for " + str(count_id[key]) + " seconds and was not tailgating")
            elif 0 < count_id[key] < tailgat.stop_second_threshold:
                Tailgating_alert = True
                print ("Vehicle ID " + str(key) + " entered the Region of Interest and stayed there for only " + str(count_id[key]) + " second/s and was probably tailgating")

        print ("Tailgating alert is " + str(Tailgating_alert))

