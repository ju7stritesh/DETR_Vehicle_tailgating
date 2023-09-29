import numpy as np
import cv2

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox,cosine_similarity,dist_travelled,dir, identities=None, offset=(0,0)):
    for i,box in enumerate(bbox):
        x1,y1,x2,y2 = [int(i) for i in box]
        y1,x1,y2,x2 = box
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0    
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
        cv2.rectangle(img,(x1, y1),(x2,y2),color,3)
        cv2.rectangle(img,(x1, y1),(x1+t_size[0]+3,y1+t_size[1]+4), color,-1)
        # print (dir, len(dir), len(bbox))
        # cv2.putText(img,str(dir[i]),(int(x1+abs(x2-x1)/2), int(y1+abs(y2-y1)/2)), cv2.FONT_HERSHEY_PLAIN, 2, [0,255,0], 2)
        cv2.putText(img,label,(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 3, [255,255,255], 3)
        # cv2.putText(img,label + '/' + str(round(cosine_similarity[i], 2)) + '/' + str(dist_travelled[i]),(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 3, [255,255,255], 3)
    return img



if __name__ == '__main__':
    for i in range(82):
        print(compute_color_for_labels(i))
