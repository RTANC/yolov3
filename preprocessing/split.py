from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBoxesOnImage
import os
from PIL import Image

def getXYcenter(bb):
    xc = bb.x1 + ((bb.x2 - bb.x1) / 2)
    yc = bb.y1 + ((bb.y2 - bb.y1) / 2)
    return xc, yc

def xyxy2xcycwh(bb, w, h):
    xc = ((bb.x1 + ((bb.x2 - bb.x1) / 2)) / w)
    yc = ((bb.y1 + ((bb.y2 - bb.y1) / 2)) / h)
    bw = (bb.x2 - bb.x1) / w
    bh = (bb.y2 - bb.y1) / h
    return xc, yc, bw, bh, bb.label

def removeBorderBox(img, bbs):
    rects = []
    bboxs_txt = ""
    h, w, c = img.shape
    for bb in bbs:
        xc, yc = getXYcenter(bb)
        if (xc >= 26 and xc <= 390) and (yc >= 26 and yc <= 390):
            rects.append(bb)
            bxc, byc, bw, bh, label = xyxy2xcycwh(bb, w, h)
            bboxs_txt += "\n"+ label + " " + str(bxc) + " " + str(byc) + " " + str(bw) + " " + str(bh)
    boxes = BoundingBoxesOnImage(rects,shape=img.shape)
    bboxs_txt = bboxs_txt[1:]
    return boxes, bboxs_txt

def split_img(img, bbs, phase_count={ "train": 1,"test": 1 }, phase="train"):
    h, w, c = img.shape
    for bb in bbs:
        if bb.label == "1":
            xc = bb.x1 / w
            yc = bb.y1 / h
            img_aug, bbs_aug = iaa.CropToFixedSize(width=416, height=416, position=(xc, yc))(image=img, bounding_boxes=bbs)
            clip_off_bbs = bbs_aug.remove_out_of_image(fully=True,partly=False).clip_out_of_image()
            bbs_aug, bboxs = removeBorderBox(img, clip_off_bbs)
            name = "pf-" + phase + "-{:010d}".format(phase_count[phase])
            print(name)
            Image.fromarray(img_aug).save(os.path.join("../malaria/images/train",name + ".jpg"))
            f = open(os.path.join("../malaria/labels/train",name + ".txt"),'w')
            bboxs = bboxs[1:]
            f.write(bboxs)
            f.close()
            phase_count[phase] += 1
    return img_aug, bbs_aug, phase_count