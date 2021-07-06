from PIL import Image
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import os
import numpy as np

def rotate(img, bbs, specie, offset=15, phase="train",phase_count={ "train": 1,"test": 1 }, degs=[45,90,135,180,225,270,315]):
    if img is None:
        return phase_count
    for deg in degs:
        name = specie + "-" + phase + "-{:010d}".format(phase_count[phase])
        image_aug, bbs_aug = iaa.Affine(rotate=deg,mode="reflect")(image=img, bounding_boxes=bbs)
        aug_h, aug_w, aug_c = image_aug.shape
        clip_off_bbs = bbs_aug.remove_out_of_image(fully=True,partly=True).clip_out_of_image()
        bboxs = ""
        for bb in clip_off_bbs.bounding_boxes:
            if bb.label == "1":
                x1 = bb.x1
                y1 = bb.y1
                x2 = bb.x2
                y2 = bb.y2
                if deg % 90 != 0:
                    x1 = x1 + offset
                    y1 = y1 + offset
                    x2 = x2 - offset
                    y2 = y2 - offset
                bb_w = (x2-x1)/aug_w
                bb_h = (y2-y1)/aug_h
                bb_xc = (((x2-x1)/2)+x1)/aug_w
                bb_yc = (((y2-y1)/2)+y1)/aug_h
                bboxs += "\n"+ str(bb.label) + " " + str(bb_xc) + " " + str(bb_yc) + " " + str(bb_w) + " " + str(bb_h)
        if len(clip_off_bbs.bounding_boxes) > 0:
            bboxs = bboxs[1:]
            im_aug = Image.fromarray(image_aug)
            im_aug.save(os.path.join("../malaria/images/train",name+".jpg"))
            f = open(os.path.join("../malaria/labels/train",name+".txt"),'w')
            f.write(bboxs)
            f.close()
            print("{} {} {}".format(phase_count[phase],deg, name))
            phase_count[phase] += 1
    return phase_count

def flipHor(img, bbs, specie, phase="train",phase_count={ "train": 1,"test": 1 }):
    if img is None:
        return None,None,phase_count
    name = specie + "-" + phase + "-{:010d}".format(phase_count[phase])
    image_aug, bbs_aug = iaa.Fliplr()(image=img, bounding_boxes=bbs)
    aug_h, aug_w, aug_c = image_aug.shape
    clip_off_bbs = bbs_aug.remove_out_of_image(fully=True,partly=True).clip_out_of_image()
    bboxs = ""
    for bb in clip_off_bbs.bounding_boxes:
        x1 = bb.x1
        y1 = bb.y1
        x2 = bb.x2
        y2 = bb.y2
        bb_w = (x2-x1)/aug_w
        bb_h = (y2-y1)/aug_h
        bb_xc = (((x2-x1)/2)+x1)/aug_w
        bb_yc = (((y2-y1)/2)+y1)/aug_h
        bboxs += "\n"+ str(bb.label) + " " + str(bb_xc) + " " + str(bb_yc) + " " + str(bb_w) + " " + str(bb_h)
    if len(clip_off_bbs.bounding_boxes) > 0:
        bboxs = bboxs[1:]
        im_aug = Image.fromarray(image_aug)
        im_aug.save(os.path.join("../malaria/images/train",name+".jpg"))
        f = open(os.path.join("../malaria/labels/train",name+".txt"),'w')
        f.write(bboxs)
        f.close()
        print("{} flipLR {}".format(phase_count[phase], name))
        phase_count[phase] += 1
        return np.array(im_aug), clip_off_bbs, phase_count
    else:
        return None,None,phase_count
    
def flipVer(img, bbs, specie, phase="train",phase_count={ "train": 1,"test": 1 }):
    if img is None:
        return None,None,phase_count
    name = specie + "-" + phase + "-{:010d}".format(phase_count[phase])
    image_aug, bbs_aug = iaa.Flipud()(image=img, bounding_boxes=bbs)
    aug_h, aug_w, aug_c = image_aug.shape
    clip_off_bbs = bbs_aug.remove_out_of_image(fully=True,partly=True).clip_out_of_image()
    bboxs = ""
    for bb in clip_off_bbs.bounding_boxes:
        x1 = bb.x1
        y1 = bb.y1
        x2 = bb.x2
        y2 = bb.y2
        bb_w = (x2-x1)/aug_w
        bb_h = (y2-y1)/aug_h
        bb_xc = (((x2-x1)/2)+x1)/aug_w
        bb_yc = (((y2-y1)/2)+y1)/aug_h
        bboxs += "\n"+ str(bb.label) + " " + str(bb_xc) + " " + str(bb_yc) + " " + str(bb_w) + " " + str(bb_h)
    if len(clip_off_bbs.bounding_boxes) > 0:
        bboxs = bboxs[1:]
        im_aug = Image.fromarray(image_aug)
        im_aug.save(os.path.join("../malaria/images/train",name+".jpg"))
        f = open(os.path.join("../malaria/labels/train",name+".txt"),'w')
        f.write(bboxs)
        f.close()
        print("{} flipUD {}".format(phase_count[phase], name))
        phase_count[phase] += 1
        return np.array(im_aug), clip_off_bbs, phase_count
    else:
        return None,None,phase_count