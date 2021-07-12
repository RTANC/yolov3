from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import os
from PIL import Image
import numpy as np

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
            bb.x1 = min(max(bb.x1, 0), w)
            bb.y1 = min(max(bb.y1, 0), h)
            bb.x2 = min(max(bb.x2, 0), w)
            bb.y2 = min(max(bb.y2, 0), h)
            rects.append(bb)
            bxc, byc, bw, bh, label = xyxy2xcycwh(bb, w, h)
            bboxs_txt += "\n"+ str(label) + " " + str(bxc) + " " + str(byc) + " " + str(bw) + " " + str(bh)
    boxes = BoundingBoxesOnImage(rects,shape=img.shape)
    bboxs_txt = bboxs_txt[1:]
    return boxes, bboxs_txt

def rotate(img, bbs, specie, offset=15, phase="train",phase_count={ "train": 1,"test": 1 }, degs=[45,90,135,180,225,270,315]):
    if img is None:
        return phase_count
    for deg in degs:
        name = specie + "-" + phase + "-{:010d}".format(phase_count[phase])
        image_aug, bbs_aug = iaa.Affine(rotate=deg, mode="reflect")(image=img, bounding_boxes=bbs)
        aug_h, aug_w, aug_c = image_aug.shape
        clip_off_bbs = bbs_aug.remove_out_of_image(fully=True,partly=False).clip_out_of_image()
        clip_off_bbs, _ = removeBorderBox(image_aug, clip_off_bbs.bounding_boxes)
        bboxs = ""
        for bb in clip_off_bbs.bounding_boxes:
            x1 = bb.x1
            y1 = bb.y1
            x2 = bb.x2
            y2 = bb.y2
            xc, yc = getXYcenter(bb)
            if (deg % 90 != 0) and ((xc >= 50 and xc <= 366) and (yc >= 50 and yc <= 366)):
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

def crop_img(img, bbs, phase_count={ "train": 1,"test": 1 }, phase="train", offset=15):
    h, w, c = img.shape
    for bb in bbs:
        if bb.label == 1:
            xc = 1 - (bb.x1 / w)
            yc = 1 - (bb.y1 / h)
            img_aug, bbs_aug = iaa.CropToFixedSize(width=416, height=416, position=(xc, yc))(image=img, bounding_boxes=bbs)
            clip_off_bbs = bbs_aug.remove_out_of_image(fully=True,partly=False).clip_out_of_image()
            bbs_aug, bboxs = removeBorderBox(img_aug, clip_off_bbs)
            name = "pf-" + phase + "-{:010d}".format(phase_count[phase])
            print(name)
            Image.fromarray(img_aug).save(os.path.join("../malaria/images/train",name + ".jpg"))
            f = open(os.path.join("../malaria/labels/train",name + ".txt"),'w')
            f.write(bboxs)
            f.close()
            phase_count[phase] += 1
            phase_count = rotate(img=img_aug,bbs=bbs_aug,specie="pf",offset=offset,phase_count=phase_count)
        
            aug_hor, bbs_hor, phase_count = flipHor(img=img_aug,bbs=bbs_aug,specie="pf",phase_count=phase_count)
            phase_count = rotate(img=aug_hor,bbs=bbs_hor,specie="pf",offset=offset,phase_count=phase_count)

            aug_ver, bbs_ver, phase_count = flipVer(img=aug_hor,bbs=bbs_hor,specie="pf",phase_count=phase_count)
            phase_count = rotate(img=aug_ver,bbs=bbs_ver,specie="pf",offset=offset,phase_count=phase_count)

            aug_ver2, bbs_ver2, phase_count = flipVer(img=img_aug,bbs=bbs_aug,specie="pf",phase_count=phase_count)
            phase_count = rotate(img=aug_ver2,bbs=bbs_ver2,specie="pf",offset=offset,phase_count=phase_count)
    return phase_count