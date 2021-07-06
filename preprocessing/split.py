from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBoxesOnImage
import os
from PIL import Image
from aug import rotate, flipHor, flipVer

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
            bboxs_txt += "\n"+ str(label) + " " + str(bxc) + " " + str(byc) + " " + str(bw) + " " + str(bh)
    boxes = BoundingBoxesOnImage(rects,shape=img.shape)
    bboxs_txt = bboxs_txt[1:]
    return boxes, bboxs_txt

def split_img(img, bbs, phase_count={ "train": 1,"test": 1 }, phase="train"):
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
            bboxs = bboxs[1:]
            f.write(bboxs)
            f.close()
            phase_count[phase] += 1
            phase_count = rotate(img=img_aug,bbs=bbs,specie="pf",offset=15,phase_count=phase_count)
        
            aug_hor, bbs_hor, phase_count = flipHor(img=img_aug,bbs=bbs,specie="pf",phase_count=phase_count)
            phase_count = rotate(img=aug_hor,bbs=bbs_hor,specie="pf",offset=15,phase_count=phase_count)

            aug_ver, bbs_ver, phase_count = flipVer(img=aug_hor,bbs=bbs_hor,specie="pf",phase_count=phase_count)
            phase_count = rotate(img=aug_ver,bbs=bbs_ver,specie="pf",offset=15,phase_count=phase_count)

            aug_ver2, bbs_ver2, phase_count = flipVer(img=img_aug,bbs=bbs,specie="pf",phase_count=phase_count)
            phase_count = rotate(img=aug_ver2,bbs=bbs_ver2,specie="pf",offset=15,phase_count=phase_count)
    return phase_count