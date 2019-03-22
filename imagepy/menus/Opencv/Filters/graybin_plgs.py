import cv2
from imagepy.core.engine import Simple
from imagepy.core.engine import Filter
class GrayScale(Simple):
    title = 'Gray-Scale'
    note = ['rgb','8-bit','preview']


    def run(self, ips, imgs, para=None):
        rst=[]
        for img in imgs:
            rst.append(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))
        ips.set_imgs(rst)


class Binary(Filter):
    title = 'Bin-Scale'
    note = ['8-bit', 'auto_msk', 'auto_snap']


    def run(self, ips, snap, img, para = None):
        return cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)[1]

plgs = [Binary,GrayScale]
