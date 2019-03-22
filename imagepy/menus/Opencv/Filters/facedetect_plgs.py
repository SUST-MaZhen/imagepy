import cv2
from imagepy.core.engine import Filter
from imagepy.core.mark import GeometryMark
import pandas as pd
from imagepy import IPy
import os

class Plugin(Filter):
    title = 'Face-Detect'
    note = ['all','not_channel','auto_msk', 'auto_snap']


    def run(self, ips, snap, img, para = None):
        # 导入训练好的人脸分类文件
        # 获取当前工作路径
        path = os.getcwd()
        face_cascade = cv2.CascadeClassifier(path+ '\menus\Opencv\Filters\haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x ,y, w, h) in faces:
            ips.mark = GeometryMark({'type': 'rectangles', 'color': (255, 0, 0), 'lw':2,'fcolor': (0, 0, 0), 'fill': False,'body': [(x, y, w, h)]})
            # 用下面这句会影响到原图像，导致不能再编辑，可以用上面的标记
            # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        IPy.show_table(pd.DataFrame([[x, y, w, h]],index=['矩形'],columns=['x','y','w','h']),'人脸参数')