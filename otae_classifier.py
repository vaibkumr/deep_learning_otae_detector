from fastai.transforms import tfms_from_model
from fastai.dataset import ImageClassifierData
from fastai.conv_learner import ConvLearner
import torchvision
from fastai.dataset import open_image
from torchvision.models import resnet34
from fastai.transforms import transforms_side_on
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os



class OtaeSanDetector():
    def __init__(self, file):
        self.PATH = "data/"
        self.MODEL = "224_all_50_better"
        self.sz = 299
        self.arch = resnet34
        self.bs = 28
        self.file = file

    def predict(self):
        print(self.file)
        if not self.file and os.path.isfile(self.file):
            return 0
        tfms = tfms_from_model(
                                self.arch, self.sz,
                                aug_tfms=transforms_side_on,
                                max_zoom=1.1
                                )
        data =  ImageClassifierData.from_paths(
                                self.PATH, tfms=tfms,
                                bs=self.bs,
                                num_workers=4
                                )
        learn = ConvLearner.pretrained(
                                self.arch, data,
                                precompute=True,
                                ps=0.5
                                )
        learn.unfreeze()
        learn.precompute = False
        learn.load(self.MODEL)
        image = open_image(self.file)
        trn_tfms, val_tfms = tfms_from_model(self.arch, self.sz)
        im = val_tfms(image)
        preds = learn.predict_array(im[None])
        return learn.data.classes[np.argmax(preds)]


# if __name__ == "__main__":
#     f = "images/1JPISO-1UFxTG3GYrACkEiw.jpeg"
#     # print(os.path.isfile(f))
#     m = OtaeSanDetector(f)
#     print(m.predict())
