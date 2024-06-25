import numpy as np
import faiss
from torchvision import models, transforms
from PIL import Image
import json
import cv2


# Define the image transformations
transform = transforms.Compose([
    # transforms.Resize(256),
    transforms.Resize(224),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class Faiss_Index():
    def __init__(self, faiss_img_list_path, img_ap_ar_path):
        print('Init Faiss Model')
        # Load the pretrained model
        self.model = models.resnet50(pretrained=True)
        self.model = self.model.eval()

        with open(faiss_img_list_path, 'r') as f:
            train_img_list = f.readlines()
            
        self.train_img_list = [i.strip() for i in train_img_list]

        for i, img_path in enumerate(self.train_img_list):  
            image_vector_add = self.trans_img(img_path)
            if i == 0:
                image_vector = image_vector_add
            else:
                image_vector = np.vstack((image_vector, image_vector_add))

        d = image_vector.shape[1]
        print('Build Faiss Index')
        self.index = faiss.IndexFlatL2(d)   # build the index

        self.index.add(image_vector)                  # add vectors to the index
        print('Search Index Generated, Num: ', self.index.ntotal)

        # 页面精度判断
        with open(img_ap_ar_path, 'r') as f:
            self.img_ap_ar = json.load(f)

    def resize_image(self, image, size, letterbox_image):
        """
            对输入图像进行resize
        Args:
            size:目标尺寸
            letterbox_image: bool 是否进行letterbox变换
        Returns:指定尺寸的图像
        """
        ih, iw, _ = image.shape
        h, w = size
        if letterbox_image:
            scale = min(w/iw, h/ih)       # 缩放比例
            nw = int(iw*scale)
            nh = int(ih*scale)
            image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
            # 生成画布
            image_back = np.ones((h, w, 3), dtype=np.uint8) * 128
            # 将image放在画布中心区域-letterbox
            image_back[(h-nh)//2: (h-nh)//2 + nh, (w-nw)//2:(w-nw)//2+nw, :] = image
        else:
            image_back = image
        return image_back

    def trans_img(self, img_path):
        # Load the image
        if isinstance(img_path,str):
            image = cv2.imread(img_path)
        else:
            image = img_path
        # print(image)
        # 加上图像预处理代码
        img = self.resize_image(image, (224, 224), True)
        # 将BGR图像转为RGB
        image = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        # Apply the transformations and get the image vector
        image = transform(image).unsqueeze(0)
        image_vector = self.model(image).detach().numpy()
        return image_vector

    def get_retrival_ap_list(self, I, D):
        ap_list = []
        for (i, d) in zip(I[0], D[0]):
            image_path = self.train_img_list[i]
            ap = float(self.img_ap_ar[image_path]['AP'])
            ar = float(self.img_ap_ar[image_path]['AR'])
            d = float(d)
            ap_list.append((d, ap, ar, image_path))
        return ap_list
    
    def score_judge(self, anns, score_threshold=0.90):
        score_all = 0
        count = 0
        for ann in anns:
            if ann['score']:
                score_all += ann['score']
                count += 1
        if count == 0:
            score_mean = 0
        else:
            score_mean = score_all / count
    
        if score_mean > score_threshold:
            score_judge = True
        else:
            score_judge = False
        return score_judge

    def low_ap_percentage(self, ap_list, low_ap_threshold=0.7, dest_threshold=250, num_threshold=3, percentage_shreshold=20):
        """
        dest_threshold  # kmeans距离小于多少才认为两个页面相似
        num_threshold    # 相似的页面大于等于多少张才认为可以评估该页面精度
        low_ap_threshold   # AP为多少以下算低精度
        percentage_shreshold  # 相似页面中，低精度占比多少认为该页面有可能为低精度
        """

        total = 0
        ap_low = 0
        cannot_find = False
        
        for (d, ap, ar, image_path) in ap_list:
            if d > dest_threshold:    # 距离大于250则认为并不相似
                continue
            total += 1
            if ap < low_ap_threshold:
                ap_low += 1
        
        if total < num_threshold:
            cannot_find = True
            search_judge = False
        else:
            ap_low_percentage = (ap_low / total) * 100    
            if ap_low_percentage <= percentage_shreshold:
                search_judge = True
            else:
                search_judge = False
            
        return search_judge, cannot_find
    