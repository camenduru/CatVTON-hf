from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import requests
import matplotlib.pyplot as plt
import torch.nn as nn


FASHION_MAP = {
    "0":"Everything Else", "1": "shirt, blouse", "2": "top, t-shirt, sweatshirt", 
    "3": "sweater", "4": "cardigan", "5": "jacket", "6": "vest", "7": "pants", 
    "8": "shorts", "9": "skirt", "10": "coat", "11": "dress", "12": "jumpsuit", 
    "13": "cape", "14": "glasses", "15": "hat", "16": "headband, head covering, hair accessory", 
    "17": "tie", "18": "glove", "19": "watch", "20": "belt", "21": "leg warmer", 
    "22": "tights, stockings", "23": "sock", "24": "shoe", "25": "bag, wallet", 
    "26": "scarf", "27": "umbrella", "28": "hood", "29": "collar", "30": "lapel", 
    "31": "epaulette", "32": "sleeve", "33": "pocket", "34": "neckline", "35": "buckle", 
    "36": "zipper", "37": "applique", "38": "bead", "39": "bow", "40": "flower", "41": "fringe", 
    "42": "ribbon", "43": "rivet", "44": "ruffle", "45": "sequin", "46": "tassel"
}


HUMAN_MAP = {
    "0":"Background","1":"shirt, blouse","2":"top, t-shirt, sweatshirt","3":"sweater",
    "4":"cardigan","5":"jacket","6":"vest","7":"pants","8":"shorts","9":"skirt",
    "10":"coat","11":"dress","12":"jumpsuit","13":"cape","14":"glasses","15":"hat",
    "16":"headband, head covering, hair accessory","17":"tie","18":"glove","19":"watch",
    "20":"belt","21":"leg warmer","22":"tights, stockings","23":"sock","24":"shoe",
    "25":"bag, wallet","26":"scarf","27":"umbrella","28":"hood","29":"collar","30":"lapel",
    "31":"epaulette","32":"sleeve","33":"pocket","34":"neckline","35":"buckle","36":"zipper",
    "37":"applique","38":"bead","39":"bow","40":"flower","41":"fringe","42":"ribbon",
    "43":"rivet","44":"ruffle","45":"sequin","46":"tassel","47":"Hair","48":"Sunglasses",
    "49":"Upper-clothes","50":"Left-shoe","51":"Right-shoe","52":"Face","53":"Left-leg",
    "54":"Right-leg","55":"Left-arm","56":"Right-arm"
}



class Segformer:
    def __init__(self, model_name, device='cuda'):
        self.device = device
        self.processor = SegformerImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(model_name).to(device)
        
        
    def predict(self, image: Image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs) 
        logits = outputs.logits.cpu()

        upsampled_logits = nn.functional.interpolate(
            logits,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )
        pred_seg = upsampled_logits.argmax(dim=1)[0]
        # to PIL image
        pred_seg = Image.fromarray(pred_seg.byte().cpu().numpy())
        return pred_seg
    
    def __call__(self, image: Image):
        return self.predict(image)



