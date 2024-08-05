from model.segformer_b2 import Segformer
from PIL import Image
from model.cloth_masker import AutoMaskerSeg
# model = Segformer("/home/chongzheng_p23/data/Projects/CatVTON-main/Models/segformer_b3_clothes")
image = Image.open("/home/chongzheng_p23/data/Projects/CatVTON-main/resource/demo/example/person/women/1-model_3.png")
# result = model(image)
# result.save("a.png")

masker = AutoMaskerSeg(
    densepose_ckpt="/home/chongzheng_p23/data/Projects/CatVTON-main/Models/densepose",
    segformer_ckpt="/home/chongzheng_p23/data/Projects/CatVTON-main/Models/segformer_b3_clothes")



result = masker(image)['mask']
result.save("b.png")
