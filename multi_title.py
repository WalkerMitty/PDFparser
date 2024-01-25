import layoutparser as lp
# from layoutparser.ocr.tesseract_agent import TesseractFeatureType
import cv2
import numpy as np
# from pdf2image import convert_from_bytes
from tqdm import tqdm
from typing import List
import json

file_name = '23.2307.14893.json'
IMAGE_LEN = 15

with open(file_name,encoding='utf-8') as f:
    data = json.load(f)
json_titles = {}
big_title = ''
for id,item in enumerate(data):
    if item['type']=='Title':
        if big_title=='':
            big_title = item['text']
        else:
            if item['text'] in big_title or big_title in item['text']:
                continue
            json_titles[item['text']] = item['metadata']['coordinates']['points'][1][1]-item['metadata']['coordinates']['points'][0][1]

# images = convert_from_bytes(open('test2.pdf', 'rb').read())
# image = np.array(image)
    # Convert the image from BGR (cv2 default loading style)
    # to RGB
# model = lp.Detectron2LayoutModel('lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config',
#                                  extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
#                                  label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"})

model = lp.PaddleDetectionLayoutModel(config_path='lp://PubLayNet/ppyolov2_r50vd_dcn_365e/config',
                                 extra_config={'threshold':0.8},
                                 label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure",},
                      )
ocr_agent = lp.TesseractAgent()
# x = lp.TesseractFeatureType()
# TesseractFeatureType.BLOCK


def smooth(reverse_list:List[tuple]):
    threshold = 0.001
    new_list = []

    current_max = reverse_list[0][1]
    new_list.append(reverse_list[0])
    for id,item in enumerate(reverse_list):
        if  id!=0:
            if (current_max-item[1])>threshold:
                current_max=item[1]
                new_list.append(item)
            else:
                new_list.append((item[0],current_max))

    return new_list

all_titles= []
all_height = []

for ii,image in enumerate(tqdm(range(IMAGE_LEN))):
    if ii<10:
        image = cv2.imread("test2_photo/test2_0" + str(ii) + ".jpg")
    else:
        image = cv2.imread("test2_photo/test2_"+str(ii)+".jpg")
    image = image[..., ::-1]

    image = np.array(image)

    layout = model.detect(image)

    text_blocks = lp.Layout([b for b in layout if b.type == 'Title'])

    for block in tqdm(text_blocks):
        segment_image = (block
                         .pad(left=5, right=5, top=5, bottom=5)
                         .crop_image(image))
        text = ocr_agent.detect(segment_image)

        text = text.replace('\n',' ')
        for key,value in json_titles.items():
            if key in text or text in key:
                print('key',key)
                print('text',text)
                all_titles.append(text)
                all_height.append(value)
                break


        block.set(text=text, inplace=True)

    for i, txt in enumerate(text_blocks.get_texts()):
        print(txt)

width_dicts = {}
for id in range(len(all_height)):
    width_dicts[id]=all_height[id]
sort_dicts = sorted(width_dicts.items(),key=lambda x:x[1],reverse=True)
sort_dicts = smooth(sort_dicts)


def return_index(sort_dicts:List[tuple]):
    title_symbol = 1
    final_result = {}  #key: title_index, value: 1,2,3...
    for i in range(len(sort_dicts)):
        if i!=len(sort_dicts)-1:
            if sort_dicts[i][1]==sort_dicts[i+1][1]:
                final_result[sort_dicts[i][0]] = title_symbol
            else:
                final_result[sort_dicts[i][0]] = title_symbol
                title_symbol +=1
        else:
            final_result[sort_dicts[i][0]] = title_symbol
    return final_result,title_symbol

result,total = return_index(sort_dicts)
# print(titles)
result = sorted(result.items(),key=lambda x:x[0])

print(big_title)
for item in result:
    print('  '*item[1],all_titles[item[0]])
