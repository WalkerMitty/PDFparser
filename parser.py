from unstructured.partition.pdf import partition_pdf
from unstructured.staging.base import elements_to_json, convert_to_dict
import layoutparser as lp
from typing import List
import fitz
from tqdm import tqdm
import cv2
import numpy as np
import time
# from paddleocr import PaddleOCR, draw_ocr
import pytesseract

#pre load the Parser
#lp://PubLayNet/mask_rcnn_R_50_FPN_3x/config
#lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config
#lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config
model = lp.PaddleDetectionLayoutModel(config_path='lp://PubLayNet/ppyolov2_r50vd_dcn_365e/config',
                                 extra_config={'threshold':0.8},
                                 label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure",},
                      )

# model = lp.Detectron2LayoutModel(config_path='config.yaml',
#                                  model_path = 'model_final.pth',
#                                  extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.6],
#                                  label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
#                                  device='cuda:0')
# model = lp.Detectron2LayoutModel(config_path='lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config',
#                                  extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
#                                  label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},)
# ocr_agent = lp.TesseractAgent()
# ocr_agent = lp.GCVAgent()
# ocr_agent = PaddleOCR(use_angle_cls=False,use_mp=True,lang="en")

def unstructured(file_name:str):
    '''
    Obtain the title and coordinates using the tool ``unstructured``
    '''
    elements = partition_pdf(
        filename=file_name,
        infer_table_structure=False,
        strategy='fast'
    )
    # print(len(elements))
    # print(type(elements))  List

    data = convert_to_dict(elements)
    '''
    {'type': 'Title', 'element_id': 'b021006a0a5387750370c9f53186f598', 'metadata': {'coordinates': {'points': ((199.0802, 115.00035000000003), (199.0802, 165.22624999999994), (416.335090809946, 165.22624999999994), (416.335090809946, 115.00035000000003)), 'system': 'PixelSpace', 'layout_width': 612, 'layout_height': 792}, 'filename': 'test2.pdf',
     'last_modified': '2023-08-19T06:07:00', 'filetype': 'application/pdf', 'page_number': 1},
     'text': 'Base-based Model Checking for Multi-Agent Only Believing (long version)⋆'}
    '''
    all_text = []
    json_titles = {}
    passage_title = ''
    for id, item in enumerate(data):
        if len(item['text'].split(' '))>1:
            all_text.append(item['text'])
        if item['type'] == 'Title':
            if passage_title == '':
                passage_title = item['text']
            else:
                if item['text'] in passage_title or passage_title in item['text']:
                    continue
                json_titles[item['text']] = item['metadata']['coordinates']['points'][1][1] - \
                                            item['metadata']['coordinates']['points'][0][1]

    return json_titles,passage_title,all_text
def smooth(reverse_list: List[tuple]):
    threshold = 0.001
    new_list = []

    current_max = reverse_list[0][1]
    new_list.append(reverse_list[0])
    for id, item in enumerate(reverse_list):
        if id != 0:
            if (current_max - item[1]) > threshold:
                current_max = item[1]
                new_list.append(item)
            else:
                new_list.append((item[0], current_max))

    return new_list

def return_index(sort_dicts:List[tuple]):
    title_symbol = 1
    final_result = {}  #key: title_index, value: 1,2,3... Indicate the level of the heading
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
def obtain_titles(last_one_index:int,json_titles:dict,passage_title,imageName):
    '''
    Parse the Body Text
    :param last_one_index:
    :param json_titles:
    :param passage_title:
    :param imageName:
    :return:
    '''

    model = lp.Detectron2LayoutModel('lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config',
                                     extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
                                     label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"})
    ocr_agent = lp.TesseractAgent()

    all_titles = []
    all_height = []

    for ii, image in enumerate(tqdm(range(last_one_index+1))):
        name = image_path +str(ii)+ '%s.jpg' % imageName
        image = cv2.imread(name)
        image = image[..., ::-1]

        image = np.array(image)

        layout = model.detect(image)

        text_blocks = lp.Layout([b for b in layout if b.type == 'Title'])  # Loop through each text box on the page.

        for block in tqdm(text_blocks):
            # print(block.to_dict())  'x_1': 372.6684265136719, 'y_1': 895.5380249023438, 'x_2': 586.5267944335938, 'y_2': 929.661865234375, 'block_type': 'rectangle', 'type': 'Title', 'score': 0.987171471118927}
            segment_image = (block
                             .pad(left=5, right=5, top=5, bottom=5)
                             .crop_image(image))
            text = ocr_agent.detect(segment_image)
            text = text.replace('\n', ' ')
            for key, value in json_titles.items():
                if key in text or text in key:
                    # print('key', key)
                    # print('text', text)
                    all_titles.append(text)
                    all_height.append(value)
                    break

            block.set(text=text, inplace=True)
    width_dicts = {}
    for id in range(len(all_height)):
        width_dicts[id] = all_height[id]
    sort_dicts = sorted(width_dicts.items(), key=lambda x: x[1], reverse=True)
    sort_dicts = smooth(sort_dicts)
    result, total = return_index(sort_dicts)
    # print(titles)
    result = sorted(result.items(), key=lambda x: x[0])
    for item in result:
        print('  ' * item[1], all_titles[item[0]])

def obtain_results(last_one_index:int,json_titles:dict,passage_title:str,imageName:str,all_text:List[str]):
    results = []
    '''
    :return: a list
    '''
    first_dict = {}
    first_dict['type'] = 'doctitle'
    first_dict['content_type'] = 'text'
    first_dict['content'] = passage_title
    first_dict['pages'] = []
    results.append(first_dict)
    #Extract other headings and body text in the order of the document.

    single = True #Determine whether it (The pdf) is single-column or double-column.

    center_point_x = 0 # The middle line of double-column

    for ii, image in enumerate(tqdm(range(last_one_index+1))):
        name = image_path +str(ii)+ '%s.jpg' % imageName
        image = cv2.imread(name)
        image = image[..., ::-1]

        image = np.array(image)
        start = time.time()
        layout_temp = model.detect(image)
        if layout_temp.page_data=={}:
            print('layout model can not apply to this kind of paper, using structured format...')
            for text in all_text:
                temp_dict = {}
                temp_dict['type'] = 'text'
                temp_dict['content_type'] = 'text'
                temp_dict['content'] = text
                temp_dict['pages'] = []
                results.append(temp_dict)

            return results
        # print('layout_temp',layout_temp)
        print('the detect model costs',time.time()-start)
        if ii==0:  #Single-column or double-column?
            all_center_x = []
            for x in layout_temp:
                all_center_x.append(x.coordinates[0])
            if (max(all_center_x)-min(all_center_x))>450:
                center_point_x=(max(all_center_x)+min(all_center_x))/2
                single=False
        if single:
            layout = layout_temp.sort(key=lambda x: x.coordinates[1], reverse=False)
        else:
            #双栏排序
            left_layout= []
            right_layout = []
            for i in layout_temp:
                if i.coordinates[0]<center_point_x:
                    left_layout.append(i)
                else:
                    right_layout.append(i)
            left = lp.Layout(left_layout).sort(key=lambda x: x.coordinates[1], reverse=False)
            right = lp.Layout(right_layout).sort(key=lambda x: x.coordinates[1], reverse=False)

            all = []
            for i in left:
                all.append(i)
            for i in right:
                all.append(i)
            layout = lp.Layout(all)



        for block in layout:

            segment_image = (lp.Layout([block])[0]
                             .pad(left=5, right=5, top=5, bottom=5)
                             .crop_image(image))

            start = time.time()
            ocr_text = pytesseract.image_to_string(segment_image)
            print('the ocr costs',time.time()-start)
            new_dict = {}
            new_dict['content_type'] = 'text'
            new_dict['pages'] = []
            if block.type=='Text' or block.type=='List':
                new_dict['type'] = 'text'
                ocr_text = ocr_text.split('\n')
                new_dict['content'] = ocr_text
                results.append(new_dict)
            elif block.type=='Title':
                new_dict['type'] = 'title'
                ocr_text = ocr_text.replace('\n', ' ')
                for key, value in json_titles.items():
                    if key in ocr_text or ocr_text in key:
                        new_dict['content'] = ocr_text
                        new_dict['title_height'] = value
                        results.append(new_dict)
                        break
            else:
                continue

    #Determine the heading level based on the title_height field of the title
    width_dicts = {}
    for id in range(len(results)):
        if results[id]['type']=='title':
            width_dicts[id]=results[id]['title_height']

    sort_dicts = sorted(width_dicts.items(),key=lambda x:x[1], reverse=True)
    sort_dicts = smooth(sort_dicts)
    order_titles,title_level_num = return_index(sort_dicts)
    for key,value in order_titles.items():
        results[key]['title_level'] = value

    for i in results:
        print(i['content'])
        if i['type']=='title':
            print(i['content'])
            print(i['title_level'])

    return results





def pdf2img(pdf_path, img_path):
    pdfDoc = fitz.open(pdf_path)
    last_one = 0
    for id,page in enumerate(pdfDoc.pages()):
        # convert the page to photos
        last_one = id
        rotate = int(0)
        zoom_x =3  # 3
        zoom_y = 3
        # (1.33333333-->1056x816)   (2-->1584x1224)  (3-->3572x2526)
        # The larger the values of x and y, the clearer the image and the larger it becomes. However, the processing time also increases, and it depends on how sharp you want the image to be
        mat = fitz.Matrix(zoom_x, zoom_y)
        pix = page.get_pixmap(matrix=mat, dpi=None, colorspace='rgb', alpha=False)
        imageName = pdf_path.split("/")[len(pdf_path.split("/"))-1]
        target_img_name = img_path +str(id)+ '%s.jpg' % imageName
        pix.save(target_img_name)
    return last_one

def main(pdf_path):
    json_titles,passage_title,all_text = unstructured(pdf_path)

    #pdf to image
    # obtain the coordinates using unstructured

    last_one = pdf2img(pdf_path,image_path)  #the last photo's index
    image_name = pdf_path.split("/")[len(pdf_path.split("/"))-1]
    result = obtain_results(last_one,json_titles,passage_title,image_name,all_text)
    print(result[:5])
    return result






if __name__ == '__main__':
    image_path = 'test2_figures/'

    main('test2.pdf')
