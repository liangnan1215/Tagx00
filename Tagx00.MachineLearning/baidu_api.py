
from aip import AipImageClassify
import os,base64
import requests as req
from io import BytesIO
import json
# 定义常量
APP_ID = '11423030'
API_KEY = 'bfT323NBGt5ke9397FairqMB'
SECRET_KEY = 'yTDojPMLXhH9M3M37Ud5RGk4oZFc69aL '

# 初始化图像
client = AipImageClassify(APP_ID, API_KEY, SECRET_KEY)


def get_path():
    # return "./data/"
    return "/Users/apple/Documents/workspace/java/SE3/Tagx00.MachineLearning/data/"

def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()

def get_baidu_results(data):
    result = {}
    result['recommendTagItemList'] = []
    for temp in data['recommendTagItemList']:
        img_src = temp['url']
        urldic = {}
        urldic['url'] = img_src
        # skimage可以直接以imread()函数来读取网页图片??这里有点小问题
        response = req.get(img_src)
        ls_f = base64.b64encode(BytesIO(response.content).read())
        imgdata = base64.b64decode(ls_f)
        file = open('1.jpg', 'wb')
        file.write(imgdata)
        image = get_file_content('1.jpg')
        """ 调用通用物体识别 """
        aipgneral = client.advancedGeneral(image)
        apiresult = aipgneral['result']
        urldic['tagConfTuples'] = []
        for a in apiresult:
            keyword = {}
            keyword['tag'] = a['keyword']
            keyword['confidence'] = a['score']
            urldic['tagConfTuples'].append(keyword.copy())
        result['recommendTagItemList'].append(urldic.copy())
    return result

def write_baidu_results(data):
    with open(get_path() + "proval/train_baidu.txt", "w") as file:
        result=get_baidu_results(data)
        file.write("{'recommendTagItemList':[")
        for b in result['recommendTagItemList']:
            file.write("{'url':'")
            file.write(b['url'])
            file.write("','tagConfTuples':[")
            for c in b['tagConfTuples']:
                file.write("{'tag':'")
                file.write(c['tag'])
                file.write("','confidence':")
                file.write(str(c['confidence']))
                file.write("},")
            file.write("]}")
            file.write('\n')
        file.write("]}")





