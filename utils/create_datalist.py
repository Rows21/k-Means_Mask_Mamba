import os
import glob
import re

ORGAN_DATASET_DIR = './dataset/demo/15_TTS'

def extract_numbers(string):
    pattern = r'\d+'  # 数字模式的正则表达式
    numbers = re.findall(pattern, string)
    return numbers

# 获取文件夹下的所有路径
file_paths = glob.glob(os.path.join(ORGAN_DATASET_DIR, '*'))
for path in file_paths:
    if 'img' in path.split('\\')[-1]:
        img_path = path
    else:
        label_path = path

# 获取子文件夹1中的文件名列表
img_files = os.listdir(img_path)
img_numbers = [extract_numbers(i)[0] for i in img_files]
# 配对成功的文件路径列表
matched_files = []
matchedimg_files = []
matchedlabel_files = []
# 遍历label中的文件名
for label in os.listdir(label_path):
    labelfile_path = '15_TTS/label/' + label
    if label.endswith('nii.gz'):
        index = extract_numbers(label)[0] # 提取文件名中的对应部分
        if index in img_numbers:
            imgpattern = index + 'ct.nii.gz'
            imgfile_path = '15_TTS/img/' + imgpattern
            matched_files.append(imgfile_path + '\t' + labelfile_path)
            matchedimg_files.append(imgfile_path)
            matchedlabel_files.append(labelfile_path)

if matched_files == []:
    raise Exception('图像列表为空，生成失败。')
    
# 保存结果到文本文件
output_file = ORGAN_DATASET_DIR + '/TTS.txt'
with open(output_file, 'w') as f:
    for matched_file in matched_files:
        f.write(matched_file + '\n')