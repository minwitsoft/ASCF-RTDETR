import os
import json
import cv2
from collections import defaultdict


def analyze_cell_annotations(folder_path):
    # 存储结果的字典
    results = {}

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 处理JSON文件
        if filename.endswith('.json'):
            image_name = filename.replace('.json', '')
            json_path = os.path.join(folder_path, filename)

            # 读取对应的图片文件（支持.jpg和.png）
            img_path = None
            for ext in ['.jpg', '.png']:
                possible_img_path = os.path.join(folder_path, image_name + ext)
                if os.path.exists(possible_img_path):
                    img_path = possible_img_path
                    break

            if img_path is None:
                continue

            # 读取JSON文件
            with open(json_path, 'r', encoding='utf-8') as f:
                annotation_data = json.load(f)

            # 统计每个类别的细胞数量
            cell_counts = defaultdict(int)

            # 这里需要根据你的JSON格式进行调整
            # 假设JSON中有一个shapes字段包含所有标注
            if 'shapes' in annotation_data:
                for shape in annotation_data['shapes']:
                    label = shape['label']
                    cell_counts[label] += 1

            # 保存结果
            results[image_name] = dict(cell_counts)

    return results


# 使用示例
folder_path = r"D:\package\study\hz-dataall"
results = analyze_cell_annotations(folder_path)

# 打印结果
for image_name, counts in results.items():
    print(f"\n图片 {image_name} 的统计结果：")
    for cell_type, count in counts.items():
        print(f"- {cell_type}: {count} 个细胞")