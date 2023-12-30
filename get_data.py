import json
import os
import numpy as np

path = 'face_data/val'
files = []
for file in os.listdir(path):
    if file[-5:] == '.json':
        files.append(file)
# print(json.load(open('face_data/face_data_json/'+files[0])))

via_region_data = {}
for file in files:
    temp_json = json.load(open('face_data/val/'+file))

    temp_image = {}
    temp_image['filename'] = file.split('.')[0] + '.jpg'

    shape = temp_json['shapes']
    regions = {}
    for i in range(len(shape)):
        pts = np.array(shape[i]['points'])
        all_pts_x = list(pts[:,0])
        all_pts_y = list(pts[:,1])

        regions[str(i)] = {}
        regions[str(i)]['region_attributes'] = {}
        regions[str(i)]['shape_attributes'] = {}

        regions[str(i)]['shape_attributes']['all_points_x'] = all_pts_x
        regions[str(i)]['shape_attributes']['all_points_y'] = all_pts_y
        regions[str(i)]['shape_attributes']['name'] = shape[i]['label']

    temp_image['regions'] = regions
    temp_image['size'] = 0

    via_region_data[file] = temp_image

with open("images/val/via_region_data.json", "w") as f:
    json.dump(via_region_data,f,sort_keys=False,ensure_ascii=True)
