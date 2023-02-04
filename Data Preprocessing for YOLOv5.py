import pandas as pd
import requests
import cv2
import json

label_file = ""  # path to annotation data csv file
comments_file = ""    # path to comment file with information on ROIs
savepath_images = ""   # path to save image files
savepath_labels = ""   # path to save label files

def extract_bndboxes(annotation_string):
    '''
    Get the coordinates of the bounding boxes from an annotation string of an image.
    Parameters
    annotation_string: (str) the annotation string under an annotated layer of an image.
    Return
    box_list: a list of bounding box coordinates in the format of [[xmin, ymin, xmax, ymax],[xmin, ymin, xmax, ymax],...,[xmin, ymin, xmax, ymax]]
    '''
    annotation_dict = json.loads(annotation_string)
    box_list = []
    for object in annotation_dict["objects"]:
        if object["type"] == "polyline":
            points = object["points"]
            x_coordinates = []
            y_coordinates = []
            for point in points:
                x_coordinates.append(point['x'])
                y_coordinates.append(point['y'])
            if (min(x_coordinates) != max(x_coordinates)) and (min(y_coordinates) != max(y_coordinates)):
                box = [min(x_coordinates),min(y_coordinates),max(x_coordinates),max(y_coordinates)]
                box_list.append(box)
    return box_list

def parse_json(comments_file):
    '''parse the comments file with annotated ROIs and get the coordinates for bounding boxes in each image.
    Parameters
    comments_file: (str) path to the json file with comments of the ROIs
    Return
    bndboxes: (dict) dictionary of the bounding boxes in each image {"img_name":[[xmin, ymin, xmax, ymax],[xmin, ymin, xmax, ymax],...,[xmin, ymin, xmax, ymax]]}
    '''
    with open(comments_file,'r') as f:
        dict_proteins = json.load(f) #{"ENSG**_protein":{"img_1":{'base':[''],'a':[''],...},"img_2":{'base':['']},..},...}
    bndboxes = {}
    for protein_name, dict_images in dict_proteins.items():
        for image_name, layers_dict in dict_images.items(): #{"img_1":{'base':[''],'a':[''],...},"img_2":{'base':['']},..}
            if "a" in layers_dict:
                annotation_string = layers_dict["a"][0]
                bndboxes[image_name] = extract_bndboxes(annotation_string)
    return bndboxes

def get_image_urls(label_file):
    '''Get urls for images labeled as 'ciliary' as a dictionary from the csv file.
    Parameter
    label_file: (str) path to the csv file with the image labels
    Return
    image_urls: (dict) dictionary of the url of images labeled as 'ciliary' {"image_name":url,...}
    '''
    df = pd.read_csv(label_file)
    image_urls = {}
    df_ciliary = df.loc[df['Ciliary'] == 'Y']
    for index, row in df_ciliary.iterrows():
        image_urls[row['annotation']] = row['uri']
    return image_urls

def download_annotated_images(image_urls,bndboxes):
    '''Download the images that have annotated ROIs.
    Parameters
    image_urls: (dict) dictionary with the image name and url
    bndboxes: (dict) dictionary with bounding box coordinates of each image
    '''
    for image_name in bndboxes:
        fullpath = "{}/{}.jpg".format(savepath_images,image_name)
        response = requests.get(image_urls[image_name])
        if response.status_code == 200:
            with open(fullpath,'wb') as f:
                f.write(response.content)

#height = img.shape[0]  width = img.shape[1]
#box = [xmin,ymin,xmax,ymax]
def scale_image(img, ratio):
    '''Scale the image to a given ratio.
    Parameters
    img: (numpy.ndarray) image loaded from a specific file
    ratio: (float)
    Return
    img_scaled: the scaled image
    '''
    img_scaled = cv2.resize(img,(0,0),fx=ratio,fy=ratio)
    return img_scaled

def translate_coordinates_scaled(box, ratio):
    '''Translate the coordinates of the bounding box after scaling the image.
    Parameters
    box: (list) a list of bounding box coordinates in the form of [xmin, ymin, xmax, ymax]
    ratio: (float) the ratio to which the image was scaled
    Return
    box_scaled: (list) new bounding box coordinates
    '''
    box_scaled = [float(box[0]) * ratio,float(box[1]) * ratio,float(box[2]) * ratio,float(box[3]) * ratio]
    #box.clear()
    #box.extend(box_scaled)
    return box_scaled

def crop_image(img,width,height):    
    '''Crop the centeral part of an image to a given width and height.
    Parameters
    img: (numpy.ndarray) image loaded from a specific file
    width: (int) width of the image after cropping
    height: (int) height of the image after cropping
    Return
    img_cropped: the cropped image
    '''
    center_x = img.shape[1] / 2 
    center_y = img.shape[0] / 2
    start_x = int(center_x - width/2)
    end_x = int(center_x + width/2)
    start_y = int(center_y - height/2)
    end_y = int(center_y + height/2)
    img_cropped = img[start_x:end_x,start_y:end_y]
    return img_cropped

def is_in_range(box,img,cropped_w,cropped_h):
    ''' Check if an annotated bounding box is still in the image after cropping.
    Parameters
    box: (list) a list of bounding box coordinates in the form of [xmin, ymin, xmax, ymax]
    img: (numpy.ndarray) the loaded image from a specific file
    cropped_w: (int) width of the image after cropping
    cropped_h: (int) height of the image after cropping
    Return
    A Boolean value: True if the bounding box is still within the image; False if it is not
    '''
    center_x = img.shape[1] / 2
    center_y = img.shape[0] / 2
    start_x = int(center_x - cropped_w/2)
    end_x = int(center_x + cropped_w/2)
    start_y = int(center_y - cropped_h/2)
    end_y = int(center_y + cropped_h/2)
    if (start_x < float(box[0])) and (end_x > float(box[2])) and (start_y < float(box[1])) and (end_y > float(box[3])):
        return True
    else:
        return False

def translate_coordinates_cropped(box,img,cropped_w,cropped_h):
    '''Translate the coordinates of the bounding box after cropping the image.
    Parameters
    box: (list) a list of bounding box coordinates in the form of [xmin, ymin, xmax, ymax]
    img: (numpy.ndarray) the loaded image from a specific file
    cropped_w: (int) width of the image after cropping
    cropped_h: (int) height of the image after cropping
    Return
    box_cropped: (list) new bounding box coordinates
    '''
    shift_x = int((img.shape[1] - cropped_w)/2)
    shift_y = int((img.shape[0] - cropped_h)/2)
    box_cropped = [float(box[0]) - shift_x,float(box[1]) - shift_y,float(box[2]) - shift_x,float(box[3]) - shift_y]
    #box.clear()
    #box.extend(box_cropped)
    return box_cropped

def normalize_yolov5(bndboxes):
    '''Get the normalized values of the bounding boxes in the label file for yolov5.
    Original: [xmin, ymin, xmax, ymax]
    yolov5: [x_center, y_center, width, height]; the values are normalized to [0,1].
    Parameters
    bndboxes: (dict) a dictionary of bounding boxes in each image
    Return
    yolov5: (dict) a dictionary of bounding boxes in yolov5 format
    '''
    yolov5 = {}
    for image_name, boxes in bndboxes.items():
        yolov5[image_name] = []
        fullpath = "{}/{}.jpg".format(savepath_images,image_name)
        img = cv2.imread(fullpath)
        img_width = img.shape[1]
        img_height = img.shape[0]
        for box in boxes:
            x_center = (box[0] + box[2])/2
            y_center = (box[1] + box[3])/2
            box_width = box[2] - box[0]
            box_height = box[3] - box[1]
            x_center_normalized = x_center/img_width
            y_center_normalized = y_center/img_height
            box_width_normalized = box_width/img_width
            box_height_normalized = box_height/img_height
            entry = "0 {} {} {} {}\n".format(x_center_normalized,y_center_normalized,box_width_normalized,box_height_normalized)
            yolov5[image_name].append(entry)
    return yolov5


# parsing label file and get image urls
image_urls = get_image_urls(label_file)

# Downloading images containing bounding boxes
bndboxes = parse_json(comments_file)
download_annotated_images(image_urls=image_urls,bndboxes=bndboxes)

# Resizing images and change coordinates of bounding boxes
ratio = 0.5
cropped_w = 1280
cropped_h = 1280
for image_name,boxes in bndboxes.items():
    fullpath = "{}/{}.jpg".format(savepath_images,image_name)
    img = cv2.imread(fullpath)
    if img.shape[0] > 3000 or img.shape[1] > 3000:
        img_3000 = crop_image(img,3000,3000)
        for box in reversed(boxes):
            if is_in_range(box,img,3000,3000):
                box_cropped = translate_coordinates_cropped(box,img,3000,3000)
                box.clear()
                box.extend(box_cropped)
            else:
                boxes.remove(box)
    else:
        img_3000 = img
    
    img_scaled = scale_image(img_3000,ratio=ratio)
    for box in reversed(boxes):
        box_scaled = translate_coordinates_scaled(box,ratio=ratio)
        box.clear()
        box.extend(box_scaled)
    
    img_scaled_cropped = crop_image(img_scaled,width=cropped_w,height=cropped_h)
    for box in reversed(boxes):
        if is_in_range(box, img=img_scaled,cropped_w=cropped_w, cropped_h=cropped_h):
            box_cropped = translate_coordinates_cropped(box,img=img_scaled,cropped_w=cropped_w, cropped_h=cropped_h)
            box.clear()
            box.extend(box_cropped)
        else:
            boxes.remove(box)
    cv2.imwrite(fullpath,img_scaled_cropped)
#if len(boxes) == 0:
    #del bndboxes[image_name]


# normalize the cooordinates to yolov5 format
yolov5 = normalize_yolov5(bndboxes)

# write the label file
for image_name, boxes in yolov5.items():
    fullpath = "{}/{}.txt".format(savepath_labels,image_name)
    with open(fullpath,'w') as f:
        for box in boxes:
            f.write(box)