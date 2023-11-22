import os
import json
from xml.dom import minidom
import re


class ConfigurationDetectClasses:
    """
    Configuration class for detection classes. Loads class ID and amount of keypoints from a configuration file.

    Attributes:
        classes_id (dict): A mapping of class names to their respective IDs.
        amount_of_keypoints (dict): A mapping of class IDs to their respective number of keypoints.
    """

    def __init__(self, configuration_file_path: str):
        self.classes_id = {}
        self.amount_of_keypoints = {}
        with open(configuration_file_path, 'r') as configuration_file:
            detecting_classes_list = json.load(configuration_file)

        for class_key, detecting_class in detecting_classes_list["class"].items():
            self.classes_id[detecting_class["name"]] = detecting_class["id"]
            self.amount_of_keypoints[detecting_class["id"]] = detecting_class["keypoints_amount"]


class DetectedObject:
    """
    Represents a detected object with bounding box and keypoints information.

    Attributes:
        bounding_box_node (minidom.Element): The XML node representing the bounding box of the detected object.
        points_node_list (minidom.NodeList): The list of XML nodes representing keypoints of the detected object.
        object_class (str): The class name of the detected object.
        object_class_id (int): The class ID of the detected object.
    """

    def __init__(self, bounding_box_node: minidom.Element, points_node_list: minidom.NodeList, object_class: str, class_info: ConfigurationDetectClasses):
        self.bounding_box_node = bounding_box_node
        self.points_node_list = self.order_points_node_list(points_node_list)
        self.object_class = object_class
        self.object_class_id = class_info.classes_id[self.object_class]

    def order_points_node_list(self, points_node_list: minidom.NodeList) -> list:
        ordered_points_node_list = []
        for i in range(1, points_node_list.length + 1):
            for current_keypoint_node in points_node_list:
                if current_keypoint_node.getAttribute("label") == str(i):
                    ordered_points_node_list.append(current_keypoint_node)
        return ordered_points_node_list

    @staticmethod
    def get_detected_object_list(bounding_box_node_list: minidom.NodeList, skeleton_node_list: minidom.NodeList, classes_info: ConfigurationDetectClasses) -> list:
        detected_object_list = []
        for bounding_box_node in bounding_box_node_list:
            bounding_box_attributes_node_list = bounding_box_node.getElementsByTagName('attribute')
            bounding_box_object_class = None
            bounding_box_id = None
            for bounding_box_attributes_node in bounding_box_attributes_node_list:
                if bounding_box_attributes_node.getAttribute('name') == 'class':
                    bounding_box_object_class = bounding_box_attributes_node.firstChild.data
                elif bounding_box_attributes_node.getAttribute('name') == 'id':
                    bounding_box_id = bounding_box_attributes_node.firstChild.data
            for skeleton_node in skeleton_node_list:
                skeleton_attributes_node_list = skeleton_node.getElementsByTagName('attribute')
                skeleton_class = None
                skeleton_id = None
                for skeleton_attributes_node in skeleton_attributes_node_list:
                    if skeleton_attributes_node.getAttribute('name') == 'class':
                        skeleton_class = skeleton_attributes_node.firstChild.data
                    elif skeleton_attributes_node.getAttribute('name') == 'id':
                        skeleton_id = skeleton_attributes_node.firstChild.data
                if skeleton_class == bounding_box_object_class and skeleton_id == bounding_box_id:
                    points = skeleton_node.getElementsByTagName('points')
                    detected_object = DetectedObject(bounding_box_node, points, bounding_box_object_class, classes_info)
                    detected_object_list.append(detected_object)
        return detected_object_list


class Image:
    """
    Represents an image with annotations for object detection.

    Attributes:
        width (int): The width of the image.
        height (int): The height of the image.
        name (str): The name of the image file.
        object_skeleton_node_list (minidom.NodeList): The list of XML nodes representing skeletons in the image.
        bounding_box_node_list (minidom.NodeList): The list of XML nodes representing bounding boxes in the image.
        image_name_with_txt_extension (str): The name of the image file with a '.txt' extension.
        detected_object_list (list): The list of detected objects in the image.
    """

    def __init__(self, image_node: minidom.Element, classes_info: ConfigurationDetectClasses):
        self.width = int(image_node.getAttribute('width'))
        self.height = int(image_node.getAttribute('height'))
        self.name = image_node.getAttribute('name')
        self.object_skeleton_node_list = image_node.getElementsByTagName('skeleton')
        self.bounding_box_node_list = image_node.getElementsByTagName('box')
        self.image_name_with_txt_extension = re.sub("\..*$", ".txt", self.name)
        self.detected_object_list = DetectedObject.get_detected_object_list(self.bounding_box_node_list,
                                                                            self.object_skeleton_node_list, classes_info)


def convert_bounding_box_from_cvat_to_yolo_format(bounding_box: minidom.Element, image_width: int, image_height: int) -> tuple:
    """
    Converts a bounding box from CVAT format to YOLO format.

    Parameters:
        bounding_box (minidom.Element): The XML node representing the bounding box.
        image_width (int): The width of the image.
        image_height (int): The height of the image.

    Returns:
        tuple: A tuple representing the bounding box in YOLO format.
    """
    xtl = int(float(bounding_box.getAttribute('xtl')))
    ytl = int(float(bounding_box.getAttribute('ytl')))
    xbr = int(float(bounding_box.getAttribute('xbr')))
    ybr = int(float(bounding_box.getAttribute('ybr')))
    w = xbr - xtl
    h = ybr - ytl
    return str((xtl + (w / 2)) / image_width), str((ytl + (h / 2)) / image_height), str(w / image_width), str(h / image_height)


def write_points_of_object(keypoint_node: minidom.Element, label_file, image_width: int, image_height: int):
    """
    Writes keypoints of an object to a label file in YOLO format.

    Parameters:
        keypoint_node (minidom.Element): The XML node representing the keypoint.
        label_file (file): The file to write keypoints data.
        image_width (int): The width of the image.
        image_height (int): The height of the image.
    """
    if keypoint_node.attributes["occluded"].value == "1":
        label_file.write('0 0 0 ')
        return

    point_in_string_format = keypoint_node.attributes['points'].value
    points_in_coco_format = []
    point_x, point_y = point_in_string_format.split(',')
    points_in_coco_format.append([int(float(point_x)), int(float(point_y))])
    for point_index, point in enumerate(points_in_coco_format):
        label_file.write('{} {}'.format(point[0] / image_width, point[1] / image_height))
        label_file.write(' 2 ' if point_index == len(points_in_coco_format) - 1 else ' ')


# Script execution
out_dir = './out'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

file = minidom.parse('annotations.xml')
images_node_list = file.getElementsByTagName('image')
classes_information = ConfigurationDetectClasses("class_configuration.json")

for image_node in images_node_list:
    if image_node.getElementsByTagName('box').length > 0:
        image = Image(image_node, classes_information)
        label_file = open(os.path.join(out_dir, image.image_name_with_txt_extension), 'w')

        for detected_object in image.detected_object_list:
            bounding_box = convert_bounding_box_from_cvat_to_yolo_format(detected_object.bounding_box_node, image.width, image.height)
            label_file.write('{} '.format(detected_object.object_class_id))
            label_file.write('{} {} {} {} '.format(*bounding_box))
            for index in range(len(classes_information.amount_of_keypoints)):
                if detected_object.object_class_id == index:
                    for keypoint_node in detected_object.points_node_list:
                        write_points_of_object(keypoint_node, label_file, image.width, image.height)
                else:
                    for j in range(classes_information.amount_of_keypoints[index]):
                        label_file.write("0 0 0 ")
            label_file.write('\n')

        label_file.close()
