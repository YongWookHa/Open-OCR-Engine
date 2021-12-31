"""
code from brooklyn1900's craft repository.
https://github.com/brooklyn1900/CRAFT_pytorch
"""

import math
import numpy as np
import cv2
from skimage import io


# RGB
NORMALIZE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
NORMALIZE_VARIANCE = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0

"""
box_util
"""
def cal_slope(p1, p2):
    return (p2[1] - p1[1]) / (p2[0] - p1[0] + 1e-5)


def above_line(p, start_point, slope):
    y = (p[0] - start_point[0]) * slope + start_point[1]
    return p[1] < y

def reorder_points(point_list):
    """
    Reorder points of quadrangle.
    (top-left, top-right, bottom right, bottom left).
    :param point_list: List of point. Point is (x, y).
    :return: Reorder points.
    """
    # Find the first point which x is minimum.
    ordered_point_list = sorted(point_list, key=lambda x: (x[0], x[1]))
    first_point = ordered_point_list[0]

    # Find the third point. The slope is middle.
    slope_list = [[cal_slope(first_point, p), p] for p in ordered_point_list[1:]]
    ordered_slope_point_list = sorted(slope_list, key=lambda x: x[0])
    first_third_slope, third_point = ordered_slope_point_list[1]

    # Find the second point which is above the line between the first point and the third point.
    # All that's left is the fourth point.
    if above_line(ordered_slope_point_list[0][1], third_point, first_third_slope):
        second_point = ordered_slope_point_list[0][1]
        fourth_point = ordered_slope_point_list[2][1]
        reverse_flag = False
    else:
        second_point = ordered_slope_point_list[2][1]
        fourth_point = ordered_slope_point_list[0][1]
        reverse_flag = True

    # Find the top left point.
    second_fourth_slope = cal_slope(second_point, fourth_point)
    if first_third_slope < second_fourth_slope:
        if reverse_flag:
            reorder_point_list = [fourth_point, first_point, second_point, third_point]
        else:
            reorder_point_list = [second_point, third_point, fourth_point, first_point]
    else:
        reorder_point_list = [first_point, second_point, third_point, fourth_point]

    return reorder_point_list


def cal_min_box_distance(box1, box2):
    box_distance = [math.pow((p2[0] - p1[0]), 2) + math.pow((p2[1] - p1[1]), 2) for p1 in box1 for p2 in box2]
    return np.min(box_distance)


def reorder_box(box_list):
    """
    Reorder character boxes.
    :param box_list: List of box. Box is a list of point. Point is (x, y).
    :return: Reorder boxes.
    """
    # Calculate the minimum distance between any two boxes.
    box_count = len(box_list)
    distance_mat = np.zeros((box_count, box_count), dtype=np.float32)
    for i in range(box_count):
        box1 = box_list[i]
        for j in range(i + 1, box_count):
            box2 = box_list[j]
            distance = cal_min_box_distance(box1, box2)
            distance_mat[i][j] = distance
            distance_mat[j][i] = distance

    # Find the boxes on the both ends.
    end_box_index = np.argmax(distance_mat)
    nan = distance_mat[end_box_index // box_count, end_box_index % box_count] + 1
    for i in range(box_count):
        distance_mat[i, i] = nan
    last_box_index = start_box_index = end_box_index // box_count
    last_box = box_list[start_box_index]

    # reorder box.
    reordered_box_list = [last_box]
    for i in range(box_count - 1):
        distance_mat[:, last_box_index] = nan
        closest_box_index = np.argmin(distance_mat[last_box_index])
        reordered_box_list.append(box_list[closest_box_index])
        last_box_index = closest_box_index

    return reordered_box_list


def cal_triangle_area(p1, p2, p3):
    """
    Calculate the area of triangle.
    S = |(x2 - x1)(y3 - y1) - (x3 - x1)(y2 - y1)| / 2
    :param p1: (x, y)
    :param p2: (x, y)
    :param p3: (x, y)
    :return: The area of triangle.
    """
    [x1, y1], [x2, y2], [x3, y3] = p1, p2, p3
    return abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)) / 2


def cal_quadrangle_area(points):
    """
    Calculate the area of quadrangle.
    :return: The area of quadrangle.
    """
    points = reorder_points(points)
    p1, p2, p3, p4 = points
    s1 = cal_triangle_area(p1, p2, p3)
    s2 = cal_triangle_area(p3, p4, p1)
    s3 = cal_triangle_area(p2, p3, p4)
    s4 = cal_triangle_area(p4, p1, p2)
    if s1 + s2 == s3 + s4:
        return s1 + s2
    else:
        return 0


def cal_intersection(points):
    """
    Calculate the intersection of diagonals.
    x=[(x3-x1)(x4-x2)(y2-y1)+x1(y3-y1)(x4-x2)-x2(y4-y2)(x3-x1)]/[(y3-y1)(x4-x2)-(y4-y2)(x3-x1)]
    y=(y3-y1)[(x4-x2)(y2-y1)+(x1-x2)(y4-y2)]/[(y3-y1)(x4-x2)-(y4-y2)(x3-x1)]+y1
    :param points: (x1, y1), (x2, y2), (x3, y3), (x4, y4).
    :return: (x, y).
    """
    [x1, y1], [x2, y2], [x3, y3], [x4, y4] = points
    x = ((x3 - x1) * (x4 - x2) * (y2 - y1) + x1 * (y3 - y1) * (x4 - x2) - x2 * (y4 - y2) * (x3 - x1)) \
        / ((y3 - y1) * (x4 - x2) - (y4 - y2) * (x3 - x1) + 1e-5)
    y = (y3 - y1) * ((x4 - x2) * (y2 - y1) + (x1 - x2) * (y4 - y2)) \
        / ((y3 - y1) * (x4 - x2) - (y4 - y2) * (x3 - x1) + 1e-5) + y1
    return [x, y]


def cal_center_point(points):
    points = np.array(points)
    return [round(np.average(points[:, 0])), round(np.average(points[:, 1]))]


def cal_point_pairs(points):
    intersection = cal_intersection(points)
    p1, p2, p3, p4 = points
    point_pairs = [[cal_center_point([p1, p2, intersection]), cal_center_point([p3, p4, intersection])],
                   [cal_center_point([p2, p3, intersection]), cal_center_point([p4, p1, intersection])]]
    return point_pairs


def cal_affinity_box(point_pairs1, point_pairs2):
    areas = [cal_quadrangle_area([p1, p2, p3, p4]) for p1, p2 in point_pairs1 for p3, p4 in point_pairs2]
    max_area_index = np.argmax(areas)
    affinity_box = [point_pairs1[max_area_index // 2][0],
                    point_pairs1[max_area_index // 2][1],
                    point_pairs2[max_area_index % 2][0],
                    point_pairs2[max_area_index % 2][1]]
    return np.int32(affinity_box)


def cal_affinity_boxes(region_box_list, reorder_point_flag=True, reorder_box_flag=True):
    if reorder_point_flag:
        region_box_list = [reorder_points(region_box) for region_box in region_box_list]
    if reorder_box_flag:
        region_box_list = reorder_box(region_box_list)
    point_pairs_list = [cal_point_pairs(region_box) for region_box in region_box_list]
    affinity_box_list = list()
    for i in range(len(point_pairs_list) - 1):
        affinity_box = cal_affinity_box(point_pairs_list[i], point_pairs_list[i + 1])
        reorder_affinity_box = reorder_points(affinity_box)
        affinity_box_list.append(reorder_affinity_box)
    return affinity_box_list


"""
fake_util
"""
def watershed(src):
    """
    Performs a marker-based image segmentation using the watershed algorithm.
    :param src: 8-bit 1-channel image.
    :return: 32-bit single-channel image (map) of markers.
    """
    # cv2.imwrite('{}.png'.format(np.random.randint(1000)), src)
    gray = src.copy()
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    # h, w = gray.shape[:2]
    # block_size = (min(h, w) // 4 + 1) * 2 + 1
    # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, 0)
    _ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    # dist_transform = opening & gray
    # cv2.imshow('dist_transform', dist_transform)
    # _ret, sure_bg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, cv2.THRESH_BINARY_INV)
    _ret, sure_fg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, cv2.THRESH_BINARY)

    # Finding unknown region
    # sure_bg = np.uint8(sure_bg)
    sure_fg = np.uint8(sure_fg)
    # cv2.imshow('sure_fg', sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker label
    lingret, marker_map = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    marker_map = marker_map + 1

    # Now, mark the region of unknown with zero
    marker_map[unknown == 255] = 0

    marker_map = cv2.watershed(img, marker_map)
    return marker_map


def find_box(marker_map):
    """
    Calculate the minimum enclosing rectangles.
    :param marker_map: Input 32-bit single-channel image (map) of markers.
    :return: A list of point.
    """
    boxes = list()
    marker_count = np.max(marker_map)

    for marker_number in range(2, marker_count + 1):
        marker_cnt = np.swapaxes(np.array(np.where(marker_map == marker_number)), axis1=0, axis2=1)[:, ::-1]
        #swapaxes交换矩阵维度
        rect = cv2.minAreaRect(marker_cnt)
        box = cv2.boxPoints(rect)
        #box = np.int0(box)
        #改变box顺序
        box = np.array([box[1,:],
                     box[2,:],
                     box[3,:],
                     box[0,:]],dtype=np.int0)

        boxes.append(box)
    return boxes

def cal_distance(p1, p2):
    return math.sqrt(math.pow((p2[0] - p1[0]), 2) + math.pow((p2[1] - p1[1]), 2))


def crop_image(src, points, dst_height=None):
    """
    Crop heat map with points.
    :param src: 8-bit single-channel image (map).
    :param points: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]].
    :return: dst_heat_map: Cropped image. 8-bit single-channel image (map) of heat map.
             src_points: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]].
             dst_points: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]].
    """
    src_image = src.copy()
    src_points = np.float32(points)
    width = round((cal_distance(points[0], points[1]) + cal_distance(points[2], points[3])) / 2)
    height = round((cal_distance(points[1], points[2]) + cal_distance(points[3], points[0])) / 2)
    if dst_height is not None:
        ratio = dst_height / min(width, height)
        width = int(width * ratio)
        height = int(height * ratio)
    crop_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    perspective_mat = cv2.getPerspectiveTransform(src=src_points, dst=crop_points)
    dst_heat_map = cv2.warpPerspective(src_image, perspective_mat, (width, height),
                                       borderValue=0, borderMode=cv2.BORDER_CONSTANT)
    return dst_heat_map, src_points, crop_points

def ic13_crop_image(src, points, dst_height=None):
    """
    Crop heat map with points.
    :param src: 8-bit single-channel image (map).
    :param points: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]].
    :return: dst_heat_map: Cropped image. 8-bit single-channel image (map) of heat map.
             src_points: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]].
             dst_points: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]].
    """
    src_image = src.copy()
    src_points = np.float32([[points[0],points[1]], [points[2], points[1]], [points[2], points[3]], [points[0], points[3]]])
    width = round(points[2] - points[0] + 1)
    height = round(points[3] - points[1] + 1)
    if dst_height is not None:
        ratio = dst_height / min(width, height)
        width = int(width * ratio)
        height = int(height * ratio)
    crop_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    perspective_mat = cv2.getPerspectiveTransform(src=src_points, dst=crop_points)
    dst_heat_map = cv2.warpPerspective(src_image, perspective_mat, (width, height),
                                       borderValue=0, borderMode=cv2.BORDER_CONSTANT)
    return dst_heat_map, src_points, crop_points


def un_warping(box, src_points, crop_points):
    """
    Unwarp the character bounding boxes.
    :param box: The character bounding box.
    :param src_points: Points before crop.
    :param crop_points: Points after crop.
    :return: The character bounding boxes after unwarp.
    """
    perspective_mat = cv2.getPerspectiveTransform(src=crop_points, dst=src_points)
    new_box = list()
    for x, y in box:
        new_x = int((perspective_mat[0][0] * x + perspective_mat[0][1] * y + perspective_mat[0][2]) /
                    (perspective_mat[2][0] * x + perspective_mat[2][1] * y + perspective_mat[2][2]))
        new_y = int((perspective_mat[1][0] * x + perspective_mat[1][1] * y + perspective_mat[1][2]) /
                    (perspective_mat[2][0] * x + perspective_mat[2][1] * y + perspective_mat[2][2]))
        new_box.append([new_x, new_y])
    return new_box


def enlarge_char_box(char_box, ratio):
    x_center, y_center = np.average(char_box[:, 0]), np.average(char_box[:, 1])
    char_box = char_box - [x_center, y_center]
    char_box = char_box * ratio
    char_box = char_box + [x_center, y_center]
    return char_box


def enlarge_char_boxes(char_boxes, crop_box):
    char_boxes = np.reshape(np.array(char_boxes), newshape=(-1, 4, 2))
    left, right, top, bottom = np.min(char_boxes[:, :, 0]), np.max(char_boxes[:, :, 0]), \
                               np.min(char_boxes[:, :, 1]), np.max(char_boxes[:, :, 1])
    width, height = crop_box[2, 0], crop_box[2, 1]
    offset = np.min([left, top, width - right, height - bottom])
    ratio = 1 + offset * 2 / min(width, height)
    char_boxes = np.array([enlarge_char_box(char_box, ratio) for char_box in char_boxes])
    char_boxes[:, :, 0] = np.clip(char_boxes[:, :, 0], 0, width)
    char_boxes[:, :, 1] = np.clip(char_boxes[:, :, 1], 0, height)
    return char_boxes


def divide_region(box, length):
    if length == 1:
        return [box]
    char_boxes = list()
    p1, p2, p3, p4 = box
    if cal_distance(p1, p2) + cal_distance(p3, p4) > cal_distance(p2, p3) + cal_distance(p4, p1):
        x_start1 = p1[0]
        y_start1 = p1[1]
        x_start2 = p4[0]
        y_start2 = p4[1]
        x_offset1 = (p2[0] - p1[0]) / length
        y_offset1 = (p2[1] - p1[1]) / length
        x_offset2 = (p3[0] - p4[0]) / length
        y_offset2 = (p3[1] - p4[1]) / length
    else:
        x_offset1 = (p4[0] - p1[0]) / length
        y_offset1 = (p4[1] - p1[1]) / length
        x_offset2 = (p3[0] - p2[0]) / length
        y_offset2 = (p3[1] - p2[1]) / length
        x_start1 = p1[0]
        y_start1 = p1[1]
        x_start2 = p2[0]
        y_start2 = p2[1]
    for i in range(length):
        char_boxes.append([
            [round(x_start1 + x_offset1 * i), round(y_start1 + y_offset1 * i)],
            [round(x_start1 + x_offset1 * (i + 1)), round(y_start1 + y_offset1 * (i + 1))],
            [round(x_start2 + x_offset2 * i), round(y_start2 + y_offset2 * i)],
            [round(x_start2 + x_offset2 * (i + 1)), round(y_start2 + y_offset2 * (i + 1))]
        ])

    return char_boxes


def cal_confidence(boxes, word_length):
    """
    Calculate the confidence score for the pseudo-GTs.
                (l(w) − min(l(w),|l(w) − lc(w)|))/l(w)
    l(w) is the word length of the sample w.
    lc(w) is the count of estimated character bounding boxes.
    :param boxes: The estimated character bounding boxes.
    :param word_length: The length of manually marked word.
    :return: Float. The confidence score for the  pseudo-GTs.
    """
    box_count = len(boxes)
    confidence = (word_length - min(word_length, abs(word_length - box_count))) / word_length
    return confidence


"""
img_util
"""
def load_image(img_path):
    """
    Load an image from file.
    :param img_path: Image file path, e.g. ``test.jpg`` or URL.
    :return: An RGB-image MxNx3.
    """
    img = io.imread(img_path)
    if img.shape[0] == 2:
        img = img[0]
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    img = np.array(img)

    return img


def img_normalize(src):
    """
    Normalize a RGB image.
    :param src: Image to normalize. Must be RGB order.
    :return: Normalized Image
    """
    img = src.copy().astype(np.float32)

    img -= NORMALIZE_MEAN
    img /= NORMALIZE_VARIANCE
    return img


def img_unnormalize(src):
    """
    Unnormalize a RGB image.
    :param src: Image to unnormalize. Must be RGB order.
    :return: Unnormalized Image.
    """
    img = src.copy()

    img *= NORMALIZE_VARIANCE
    img += NORMALIZE_MEAN

    return img.astype(np.uint8)


def img_resize(src, ratio, max_size, interpolation):
    """
    Resize image with a ratio.
    :param src: Image to resize.
    :param ratio: Scaling ratio.
    :param max_size: Maximum size of Image.
    :param interpolation: Interpolation method. See OpenCV document.
    :return: dst: Resized image.
             target_ratio: Actual scaling ratio.
    """
    img = src.copy()
    height, width, channel = img.shape

    target_ratio = min(max_size / max(height, width), ratio)
    target_h, target_w = int(height * target_ratio), int(width * target_ratio)
    dst = cv2.resize(img, (target_w, target_h), interpolation=interpolation)

    return dst, target_ratio


def score_to_heat_map(score):
    """
    Convert region score or affinity score to heat map.
    :param score: Region score or affinity score.
    :return: Heat map.
    """
    heat_map = (np.clip(score, 0, 1) * 255).astype(np.uint8)
    heat_map = cv2.applyColorMap(heat_map, cv2.COLORMAP_JET)
    return heat_map


def create_affinity_box(boxes):
    affinity_boxes = cal_affinity_boxes(boxes)
    return affinity_boxes


def create_score_box(boxes_list):
    region_box_list = list()
    affinity_box_list = list()

    for boxes in boxes_list:
        region_box_list.extend(boxes)
        if len(boxes) > 0:
            affinity_box_list.extend(create_affinity_box(boxes))

    return region_box_list, affinity_box_list


def load_sample(img_path, img_size, word_boxes, boxes_list):
    img = load_image(img_path)

    height, width = img.shape[:2]
    ratio = img_size / max(height, width)
    target_height = int(height * ratio)
    target_width = int(width * ratio)
    img = cv2.resize(img, (target_width, target_height))

    normalized_img = img_normalize(img)
    # padding
    img = np.zeros((img_size, img_size, 3), dtype=np.float32)
    img[:target_height, :target_width] = normalized_img

    word_boxes = [[[int(x * ratio), int(y * ratio)] for x, y in box] for box in word_boxes]

    if len(boxes_list) == 0:
        return img, word_boxes, boxes_list, [], [], (target_width, target_height)

    boxes_list = [[[[int(x * ratio), int(y * ratio)] for x, y in box] for box in boxes] for boxes in boxes_list]
    region_box_list, affinity_box_list = create_score_box(boxes_list)

    return img, word_boxes, boxes_list, region_box_list, affinity_box_list, (target_width, target_height)


def to_heat_map(img):
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return img

"""
Gaussian
"""
def gaussian_2d():
    """
    Create a 2-dimensional isotropic Gaussian map.
    :return: a 2D Gaussian map. 1000x1000.
    """
    mean = 0
    radius = 2.5
    # a = 1 / (2 * np.pi * (radius ** 2))
    a = 1.
    x0, x1 = np.meshgrid(np.arange(-5, 5, 0.01), np.arange(-5, 5, 0.01))
    x = np.append([x0.reshape(-1)], [x1.reshape(-1)], axis=0).T

    m0 = (x[:, 0] - mean) ** 2
    m1 = (x[:, 1] - mean) ** 2
    gaussian_map = a * np.exp(-0.5 * (m0 + m1) / (radius ** 2))
    gaussian_map = gaussian_map.reshape(len(x0), len(x1))

    max_prob = np.max(gaussian_map)
    min_prob = np.min(gaussian_map)
    gaussian_map = (gaussian_map - min_prob) / (max_prob - min_prob)
    gaussian_map = np.clip(gaussian_map, 0., 1.)
    return gaussian_map

class GaussianGenerator:
    def __init__(self):
        self.gaussian_img = gaussian_2d()

    @staticmethod
    def perspective_transform(src, dst_shape, dst_points):
        """
        Perspective Transform
        :param src: Image to transform.
        :param dst_shape: Tuple of 2 intergers(rows and columns).
        :param dst_points: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]].
        :return: Image after perspective transform.
        """
        img = src.copy()
        h, w = img.shape[:2]

        src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst_points = np.float32(dst_points)
        perspective_mat = cv2.getPerspectiveTransform(src=src_points, dst=dst_points)
        dst = cv2.warpPerspective(img, perspective_mat, (dst_shape[1], dst_shape[0]),
                                  borderValue=0, borderMode=cv2.BORDER_CONSTANT)
        return dst

    def gen(self, score_shape, points_list):
        score_map = np.zeros(score_shape, dtype=np.float32)
        for points in points_list:
            tmp_score_map = self.perspective_transform(self.gaussian_img, score_shape, points)
            score_map = np.where(tmp_score_map > score_map, tmp_score_map, score_map)
        score_map = np.clip(score_map, 0, 1.)
        return score_map