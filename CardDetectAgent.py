import os

import cv2
import numpy as np

import Cards


def resize_image(image, width=200, height=300, edge_width=0, edge_height=0, output_path=None):
    """
    將圖片縮放到指定的寬度和高度。
    :param image: 輸入圖片
    :param width: 縮放後的寬度
    :param height: 縮放後的高度
    :param output_path: 如果提供，將縮放後的圖片保存到此路徑
    :param edge_width: 要去除的邊緣寬度（像素）
    :param edge_height: 要去除的邊緣高度（像素）
    :return: 縮放後的圖片數據（numpy.ndarray）
    """

    # 去除邊緣
    image_no_edges = remove_image_edges(image, edge_width=edge_width, edge_height=edge_height)

    # 調整尺寸
    resized_image = cv2.resize(image_no_edges, (width, height), interpolation=cv2.INTER_AREA)

    # 如果指定了輸出路徑，保存圖片
    if output_path:
        cv2.imwrite(output_path, resized_image)

    return resized_image


def remove_image_edges(image, edge_height=0, edge_width=0):
    """
    去除圖片的邊緣，支持單獨設置去除高度和寬度的參數。

    :param image: 輸入圖片（numpy.ndarray）
    :param edge_height: 要去除的邊緣高度（像素）
    :param edge_width: 要去除的邊緣寬度（像素）
    :return: 去除邊緣後的圖片
    """
    if edge_height <= 0 and edge_width <= 0:
        return image

    # 檢查圖片尺寸是否足夠去除邊緣
    if image.shape[0] <= 2 * edge_height or image.shape[1] <= 2 * edge_width:
        raise ValueError("圖片太小，無法去除指定寬度或高度的邊緣")
        # 計算裁剪範圍
    top = edge_height if edge_height > 0 else 0
    bottom = -edge_height if edge_height > 0 else image.shape[0]
    left = edge_width if edge_width > 0 else 0
    right = -edge_width if edge_width > 0 else image.shape[1]
    # 裁剪圖片邊緣
    return image[top:bottom, left:right]

def create_black_image(width, height, output_path=None):
    """
    創建一個指定寬度和高度的黑色圖片。

    :param width: 圖片的寬度（像素）
    :param height: 圖片的高度（像素）
    :return: 創建的黑色圖片數據（numpy.ndarray）
    """
    # 創建黑色圖片 (高度, 寬度, 通道數)
    black_image = np.zeros((height, width, 3), dtype=np.uint8)
    return black_image

def place_images_on_black_canvas(black_canvas, img, interval=10):
    """
    將多張縮放後的圖片依次放置到黑色背景中，每張圖片之間間隔一定像素。

    :param black_canvas: 黑色背景（numpy.ndarray）
    :param resized_images: 縮放後的圖片列表
    :param interval: 圖片之間的間隔像素
    :return: 放置後的黑色背景
    """
    x, y = interval, interval  # 初始放置位置

    h, w = img.shape[:2]
    canvas_h, canvas_w = black_canvas.shape[:2]

    if y + h > canvas_h:  # 如果超出黑色背景的高度，換行放置
        y = interval
        x += w + interval

    if x + w > canvas_w:  # 如果超出黑色背景的寬度，報錯
        raise ValueError("圖片總寬度超出黑色背景，請增加背景寬度或減少圖片數量")

    # 將圖片放置到黑色背景的指定位置
    black_canvas[y:y+h, x:x+w] = img
    return black_canvas

def rec_card_with_black_canvas(img_path, img_name, edge_width, edge_height, is_rotate=False):
    # global train_ranks, train_suits, width, height
    path = os.path.dirname(os.path.abspath(__file__))

    train_ranks = Cards.load_ranks(path + '/Card_Imgs/')
    train_suits = Cards.load_suits(path + '/Card_Imgs/')
    width, height = 200, 300
    image = cv2.imread(img_path)
    if is_rotate:
        image = cv2.rotate(image, cv2.ROTATE_180)
    image = resize_image(image, width, height, edge_width=edge_width, edge_height=edge_height)
    # 創建黑色背景
    black_img = create_black_image(220, 320)
    image = place_images_on_black_canvas(black_img, image)
    # 預處理圖片
    pre_proc = Cards.preprocess_image(image)

    # 找到所有卡片的輪廓
    cnts_sort, cnt_is_card = Cards.find_cards(pre_proc)

    # 如果有卡片輪廓，開始處理 (這個方法只有一張輪廓)
    if len(cnts_sort) != 0:
        for i in range(len(cnts_sort)):
            if cnt_is_card[i] == 1:
                cards = (Cards.preprocess_card(cnts_sort[i], image))
                cards.name = img_name
                (cards.best_rank_match,
                 cards.best_suit_match,
                 cards.rank_diff,
                 cards.suit_diff) = Cards.match_card(cards, train_ranks, train_suits)
                cards.result_img = Cards.draw_results(image, cards)
                return cards
    else:
        card = Cards.Query_card()
        card.name = img_name
        return card

def rotate_image(image, angle):
    """
    旋轉圖片一定角度（逆時針）。

    :param image: 輸入圖片（numpy.ndarray）
    :param angle: 旋轉角度，正數為逆時針
    :return: 旋轉後的圖片
    """
    # 獲取圖片尺寸
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)  # 圖片中心

    # 計算旋轉矩陣
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 應用仿射變換進行旋轉
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    # cv2.imshow('rotated_image', rotated_image)
    # cv2.waitKey(0)
    return rotated_image


def rec_card(img_path, img_name, edge_width, edge_height, is_rotate=False, rotate_degree=180):
    # global train_ranks, train_suits, width, height
    path = os.path.dirname(os.path.abspath(__file__))

    train_ranks = Cards.load_ranks(path + '/Card_Imgs/')
    train_suits = Cards.load_suits(path + '/Card_Imgs/')
    width, height = 200, 300

    image = cv2.imread(img_path)
    if is_rotate and rotate_degree == 180:
        image = cv2.rotate(image, cv2.ROTATE_180)
    elif is_rotate:
        image = rotate_image(image, rotate_degree)


    image = resize_image(image, width, height, edge_width=edge_width, edge_height=edge_height)

    contour = []
    contour = np.array([[[0, 0]], [[width - 1, 0]], [[width - 1, height - 1]], [[0, height - 1]], ])

    cards = Cards.preprocess_card(contour, image)
    cards.name = img_name
    (cards.best_rank_match,
     cards.best_suit_match,
     cards.rank_diff,
     cards.suit_diff) = Cards.match_card(cards, train_ranks, train_suits)
    cards.result_img = Cards.draw_results(image, cards)


    return cards




