import cv2
import time
import os
import json

import numpy as np

import Cards

# 全局變量，用於保存已加載的 Ranks 和 Suits
train_ranks, train_suits = None, None


def remove_image_edges(image, edge_width=0):
    """
    去除圖片的邊緣。

    :param image: 輸入圖片（numpy.ndarray）
    :param edge_width: 要去除的邊緣寬度（像素）
    :return: 去除邊緣後的圖片
    """
    # 檢查圖片尺寸是否足夠去除邊緣
    if image.shape[0] <= 2 * edge_width or image.shape[1] <= 2 * edge_width:
        raise ValueError("圖片太小，無法去除指定寬度的邊緣")

    # 裁剪圖片邊緣
    return image[edge_width:-edge_width, edge_width:-edge_width]

def resize_image(image_path, width, height, output_path=None):
    """
    將圖片縮放到指定的寬度和高度。

    :param image_path: 輸入圖片的路徑
    :param width: 縮放後的寬度
    :param height: 縮放後的高度
    :param output_path: 如果提供，將縮放後的圖片保存到此路徑
    :return: 縮放後的圖片數據（numpy.ndarray）
    """
    # 讀取圖片
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"無法讀取圖片，路徑可能無效: {image_path}")


    # 去除邊緣
    image_no_edges = remove_image_edges(image, edge_width=6)

    # 調整尺寸
    resized_image = cv2.resize(image_no_edges, (width, height), interpolation=cv2.INTER_AREA)

    # 如果指定了輸出路徑，保存圖片
    if output_path:
        cv2.imwrite(output_path, resized_image)

    return resized_image


def create_black_image(width, height, output_path=None):
    """
    創建一個指定寬度和高度的黑色圖片。

    :param width: 圖片的寬度（像素）
    :param height: 圖片的高度（像素）
    :param output_path: 如果提供，將創建的圖片保存到此路徑
    :return: 創建的黑色圖片數據（numpy.ndarray）
    """
    # 創建黑色圖片 (高度, 寬度, 通道數)
    black_image = np.zeros((height, width, 3), dtype=np.uint8)

    # 如果指定了輸出路徑，保存圖片
    if output_path:
        cv2.imwrite(output_path, black_image)

    return black_image


def place_images_on_black_canvas(black_canvas, resized_images, interval=10):
    """
    將多張縮放後的圖片依次放置到黑色背景中，每張圖片之間間隔一定像素。

    :param black_canvas: 黑色背景（numpy.ndarray）
    :param resized_images: 縮放後的圖片列表
    :param interval: 圖片之間的間隔像素
    :return: 放置後的黑色背景
    """
    x, y = interval, interval  # 初始放置位置

    for img in resized_images:
        h, w = img.shape[:2]
        canvas_h, canvas_w = black_canvas.shape[:2]

        if y + h > canvas_h:  # 如果超出黑色背景的高度，換行放置
            y = interval
            x += w + interval

        if x + w > canvas_w:  # 如果超出黑色背景的寬度，報錯
            raise ValueError("圖片總寬度超出黑色背景，請增加背景寬度或減少圖片數量")

        # 將圖片放置到黑色背景的指定位置
        black_canvas[y:y+h, x:x+w] = img
        y += h + interval  # 更新下一張圖片的 y 坐標

    return black_canvas




def process_card_images(image_paths, card_image_width, card_image_height, black_canvas_width, black_canvas_height, interval=10, folder_path=None):
    """
    處理多張卡片圖片，將縮放後的圖片依次放置到黑色背景，檢測卡牌並標註結果。

    :param image_paths: 輸入圖片的路徑列表
    :param card_image_width: 卡片圖片縮放後的寬度
    :param card_image_height: 卡片圖片縮放後的高度
    :param black_canvas_width: 黑色背景的寬度
    :param black_canvas_height: 黑色背景的高度
    :param interval: 圖片之間的間隔像素
    :param folder_path: 當前資料夾的路徑
    :return: 標註後的黑色背景圖片
    """
    global train_ranks, train_suits
    path = os.path.dirname(os.path.abspath(__file__))

    if train_ranks is None or train_suits is None:
        train_ranks = Cards.load_ranks(path + '/Card_Imgs/')
        train_suits = Cards.load_suits(path + '/Card_Imgs/')

    # 縮放所有圖片
    resized_images = [resize_image(folder_path + '/'+image_path, card_image_width, card_image_height) for image_path in image_paths]



    # 創建黑色背景
    black_img = create_black_image(black_canvas_width, black_canvas_height)

    # 將圖片依次放置到黑色背景
    image = place_images_on_black_canvas(black_img, resized_images, interval=interval)

    # 預處理圖片
    pre_proc = Cards.preprocess_image(image)

    # 找到所有卡片的輪廓
    cnts_sort, cnt_is_card = Cards.find_cards(pre_proc)

    # 存儲檢測到的卡片信息
    detected_cards = []

    # 如果有卡片輪廓，開始處理
    if len(cnts_sort) != 0:
        cards = []
        k = 0
        for i in range(len(cnts_sort)):
            if cnt_is_card[i] == 1:
                cards.append(Cards.preprocess_card(cnts_sort[i], image))
                (cards[k].best_rank_match,
                 cards[k].best_suit_match,
                 cards[k].rank_diff,
                 cards[k].suit_diff) = Cards.match_card(cards[k], train_ranks, train_suits)
                image = Cards.draw_results(image, cards[k])
                # 保存 Rank 和 Suit 到檢測結果
                detected_cards.append({
                    "rank": cards[k].best_rank_match,
                    "suit": cards[k].best_suit_match
                })
                k += 1

        # 畫出卡片輪廓
        if len(cards) != 0:
            temp_cnts = [card.contour for card in cards]
            cv2.drawContours(image, temp_cnts, -1, (255, 0, 0), 2)

    if folder_path is not None:
        # 保存結果圖片
        cv2.imwrite(os.path.join(folder_path, 'newResultPic.jpg'), image)
        # 保存檢測到的卡牌信息為 JSON 文件
        report_path = os.path.join(folder_path, 'newReport.txt')
        with open(report_path, 'w') as report_file:
            json.dump(detected_cards, report_file, indent=4)

    # # 保存結果圖片
    # cv2.imwrite('folder_path.result.jpg', image)

    return image


def find_and_process_images(root_dir, target_prefixes, card_image_width, card_image_height, black_canvas_width, black_canvas_height, interval=10):
    """
    遍歷所有資料夾，找到以指定前綴開頭的照片，並處理所在資料夾中的所有照片。

    :param root_dir: 根目錄，將從這裡開始遍歷所有資料夾
    :param target_prefixes: 目標照片的前綴列表（如 ['banker', 'player']）
    :param card_image_width: 卡片圖片縮放後的寬度
    :param card_image_height: 卡片圖片縮放後的高度
    :param black_canvas_width: 黑色背景的寬度
    :param black_canvas_height: 黑色背景的高度
    :param interval: 圖片之間的間隔像素
    :return: 處理過的圖片結果列表
    """
    processed_results = []

    for root, dirs, files in os.walk(root_dir):

        # 找到符合條件的照片
        target_files = [
            file for file in files
            if any(file.lower().startswith(prefix.lower()) for prefix in target_prefixes)
        ]

        if target_files:  # 如果有符合條件的照片
            # 構建該資料夾中所有照片的完整路徑
            # all_image_paths = [os.path.join(root, file) for file in files]
            print(target_files)
            # 呼叫處理方法處理所有照片
            result_image = process_card_images(
                image_paths=target_files,
                card_image_width=card_image_width,
                card_image_height=card_image_height,
                black_canvas_width=black_canvas_width,
                black_canvas_height=black_canvas_height,
                interval=interval,
                folder_path=root  # 傳遞資料夾路徑
            )
            processed_results.append((root, result_image))  # 保存結果

    return processed_results

# 測試方法
root_directory = 'C:/Users/LeoAlliance/Desktop/analysis1/fail/2024-11-20_09-58-05'  # 替換為你的根目錄路徑
result_images = find_and_process_images(
    root_dir=root_directory,               # 根目錄
    target_prefixes=['banker', 'player'],  # 符合條件的檔案前綴
    card_image_width=200,                  # 卡片寬度
    card_image_height=300,                 # 卡片高度
    black_canvas_width=1280,               # 黑色背景寬度
    black_canvas_height=720,               # 黑色背景高度
    interval=40                            # 圖片間隔
)



# # 測試方法
# image_paths = ['player2.png']  # 替換為多張圖片的路徑
# result_image = process_card_images(
#     image_paths=image_paths,                  # 輸入圖片路徑列表
#     card_image_width=200,                    # 卡片寬度
#     card_image_height=300,                   # 卡片高度
#     black_canvas_width=1280,                 # 黑色背景寬度
#     black_canvas_height=720,                 # 黑色背景高度
#     interval=10,                              # 圖片間隔
#     folder_path="img/"
# )

