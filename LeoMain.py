import os

import cv2
import json

import numpy as np

import Cards
import Leo001

def merge_cards(original_card, new_card):
    """
        合併兩個 Query_card 對象，保留非預設值的屬性。

        :param original_card: 原始的 Query_card 對象
        :param new_card: 新的 Query_card 對象
        :return: 合併後的 Query_card 對象
        """
    if new_card is None: return original_card

    # 遍歷 Query_card 的所有屬性
    for attr in vars(original_card).keys():
        original_value = getattr(original_card, attr)
        new_value = getattr(new_card, attr)

        # 判斷新值是否是預設值
        if is_default_value(new_value):
            # 如果是預設值，保留原值
            setattr(new_card, attr, original_value)

    return new_card
def is_default_value(value):
    """
    判斷屬性是否是預設值。

    :param value: 屬性值
    :return: 如果是預設值，返回 True；否則返回 False
    """
    if isinstance(value, str) and value == "Unknown":
        return True
        # 空列表
    if isinstance(value, list) and not value:
        return True
    if isinstance(value, (int, float)) and value == 0:  # 數字型預設值
        return True
    # NumPy array 預設值檢查
    if isinstance(value, np.ndarray) and value.size == 0:
        return True
    return False


def write_result(root, processed_results):

    str_path = "myhahapoint"
    str_file = []
    # 構建目標資料夾的完整路徑
    path = os.path.join(root, str_path)

    # 如果資料夾不存在，則創建它
    if not os.path.exists(path):
        os.makedirs(path)


    for card in processed_results:
        name = card.name
        res_name = card.name
        if len(name)>=6:
            res_name = name[0]+name[-5]

        new_name = res_name +"_new.jpg"
        img = card.result_img
        rank = card.best_rank_match
        suit = card.best_suit_match

        # 保存結果圖片
        path = os.path.join(root, str_path)
        cv2.imwrite(os.path.join(path, str(new_name)), img)

        rk_img = card.rank_img
        su_img = card.suit_img
        cv2.imwrite(os.path.join(path, str(res_name + 'rank.jpg')), rk_img)
        cv2.imwrite(os.path.join(path, str(res_name + 'suit .jpg')), su_img)

        str_file.append(
            {
                "name": name,
                "rank": rank,
                "suit": suit
            }
        )

    # 保存檢測到的卡牌信息為 JSON 文件
    report_path = os.path.join(root, 'newReport.txt')

    with open(report_path, 'w') as report_file:
        json.dump(str_file, report_file, indent=4)



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


    for root, dirs, files in os.walk(root_dir):

        # 如果資料夾開頭名稱有"myhahapoint" 跳過
        if root.find("myhahapoint") != -1:
            continue

        # 找到符合條件的照片
        target_files = [
            file for file in files
            if any(file.lower().startswith(prefix.lower()) for prefix in target_prefixes)
        ]

        if target_files:  # 如果有符合條件的照片
            # 結果圖片
            processed_results = []
            # 呼叫處理方法處理所有照片
            for target_file in target_files:
                edge = 0
                max_attempts = 50

                origin_card = Cards.Query_card()
                origin_card.name = target_file
                while edge < max_attempts:
                    edge_w = edge // 10
                    edge_h = edge % 10
                    card = Leo001.rec_card(root + '/' + target_file, target_file, edge_w, edge_h)
                    origin_card = merge_cards(origin_card, card)
                    if is_card_is_match(origin_card):
                        break
                    card = Leo001.rec_card(root + '/' + target_file, target_file, edge_w, edge_h, is_rotate=True)
                    origin_card = merge_cards(origin_card, card)
                    if is_card_is_match(origin_card):
                        break
                    card = Leo001.rec_card_with_black_canvas(root + '/' + target_file, target_file, edge_w, edge_h)
                    if card is not None:
                        origin_card = merge_cards(origin_card, card)
                        if is_card_is_match(origin_card):
                            break
                    card = Leo001.rec_card_with_black_canvas(root + '/' + target_file, target_file, edge_w, edge_h, is_rotate = True)
                    if card is not None:
                        origin_card = merge_cards(origin_card, card)
                        if is_card_is_match(origin_card):
                            break
                    edge += 1

                if edge == max_attempts:
                    print(f"Failed to recognize card: {target_file} after {max_attempts} attempts., root: {root}")
                # 嘗試旋轉0
                rotate_degree = 1
                while rotate_degree <= 10:
                    card = Leo001.rec_card(root + '/' + target_file, target_file, 0, 0, is_rotate=True, rotate_degree=rotate_degree)
                    origin_card = merge_cards(origin_card, card)
                    if is_card_is_match(origin_card):
                        break
                    rotate_degree += 1


                if origin_card is not None:
                    processed_results.append(origin_card)
            write_result(root, processed_results)
            # processed_results.append((root, result_image))  # 保存結果
    # return processed_results

def is_card_is_match(origin_card):
    if origin_card.best_rank_match != "Unknown" and origin_card.best_suit_match != "Unknown":
        return True
    return False

# 測試方法
root_directory = 'C:/Users/LeoAlliance/Desktop/analysis/analysis/2024-11-21_18-00-55'  # 替換為你的根目錄路徑
result_images = find_and_process_images(
    root_dir=root_directory,               # 根目錄
    target_prefixes=['banker', 'player'],  # 符合條件的檔案前綴
    card_image_width=200,                  # 卡片寬度
    card_image_height=300,                 # 卡片高度
    black_canvas_width=1280,               # 黑色背景寬度
    black_canvas_height=720,               # 黑色背景高度
    interval=40                            # 圖片間隔
)