import os
import time

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
        if is_default_value_without_diff_and_match(new_value, attr):
            # 如果是預設值，保留原值
            setattr(new_card, attr, original_value)
    # set match and diff

    if new_card.rank_diff > original_card.rank_diff != 0:
        setattr(new_card, "rank_diff", original_card.rank_diff)
        setattr(new_card, "best_rank_match", original_card.best_rank_match)
    if new_card.suit_diff > original_card.suit_diff != 0:
        setattr(new_card, "suit_diff", original_card.suit_diff)
        setattr(new_card, "best_suit_match", original_card.best_suit_match)


    return new_card
def is_default_value_without_diff_and_match(value, key):
    """
    判斷屬性是否是預設值。

    :param value: 屬性值
    :return: 如果是預設值，返回 True；否則返回 False
    """
    # print(f"key: {key}, value: {value}")

    if key == "best_rank_match" or key == "best_suit_match" or key == "rank_diff" or key == "suit_diff":
        return False

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
        if len(card.rank_img) > 0:
            if card.best_rank_index >= len(card.rank_img) :
                card.best_rank_index = 0

            rk_img = card.rank_img[card.best_rank_index]
            cv2.imwrite(os.path.join(path, str(res_name + 'rank.jpg')), rk_img)
        if len(card.suit_img) > 0:
            if card.best_suit_index >= len(card.suit_img):
                card.best_suit_index = 0
            su_img = card.suit_img[card.best_suit_index]
            cv2.imwrite(os.path.join(path, str(res_name + 'suit.jpg')), su_img)

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


def print_origin_card(origin_card):
    print(f"Card name: {origin_card.name}")
    print(f"Best rank match: {origin_card.best_rank_match}")
    print(f"Best suit match: {origin_card.best_suit_match}")
    print(f"Rank diff: {origin_card.rank_diff}")
    print(f"Suit diff: {origin_card.suit_diff}")
    print(f"Best rank index: {origin_card.best_rank_index}")
    print(f"Best suit index: {origin_card.best_suit_index}")
    print(f"Rank images: {len(origin_card.rank_img)}")
    print(f"Suit images: {len(origin_card.suit_img)}")
    print(f"Result image shape: {origin_card.result_img.shape}")
    print("")
    pass

def rec_card_process(img_path, img_name):
    edge = 0
    max_attempts = 50
    origin_card = Cards.Query_card()
    origin_card.name = img_name
    while edge < max_attempts:
        edge_w = edge // 10
        edge_h = edge % 10
        card = Leo001.rec_card(img_path, img_name, edge_w, edge_h)
        origin_card = merge_cards(origin_card, card)
        if is_card_is_match(origin_card):
            break
        card = Leo001.rec_card(img_path, img_name, edge_w, edge_h, is_rotate=True)
        origin_card = merge_cards(origin_card, card)
        if is_card_is_match(origin_card):
            break
        card = Leo001.rec_card_with_black_canvas(img_path, img_name, edge_w, edge_h)
        if card is not None:
            origin_card = merge_cards(origin_card, card)
            if is_card_is_match(origin_card):
                break
        card = Leo001.rec_card_with_black_canvas(img_path, img_name, edge_w, edge_h, is_rotate=True)
        if card is not None:
            origin_card = merge_cards(origin_card, card)
            if is_card_is_match(origin_card):
                break
        edge += 1

    # if edge == max_attempts:
    #     print(f"Failed to recognize card: {target_file} after {max_attempts} attempts., root: {root}")

    if not is_card_is_match(origin_card):
        # 嘗試旋轉0
        rotate_degree = -1
        rotate_180 = 180
        while rotate_degree <= 10:
            # 轉1
            rot = rotate_degree
            card = Leo001.rec_card(img_path, img_name, 0, 0, is_rotate=True,
                                   rotate_degree=rot)
            origin_card = merge_cards(origin_card, card)
            if is_card_is_match(origin_card):
                break
            # 180度 轉1
            rot = rotate_180 + rotate_degree
            card = Leo001.rec_card(img_path, img_name, 0, 0, is_rotate=True,
                                   rotate_degree=rot)
            origin_card = merge_cards(origin_card, card)
            if is_card_is_match(origin_card):
                break
            # 倒轉1
            rot = -rotate_degree
            card = Leo001.rec_card(img_path, img_name, 0, 0, is_rotate=True,
                                   rotate_degree=rot)
            origin_card = merge_cards(origin_card, card)
            if is_card_is_match(origin_card):
                break
            # 180度 倒轉1
            rot = -rotate_degree - rotate_180
            card = Leo001.rec_card(img_path, img_name, 0, 0, is_rotate=True,
                                   rotate_degree=rot)
            origin_card = merge_cards(origin_card, card)
            if is_card_is_match(origin_card):
                break
            rotate_degree += 1
    return origin_card


def find_and_process_images(root_dir, target_prefixes):
    """
    遍歷所有資料夾，找到以指定前綴開頭的照片，並處理所在資料夾中的所有照片。

    :param root_dir: 根目錄，將從這裡開始遍歷所有資料夾
    :param target_prefixes: 目標照片的前綴列表（如 ['banker', 'player']）
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
                origin_card = rec_card_process(root + '/' + target_file, target_file)
                if origin_card is not None:
                    # print_origin_card(origin_card)
                    processed_results.append(origin_card)
            write_result(root, processed_results)

def is_card_is_match(origin_card):
    if origin_card.best_rank_match != "Unknown" and origin_card.best_suit_match != "Unknown":
        return True
    return False

# 測試方法1 (遍歷所有資料夾, 找到以指定前綴開頭的照片, 並處理所在資料夾中的所有照片, 最後將結果保存到指定資料夾)
# root_directory = 'C:/Users/LeoAlliance/Desktop/analysis/analysis/2024-11-20_09-47-08'  # 替換為你的根目錄路徑
# # root_directory = 'C:/Users/LeoAlliance/Desktop/fail/test'  # 替換為你的根目錄路徑
# result_images = find_and_process_images(
#     root_dir=root_directory,               # 根目錄
#     target_prefixes=['banker', 'player'],  # 符合條件的檔案前綴
# )

# 測試方法2 (處理單張圖片, 回傳處理後的 Query_card 對象)
img_path = 'C:/Users/LeoAlliance/Desktop/analysis/analysis/2024-11-20_09-47-08/player3.png'  # 圖片路徑
img_name = 'player3.png'  # 圖片名稱
card = rec_card_process(img_path, img_name)
print_origin_card(card)


