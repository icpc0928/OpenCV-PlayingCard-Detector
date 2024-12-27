############## Playing Card Detector Functions ###############
#
# Author: Evan Juras
# Date: 9/5/17
# Description: Functions and classes for CardDetector.py that perform 
# various steps of the card detection algorithm


# Import necessary packages
import numpy as np
import cv2
import time


### Constants ###

# Adaptive threshold levels
BKG_THRESH = 60
CARD_THRESH = 30

# Width and height of card corner, where rank and suit are
# CORNER_WIDTH = 36
CORNER_WIDTH = 40
CORNER_HEIGHT = 84

# Dimensions of rank train images
RANK_WIDTH = 70
RANK_HEIGHT = 125

# Dimensions of suit train images
SUIT_WIDTH = 70
SUIT_HEIGHT = 100

RANK_DIFF_MAX = 2000
SUIT_DIFF_MAX = 700

CARD_MAX_AREA = 120000
CARD_MIN_AREA = 25000

# 輪廓最小判斷面積
MIN_AREA_THRESHOLD = 2000

font = cv2.FONT_HERSHEY_SIMPLEX

### Structures to hold query card and train card information ###

class Query_card:
    """Structure to store information about query cards in the camera image."""

    def __init__(self):
        self.name = "Unknown"
        self.contour = [] # Contour of card
        self.width, self.height = 0, 0 # Width and height of card
        self.corner_pts = [] # Corner points of card
        self.center = [] # Center point of card
        self.warp = [] # 200x300, flattened, grayed, blurred image
        self.rank_img = [] # Thresholded, sized image of card's rank            //改成多個圖片
        self.best_rank_index = 0 #這個是最好的rank圖片 只會有一張
        self.suit_img = [] # Thresholded, sized image of card's suit            //改成多個圖片
        self.best_suit_index = 0 #這個是最好的suit圖片 只會有一張
        self.best_rank_match = "Unknown" # Best matched rank
        self.best_suit_match = "Unknown" # Best matched suit
        self.rank_diff = 0 # Difference between rank image and best matched train rank image
        self.suit_diff = 0 # Difference between suit image and best matched train suit image
        self.result_img = [] # Drawing of best match

class Train_ranks:
    """Structure to store information about train rank images."""

    def __init__(self):
        self.img = [] # Thresholded, sized rank image loaded from hard drive
        self.name = "Placeholder"

class Train_suits:
    """Structure to store information about train suit images."""

    def __init__(self):
        self.img = [] # Thresholded, sized suit image loaded from hard drive
        self.name = "Placeholder"

### Functions ###
def load_ranks(filepath):
    """Loads rank images from directory specified by filepath. Stores
    them in a list of Train_ranks objects."""

    train_ranks = []
    i = 0
    
    for Rank in ['Ace','Two','Three','Four','Five','Six','Seven',
                 'Eight','Nine','Ten','Jack','Queen','King']:
        train_ranks.append(Train_ranks())
        train_ranks[i].name = Rank
        filename = Rank + '.jpg'
        train_ranks[i].img = cv2.imread(filepath+filename, cv2.IMREAD_GRAYSCALE)
        i = i + 1

    return train_ranks

def load_suits(filepath):
    """Loads suit images from directory specified by filepath. Stores
    them in a list of Train_suits objects."""

    train_suits = []
    i = 0
    
    for Suit in ['Spades','Diamonds','Clubs','Hearts']:
        train_suits.append(Train_suits())
        train_suits[i].name = Suit
        filename = Suit + '.jpg'
        train_suits[i].img = cv2.imread(filepath+filename, cv2.IMREAD_GRAYSCALE)
        i = i + 1

    return train_suits

def preprocess_image(image):
    """Returns a grayed, blurred, and adaptively thresholded camera image."""

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)

    # The best threshold level depends on the ambient lighting conditions.
    # For bright lighting, a high threshold must be used to isolate the cards
    # from the background. For dim lighting, a low threshold must be used.
    # To make the card detector independent of lighting conditions, the
    # following adaptive threshold method is used.
    #
    # A background pixel in the center top of the image is sampled to determine
    # its intensity. The adaptive threshold is set at 50 (THRESH_ADDER) higher
    # than that. This allows the threshold to adapt to the lighting conditions.
    img_w, img_h = np.shape(image)[:2]
    bkg_level = gray[int(img_h/100)][int(img_w/2)]
    thresh_level = bkg_level + BKG_THRESH


    retval, thresh = cv2.threshold(blur,thresh_level,255,cv2.THRESH_BINARY)
    # retval, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
    # cv2.imwrite('t1.jpg', image)
    # cv2.imwrite('thresh2.jpg', thresh)
    return thresh

def find_cards(thresh_image):
    """Finds all card-sized contours in a thresholded camera image.
    Returns the number of cards, and a list of card contours sorted
    from largest to smallest.
    找出卡片輪廓
    """

    # Find contours and sort their indices by contour size
    # dummy,cnts,hier = cv2.findContours(thresh_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnts, hier = cv2.findContours(thresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    index_sort = sorted(range(len(cnts)), key=lambda i : cv2.contourArea(cnts[i]),reverse=True)

    # If there are no contours, do nothing
    if len(cnts) == 0:
        return [], []
    
    # Otherwise, initialize empty sorted contour and hierarchy lists
    cnts_sort = []
    hier_sort = []
    cnt_is_card = np.zeros(len(cnts),dtype=int)

    # Fill empty lists with sorted contour and sorted hierarchy. Now,
    # the indices of the contour list still correspond with those of
    # the hierarchy list. The hierarchy array can be used to check if
    # the contours have parents or not.
    for i in index_sort:
        cnts_sort.append(cnts[i])
        hier_sort.append(hier[0][i])

    # Determine which of the contours are cards by applying the
    # following criteria:
    # 1) Smaller area than the maximum card size,
    # 2), bigger area than the minimum card size,
    # 3) have no parents,
    # and 4) have four corners

    for i in range(len(cnts_sort)):
        size = cv2.contourArea(cnts_sort[i])
        peri = cv2.arcLength(cnts_sort[i],True)
        approx = cv2.approxPolyDP(cnts_sort[i],0.01*peri,True)

        
        if ((size < CARD_MAX_AREA) and (size > CARD_MIN_AREA)
            and (hier_sort[i][3] == -1) and (len(approx) == 4)):
            cnt_is_card[i] = 1

    return cnts_sort, cnt_is_card


def get_best_cut_line(query_thresh):
    temp_y = 185
    t_range = 65
    top_bottom_threshold = 40
    width_threshold = 10  # 左右邊界判斷的閾值
    default_cut_line = temp_y  # 預設值

    # 複製 query_thresh 並畫輔助線
    temp_Q = query_thresh.copy()

    # cv2.line(temp_Q, (0, temp_y+t_range), (temp_Q.shape[1], temp_y+t_range), (255,255,255), 2)
    # cv2.line(temp_Q, (0, temp_y), (temp_Q.shape[1], temp_y), (255,255,255), 2)
    # cv2.line(temp_Q, (0, temp_y-t_range), (temp_Q.shape[1], temp_y-t_range), (255,255,255), 2)
    # 找輪廓和層次結構
    temp_query_cnts, hierarchy = cv2.findContours(query_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 將外部輪廓與內部輪廓分開
    external_contours = []
    internal_contours = []
    for idx, cnt in enumerate(temp_query_cnts):
        _, _, _, parent = hierarchy[0][idx]
        if parent == -1:  # 沒有父輪廓，是外部輪廓
            external_contours.append((cnt, idx))
        else:  # 有父輪廓，是內部輪廓
            internal_contours.append((cnt, idx))

    # 對外部輪廓按面積從大到小排序
    external_contours = sorted(external_contours, key=lambda x: cv2.contourArea(x[0]), reverse=True)

    # 儲存結果
    points_within_range = []  # 儲存點數與花色的座標
    left_right_pairs = []    # 儲存所有輪廓的 Left 和 Right
    # 影像寬高
    img_height, img_width = temp_Q.shape

    # 處理外部輪廓
    for cnt, idx in external_contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA_THRESHOLD:  # 忽略小面積輪廓
            continue

        # 計算四個極值點
        points = cnt[:, 0, :]  # 獲取輪廓點的 x 和 y 值
        top = tuple(points[points[:, 1].argmin()])    # y 最小
        bottom = tuple(points[points[:, 1].argmax()]) # y 最大
        left = tuple(points[points[:, 0].argmin()])   # x 最小
        right = tuple(points[points[:, 0].argmax()])  # x 最大

        # 忽略 top 在 0~50 且 bottom 在影像底部最後 50 像素的輪廓
        if 0 <= top[1] <= top_bottom_threshold and (img_height - top_bottom_threshold) <= bottom[1] <= img_height:
            continue
        # 忽略寬度覆蓋整個左邊和右邊的輪廓
        if left[0] <= width_threshold and (img_width - width_threshold) <= right[0]:
            continue
        if  bottom[1] - top[1] > 180:  # 忽略高度大於 185 的輪廓
            continue

        # 檢查點是否在 temp_y +- t_range 中
        if temp_y - t_range <= bottom[1] <= temp_y + t_range and top[1] <= temp_y - t_range:
            points_within_range.append(('bottom', bottom[1]))

        if temp_y - t_range <= top[1] <= temp_y + t_range and bottom[1] >= temp_y + t_range:
            points_within_range.append(('top', top[1]))


        # 保存 Left 和 Right
        left_right_pairs.append((left, right))

        # 在影像上標註這些點（可視化輔助）
        cv2.circle(temp_Q, top, 3, (82, 157, 255), -1)
        cv2.circle(temp_Q, bottom, 3, (82, 157, 255), -1)
        cv2.circle(temp_Q, left, 3, (82, 157, 255), -1)
        cv2.circle(temp_Q, right, 3, (82, 157, 255), -1)

    # 第一判斷：檢查是否有點數與花色
    if len(points_within_range) == 2:
        # 找到點數的底部與花色的頭部，取中間值
        bottom_y = [p[1] for p in points_within_range if p[0] == 'bottom']
        top_y = [p[1] for p in points_within_range if p[0] == 'top']



        if bottom_y and top_y:
            result1 = (min(bottom_y) + max(top_y)) // 2
            # print(f"result1: {result1}")
            # # 顯示結果影像
            # cv2.line(temp_Q, (0, result1), (temp_Q.shape[1], result1), (82, 157, 255), 1)
            # cv2.imshow('Contours with Extreme Points', temp_Q)
            # cv2.waitKey(0)
            return result1

    # 第二判斷：檢查斜率相近的 Left 和 Right
    if len(left_right_pairs) >= 2:
        for i in range(len(left_right_pairs)):
            for j in range(i + 1, len(left_right_pairs)):
                # 計算兩對點的斜率
                left1, right1 = left_right_pairs[i]
                left2, right2 = left_right_pairs[j]

                slope1 = (right1[1] - left1[1]) / (right1[0] - left1[0] + 1e-6)
                slope2 = (right2[1] - left2[1]) / (right2[0] - left2[0] + 1e-6)
                if abs(slope1 - slope2) < 0.1:  # 斜率接近閾值
                    top_pair = left_right_pairs[i] if left1[1] < left2[1] else left_right_pairs[j]
                    bottom_pair = left_right_pairs[j] if left1[1] < left2[1] else left_right_pairs[i]
                    result2 = (top_pair[0][1] + bottom_pair[1][1]) // 2
                    # print(f"result2: {result2}")
                    # cv2.line(temp_Q, (0, result2), (temp_Q.shape[1], result2), (82, 157, 255), 1)
                    # cv2.imshow('Contours with Extreme Points', temp_Q)
                    # cv2.waitKey(0)
                    return result2
    # print("default")

    # cv2.line(temp_Q, (0, default_cut_line), (temp_Q.shape[1], default_cut_line), (82, 157, 255), 1)
    # cv2.imshow('Contours with Extreme Points', temp_Q)
    # cv2.waitKey(0)
    # 如果都無法判斷，返回預設值
    return default_cut_line



def preprocess_card(contour, image):
    """Uses contour to find information about the query card. Isolates rank
    and suit images from the card.
    """

    qCard = Query_card()

    qCard.contour = contour

    # Find perimeter of card and use it to approximate corner points 獲取卡片的四個角點
    peri = cv2.arcLength(contour,True) # 計算輪廓的周長
    approx = cv2.approxPolyDP(contour, 0.01 * peri,True) # 近似多邊形
    pts = np.float32(approx) # 將結果轉為浮點型數據
    qCard.corner_pts = pts

    # Find width and height of card's bounding rectangle
    x,y,w,h = cv2.boundingRect(contour) # 計算邊界框

    qCard.width, qCard.height = w, h

    # Find center point of card by taking x and y average of the four corners.  計算卡片的邊界框和中心點
    average = np.sum(pts, axis=0)/len(pts) # 計算四個角點的平均值
    cent_x = int(average[0][0])
    cent_y = int(average[0][1])
    qCard.center = [cent_x, cent_y]

    # Warp card into 200x300 flattened image using perspective transform  將0
    qCard.warp = flattener(image, pts, w, h)
    # cv2.imshow('warp', qCard.warp)
    # key = cv2.waitKey(0)


    # Grab corner of warped card image and do a 4x zoom  提取卡片左上角
    Qcorner = qCard.warp[0:CORNER_HEIGHT, 0:CORNER_WIDTH]  # 功能: 從變換後的卡片圖像中提取左上角，並進行 4 倍縮放。
    Qcorner_zoom = cv2.resize(Qcorner, (0,0), fx=4, fy=4)
    # cv2.imshow('Qcorner_zoom', Qcorner_zoom)
    # key = cv2.waitKey(0)

    # Sample known white pixel intensity to determine good threshold level  設置二值化閾值
    white_level = Qcorner_zoom[15,int((CORNER_WIDTH*4)/2)]  # 採樣白色強度
    thresh_level = white_level - CARD_THRESH
    if (thresh_level <= 0):
        thresh_level = 1
    retval, query_thresh = cv2.threshold(Qcorner_zoom, thresh_level, 255, cv2. THRESH_BINARY_INV)

    temp_y = get_best_cut_line(query_thresh)


    # Split in to top and bottom half (top shows rank, bottom shows suit)  分離點數和花色部分
    Qrank = query_thresh[10:temp_y, 0:CORNER_WIDTH*4]  # 提取點數部分
    Qsuit = query_thresh[temp_y+1:CORNER_HEIGHT*4, 0:CORNER_WIDTH*4]  # 提取花色部分
    # cv2.imshow('Qrank.jpg', Qrank)
    # cv2.imshow('Qsuit.jpg', Qsuit)
    # key = cv2.waitKey(0)


    # Find rank contour and bounding rectangle, isolate and find largest contour  點數部分的輪廓提取和處理
    Qrank_cnts, hier = cv2.findContours(Qrank, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    Qrank_cnts = sorted(Qrank_cnts, key=cv2.contourArea,reverse=True)

    # Find bounding rectangle for largest contour, use it to resize query rank
    # image to match dimensions of the train rank image
    # cv2.imshow('Qrank', Qrank)
    if len(Qrank_cnts) != 0:
        for ri in range(len(Qrank_cnts)):
            area = cv2.contourArea(Qrank_cnts[ri])
            if area < MIN_AREA_THRESHOLD:   # 面積太小的輪廓不處理
                continue
            # 不應該只算一個輪廓 搞不好有多個
            x1,y1,w1,h1 = cv2.boundingRect(Qrank_cnts[ri])
            # print("面積: ", cv2.contourArea(Qrank_cnts[ri]))


            Qrank_roi = Qrank[y1:y1+h1, x1:x1+w1]
            Qrank_sized = cv2.resize(Qrank_roi, (RANK_WIDTH,RANK_HEIGHT), 0, 0)
            # qCard.rank_img = Qrank_sized
            qCard.rank_img.append(Qrank_sized)
            # cv2.imshow('Qrank_sized', Qrank_sized)
            # cv2.waitKey(0)


    # 花色部分的輪廓提取和處理
    Qsuit_cnts, hier = cv2.findContours(Qsuit, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    Qsuit_cnts = sorted(Qsuit_cnts, key=cv2.contourArea,reverse=True)

    # Find bounding rectangle for largest contour, use it to resize query suit
    # image to match dimensions of the train suit image
    if len(Qsuit_cnts) != 0:
        for si in range(len(Qsuit_cnts)):
            area = cv2.contourArea(Qsuit_cnts[si])
            if area < MIN_AREA_THRESHOLD:   # 面積太小的輪廓不處理
                continue

            x2,y2,w2,h2 = cv2.boundingRect(Qsuit_cnts[0])
            Qsuit_roi = Qsuit[y2:y2+h2, x2:x2+w2]
            Qsuit_sized = cv2.resize(Qsuit_roi, (SUIT_WIDTH, SUIT_HEIGHT), 0, 0)
            qCard.suit_img.append(Qsuit_sized)
            # cv2.imshow('Qsuit_sized', Qsuit_sized)
            # print("花色面積: ", area)
            # cv2.waitKey(0)
    return qCard

def match_card(qCard, train_ranks, train_suits):
    """Finds best rank and suit matches for the query card. Differences
    the query card rank and suit images with the train rank and suit images.
    The best match is the rank or suit image that has the least difference."""

    best_rank_match_diff = 10000
    best_suit_match_diff = 10000
    best_rank_match_name = "Unknown"
    best_suit_match_name = "Unknown"
    i = 0

    # If no contours were found in query card in preprocess_card function,
    # the img size is zero, so skip the differencing process
    # (card will be left as Unknown)
    if (len(qCard.rank_img) != 0) and (len(qCard.suit_img) != 0):

        for ri in range(len(qCard.rank_img)):
            # 判斷點數
            for Trank in train_ranks:
                    # print("------")
                    # 這裡的rank_img可能是多組輪廓
                    # 計算兩張影像對應像素的絕對差異
                    diff_img = cv2.absdiff(qCard.rank_img[ri], Trank.img)
                    rank_diff = int(np.sum(diff_img)/255)
                    # print(f"ri:{ri}, 比較:{Trank.name} rank_diff: {rank_diff}" )
                    # cv2.imshow('diff_img', diff_img)
                    # cv2.waitKey(0)
                    if rank_diff < best_rank_match_diff:
                        best_rank_diff_img = diff_img
                        best_rank_match_diff = rank_diff
                        best_rank_name = Trank.name
                        qCard.best_rank_index = ri

        for si in range(len(qCard.suit_img)):
            # Same process with suit images
            for Tsuit in train_suits:
                    # print("=====")
                    diff_img = cv2.absdiff(qCard.suit_img[si], Tsuit.img)
                    suit_diff = int(np.sum(diff_img)/255)
                    # print(f"ri:{ri}, 比較:{Trank.name} rank_diff: {suit_diff}" )
                    # cv2.imshow('diff_img', diff_img)
                    # cv2.waitKey(0)
                    if suit_diff < best_suit_match_diff:
                        best_suit_diff_img = diff_img
                        best_suit_match_diff = suit_diff
                        best_suit_name = Tsuit.name
                        qCard.best_suit_index = si

    # Combine best rank match and best suit match to get query card's identity.
    # If the best matches have too high of a difference value, card identity
    # is still Unknown
    if (best_rank_match_diff < RANK_DIFF_MAX):
        best_rank_match_name = best_rank_name

    if (best_suit_match_diff < SUIT_DIFF_MAX):
        best_suit_match_name = best_suit_name

    # Return the identiy of the card and the quality of the suit and rank match
    return best_rank_match_name, best_suit_match_name, best_rank_match_diff, best_suit_match_diff
    
    
def draw_results(image, qCard):
    """Draw the card name, center point, and contour on the camera image."""

    x = qCard.center[0]
    y = qCard.center[1]
    cv2.circle(image,(x,y),5,(255,0,0),-1)

    rank_name = qCard.best_rank_match
    suit_name = qCard.best_suit_match

    # Draw card name twice, so letters have black outline
    cv2.putText(image,(rank_name+' of'),(x-60,y-10),font,1,(0,0,0),3,cv2.LINE_AA)
    cv2.putText(image,(rank_name+' of'),(x-60,y-10),font,1,(50,200,200),2,cv2.LINE_AA)

    cv2.putText(image,suit_name,(x-60,y+25),font,1,(0,0,0),3,cv2.LINE_AA)
    cv2.putText(image,suit_name,(x-60,y+25),font,1,(50,200,200),2,cv2.LINE_AA)


    # Can draw difference value for troubleshooting purposes
    # (commented out during normal operation)
    #r_diff = str(qCard.rank_diff)
    #s_diff = str(qCard.suit_diff)
    #cv2.putText(image,r_diff,(x+20,y+30),font,0.5,(0,0,255),1,cv2.LINE_AA)
    #cv2.putText(image,s_diff,(x+20,y+50),font,0.5,(0,0,255),1,cv2.LINE_AA)

    return image

def flattener(image, pts, w, h):
    """Flattens an image of a card into a top-down 200x300 perspective.
    Returns the flattened, re-sized, grayed image.
    See www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/"""
    temp_rect = np.zeros((4,2), dtype = "float32")
    
    s = np.sum(pts, axis = 2)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]

    diff = np.diff(pts, axis = -1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    # Need to create an array listing points in order of
    # [top left, top right, bottom right, bottom left]
    # before doing the perspective transform

    if w <= 0.8*h: # If card is vertically oriented
        temp_rect[0] = tl
        temp_rect[1] = tr
        temp_rect[2] = br
        temp_rect[3] = bl

    if w >= 1.2*h: # If card is horizontally oriented
        temp_rect[0] = bl
        temp_rect[1] = tl
        temp_rect[2] = tr
        temp_rect[3] = br

    # If the card is 'diamond' oriented, a different algorithm
    # has to be used to identify which point is top left, top right
    # bottom left, and bottom right.
    
    if w > 0.8*h and w < 1.2*h: #If card is diamond oriented
        # If furthest left point is higher than furthest right point,
        # card is tilted to the left.
        if pts[1][0][1] <= pts[3][0][1]:
            # If card is titled to the left, approxPolyDP returns points
            # in this order: top right, top left, bottom left, bottom right
            temp_rect[0] = pts[1][0] # Top left
            temp_rect[1] = pts[0][0] # Top right
            temp_rect[2] = pts[3][0] # Bottom right
            temp_rect[3] = pts[2][0] # Bottom left

        # If furthest left point is lower than furthest right point,
        # card is tilted to the right
        if pts[1][0][1] > pts[3][0][1]:
            # If card is titled to the right, approxPolyDP returns points
            # in this order: top left, bottom left, bottom right, top right
            temp_rect[0] = pts[0][0] # Top left
            temp_rect[1] = pts[3][0] # Top right
            temp_rect[2] = pts[2][0] # Bottom right
            temp_rect[3] = pts[1][0] # Bottom left
            
        
    maxWidth = 200
    maxHeight = 300

    # Create destination array, calculate perspective transform matrix,
    # and warp card image
    dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0, maxHeight-1]], np.float32)
    M = cv2.getPerspectiveTransform(temp_rect,dst)
    warp = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    warp = cv2.cvtColor(warp,cv2.COLOR_BGR2GRAY)

        

    return warp
