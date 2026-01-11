import cv2
import dlib
import numpy as np

# 臉部特徵點模型檔案路徑
# 確保 'shape_predictor_68_face_landmarks.dat' 檔案存在
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

def get_landmarks(im):
    """
    偵測人臉並回傳 68 個特徵點座標。
    """
    rects = detector(im, 1)
    if len(rects) == 0:
        return None, None
    
    # 假設只有一張臉，選擇第一張臉
    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()]), rects[0]

def apply_slimming(img, landmarks, slimming_factor=0.3):
    """
    優化版：僅針對下顎邊緣進行變形，並保護嘴巴區域
    """
    if landmarks is None: return img
    
    # 將 landmarks 轉換為整數座標
    pts = np.array(landmarks, dtype=np.int32)
    
    # 定義下顎線的關鍵點 (左臉 3-6 號, 右臉 10-13 號)
    left_cheek = pts[3:7]
    right_cheek = pts[10:14]
    
    # 鼻尖 (30號) 作為推移的目標方向參考點
    nose_tip = pts[30]
    
    # 建立一個與原圖一樣大的偏移地圖 (Map)
    rows, cols = img.shape[:2]
    map_x = np.tile(np.arange(cols), (rows, 1)).astype(np.float32)
    map_y = np.tile(np.arange(rows), (cols, 1)).T.astype(np.float32)
    
    # 我們只針對下顎附近的像素進行位移
    # 為了避免嘴巴糊掉，我們設定變形半徑只影響邊緣
    radius = np.linalg.norm(pts[3] - pts[13]) / 6 # 縮小影響範圍
    
    # 左臉颊
    for i in range(len(left_cheek)):
        pt = left_cheek[i]
        # 計算推移方向：往鼻尖方向推一點
        direction = nose_tip - pt
        dist_to_nose = np.linalg.norm(direction)
        move_vec = (direction / dist_to_nose) * (dist_to_nose * slimming_factor * 0.5)
        
        # 局部扭曲邏輯
        mask = np.sqrt((map_x - pt[0])**2 + (map_y - pt[1])**2) < radius
        map_x[mask] -= move_vec[0] * (1 - np.sqrt((map_x[mask] - pt[0])**2 + (map_y[mask] - pt[1])**2) / radius)
        map_y[mask] -= move_vec[1] * (1 - np.sqrt((map_x[mask] - pt[0])**2 + (map_y[mask] - pt[1])**2) / radius)

    # 右臉颊 (邏輯同上)
    for i in range(len(right_cheek)):
        pt = right_cheek[i]
        direction = nose_tip - pt
        dist_to_nose = np.linalg.norm(direction)
        move_vec = (direction / dist_to_nose) * (dist_to_nose * slimming_factor * 0.5)
        
        mask = np.sqrt((map_x - pt[0])**2 + (map_y - pt[1])**2) < radius
        map_x[mask] -= move_vec[0] * (1 - np.sqrt((map_x[mask] - pt[0])**2 + (map_y[mask] - pt[1])**2) / radius)
        map_y[mask] -= move_vec[1] * (1 - np.sqrt((map_x[mask] - pt[0])**2 + (map_y[mask] - pt[1])**2) / radius)

    # 使用 OpenCV 的 remap 進行高品質重繪，這比手動 for 迴圈快且不糊
    output = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    
    return output

def draw_landmarks(img, landmarks):
    """
    在圖片上繪製特徵點。
    """
    if landmarks is None:
        return img
    
    # 修正處：將 matrix 轉換為 list，這樣 (x, y) 才能正確拆解
    for pt in landmarks.tolist():
        x, y = pt[0], pt[1]
        cv2.circle(img, (int(x), int(y)), 2, (0, 255, 0), -1)
    return img

# --- 主要執行區塊 ---
# --- 修改後的主要執行區塊 ---
if __name__ == "__main__":
    IMAGE_PATH = 'input.jpg'
    img = cv2.imread(IMAGE_PATH)
    
    if img is None:
        print(f"錯誤: 無法讀取圖片 {IMAGE_PATH}")
    else:
        landmarks, face_rect = get_landmarks(img)
        
        if landmarks is not None:
            print("成功偵測到人臉，正在進行瘦臉處理...")
            
            # 將 matrix 轉為 numpy array 以供運算
            landmarks_points = np.array(landmarks)
            
            # 您可以調整這個參數：0.1 輕微，0.5 明顯，0.8 強力
            SLIMMING_FACTOR = 0.13
            
            # 直接進行瘦臉，但不呼叫繪製點點的函數
            final_img = apply_slimming(img, landmarks_points, SLIMMING_FACTOR)
            
            # 顯示結果
            cv2.imshow("Original", img)
            cv2.imshow("Slimmed Result (No Dots)", final_img)
            
            print("處理完成！按任意鍵關閉視窗。")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            # 儲存一張沒有點點的乾淨結果圖
            cv2.imwrite("result_clean.jpg", final_img)
        else:
            print("圖片中未偵測到人臉。")
