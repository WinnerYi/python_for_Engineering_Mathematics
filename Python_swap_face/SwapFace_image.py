import cv2
import insightface
from insightface.app import FaceAnalysis

# 使用輕量模型 buffalo_s（比較好下載，也比較快）
app = FaceAnalysis(name='buffalo_s')
app.prepare(ctx_id=0, det_size=(640, 640))

# 載入 inswapper 模型（臉部替換）
swapper = insightface.model_zoo.get_model('./inswapper_128.onnx', download=False)

# 來源與目標圖片檔名（依照你的檔案名稱修改）
source_img_path = "source.jpg"   # 來源人臉
target_img_path = "target.jpg"   # 想被換臉的圖片
output_img_path = "output.jpg"   # 儲存結果

# 讀取來源圖片
source_img = cv2.imread(source_img_path)
source_faces = app.get(source_img)
if not source_faces:
    raise ValueError("來源圖片中偵測不到人臉")
source_face = source_faces[0]

# 讀取目標圖片
target_img = cv2.imread(target_img_path)
target_faces = app.get(target_img)
if not target_faces:
    raise ValueError("目標圖片中偵測不到人臉")

# 對每一張偵測到的人臉進行替換
for face in target_faces:
    target_img = swapper.get(target_img, face, source_face, paste_back=True)

# 儲存結果
cv2.imwrite(output_img_path, target_img)
print(f"已完成換臉，儲存於 {output_img_path}")

