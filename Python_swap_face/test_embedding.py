import cv2
from insightface.app import FaceAnalysis

img = cv2.imread("source_face.jpg")
if img is None:
    raise SystemExit("❌ source_face.jpg 讀不到")

app = FaceAnalysis(name='buffalo_s')
app.prepare(ctx_id=-1, det_size=(640, 640))

faces = app.get(img)
if not faces:
    raise SystemExit("❌ 圖片裡沒有偵測到人臉")

face = faces[0]

# 嘗試計算 embedding
rec_model = app.models.get('recognition', None)
if rec_model and hasattr(rec_model, 'compute_embedding'):
    rec_model.compute_embedding(face)
else:
    print("⚠️ recognition 模型不存在或不支援 compute_embedding")

emb = getattr(face, "embedding", None) or getattr(face, "normed_embedding", None) or getattr(face, "feat", None)

print("👉 embedding = ", type(emb), emb.shape if emb is not None else None)
