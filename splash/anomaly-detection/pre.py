import cv2
import matplotlib.pyplot as plt

def img_crop2(img, top=500, bottom=4000, left=580, right=4000):
    h, w = img.shape[:2]

    # 安全な範囲に制限
    top = min(max(0, top), h)
    bottom = min(max(top + 1, bottom), h)
    left = min(max(0, left), w)
    right = min(max(left + 1, right), w)  # right > left を保証

    print(f"Cropping: top={top}, bottom={bottom}, left={left}, right={right}")
    cropped = img[top:bottom, left:right]

    if cropped.size == 0:
        raise ValueError("Cropped image is empty. Adjust cropping parameters.")

    return cropped


# テスト用画像パス
img_path = "/net/nfs3/export/dataset/morita/tlo/bestex-splashguard/20250326/OK/1.png"

# 元画像読み込み
img = cv2.imread(img_path)
assert img is not None, f"Failed to load image: {img_path}"
print(f"Original image shape: {img.shape}")

# クロップ適用
cropped = img_crop2(img)

# 表示
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
plt.title("Cropped")

plt.tight_layout()
plt.show()


