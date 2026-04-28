import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# =========================
# LOAD IMAGE
# =========================
# Ganti path sesuai file kamu
img_text = cv2.imread("Citra_A.png", cv2.IMREAD_GRAYSCALE)
img_obj = cv2.imread("Citra_B.png")

# Resize biar konsisten
img_text = cv2.resize(img_text, (600, 400))
img_obj = cv2.resize(img_obj, (600, 400))

# =========================
# FUNCTION: SHOW IMAGE
# =========================
def show(title, image, cmap='gray'):
    plt.figure()
    plt.title(title)
    if len(image.shape) == 2:
        plt.imshow(image, cmap=cmap)
    else:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# =========================
# STRUCTURING ELEMENT
# =========================
def get_kernels():
    sizes = [3, 5, 7]
    shapes = {
        "square": cv2.MORPH_RECT,
        "cross": cv2.MORPH_CROSS,
        "ellipse": cv2.MORPH_ELLIPSE
    }

    kernels = {}
    for size in sizes:
        for name, shape in shapes.items():
            kernels[f"{name}_{size}"] = cv2.getStructuringElement(shape, (size, size))
    return kernels

kernels = get_kernels()

# =========================
# MORPHOLOGY BASIC
# =========================
def basic_morphology(img, kernel):
    start = time.time()

    erosion = cv2.erode(img, kernel, iterations=1)
    dilation = cv2.dilate(img, kernel, iterations=1)

    end = time.time()
    return erosion, dilation, end - start

# =========================
# MORPHOLOGY ADVANCED
# =========================
def advanced_morphology(img, kernel):
    start = time.time()

    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

    end = time.time()

    return opening, closing, gradient, tophat, blackhat, end - start

# =========================
# OCR PREPROCESSING PIPELINE
# =========================
def ocr_preprocessing(img):
    start = time.time()

    # Blur
    blur = cv2.GaussianBlur(img, (5,5), 0)

    # Threshold
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphology (clean noise)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    end = time.time()
    return thresh, clean, end - start

# =========================
# COUNTING OBJECT (WATERSHED)
# =========================
def count_objects(img):
    start = time.time()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Noise removal
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Background
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Foreground
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.7 * dist.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labeling
    _, markers = cv2.connectedComponents(sure_fg)

    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(img, markers)

    # Count objects
    count = len(np.unique(markers)) - 2  # exclude background & boundary

    end = time.time()
    return markers, count, end - start

# =========================
# RUN EXPERIMENT
# =========================
results = []

for name, kernel in kernels.items():
    print(f"Processing kernel: {name}")

    # BASIC
    erosi, dilasi, t_basic = basic_morphology(img_text, kernel)

    # ADVANCED
    opening, closing, gradient, tophat, blackhat, t_adv = advanced_morphology(img_text, kernel)

    # Save result
    results.append({
        "kernel": name,
        "basic_time": t_basic,
        "advanced_time": t_adv
    })

    # Visualisasi contoh (biar gak terlalu banyak, tampilkan sebagian)
    if name == "square_3":
        show("Original Text", img_text)
        show("Erosion", erosi)
        show("Dilation", dilasi)
        show("Opening", opening)
        show("Closing", closing)
        show("Gradient", gradient)
        show("Top Hat", tophat)
        show("Black Hat", blackhat)

# =========================
# OCR PIPELINE
# =========================
thresh, clean, t_ocr = ocr_preprocessing(img_text)

show("OCR - Threshold", thresh)
show("OCR - Cleaned", clean)

# =========================
# OBJECT COUNTING
# =========================
markers, count, t_count = count_objects(img_obj)

print(f"Jumlah objek terdeteksi: {count}")

# Visualisasi boundary
boundary = img_obj.copy()
boundary[markers == -1] = [255, 0, 0]

show("Watershed Boundary", boundary)

# =========================
# PRINT HASIL WAKTU
# =========================
print("\n=== WAKTU KOMPUTASI ===")
for r in results:
    print(r)

print(f"OCR Time: {t_ocr}")
print(f"Counting Time: {t_count}")