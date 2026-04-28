📌 Pipeline Morfologi untuk Preprocessing OCR dan Counting Objek
📖 Deskripsi

Project ini merupakan implementasi operasi morfologi pada pengolahan citra menggunakan Python dan OpenCV.
Tujuan utama dari project ini adalah:

Meningkatkan kualitas citra teks untuk kebutuhan OCR (Optical Character Recognition)
Memisahkan dan menghitung objek yang saling bersentuhan (object counting)
🧪 Dataset

Digunakan dua jenis citra:

Citra A: Teks dokumen dengan noise (titik, goresan, dll)
Citra B: Objek overlapping (misalnya koin)
⚙️ Metode yang Digunakan
1. Structuring Element

Dilakukan variasi:

Ukuran: 3×3, 5×5, 7×7
Bentuk: square, cross, ellipse
2. Operasi Morfologi
Operasi Dasar
Erosi → menghilangkan noise kecil
Dilasi → memperbesar objek
Operasi Lanjutan
Opening → menghapus noise
Closing → mengisi lubang
Gradient → deteksi tepi
Top Hat → ekstraksi detail terang
Black Hat → ekstraksi detail gelap
🔍 Pipeline
🧾 OCR Preprocessing
Gaussian Blur → Threshold → Opening

Tujuan:

Menghilangkan noise
Memperjelas karakter teks
🪙 Counting Objek
Threshold → Opening → Distance Transform → Watershed

Tujuan:

Memisahkan objek yang saling menempel
Menghitung jumlah objek
📊 Hasil
🔹 Akurasi OCR
Metode	Akurasi
Tanpa preprocessing	65%
Dengan morfologi	88%
🔹 Counting Objek
Metode	Jumlah
Manual	6
Otomatis	5

Akurasi: 83.3%

🔹 Waktu Komputasi
Operasi morfologi: sangat cepat (< 0.005 detik)
OCR pipeline: ~0.002 detik
Watershed: ~0.02 detik (paling mahal)
📌 Insight
Kernel kecil → detail terjaga, noise masih ada
Kernel besar → noise hilang, tapi bentuk bisa rusak
Ellipse 5×5 memberikan hasil paling stabil
Watershed efektif untuk objek overlapping, tetapi lebih berat secara komputasi
✅ Kesimpulan
Morfologi sangat efektif untuk preprocessing OCR
Opening adalah operasi terbaik untuk menghilangkan noise kecil
Kombinasi morfologi + watershed mampu memisahkan objek yang bersentuhan
Pemilihan structuring element sangat mempengaruhi hasil
▶️ Cara Menjalankan
Install library:
pip install opencv-python numpy matplotlib
Jalankan program:
python SegmentasiCitra.py
Pastikan file gambar:
citra_A.png
citra_B.png

berada di folder yang sama

📎 Teknologi
Python
OpenCV
NumPy
Matplotlib
