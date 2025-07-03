# Prediksi Harga Saham NIFTY 500 Menggunakan Machine Learning

## Deskripsi Proyek

Proyek ini bertujuan untuk mengembangkan sistem prediksi harga saham berdasarkan data snapshot saham dari indeks NIFTY 500. Model machine learning digunakan untuk memprediksi harga terakhir diperdagangkan (`Last Traded Price`) dari saham-saham di NIFTY 500 berdasarkan berbagai fitur yang tersedia dalam dataset. Beberapa algoritma regresi digunakan untuk mencapai prediksi yang akurat.

## Pemahaman Bisnis

### Latar Belakang

Pasar saham adalah instrumen investasi yang sangat dinamis dan menjadi perhatian utama bagi investor. NIFTY 500 adalah salah satu indeks pasar saham terbesar di India yang mencakup 500 perusahaan teratas di National Stock Exchange (NSE). Dengan memahami faktor-faktor yang mempengaruhi harga saham, para investor dapat membuat keputusan investasi yang lebih baik.

### Rumusan Masalah

1. Bagaimana cara membangun model machine learning yang dapat memprediksi harga terakhir diperdagangkan (`Last Traded Price`) dari saham NIFTY 500 berdasarkan fitur yang tersedia dalam dataset?
2. Fitur-fitur apa saja yang paling berpengaruh dalam menentukan harga terakhir diperdagangkan (`Last Traded Price`)?
3. Seberapa akurat model machine learning yang dikembangkan dalam memprediksi harga saham dibandingkan dengan nilai aktual?

### Tujuan Proyek

1. Melakukan analisis data eksploratif (EDA) untuk memahami karakteristik data dan hubungan antar variabel.
2. Mengembangkan beberapa model regresi untuk memprediksi `Last Traded Price`.
3. Mengevaluasi performa model-model tersebut menggunakan metrik evaluasi yang relevan seperti RMSE, MAE, dan R2 Score.
4. Menginterpretasikan hasil dari model terbaik untuk mendapatkan wawasan mengenai faktor-faktor yang mempengaruhi harga saham.

### Solusi yang Diusulkan

Untuk memecahkan masalah ini, beberapa algoritma regresi digunakan:
- **Linear Regression**: Sebagai model dasar untuk memahami hubungan linier antara fitur dan target.
- **Random Forest Regressor**: Model ensemble berbasis pohon keputusan yang mampu menangani hubungan non-linear.
- **XGBoost Regressor**: Model boosting berbasis gradient yang sering digunakan untuk meningkatkan performa model.

## Pemahaman Data

### Deskripsi Dataset

Dataset yang digunakan adalah `nifty_500.csv`, yang berisi berbagai parameter untuk masing-masing saham dalam indeks NIFTY 500. Dataset ini memiliki **501 baris** dan **17 kolom**, yang berisi informasi tentang saham-saham tersebut.

**Kolom Dataset:**
- `Company Name`, `Symbol`, `Industry`, `Series`
- Fitur numerik seperti `Open`, `High`, `Low`, `Previous Close`, `Last Traded Price`, dll.

### Exploratory Data Analysis (EDA)

1. **Pengecekan Nilai Null dan Pembersihan Data**: Beberapa kolom memiliki nilai hilang dan telah diisi dengan nilai median.
2. **Distribusi Target**: Distribusi `Last Traded Price` menunjukkan kemiringan ke kanan (right-skewed).
3. **Korelasi**: Fitur seperti `High`, `Low`, dan `Previous Close` memiliki korelasi yang sangat kuat dengan `Last Traded Price`.

## Persiapan Data

1. **Pembersihan dan Pengisian Nilai Hilang**: Menghapus baris dengan nilai kosong pada kolom target dan mengisi NaN dengan median pada kolom numerik.
2. **Pengkodean Fitur Kategorikal**: Fitur seperti `Industry` dan `Series` di-encode menggunakan One-Hot Encoding.
3. **Penskalaan Fitur**: Fitur numerik distandarisasi menggunakan `StandardScaler` untuk memastikan skala yang seragam pada data.

## Modeling

Model yang digunakan:
1. **Linear Regression**
2. **Random Forest Regressor**
3. **XGBoost Regressor**

### Parameter Model
- **Random Forest Regressor**: `n_estimators=100`, `random_state=42`
- **XGBoost**: `n_estimators=100`, `random_state=42`

## Evaluasi

Performa model dievaluasi menggunakan metrik **MSE**, **RMSE**, **MAE**, dan **R2 Score**.

### Hasil Evaluasi

| Model              | MSE           | RMSE      | MAE        | R2 Score |
|--------------------|---------------|-----------|------------|----------|
| Linear Regression  | 287.76        | 16.96     | 9.04       | 1.00     |
| Random Forest      | 133,302.50    | 365.11    | 86.25      | 0.99     |
| XGBoost            | 119,203.85    | 345.26    | 110.40     | 0.99     |

### Kesimpulan

- **Linear Regression** memberikan hasil terbaik dengan RMSE terendah dan R2 Score mendekati 1.0.
- Meskipun **Linear Regression** menunjukkan hasil yang sangat baik, perlu investigasi lebih lanjut untuk memastikan tidak adanya *data leakage* atau fitur yang sangat kuat.

## Rekomendasi

1. **Investigasi Data Leakage**: Periksa kembali fitur-fitur seperti `Change` dan `Percentage Change` yang mungkin menyebabkan *data leakage*.
2. **Pengembangan Fitur Lebih Lanjut**: Pertimbangkan untuk menambahkan fitur lain seperti rasio keuangan untuk meningkatkan model.
3. **Hyperparameter Tuning**: Lakukan tuning hyperparameter untuk **Random Forest** dan **XGBoost**.
4. **Analisis Feature Importance**: Setelah model divalidasi, lakukan analisis fitur penting untuk mengetahui faktor dominan dalam prediksi harga saham.
