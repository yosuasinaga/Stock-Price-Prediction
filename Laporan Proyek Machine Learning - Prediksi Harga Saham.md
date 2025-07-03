# Laporan Proyek Machine Learning - Prediksi Harga Saham NIFTY 500 (Snapshot)

- Nama: Yosua Samuel Edlyn Sinaga
- Email: yosuas282@gmail.com
- ID Dicoding: yosuasinaga
---
## Domain Proyek

### Latar Belakang
Pasar saham merupakan instrumen investasi yang dinamis dan menjadi perhatian banyak investor. NIFTY 500 adalah salah satu indeks pasar saham utama di India, yang terdiri dari 500 perusahaan teratas yang terdaftar di National Stock Exchange (NSE). Indeks ini merepresentasikan sekitar 96.1% dari kapitalisasi pasar _free-float_ dan sekitar 96.5% dari total omset di NSE, menjadikannya barometer penting bagi kesehatan pasar saham India secara keseluruhan.

Memahami faktor-faktor apa saja yang mempengaruhi harga saham dan bagaimana memprediksi nilai wajar saham berdasarkan karakteristiknya saat ini adalah hal yang krusial bagi investor dan analis. Proyek ini berfokus pada prediksi harga saham NIFTY 500 berdasarkan data _snapshot_, yaitu data yang menangkap berbagai parameter saham pada satu titik waktu tertentu, bukan sebagai analisis time series historis harian.

### Mengapa Masalah Ini Perlu Diselesaikan
Prediksi harga saham berdasarkan fitur-fitur terkini dapat memberikan beberapa manfaat:
1.  **Estimasi Nilai Wajar**: Membantu investor mendapatkan gambaran apakah suatu saham dihargai secara wajar berdasarkan fitur-fitur fundamental dan pasar saat ini.
2.  **Alat Bantu Keputusan**: Menyediakan informasi tambahan yang dapat menjadi salah satu pertimbangan dalam pengambilan keputusan investasi jangka pendek atau menengah.
3.  **Pemahaman Faktor Penggerak Harga**: Mengidentifikasi fitur-fitur mana (misalnya, kinerja historis jangka pendek/panjang, volume, sektor industri) yang memiliki pengaruh signifikan terhadap harga terakhir saham.
4.  **Skrining Awal**: Memungkinkan investor untuk melakukan skrining awal terhadap sejumlah besar saham untuk menemukan saham yang berpotensi menarik untuk dianalisis lebih lanjut.

## Business Understanding

### Rumusan Masalah (Problem Statements)
Berdasarkan latar belakang di atas, rumusan masalah untuk proyek ini adalah:
1.  Bagaimana cara membangun model machine learning yang dapat memprediksi `Last Traded Price` (Harga Terakhir Diperdagangkan) dari saham-saham dalam NIFTY 500 berdasarkan fitur-fitur yang tersedia dalam dataset snapshot?
2.  Fitur-fitur apa saja yang paling berpengaruh dalam menentukan `Last Traded Price` saham dalam konteks dataset yang digunakan?
3.  Seberapa akurat model machine learning yang dikembangkan dalam memprediksi `Last Traded Price` dibandingkan dengan nilai aktualnya?

### Tujuan Proyek (Goals)
Proyek ini bertujuan untuk:
1.  Melakukan analisis data eksploratif (EDA) pada dataset NIFTY 500 snapshot untuk memahami karakteristik data dan hubungan antar variabel.
2.  Melakukan pra-pemrosesan data yang sesuai untuk menyiapkan data agar dapat digunakan oleh model machine learning.
3.  Mengembangkan beberapa model regresi machine learning untuk memprediksi `Last Traded Price` saham.
4.  Mengevaluasi performa model-model tersebut menggunakan metrik evaluasi yang relevan (seperti RMSE, MAE, dan R2 Score) dan memilih model terbaik.
5.  Menginterpretasikan hasil dari model terbaik untuk mendapatkan wawasan mengenai faktor-faktor yang mempengaruhi harga saham.

### Solusi yang Diusulkan (Solution Statement)
Untuk menyelesaikan permasalahan di atas, solusi yang diusulkan adalah dengan menerapkan teknik machine learning regresi. Beberapa algoritma yang akan dieksplorasi dan dibandingkan meliputi:
1.  **Linear Regression**: Sebagai model dasar untuk memahami hubungan linear antara fitur dan target.
2.  **Random Forest Regressor**: Model ensemble berbasis pohon keputusan yang mampu menangani hubungan non-linear dan interaksi antar fitur.
3.  **XGBoost Regressor**: Model ensemble berbasis gradient boosting yang dikenal memiliki performa tinggi dan robust.

Performa dari ketiga model ini akan dievaluasi untuk menentukan solusi terbaik dalam memprediksi `Last Traded Price`.
---
## Data Understanding

### Deskripsi Dataset
Dataset yang digunakan adalah `nifty_500.csv`. Dataset ini merupakan _snapshot_ yang berisi berbagai parameter dan fitur untuk masing-masing dari 501 Saham NIFTY. Dataset ini memiliki **501 baris** dan **17 kolom**, yang berisi informasi tentang saham-saham tersebut. Kolom-kolom yang teridentifikasi adalah: `['Company Name', 'Symbol', 'Industry', 'Series', 'Open', 'High', 'Low', 'Previous Close', 'Last Traded Price', 'Change', 'Percentage Change', 'Share Volume', 'Value (Indian Rupee)', '52 Week High', '52 Week Low', '365 Day Percentage Change', '30 Day Percentage Change']`.

**Sumber Data:** [Stock Market Dataset (NIFTY-500)](https://www.kaggle.com/datasets/iamsouravbanerjee/nifty500-stocks-dataset)

### Variabel-variabel pada Dataset
* **Company Name**: Nama Perusahaan.
* **Symbol**: Simbol unik saham.
* **Industry**: Industri perusahaan. (Fitur Kategorikal)
* **Series**: Seri perdagangan saham. (Fitur Kategorikal)
* **Open**: Harga pembukaan. (Fitur Numerik)
* **High**: Harga tertinggi harian. (Fitur Numerik)
* **Low**: Harga terendah harian. (Fitur Numerik)
* **Previous Close**: Harga penutupan hari sebelumnya. (Fitur Numerik)
* **Last Traded Price (LTP)**: Harga aktual terakhir diperdagangkan. **Ini adalah variabel target (y).** (Fitur Numerik)
* **Change**: Perbedaan LTP dengan Previous Close. (Fitur Numerik)
* **Percentage Change**: Perubahan harga dalam persentase. (Fitur Numerik)
* **Share Volume**: Jumlah saham diperdagangkan. (Fitur Numerik)
* **Value (Indian Rupee)**: Nilai total perdagangan atau kapitalisasi pasar. (Fitur Numerik)
* **52 Week High**: Harga tertinggi 52 minggu. (Fitur Numerik)
* **52 Week Low**: Harga terendah 52 minggu. (Fitur Numerik)
* **365 Day Percentage Change**: Perubahan persentase harga 365 hari. (Fitur Numerik)
* **30 Day Percentage Change**: Perubahan persentase harga 30 hari. (Fitur Numerik)

### Exploratory Data Analysis (EDA) Singkat
1.  **Informasi Dasar dan Statistik Deskriptif**: `df.info()` menunjukkan tipe data per kolom dan jumlah entri. `df.describe(include='all')` memberikan ringkasan statistik awal.
2.  **Pengecekan Nilai Null Awal**: Dilakukan pengecekan nilai null per kolom.
3.  **Pembersihan Data untuk EDA**: Kolom-kolom yang seharusnya numerik dikonversi paksa menjadi numerik pada `df_eda`. Nilai placeholder seperti `'-'` atau string lain diubah menjadi `NaN`, lalu `NaN` tersebut diisi dengan **median** kolom masing-masing.
4.  **Deteksi Outlier**: Boxplot pada beberapa fitur numerik kunci menunjukkan adanya outlier. Outlier ini dianggap sebagai data valid yang mencerminkan volatilitas pasar dan dipertahankan.
5.  **Distribusi Target**: Plot histogram untuk `Last Traded Price` dibuat. Hasilnya menunjukkan bahwa distribusi 'Last Traded Price' **terlihat miring ke kanan (right-skewed), dengan nilai skewness sebesar 10.21**.
6.  **Heatmap Korelasi**: Dibuat untuk fitur-fitur numerik yang telah dibersihkan. Hasil heatmap menunjukkan bahwa fitur seperti **'High' (0.999973), 'Low' (0.999957), 'Previous Close' (0.999890), 'Open' (0.999832), '52 Week Low' (0.994556), dan '52 Week High' (0.989987)** memiliki korelasi positif yang sangat kuat dengan 'Last Traded Price' (target). Fitur lain seperti **'Share Volume' menunjukkan korelasi negatif lemah (-0.072868)**, sedangkan '30 Day Percentage Change' (0.078694) dan '365 Day Percentage Change' (0.069566) menunjukkan korelasi positif yang sangat lemah.
7.  **Analisis Bivariat**:
    * **Numerik vs Target**: Korelasi antara **'Open' dan 'Last Traded Price' adalah 1.00**. Korelasi antara '30 Day Percentage Change' dan 'Last Traded Price' adalah 0.08. Korelasi antara 'Change' dan 'Last Traded Price' adalah 0.05. Korelasi antara 'Share Volume' dan 'Last Traded Price' adalah -0.07.
    * **Kategorikal vs Target**: Rata-rata 'Last Traded Price' berdasarkan 'Industry' (Top 5 berdasarkan mean LTP): **Diversified (11261.40), Automobile and Auto Components (4759.36), Textiles (4756.17), Construction Materials (2673.76), Capital Goods (2009.81)**.
---
## Data Preparation

Proses persiapan data dilakukan untuk mengubah data mentah menjadi format yang siap digunakan oleh model machine learning. Langkah-langkah yang dilakukan adalah sebagai berikut:

1. **Menggunakan `df_prep` sebagai Sumber**: Tahap ini dimulai dengan salinan baru dari dataset asli (`df_prep = df.copy()`) untuk persiapan model. Dataset `df_prep` kemudian melalui proses pembersihan lebih lanjut, termasuk penghapusan nilai NaN pada kolom target (`Last Traded Price`) menggunakan teknik `dropna()` untuk memastikan tidak ada baris dengan nilai kosong pada kolom target.
    ```python
    df_prep.dropna(subset=[target_col_name], inplace=True)
    ```
2. **Definisi Kolom Utama**: Nama-nama kolom untuk target (`Last Traded Price`) dan fitur-fitur (numerik dan kategorikal) ditetapkan secara eksplisit.
3. **Finalisasi Target (y) dan Fitur Numerik (X_numerical)**:
    * Variabel target `y` diambil langsung dari kolom `Last Traded Price`.
    * Fitur numerik `X_numerical` diambil dari kolom-kolom numerik yang relevan. Dilakukan pengecekan dan pengisian NaN terakhir pada `X_numerical` menggunakan median kolom sebagai jaring pengaman.
4. **Encoding Fitur Kategorikal (`X_categorical_encoded`)**:
    * Fitur kategorikal yang dipilih (`Industry`, `Series`) di-encode menggunakan teknik One-Hot Encoding (`pd.get_dummies`).
    * Nilai NaN pada fitur kategorikal diisi dengan modus atau placeholder "UNKNOWN" sebelum encoding. Proses ini menghasilkan **23** fitur tambahan dari kolom kategorikal.
5. **Penggabungan Fitur menjadi X Final**: Fitur numerik (`X_numerical`) dan fitur kategorikal yang sudah di-encode (`X_categorical_encoded`) digabungkan untuk membentuk dataset fitur final `X`. Index dari `X` dan `y` direset.
6. **Pembagian Data Latih dan Uji**: Dataset `X` dan `y` dibagi menjadi data pelatihan (80%) dan data pengujian (20%) menggunakan `train_test_split`, dengan `shuffle=True`
7. **Penskalaan Fitur Numerik**: Fitur-fitur numerik dalam data pelatihan dan pengujian diskalakan menggunakan `StandardScaler`. Scaler di-`fit` hanya pada bagian numerik data pelatihan.
---
## Modeling

Pada tahap ini, dilakukan pengembangan model machine learning untuk memprediksi `Last Traded Price`. Tiga model regresi dipilih dan dilatih:

1. **Linear Regression**:
   Linear regression adalah algoritma yang digunakan untuk memodelkan hubungan linier antara variabel dependen (target) dan variabel independen (fitur). Dalam konteks proyek ini, linear regression digunakan untuk memprediksi harga saham berdasarkan fitur-fitur yang diberikan.

2. **Random Forest Regressor**:
   Random Forest adalah algoritma ensemble yang menggabungkan hasil dari beberapa pohon keputusan (decision trees). Setiap pohon dilatih menggunakan subset data dan subset fitur secara acak. Hal ini membantu mengurangi overfitting yang sering terjadi pada pohon keputusan tunggal. Random Forest digunakan di sini untuk menangani hubungan non-linier antara fitur dan target.

3. **XGBoost Regressor**:
   XGBoost adalah algoritma boosting yang menggabungkan pohon keputusan lemah (weak learners) menjadi model yang kuat. Algoritma ini meminimalkan error melalui pembelajaran bertahap dan sering digunakan dalam berbagai masalah prediksi. Pada proyek ini, XGBoost digunakan untuk mengeksplorasi pengaruh fitur terhadap harga saham dan memberikan prediksi yang lebih robust.

### Parameter Model

1. **Random Forest Regressor**:
   - `n_estimators=100` - Jumlah pohon keputusan yang digunakan.
   - `random_state=42` - Menetapkan nilai acak untuk memastikan hasil yang konsisten.
   - `n_jobs=-1` - Menggunakan semua core CPU yang tersedia untuk mempercepat pelatihan model.

2. **XGBoost Regressor**:
   - `objective='reg:squarederror'` - Menentukan bahwa model ini digunakan untuk regresi dan meminimalkan error kuadrat.
   - `n_estimators=100` - Jumlah pohon yang akan dibangun.
   - `random_state=42` - Menetapkan nilai acak untuk memastikan hasil yang konsisten.
   - `n_jobs=-1` - Menggunakan semua core CPU yang tersedia untuk mempercepat pelatihan model.


Semua model dilatih menggunakan data `X_train_scaled` dan `y_train`.
---
## Evaluation

Performa dari setiap model dievaluasi pada data uji (`X_test_scaled` dan `y_test`) menggunakan metrik MSE, RMSE, MAE, dan R2 Score.

### Hasil Metrik Evaluasi
Berikut adalah ringkasan hasil evaluasi untuk setiap model:

| Model              | MSE           | RMSE      | MAE        | R2 Score |
|--------------------|---------------|-----------|------------|----------|
| Linear Regression  | 287.7629      | 16.9636   | 9.0441     | 1.0000   |
| Random Forest      | 133,302.4973  | 365.1061  | 86.2497    | 0.9871   |
| XGBoost            | 119,203.8505  | 345.2591  | 110.3963   | 0.9885   |

Model terbaik dipilih berdasarkan nilai RMSE yang terendah dan R2 Score yang tertinggi. Model **LinearRegression** menunjukkan performa paling unggul dengan nilai RMSE sebesar **16.9636** dan R2 Score sebesar **1.0000**. Ini mengindikasikan bahwa model tersebut mampu menjelaskan sekitar **100.0%** varians dalam data `Last Traded Price`. *(Catatan: R2 Score 1.0000 dan RMSE yang sangat rendah untuk Linear Regression bisa jadi indikasi adanya data leakage atau fitur yang sangat kuat dan mungkin secara langsung menghitung target. Perlu investigasi lebih lanjut pada fitur yang digunakan, terutama 'Change' dan 'Percentage Change'.)*

### Uji Prediksi pada Satu Sampel
Pengujian prediksi pada satu sampel data dari set pengujian menunjukkan perbandingan nilai aktual (`y_true`) dengan prediksi dari masing-masing model:

| y_true     | pred_LinearRegression | pred_RandomForest | pred_XGBoost |
|------------|-----------------------|-------------------|--------------|
| 3,100.0000 | 3,127.1791            | 3,138.6225        | 3,190.7263   |

### Analisis Visual Model Terbaik
Untuk model terbaik (**LinearRegression**):
1.  **Plot Aktual vs. Prediksi**: Scatter plot antara `Last Traded Price` aktual dengan nilai prediksi menunjukkan titik-titik data **prediksi sempurna** (mengikuti garis y=x).
2.  **Plot Distribusi Residual**: Histogram residual menunjukkan bahwa distribusi error **mendekati normal dan terpusat di sekitar nol**. Rata-rata residual adalah **0.2768**.
3.  **Plot Residual vs. Prediksi**: Scatter plot antara nilai prediksi dan residual menunjukkan **tersebar acak tanpa pola jelas di sekitar garis horizontal nol**.
---
## Kesimpulan
Proyek ini bertujuan untuk mengembangkan model machine learning yang mampu memprediksi `Last Traded Price` saham dari dataset snapshot NIFTY 500. Setelah melalui tahapan EDA, Data Preparation, Modeling, dan Evaluasi, dapat disimpulkan:

1.  Dataset `nifty_500.csv` adalah data snapshot perusahaan, sehingga pendekatan regresi standar lebih sesuai.
2.  Fitur-fitur numerik dan kategorikal (`Industry`, `Series` setelah di-encode) digunakan untuk membangun model prediksi.
3.  Dari ketiga model yang diuji, model **LinearRegression** menunjukkan performa terbaik dengan RMSE **16.9636** dan R2 Score **1.0000**.
4.  Hasil R2 Score 1.0000 mengindikasikan bahwa model Linear Regression dapat menjelaskan **100.0%** variabilitas harga. **Perlu diperhatikan bahwa hasil R2 Score yang sempurna ini seringkali mengindikasikan adanya *data leakage* atau fitur yang terlalu prediktif (misalnya, fitur 'Change' atau 'Percentage Change' yang mungkin dihitung menggunakan harga target itu sendiri atau informasi yang tidak akan tersedia pada saat prediksi nyata). Hal ini memerlukan investigasi lebih lanjut terhadap definisi dan penggunaan fitur.**

---
## Rekomendasi
1. **Investigasi Data Leakage**: Sangat penting untuk menyelidiki penyebab R2 Score 1.0000 pada Linear Regression. Periksa kembali definisi fitur 'Change' dan 'Percentage Change' dan pastikan tidak ada kebocoran informasi dari target. Jika ada, fitur tersebut harus dihilangkan atau dimodifikasi.
2. **Pengembangan Fitur Lebih Lanjut**: Jika leakage teratasi dan performa model menjadi lebih realistis, eksplorasi penambahan fitur lain seperti rasio keuangan dapat dipertimbangkan.
3. **Hyperparameter Tuning**: Setelah memastikan tidak ada leakage, lakukan tuning hyperparameter untuk Random Forest dan XGBoost.
4. **Analisis Feature Importance**: Setelah model yang valid didapatkan, analisis *feature importance* akan memberikan wawasan mengenai faktor-faktor yang paling dominan.
5. **Konteks Penggunaan**: Model ini (setelah validasi lebih lanjut) dapat digunakan sebagai alat bantu analisis, bukan satu-satunya dasar keputusan investasi.