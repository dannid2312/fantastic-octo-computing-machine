# Laporan Proyek Machine Learning - Danni Dwicahyo

## Project Overview

Selama pandemi COVID-19, terjadi peningkatan signifikan dalam permintaan hiburan di rumah, yang mencakup peningkatan penggunaan platform streaming seperti Netflix, Disney Plus, dan Amazon Prime. Pembatasan sosial dan penutupan bioskop telah mendorong masyarakat untuk beralih ke media digital untuk hiburan. [The Impact COVID-19 Had On The Entertainment Industry In 2020](https://www.forbes.com/sites/bradadgate/2021/04/13/the-impact-covid-19-had-on-the-entertainment-industry-in-2020/?sh=4ca49c19250f)

Platform streaming seperti Netflix menawarkan ribuan film kepada penggunanya yang dapat membuat pengguna kesulitan menemukan film yang sesuai dengan selera mereka. Netflix berhasil menjawab permasalahan tersebut dengan menggunakan machine learning untuk meningkatkan retensi pelanggan dan kepuasan pengguna dengan melakukan pembangunan sistem rekomendasi film yang efektif. Dengan mengadopsi pendekatan yang terinspirasi dari praktik terbaik yang ditunjukkan oleh Netflix, movie recommendation system dapat meningkatkan kemampuan dalam menyajikan rekomendasi film yang relevan dan memuaskan bagi pengguna. [The Netflix Recommender System: Algorithms, Business Value, and Innovation](https://dl.acm.org/doi/pdf/10.1145/2843948)

Proyek movie recommendation system memiliki manfaat yang sangat besar, yaitu: 
- Meningkatkan kepuasan pengguna: Sistem rekomendasi yang akurat dan personal dapat membantu pengguna menemukan film yang sesuai dengan selera mereka dengan lebih mudah dan cepat. Hal ini dapat meningkatkan kepuasan pengguna dan mendorong mereka untuk terus menggunakan platform streaming atau situs web rekomendasi film.
- Meningkatkan efisiensi pencarian film: Sistem rekomendasi dapat membantu pengguna menghemat waktu dan tenaga dalam mencari film yang ingin ditonton. Hal ini dapat meningkatkan efisiensi dan produktivitas pengguna.
- Meningkatkan pendapatan platform: Sistem rekomendasi dapat membantu platform streaming dan situs web rekomendasi film meningkatkan pendapatan mereka dengan mendorong pengguna untuk menonton lebih banyak film.

## Business Understanding

Sistem rekomendasi film merupakan alat penting bagi bisnis di industri hiburan. Sistem ini memberikan manfaat bagi pengguna dengan menghemat waktu dan tenaga dalam mencari film yang sesuai selera, serta membantu mereka menemukan film baru yang menarik. Bagi bisnis, sistem ini meningkatkan engagement dan konversi pengguna, serta membantu monetisasi platform dengan lebih efektif. Meskipun terdapat beberapa tantangan, namun seiring dengan perkembangan teknologi AI dan machine learning memungkinkan sistem ini terus berkembang dan menjadi lebih personal dan akurat. Dengan memanfaatkan peluang dan mengatasi tantangan tersebut, sistem rekomendasi film akan terus memainkan peran penting dalam industri hiburan.

### Problem Statements

Berdasarkan kondisi yang telah diuraikan sebelumnya, sistem rekomendasi film dikembangkan untuk menjawab permasalahan sebagai berikut:
- Diperlukan data yang lengkap, akurat, dan/atau tidak mengandung bias tentang rating film dari banyak penonton atau pelanggan.
- Diperlukan pengukuran yang baik dalam menentukan akurasi dan personalisasi rekomendasi untuk menghasilkan rekomendasi yang relevan.
- Diperlukan sistem rekomendasi film yang dapat memberikan rekomendasi kepada pengguna baru yang belum memberikan banyak data tentang preferensinya.

### Goals

Untuk menjawab permasalahan tersebut, sistem rekomendasi film dikembangkan dengan tujuan atau goals sebagai berikut:
- Menggunakan data rating dari film lama untuk mendapatkan data yang lengkap, serta melakukan data preparation yang baik untuk menghilangkan data yang tidak berkualitas atau data yang bias.
- Menggunakan metrik evaluasi root mean squared error (RMSE) untuk menentukan seberapa baik sistem memberikan rekomendasi yang relevan dan terpersonalisasi.
- Menggunakan metode collaborative filtering untuk memberikan rekomendasi film baik kepada pengguna baru maupun pengguna lama yang telah terpersonalisasi.

### Solution Statement
Untuk mencapai tujuan atau goals tersebut, dilakukan beberapa tahapan sebagai berikut:
- Menggunakan database film yang memiliki data rating yang lengkap dan melimpah.
- Melakukan univariate analysis untuk memahami kualitas data yang dimiliki.
- Melakukan penggabungan terhadap dataset database film dengan dataset database rating.
- Melakukan label encoding untuk merubah categorical features seperti userId dan movieId menjadi numerical features untuk memudahkan model mencapai konvergensi selama pelatihan.
- Melakukan standardization berupa min max scaling untuk merubah data rating dari 0-5 menjadi 0-1 untuk memudahkan model mencapai konvergensi selama pelatihan.
- Melakukan pengacakan terhadap dataset serta pembagian untuk menentukan data train dan data validasi.
- Melakukan perhitungan untuk menentukan skor kecocokan antara pengguna dan film yang ditonton dengan teknik embedding.
- Melakukan operasi perkalian dot product antara embedding pengguna dan film.
- Melakukan penambahan bias untuk setiap user dan resto. 
- Menetapkan sko kecocolan dalam skala [0,1] dengan fungsi aktivasi sigmoid.
- Melakukan permodelan dengan metode collaborative filtering untuk memproses model pengguna dan rating filmnya.
- Memberikan rekomendasi film baik terhadap pengguna baru maupun pengguna lama.

## Data Understanding
Data yang digunakan pada proyek kali ini adalah Movies & Ratings for Recommendation System dataset yang diunduh dari website [Kaggle](https://www.kaggle.com/datasets/nicoletacilibiu/movies-and-ratings-for-recommendation-system/code). Dataset ini terdiri dari dua file csv berupa movies.csv dan ratings.csv. File movies.csv merupakan dataset tentang database film yang memiliki 9742 baris yang terdiri dari tiga kolom yaitu movieId, title, dan genres. File ratings.csv merupakan dataset tentang rating film yang memiliki 100836 baris yang terdiri dari empat kolom, yaitu userId, movieId, rating, dan timestamp. Dataset masih perlu dilakukan beberapa penyesuaian dalam tahap data preparation untuk menghasilkan dataset yang berkualitas. Kedua dataset yang tersedia kemudian digabungkan menjadi satu dataset dengan menggunakan movieId sebagai acuan penggabungan.

Variabel-variabel yang terdapat pada dataset gabungan antara file movies.csv dan ratings.csv adalah sebagai berikut:
- accepts : merupakan jenis pembayaran yang diterima pada restoran tertentu.
- cuisine : merupakan jenis masakan yang disajikan pada restoran.
- dst

**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data beserta insight atau exploratory data analysis.

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model sisten rekomendasi yang Anda buat untuk menyelesaikan permasalahan. Sajikan top-N recommendation sebagai output.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menyajikan dua solusi rekomendasi dengan algoritma yang berbeda.
- Menjelaskan kelebihan dan kekurangan dari solusi/pendekatan yang dipilih.

## Evaluation
Pada bagian ini Anda perlu menyebutkan metrik evaluasi yang digunakan. Kemudian, jelaskan hasil proyek berdasarkan metrik evaluasi tersebut.

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.
