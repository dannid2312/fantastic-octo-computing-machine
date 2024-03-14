# Laporan Proyek Machine Learning - Danni Dwicahyo

## Project Overview

Selama pandemi COVID-19, terjadi peningkatan signifikan dalam permintaan hiburan di rumah, yang mencakup peningkatan penggunaan platform streaming. Pembatasan sosial dan penutupan bioskop telah mendorong masyarakat untuk beralih ke media digital untuk hiburan [[1](https://mapub.org/ojs/index.php/mapeh/article/view/72)]. Platform streaming seperti Netflix menawarkan ribuan film kepada penggunanya yang dapat membuat pengguna kesulitan menemukan film yang sesuai dengan selera mereka. Netflix berhasil menjawab permasalahan tersebut dengan menggunakan machine learning untuk meningkatkan retensi pelanggan dan kepuasan pengguna dengan melakukan pembangunan sistem rekomendasi film yang efektif [[2](https://dl.acm.org/doi/pdf/10.1145/2843948)]. Dengan mengadopsi pendekatan yang terinspirasi dari praktik terbaik yang ditunjukkan oleh Netflix, movie recommendation system dapat meningkatkan kemampuan dalam menyajikan rekomendasi film yang relevan dan memuaskan bagi pengguna.

Proyek movie recommendation system memiliki manfaat yang sangat besar, yaitu: 
- Meningkatkan kepuasan pengguna: Sistem rekomendasi yang akurat dan personal dapat membantu pengguna menemukan film yang sesuai dengan selera mereka dengan lebih mudah dan cepat. Hal ini dapat meningkatkan kepuasan pengguna dan mendorong mereka untuk terus menggunakan platform streaming atau situs web rekomendasi film.
- Meningkatkan efisiensi pencarian film: Sistem rekomendasi dapat membantu pengguna menghemat waktu dan tenaga dalam mencari film yang ingin ditonton. Hal ini dapat meningkatkan efisiensi dan produktivitas pengguna.
- Meningkatkan pendapatan platform: Sistem rekomendasi dapat membantu platform streaming dan situs web rekomendasi film meningkatkan pendapatan mereka dengan mendorong pengguna untuk menonton lebih banyak film.

## Business Understanding

Sistem rekomendasi film merupakan alat penting bagi bisnis di industri hiburan. Sistem ini memberikan manfaat bagi pengguna dengan menghemat waktu dan tenaga dalam mencari film yang sesuai selera, serta membantu mereka menemukan film baru yang menarik. Bagi bisnis, sistem ini meningkatkan engagement dan konversi pengguna, serta membantu monetisasi platform dengan lebih efektif. Meskipun terdapat beberapa tantangan, namun seiring dengan perkembangan teknologi AI dan machine learning memungkinkan sistem ini terus berkembang dan menjadi lebih personal dan akurat. Dengan memanfaatkan peluang dan mengatasi tantangan tersebut, sistem rekomendasi film akan terus memainkan peran penting dalam industri hiburan.

### Problem Statements

Berdasarkan kondisi yang telah diuraikan sebelumnya, sistem rekomendasi film dikembangkan untuk menjawab permasalahan sebagai berikut:
- Diperlukan data yang lengkap, akurat, dan/atau tidak mengandung bias tentang rating film dari banyak penonton atau pelanggan.
- Diperlukan sistem rekomendasi film yang dapat memberikan rekomendasi kepada pengguna lama maupun pengguna baru yang belum memberikan banyak data tentang preferensinya berdasarkan rating yang diberikan oleh pengguna.
- Diperlukan sistem rekomendasi film yang tidak hanya memperhitungkan rating namun juga fitur yang ada di dalam filmnya, contohnya genre.

### Goals

Untuk menjawab permasalahan tersebut, sistem rekomendasi film dikembangkan dengan tujuan atau goals sebagai berikut:
- Menggunakan data rating dari film lama untuk mendapatkan data yang lengkap, serta melakukan data preparation yang baik untuk menghilangkan data yang tidak berkualitas atau data yang bias.
- Menggunakan metode collaborative filtering untuk memberikan rekomendasi film baik kepada pengguna baru maupun pengguna lama yang telah terpersonalisasi.
- Menggunakan metode hybrid content based filtering dan collaborative filtering untuk memberikan rekomendasi film berdasarkan rating dan genre filmnya.

### Solution Statement
Untuk mencapai masing-masing tujuan atau goals tersebut, dilakukan tahapan sebagai berikut:
- Menggunakan database film yang memiliki data rating yang lengkap dan melimpah untuk kemudian dilakukan univariate analysis, preprocessing, dan data preparation berupa encoding dan standardization untuk mendapatkan data yang berkualitas.
- Melakukan permodelan dengan metode collaborative filtering untuk memproses model pengguna dan rating filmnya dengan menggunakan teknik embedding dan perkalian dot product untuk memberikan rekomendasi film baik terhadap pengguna baru maupun pengguna lama berdasarkan skor kecocokan yang dihitung dengan fungsi aktivasi sigmoid.
- Melakukan permodelan dengan metode yang menggabungkan collaborative filtering dengan content based filtering untuk memproses model film dan genre filmnya dengan bantuan fungsi sentence transformer dan matriks factorization untuk memberikan rekomendasi film berdasarkan rating film dan kemiripannya dengan film yang sudah pernah ditonton sebelumnya .

## Data Understanding
Data yang digunakan pada proyek kali ini adalah Movies & Ratings for Recommendation System dataset yang diunduh dari website [Kaggle](https://www.kaggle.com/datasets/nicoletacilibiu/movies-and-ratings-for-recommendation-system/code). Dataset ini terdiri dari dua file csv berupa movies.csv dan ratings.csv. File movies.csv merupakan dataset tentang database film yang memiliki 9742 baris yang terdiri dari tiga kolom yaitu movieId, title, dan genres. File ratings.csv merupakan dataset tentang rating film yang memiliki 100836 baris yang terdiri dari empat kolom, yaitu userId, movieId, rating, dan timestamp. Dataset masih perlu dilakukan beberapa penyesuaian dalam tahap data preparation untuk menghasilkan dataset yang berkualitas. Kedua dataset yang tersedia kemudian digabungkan menjadi satu dataset dengan menggunakan movieId sebagai acuan penggabungan. Hasil akhir dari penggabungan dataset terdiri dari 100836 baris dan 6 kolom, serta tidak terdapat missing values yang ditunjukkan pada Tabel 1.

(Tabel 1. Dataset Gabungan)

| # | Column    | Non-Null Count  | Dtype   |
|---|-----------|-----------------|---------|
| 1 | movieId   | 100836 non-null | int64   |
| 2 | title     | 100836 non-null | object  |
| 3 | genres    | 100836 non-null | object  |
| 4 | userId    | 100836 non-null | int64   |
| 5 | rating    | 100836 non-null | float64 |
| 6 | timestamp | 100836 non-null | int64   |

### Variabel-variabel yang terdapat pada dataset gabungan antara file movies.csv dan ratings.csv adalah sebagai berikut:
- movieId: menunjukkan nomor identitas atau index dari suatu film, merupakan kolom yang menjadi acuan dalam penggabungan antara dua dataset.
- title: menunjukkan judul film
- genres: menunjukkan genre / ragam / tipe film
- userId: menunjukkan nomor identitas atau index dari pengguna atau penonton film yang memberikan penilaian (rating)
- rating: menunjukkan nilai yang diperoleh dari pengguna atau penonton film
- timestamp: menunjukkan waktu pengguna melakukan penilaian terhadap suatu film

### Univariate Analysis
Dari data movies.csv, dari jumlah film sebanyak 9742, terbagi kedalam 951 genre yang berbeda. Namun bila dilihat lebih detail, setiap film bisa terdiri dari berbagai genre, sehingga perlu dilakukan pemrosesan lebih lanjut terhadap kolom genre. Pertama, adalah mengubah tanda "|" dan mengganti dengan " ", lalu menghilangkan tanda "-", dan merubah film yang tidak bergenre "non genres listed" dengan "None". Rangkaian tersebut diperlukan untuk memudahkan nantinya ketika melakukan ekstraksi fitur genre dengan fungsi TF-IDF Vectorizer.

Berdasarkan pengecekan setiap variabel dari data ratings.csv, didapatkan bahwa dari 9742 film yang tersedia di database, hanya 9724 film yang sudah diberikan rating, sehingga masih ada 18 film yang belum pernah diberikan penilaian. Kemudian, pengguna atau penonton (user) yang melakukan penilaian adalah sebanyak 610 orang, dengan rentang penilaian antara 0.5 sampai 5.0.
```
Jumlah film berdasarkan movie ID 9742
Jumlah user yang memberikan penilaian 610
Jumlah film yang dinilai berdasarkan movie ID 9724
Jumlah nilai minimum rating 0.5
Jumlah nilai minimum rating 5.0
```

![box-rating](https://github.com/dannid2312/fantastic-octo-computing-machine/assets/123451351/3fe38a57-94ab-4451-9a0b-21b5904b5ffc)

(Gambar 1. Boxplot Rating)

Berdasarkan gambar 1 berupa boxplot sebaran nilai rating yang diberikan pengguna, nilai yang paling banyak diberikan adalah antara 3.0 dan 4.0, serta terdapat nilai outlier yaitu 0.5 dan 1, sehingga diperlukan treatment terhadap outlier tersebut. Outlier dihilangkan dengan menggunakan "the 1.5 IQR rule" yaitu menghilangkan outlier yang berada diluar quartile 1 dan quartile tiga dengan jarak 1.5 kali dari selisih quartile tiga dan quartile satu, sehingga menghasilkan dataset akhir gabungan sejumlah 74639 baris seperti ditunjukkan pada tabel 2. Dengan deskripsi dataset sebagai berikut:

```
Jumlah film berdasarkan movieId: 5284
Jumlah film berdasarkan title: 5284
Jumlah genre berdasarkan genres: 585
Jumlah user berdasarkan userId: 610
Jumlah nilai minimum rating 1.5
Jumlah nilai minimum rating 5.0
```

(Tabel 2. Dataset Gabungan Dikurangi Outlier Rating)

| # | Column    | Non-Null Count | Dtype   |
|---|-----------|----------------|---------|
| 1 | movieId   | 74639 non-null | int64   |
| 2 | title     | 74639 non-null | object  |
| 3 | genres    | 74639 non-null | object  |
| 4 | userId    | 74639 non-null | int64   |
| 5 | rating    | 74639 non-null | float64 |
| 6 | timestamp | 74639 non-null | int64   |

## Data Preparation
Pada bagian ini akan dilakukan tiga tahap persiapan data, yaitu:

### Encoding fitur kategori
Encoding dilakukan terhadap variabel userId dan movieId. Meskipun kedua variabel tersebut sudah memiliki nilai integer, namun encoding tetap dilakukan untuk mempermudah model dalam melakukan pelatihan sehingga konvergensi lebih mudah dicapai. Encoding dilakukan secara manual dengan menggunakan fungsi enumerate().

### Standarisasi
Algoritma machine learning memiliki performa lebih baik dan konvergen lebih cepat ketika dimodelkan pada data dengan skala relatif sama atau mendekati distribusi normal. Proses scaling dan standarisasi membantu untuk membuat fitur data menjadi bentuk yang lebih mudah diolah oleh algoritma. Pada proyek ini standarisasi yang dilakukan adalah dengan menggunakan min max scaling untuk mengubah variabel rating dari yang semula memiliki skala [1.5,5] menjadi skala [0,1].

### Pembagian dataset dengan fungsi train_test_split dari library sklearn
Sebelum melakukan permodelan, perlu dilakukan pembagian antara dataset untuk dilatih (train) pada model dan dataset untuk menguji (test) performa model. Dalam project ini akan digunakan proporsi pembagian sebesar 80:20 secara manual dengan melakukan split terhadap dataframe yang sebelumnya sudah diacak.

## Modeling
### Model Development: Collaborative Filtering

Metode Collaborative filtering (CF) adalah teknik yang digunakan dalam sistem rekomendasi untuk memprediksi preferensi pengguna berdasarkan rating dan interaksi pengguna lain. CF didasarkan pada asumsi bahwa pengguna dengan preferensi yang sama di masa lalu kemungkinan besar akan memiliki preferensi yang sama di masa depan. Oleh sebab itu, sistem CF digunakan dalam memberikan rekomendasi film yang cocok berdasarkan rating yang diberikan pengguna sebelumnya.

Tahapan dalam metode CF adalah dimulai dengan menghitung skor kecocokan antara pengguna dan film dengan teknik embedding. Pertama, kita melakukan proses embedding terhadap data user dan film. Selanjutnya, lakukan operasi perkalian dot product antara embedding user dan film. Selain itu, kita juga dapat menambahkan bias untuk setiap user dan film. Skor kecocokan ditetapkan dalam skala [0,1] dengan fungsi aktivasi sigmoid. Permodelan dilakukan dengan membuat class RecommenderNet dengan keras Model class yang disesuaikan dengan movie recommendation system. Model ini menggunakan Binary Crossentropy untuk menghitung loss function, Adam (Adaptive Moment Estimation) sebagai optimizer, dan root mean squared error (RMSE) sebagai metrics evaluation. Berikut ini adalah contoh hasil rekomendasi dengan metode collaborative filtering untuk users 177:

(Tabel 3. Movie with high ratings from user)

| Title (Year)                                                                                   | Genre                                           |
|------------------------------------------------------------------------------------------------|-------------------------------------------------|
| Beauty and the Beast (1991)                                                                    | Animation Children Fantasy Musical Romance IMAX |
| Apartment, The (1960)                                                                          | Comedy Drama Romance                            |
| Graduate, The (1967)                                                                           | Comedy Drama Romance                            |
| Harry Potter and the Sorcerer's Stone (a.k.a. Harry Potter and the Philosopher's Stone) (2001) | Adventure Children Fantasy                      |
| Adam's Rib (1949)                                                                              | Comedy Romance                                  |


(Tabel 4. Top 10 Rekomendasi Metode Collaborative Filtering)

| Title (Year)                                        | Genre                       |
|-----------------------------------------------------|-----------------------------|
| Crumb (1994)                                        |  Documentary                |
| Three Colors: Red (Trois couleurs: Rouge)   (1994)  |  Drama                      |
| In the Name of the Father (1993)                    |  Drama                      |
| His Girl Friday (1940)                              |  Comedy Romance             |
| Great Escape, The (1963)                            |  Action Adventure Drama War |
| Batman: Mask of the Phantasm (1993)                 |  Animation Children         |
| Kelly's Heroes (1970)                               |  Action Comedy War          |
| You Can Count on Me (2000)                          |  Drama Romance              |
| Dogville (2003)                                     |  Drama Mystery Thriller     |
| Blind Swordsman: Zatoichi, The (Zatôichi)   (2003)  |  Action Comedy Crime Drama  |

### Model Development: Hybrid Model
Hybrid model adalah menggabungkan antara metode Collaborative Filtering (CF) dengan Content Based Filtering (CBF), dimana model akan merekomendasikan film tidak hanya dari rating yang diberikan pengguna sebelumnya, namun juga mempertimbangkan fitur yang ada dalam film tersebut. Fitur yang dimaksud dan digunakan dalam model hybrid pada project ini adalah fitur genre. Dimana fitur genre dimasukkan sebagai input dalam melakukan permodelan.

Tahapan dalam melakukan model hybrid kurang lebih sama dengan model CF di atas, namun terdapat input tambahan berupa fitur genre yang ditransformasi dengan menggunakan fungsi statement transformer. Fungsi tersebut merubah genre film menjadi matriks yang terdiri dari angka yang menunjukkan sebaran genre tertentu dalam dataset. Untuk tujuan perbandingan, model hybrid juga menggunakan Binary Crossentropy untuk menghitung loss function, Adam (Adaptive Moment Estimation) sebagai optimizer, dan root mean squared error (RMSE) sebagai metrics evaluation. Berikut ini adalah contoh hasil rekomendasi dengan hybrid model:

(Tabel 5. Top 10 Rekomendasi Metode Hybrid)

| Title (Year)                                                         | Genre                     |
|----------------------------------------------------------------------|---------------------------|
| Billy Madison (1995)                                                 |  Comedy                   |
| Terminator, The (1984)                                               |  Action SciFi Thriller    |
| Dreamlife of Angels, The (Vie rêvée des   anges, La) (1998)          |  Drama                    |
| Universal Soldier: The Return (1999)                                 |  Action SciFi             |
| Bats (1999)                                                          |  Horror Thriller          |
| Hurricane, The (1999)                                                |  Drama                    |
| Death Wish (1974)                                                    |  Action Crime Drama       |
| Heist (2001)                                                         |  Crime Drama              |
| Triplets of Belleville, The (Les   triplettes de Belleville) (2003)  |  Animation Comedy Fantasy |
| Dernier Combat, Le (Last Battle, The)   (1983)                       |  Drama SciFi              |

## Evaluation
Root Mean Square Error (RMSE) adalah salah satu metrik evaluasi yang umum digunakan untuk mengukur tingkat kesalahan prediksi dalam konteks sistem rekomendasi. Metrik ini mengukur akurasi dari prediksi yang dihasilkan oleh sistem terhadap nilai sebenarnya yang diberikan oleh pengguna. Nilai RMSE dihitung dari selisih antara nilai sebenarnya dan nilai prediksi untuk setiap item dalam dataset, kemudian melakukan kuadrat terhadap masing-masing selisih. Nilai RMSE adalah akar dari rata-rata kuadrat selisih tersebut. Untuk lebih jelas, nilai RMSE bisa dihitung dengan rumus berikut:

$$RMSE = \sqrt{{\frac{1}{N}}{\sum_{ i = 1 } ^ { N }((r_i - \hat{r}_i)^2)}}$$

- N adalah jumlah total prediksi.
- $r_i$ adalah rating yang sebenarnya oleh pengguna untuk item ke-i
- $\hat{r}_i$ adalah rating yang diprediksi oleh model untuk item ke-i

RMSE memberikan gambaran tentang seberapa dekat prediksi sistem dengan nilai sebenarnya. Semakin rendah nilai RMSE, semakin baik kinerja sistem dalam memprediksi preferensi pengguna. Nilai RMSE yang lebih rendah menunjukkan bahwa sistem memberikan prediksi yang lebih akurat dan dekat dengan nilai sebenarnya, sementara nilai RMSE yang lebih tinggi menunjukkan adanya kesalahan prediksi yang lebih besar. Oleh karena itu, RMSE adalah salah satu metrik yang penting dalam mengevaluasi kinerja dan akurasi dari sistem rekomendasi.

Berdasarkan gambar 2 yang menunjukkan hasil matrik RMSE pada metode Collaborative Filtering, didapatkan bahwa nilai rmse pada dataset train konvergen pada sekitar angka 0.203, sedangkan nilai rmse pada dataset test / validation konvergen pada sekitar 0.22, yang mana nilai tersebut tercapai pada sekitar epoch ke 9. 

![cf-rmse](https://github.com/dannid2312/fantastic-octo-computing-machine/assets/123451351/347c7d00-2436-4ae7-8dfb-b7f588a9087a)

(Gambar 2. Metrik RMSE Model Collaborative Filtering)

Berdasarkan gambar 3 yang menunjukkan hasil matrik EMSE pada metode Hybrid, didapatkan bahwa nilai rmse pada dataset train dan dataset test / validation konvergen dinilai 1.55, yang mana nilai tersebut tercapai pada epoch ke 3.

![hybrid-rmse](https://github.com/dannid2312/fantastic-octo-computing-machine/assets/123451351/eb0bde5d-cc67-4d54-a4f7-8ec68feec352)

(Gambar 3. Metrik RMSE Model Hybrid)

## Kesimpulan

Berdasarkan hasil metrik evaluasi RMSE pada kedua model, didapatkan bahwa model Collaborative Filtering menghasilkan data yang lebih akurat karena nilai RMSE yang jauh lebih kecil sebesar 0.2 dibandingkan dengan nilai RMSE model Hybrid sebesar 1.55. Hal tersebut bisa disebabkan oleh fitur genre yang ditambahkan pada model Hybrid justru membuat prediksi model tidak lebih baik dibandingkan model Collaborative Filtering yang memang fokus pada kolom user dan rating dalam permodelan.

Berdasarkan hasil top 10 rekomendasi film yang dihasilkan oleh model Collaborative Filtering dan model Hybrid pada sampel user 117 menunjukkan bahwa user menyukai film Drama dengan variasi sentuhan genre lain seperti Comedy, Thriller, dan Action. Selain itu, bila ditinjau lebih detail, bahwa hasil dari masing-masing model memberikan top 10 rekomendasi yang seluruhnya berbeda, hal itu dapat mendukung fakta terkait perbedaan nilai RMSE antara kedua model yang juga cukup signifikan, sehingga memberikan hasil rekomendasi yang berbeda, meskipun masih masuk ke dalam kelompok genre yang sama.

Pemberian rekomendasi yang akurat dapat menguntungkan kedua pihak baik yaitu perusahaan platform streaming yang mendapatkan pelanggan yang setia dengan film-film yg direkomendasikan dan juga pelanggan yang dapat menikmati film-film yang disukai tanpa kesulitan dalam melakukan pencarian.

## Referensi
[1] Soldo, L., & Schagerl, C. (2023). Impact of the Covid-19 Pandemic on Netflix. MAP Education and Humanities, 3(1), 75–82. [https://doi.org/10.53880/2744-2373.2023.3.1.75](https://mapub.org/ojs/index.php/mapeh/article/view/72)

[2] Carlos A. Gomez-Uribe and Neil Hunt. 2015. The Netflix recommender system: Algorithms, business value, and innovation. ACM Trans. Manage. Inf. Syst. 6, 4, Article 13 (December 2015), 19 pages. DOI: [http://dx.doi.org/10.1145/2843948](https://dl.acm.org/doi/pdf/10.1145/2843948)
