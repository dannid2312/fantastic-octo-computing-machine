# Laporan Proyek Machine Learning - Danni Dwicahyo

## Project Overview

Selama pandemi COVID-19, terjadi peningkatan signifikan dalam permintaan hiburan di rumah, yang mencakup peningkatan penggunaan platform streaming. Pembatasan sosial dan penutupan bioskop telah mendorong masyarakat untuk beralih ke media digital untuk hiburan [[1](https://mapub.org/ojs/index.php/mapeh/article/view/72)]. Platform streaming seperti Netflix menawarkan ribuan film kepada penggunanya yang dapat membuat pengguna kesulitan menemukan film yang sesuai dengan selera mereka. Netflix berhasil menjawab permasalahan tersebut dengan menggunakan machine learning untuk meningkatkan retensi pelanggan dan kepuasan pengguna dengan melakukan pembangunan sistem rekomendasi film yang efektif [[2](https://dl.acm.org/doi/pdf/10.1145/2843948)]. Dengan mengadopsi pendekatan yang terinspirasi dari praktik terbaik yang ditunjukkan oleh Netflix, movie recommendation system dapat meningkatkan kemampuan dalam menyajikan rekomendasi film yang relevan dan memuaskan bagi pengguna.

Proyek movie recommendation system memiliki manfaat yang sangat besar, yaitu: 
- Meningkatkan kepuasan pengguna: Sistem rekomendasi yang akurat dan personal dapat membantu pengguna menemukan film yang sesuai dengan selera mereka dengan lebih mudah dan cepat. Hal ini dapat meningkatkan kepuasan pengguna dan mendorong mereka untuk terus menggunakan platform streaming atau situs web rekomendasi film.
- Meningkatkan efisiensi pencarian film: Sistem rekomendasi dapat membantu pengguna menghemat waktu dan tenaga dalam mencari film yang ingin ditonton. Hal ini dapat meningkatkan efisiensi dan produktivitas pengguna.
- Meningkatkan pendapatan platform: Sistem rekomendasi dapat membantu platform streaming dan situs web rekomendasi film meningkatkan pendapatan mereka dengan mendorong pengguna untuk menonton lebih banyak film.

### Referensi
[1] Soldo, L., & Schagerl, C. (2023). Impact of the Covid-19 Pandemic on Netflix. MAP Education and Humanities, 3(1), 75–82. [https://doi.org/10.53880/2744-2373.2023.3.1.75](https://mapub.org/ojs/index.php/mapeh/article/view/72)
[2] Carlos A. Gomez-Uribe and Neil Hunt. 2015. The Netflix recommender system: Algorithms, business value,
and innovation. ACM Trans. Manage. Inf. Syst. 6, 4, Article 13 (December 2015), 19 pages.
DOI: [http://dx.doi.org/10.1145/2843948](https://dl.acm.org/doi/pdf/10.1145/2843948)

## Business Understanding

Sistem rekomendasi film merupakan alat penting bagi bisnis di industri hiburan. Sistem ini memberikan manfaat bagi pengguna dengan menghemat waktu dan tenaga dalam mencari film yang sesuai selera, serta membantu mereka menemukan film baru yang menarik. Bagi bisnis, sistem ini meningkatkan engagement dan konversi pengguna, serta membantu monetisasi platform dengan lebih efektif. Meskipun terdapat beberapa tantangan, namun seiring dengan perkembangan teknologi AI dan machine learning memungkinkan sistem ini terus berkembang dan menjadi lebih personal dan akurat. Dengan memanfaatkan peluang dan mengatasi tantangan tersebut, sistem rekomendasi film akan terus memainkan peran penting dalam industri hiburan.

### Problem Statements

Berdasarkan kondisi yang telah diuraikan sebelumnya, sistem rekomendasi film dikembangkan untuk menjawab permasalahan sebagai berikut:
- Diperlukan data yang lengkap, akurat, dan/atau tidak mengandung bias tentang rating film dari banyak penonton atau pelanggan.
- Diperlukan sistem rekomendasi film yang dapat memberikan rekomendasi kepada pengguna baru yang belum memberikan banyak data tentang preferensinya.
- Diperlukan sistem rekomendasi film yang dapat memberikan kesempatan rekomendasi kepada film serupa yang belum pernah mendapatkan rating.

### Goals

Untuk menjawab permasalahan tersebut, sistem rekomendasi film dikembangkan dengan tujuan atau goals sebagai berikut:
- Menggunakan data rating dari film lama untuk mendapatkan data yang lengkap, serta melakukan data preparation yang baik untuk menghilangkan data yang tidak berkualitas atau data yang bias.
- Menggunakan metode collaborative filtering untuk memberikan rekomendasi film baik kepada pengguna baru maupun pengguna lama yang telah terpersonalisasi.
- Menggunakan metode content based filtering untuk memberikan rekomendasi film berdasarkan genre film yang sudah pernah ditonton sebelumnya.

### Solution Statement
Untuk mencapai masing-masing tujuan atau goals tersebut, dilakukan tahapan sebagai berikut:
- Menggunakan database film yang memiliki data rating yang lengkap dan melimpah untuk kemudian dilakukan univariate analysis, preprocessing, dan data preparation berupa encoding dan standardization untuk mendapatkan data yang berkualitas.
- Melakukan permodelan dengan metode collaborative filtering untuk memproses model pengguna dan rating filmnya dengan menggunakan teknik embedding dan perkalian dot product untuk memberikan rekomendasi film baik terhadap pengguna baru maupun pengguna lama berdasarkan skor kecocokan yang dihitung dengan fungsi aktivasi sigmoid.
- Melakukan permodelan dengan motede content based filtering untuk memproses model film dan genre filmnya dengan bantuan fungsi tfidfvectorizer dan cosine_similarity dari library sklearn untuk memberikan rekomendasi film berdasarkan kemiripannya dengan film yang sudah pernah ditonton sebelumnya.

## Data Understanding
Data yang digunakan pada proyek kali ini adalah Movies & Ratings for Recommendation System dataset yang diunduh dari website [Kaggle](https://www.kaggle.com/datasets/nicoletacilibiu/movies-and-ratings-for-recommendation-system/code). Dataset ini terdiri dari dua file csv berupa movies.csv dan ratings.csv. File movies.csv merupakan dataset tentang database film yang memiliki 9742 baris yang terdiri dari tiga kolom yaitu movieId, title, dan genres. File ratings.csv merupakan dataset tentang rating film yang memiliki 100836 baris yang terdiri dari empat kolom, yaitu userId, movieId, rating, dan timestamp. Dataset masih perlu dilakukan beberapa penyesuaian dalam tahap data preparation untuk menghasilkan dataset yang berkualitas. Kedua dataset yang tersedia kemudian digabungkan menjadi satu dataset dengan menggunakan movieId sebagai acuan penggabungan. Hasil akhir dari penggabungan dataset terdiri dari 100836 baris dan 6 kolom, serta tidak terdapat missing values.

```
<class 'pandas.core.frame.DataFrame'>
Int64Index: 100836 entries, 0 to 100835
Data columns (total 6 columns):
 #   Column     Non-Null Count   Dtype  
---  ------     --------------   -----  
 0   movieId    100836 non-null  int64  
 1   title      100836 non-null  object 
 2   genres     100836 non-null  object 
 3   userId     100836 non-null  int64  
 4   rating     100836 non-null  float64
 5   timestamp  100836 non-null  int64  
dtypes: float64(1), int64(3), object(2)
memory usage: 5.4+ MB
```

### Variabel-variabel yang terdapat pada dataset gabungan antara file movies.csv dan ratings.csv adalah sebagai berikut:
- movieId: menunjukkan nomor identitas atau index dari suatu film, merupakan kolom yang menjadi acuan dalam penggabungan antara dua dataset.
- title: menunjukkan judul film
- genres: menunjukkan genre / ragam / tipe film
- userId: menunjukkan nomor identitas atau index dari pengguna atau penonton film yang memberikan penilaian (rating)
- rating: menunjukkan nilai yang diperoleh dari pengguna atau penonton film
- timestamp: menunjukkan waktu pengguna melakukan penilaian terhadap suatu film

### Univariate Analysis
Dari data movies.csv, dari jumlah film sebanyak 9742, terbagi kedalam 951 genre yang berbeda. Namun bila dilihat lebih detail, setiap film bisa terdiri dari berbagai genre, sehingga perlu dilakukan pemrosesan lebih lanjut terhadap kolom genre.

![before](https://github.com/dannid2312/fantastic-octo-computing-machine/assets/123451351/faaa737f-7d8d-49d4-97d0-dadee07065f0)

Pertama, adalah mengubah tanda "|" dan mengganti dengan " ", lalu menghilangkan tanda "-", dan merubah film yang tidak bergenre "non genres listed" dengan "None". Rangkaian tersebut diperlukan untuk memudahkan nantinya ketika melakukan ekstraksi fitur genre dengan fungsi TF-IDF Vectorizer.

![after](https://github.com/dannid2312/fantastic-octo-computing-machine/assets/123451351/1f74bc59-57b0-4089-b07d-b52892feb7a1)

Berdasarkan pengecekan setiap variabel diatas, didapatkan bahwa dari 9742 film yang tersedia di database, hanya 9724 film yang sudah diberikan rating, sehingga masih ada 18 film yang belum pernah diberikan penilaian. Kemudian, pengguna atau penonton (user) yang melakukan penilaian adalah sebanyak 610 orang, dengan rentang penilaian antara 0.5 sampai 5.0.
```
Jumlah film berdasarkan movie ID 9742
Jumlah user yang memberikan penilaian 610
Jumlah film yang dinilai berdasarkan movie ID 9724
Jumlah nilai minimum rating 0.5
Jumlah nilai minimum rating 5.0
```

![box-rating](https://github.com/dannid2312/fantastic-octo-computing-machine/assets/123451351/3fe38a57-94ab-4451-9a0b-21b5904b5ffc)

Berdasarkan grafik, nilai yang paling banyak diberikan adalah 4.0, sedangkan nilai paling sedikit diberikan adalah 0.5.

![bar-rating](https://github.com/dannid2312/fantastic-octo-computing-machine/assets/123451351/d7222b3e-63f1-4046-acd6-e369b9a125c6)

## Data Preparation
Pada bagian ini akan dilakukan tiga tahap persiapan data, yaitu:
### Encoding fitur kategori
Encoding dilakukan terhadap variabel userId dan movieId. Meskipun kedua variabel tersebut sudah memiliki nilai integer, namun encoding tetap dilakukan untuk mempermudah model dalam melakukan pelatihan sehingga konvergensi lebih mudah dicapai. Encoding dilakukan secara manual dengan menggunakan fungsi enumerate().

### Standarisasi
Algoritma machine learning memiliki performa lebih baik dan konvergen lebih cepat ketika dimodelkan pada data dengan skala relatif sama atau mendekati distribusi normal. Proses scaling dan standarisasi membantu untuk membuat fitur data menjadi bentuk yang lebih mudah diolah oleh algoritma. Pada proyek ini standarisasi yang dilakukan adalah dengan menggunakan min max scaling untuk mengubah variabel rating dari yang semula memiliki skala [0,5] menjadi skala [0,1].

### Pembagian dataset dengan fungsi train_test_split dari library sklearn
Sebelum melakukan permodelan, perlu dilakukan pembagian antara dataset untuk dilatih (train) pada model dan dataset untuk menguji (test) performa model. Dalam project ini akan digunakan proporsi pembagian sebesar 80:20 secara manual dengan melakukan split terhadap dataframe yang sebelumnya sudah diacak.

## Modeling
### Model Development: Collaborative Filtering
Pada tahap ini, model menghitung skor kecocokan antara pengguna dan film dengan teknik embedding. Pertama, kita melakukan proses embedding terhadap data user dan film. Selanjutnya, lakukan operasi perkalian dot product antara embedding user dan film. Selain itu, kita juga dapat menambahkan bias untuk setiap user dan film. Skor kecocokan ditetapkan dalam skala [0,1] dengan fungsi aktivasi sigmoid. Permodelan dilakukan dengan membuat class RecommenderNet dengan keras Model class yang disesuaikan dengan movie recommendation system. Model ini menggunakan Binary Crossentropy untuk menghitung loss function, Adam (Adaptive Moment Estimation) sebagai optimizer, dan root mean squared error (RMSE) sebagai metrics evaluation. Berikut ini adalah contoh hasil rekomendasi dengan metode collaborative filtering:

![collaborative](https://github.com/dannid2312/fantastic-octo-computing-machine/assets/123451351/141d6554-aec0-4e25-bef0-164a20d8eecc)

### Model Development: Content Based Filtering
Dalam membuat model content based filtering, dataset yang digunakan akan berfokus pada file movies.csv. Langkah pertama yang dilakukan adalah melakukan ekstraksi fitur dari kolom genres yang akan digunakan untuk mengukur derajat kesamaan antar film satu dengan film lainnya dengan menggunakan fungsi tfidfvectorizer() dari library sklearn. Dari tahapan tersebut, genre yang diawal berjumlah 951 variasi menjadi hanya 20 variasi genre yang terdiri dari 'action', 'adventure', 'animation', 'children', 'comedy', 'crime', 'documentary', 'drama', 'fantasy', 'filmnoir', 'horror', 'imax', 'musical', 'mystery', 'none', 'romance', 'scifi', 'thriller', 'war', dan 'western'.

Langkah selanjutnya adalah mengubah vektor yang dihasilkan dari tfidf menjadi ke dalam bentuk matriks dengan melakukan fit dan transformasi, serta fungsi todense(). Matriks yang dihasilkan akan menunjukkan korelasi antara film dengan genre filmnya. Kemudian, menghitung derajat kesamaan antara satu film dengan film lainnya untuk menghasilkan kandidat film yang akan direkomendasikan berdasarkan kesamaan genre filmnya dengan menggunakan fungsi cosine_similarity dari library sklearn. Sehingga didapatkan hasil akhir sebagai berikut:

![similarity](https://github.com/dannid2312/fantastic-octo-computing-machine/assets/123451351/358649c0-0c7c-43c1-8135-a59fe22daa4c)

Berikut ini adalah contoh hasil rekomendasi dengan metode content based filtering:

![content1](https://github.com/dannid2312/fantastic-octo-computing-machine/assets/123451351/dc974993-5366-4b4b-a80e-1a953f1af0aa)
![content2](https://github.com/dannid2312/fantastic-octo-computing-machine/assets/123451351/aa417abe-6529-4d07-91a4-5898856da2db)

### Collaborative Filtering vs Content Based Filtering
Berdasarkan hasil rekomendasi yang dihasilkan dari metode collaborative filtering dan content based filtering, cukup jelas terlihat bahwa hasil rekomendasi yang diberikan oleh collaborative filtering lebih bervariasi dibandingkan dengan content based filtering yang terpaku pada genre-genre tertentu. Hal tersebut memberikan keuntungan untuk penyedia jasa streaming agar film-film yang direkomendasikan lebih bervariasi sehingga kedepannya bisa lebih banyak film yang diexplorasi oleh pengguna, begitupun sebaliknya pengguna mendapatkan variasi film yang lebih menarik dan tidak mudah bosan karena menonton film yang serupa terus menerus.

Secara umum, metode collaborative filtering memberikan keuntungan berupa rekomendasi lintas domain, yang berarti dapat digunakan untuk merekomendasikan item dari berbagai kategori atau jenis karena tidak hanya bergantung pada preferensi pengguna sendiri. Collaborative filtering juga dianggap  dapat memberikan rekomendasi yang lebih akurat dan dapat memberikan rekomendasi untuk pengguna baru karena mempertimbangkan preferensi pengguna lain. Namun kekurangan dari collaborative filtering adalah cenderung memberikan rekomendasi yang populer saja sehingga mengabaikan preferensi individu yang unik. Selain itu, collaborative filtering juga sangat bergantung pada data interaksi antara user dan item, sebagai contoh dalam proyek ini adalah rating film, sehingga metode ini akan sulit diimplementasikan bila data masih minim.

Sedangkan untuk metode content based filtering, secara umum memiliki keuntungan berupa personalisasi yang kuat karena rekomendasi didasarkan pada preferensi individual pengguna dan tidak terlalu dipengaruhi oleh popularitas item karena rekomendasi didasarkan pada kesamaan fitur antara item yang sudah disukai pengguna dan item yang akan direkomendasikan. Namun kekurangan dari metode ini adalah adanya overfitting dimana rekomendasi yang diberikan terlalu mirip dengan item yang telah disukai pengguna sebelumnya dan mungkin gagal untuk mengungkapkan item baru atau mengeksplorasi minat yang berbeda dari pengguna.

## Evaluation
Root Mean Square Error (RMSE) adalah salah satu metrik evaluasi yang umum digunakan untuk mengukur tingkat kesalahan prediksi dalam konteks sistem rekomendasi. Metrik ini mengukur akurasi dari prediksi yang dihasilkan oleh sistem terhadap nilai sebenarnya yang diberikan oleh pengguna. Nilai RMSE dihitung dari selisih antara nilai sebenarnya dan nilai prediksi untuk setiap item dalam dataset, kemudian melakukan kuadrat terhadap masing-masing selisih. Nilai RMSE adalah akar dari rata-rata kuadrat selisih tersebut.

RMSE memberikan gambaran tentang seberapa dekat prediksi sistem dengan nilai sebenarnya. Semakin rendah nilai RMSE, semakin baik kinerja sistem dalam memprediksi preferensi pengguna. Nilai RMSE yang lebih rendah menunjukkan bahwa sistem memberikan prediksi yang lebih akurat dan dekat dengan nilai sebenarnya, sementara nilai RMSE yang lebih tinggi menunjukkan adanya kesalahan prediksi yang lebih besar. Oleh karena itu, RMSE adalah salah satu metrik yang penting dalam mengevaluasi kinerja dan akurasi dari sistem rekomendasi.

Dari hasil permodelan dengan metode collaborative filtering, didapatkan hasil yang sangat memuaskan dengan nilai RMSE kurang dari 0.2 pada dataset train maupun dataset validation. Nilai tersebut menyatakan bahwa data yang digunakan memberikan hasil rekomendasi yang akurat dan terpersonalisasi dengan baik terhadap pengguna sehingga menghasilkan rekomendasi yang dianggap relevan.

![metrik](https://github.com/dannid2312/fantastic-octo-computing-machine/assets/123451351/4d287b2f-597f-48d7-b25c-1c5bbdccb6de)
