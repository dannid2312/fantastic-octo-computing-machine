# Laporan Proyek Machine Learning - Danni Dwicahyo

## Domain Proyek

Kebutuhan akan mobil di Indonesia mengalami peningkatan signifikan dari tahun ke tahun. Hal ini didorong oleh meningkatnya perekonomian di masyarakat, yaitu meningkatnya jumlah penduduk usia produktif (15-64 tahun) yang mencapai 75.94% pada tahun 2022 meningkatkan permintaan mobilitas individual dan meningkatnya rata-rata upah / gaji bersih sebesar 12.21% pada tahun 2022 dan dibarengi dengan pertumbuhan ekonomi sebesar 5.31% atau meningkat 1.61% dari tahun 2021 ([BPS, 2023](https://www.bps.go.id/id/publication/2023/09/26/0e70a59af34c8964e775f4b7/statistik-indonesia-dalam-infografis-2023.html)). Serta adanya pembangunan infrastruktur jalan tol yang masif di berbagai daerah mencapai 2618 km hingga pertengahan Januari 2024 ([BPJT,2024](https://bpjt.pu.go.id/berita/jalan-tol-beroperasi-di-indonesia-telah-mencapai-2816-km#:~:text=Sejak%20tahun%201978%20hingga%20pertengahan,Pulau%20Bali%2010%2C07%20Km.)) memperlancar mobilitas dan meningkatkan minat masyarakat terhadap penggunaan mobil.

Meskipun kebutuhan mobil meningkat, harga mobil baru yang terus meningkat menjadi hambatan bagi banyak masyarakat. Hal ini mendorong peralihan ke pasar mobil bekas yang menawarkan alternatif kendaraan dengan harga lebih terjangkau. Bisnis mobil bekas di Indonesia berkembang pesat, menandakan tingginya permintaan dan peluang pasar. Di tengah persaingan yang ketat, perusahaan dituntut untuk menyediakan mobil bekas dengan harga yang kompetitif dan menawarkan nilai terbaik bagi konsumen (value for money). 

Penggunaan machine learning dalam menentukan harga yang tepat untuk sebuah mobil bekas memiliki potensi besar untuk membantu pembeli, penjual, dan industri mobil bekas. Program ini dapat membantu mempermudah proses jual beli mobil bekas dan meningkatkan efisiensi pasar sehingga membuat persaingan usaha menjadi lebih sehat.

## Business Understanding

Penggunaan machine learning dapat berkontribusi pada pasar mobil bekas di Indonesia yang sedang berkembang pesat. Salah satu poin penting yang jadi perhatian di pasar mobil bekas adalah penentuan harga mobil yang masih diperkirakan secara manual dapat membuat mobil dijual terlalu murah yang dapat merugikan penjual atau terlalu mahal hingga membuat penjualan tidak laku. Penentuan harga mobil bekas dengan bantuan machine learning diharapkan dapat membantu pembeli maupun penjual untuk menentukan harga mobil bekas yang sesuai dengan harga pasaran.

### Problem Statements

Berdasarkan kondisi yang telah diuraikan sebelumnya, sistem prediksi harga mobil bekas dikembangkan untuk menjawab permasalahan sebagai berikut:
- Ketersediaan data mobil bekas yang lengkap, berkualitas, dan terstruktur masih terbatas, apakah ada data yang bisa digunakan?
- Penentuan harga mobil bekas dipengaruhi oleh banyak faktor, diantaranya merek, model, tahun pembuatan, jarak tempuh, dan faktor lainnya, faktor apa yang paling berpengaruh terhadap penentuan harga mobil bekas?
- Berapa harga mobil bekas dengan konfigurasi faktor tertentu?  

### Goals

Untuk  menjawab permasalahan tersebut, sistem machine learning predictive modelling dikembangkan dengan tujuan atau goals sebagai berikut:
- Menggunakan data harga mobil bekas yang lengkap, berkualitas, dan terstruktur dari open source sebagai benchmark awal dalam model
- Mengetahui faktor-faktor yang paling mempengaruhi harga mobil bekas
- Membuat model machine learning yang dapat memprediksi harga mobil bekas seakurat mungkin menggunakan serangkaian konfigurasi tertentu

### Solution Statement

Untuk mencapai tujuan atau goals tersebut, dilakukan beberapa tahapan sebagai berikut:
- Dengan menggunakan data harga mobil bekas yang terdapat di kaggle, dilakukan penyesuaikan dengan mengubah harga dari sebelumnya GBP ke IDR dengan konversi 1 BGP = 20000 IDR
- Dilakukan cleansing terhadap data-data yang belum terstruktur dan menghilangkan outliers untuk meningkatkan kualitas distribusi data
- Dilakukan multivariate analysis untuk menentukan faktor apa saja yang diduga paling mempengaruhi harga mobil bekas
- Dilakukan one-hot-encoding terhadap categorical features yang termasuk faktor yang diduga paling mempengaruhi harga mobil bekas
- Dilakukan standardization terhadap numerical features yang termasuk faktor yang diduga paling mempengaruhi harga mobil bekas
- Dilakukan percobaan permodelan dengan K-Nearest Neighbor, Random Forest, dan Boosting Algorithm untuk menentukan model mana yang memiliki akurasi paling tinggi 

## Data Understanding

Data yang digunakan pada proyek kali ini adalah Car Price Prediction dataset yang diunduh dari website [Kaggle](https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes/data). Dataset ini memiliki 108540 baris dan terdiri dari 10 kolom yaitu model, year, price, transmission, mileage, fuelType, tax, mpg, engineSize, dan brand. Dataset masih perlu dilakukan beberapa penyesuaian berupa data cleansing dan data preparation untuk menghasilkan dataset yang berkualitas. Beberapa baris dataset dihilangkan karena terdapat kolom yang kosong, serta dihilangkan data-data yang merupakan outlier, sehingga menghasilkan data akhir yang dilanjutkan pada tahap permodelan sebanyak 65039 baris.

### Variabel-variabel pada Car Price Prediction dataset adalah sebagai berikut:
#### Variabel yang termasuk ke dalam categorical features terdiri diri:
- model : nama spesifik dari model mobil yang dikeluarkan oleh brand tertentu
- transmission : tipe transmisi dari mobil bekas, terdiri dari automatic, manual, triptonic, dan lainnya
- fuelType : jenis bahan bakar yang digunakan oleh mobil bekas
- brand : perusahaan atau brand yang memproduksi mobil

#### Variabel yang termasuk ke dalam numerical features terdiri diri:
- year : tahun pembuatan mobil, terdapat 2 data outlier yaitu tahun 1970 dan 2060, yang kemudian dihilangkan pada tahap data cleansing  <br> ![year](https://github.com/dannid2312/fantastic-octo-computing-machine/assets/123451351/f652aacc-c24a-4182-a427-b787dac8a1b2)
- price : merupakan kolom target dalam melakukan modelling machine learning yang menyatakan harga dalam mobil bekas dalam satuan GBP, dalam proyek kali ini akan diubah ke IDR dengan nilai konversi 1 GBP = 20000 IDR <br> ![price](https://github.com/dannid2312/fantastic-octo-computing-machine/assets/123451351/02a210a5-ada8-4031-9563-9c2c74a0dc81)
- mileage : jarak yang telah ditempuh oleh mobil bekas dalam satuan mile diubah menjadi satuan km dengan konversi 1 mile = 1.6 km <br> ![mileage](https://github.com/dannid2312/fantastic-octo-computing-machine/assets/123451351/9c2df07b-af5e-4487-9807-b247f3df132a)
- tax : nilai pajak terkait dengan administrasi mobil bekas dalam satuan GBP, dalam proyek kali ini akan diubah ke IDR dengan nilai konversi 1 GBP = 20000 IDR <br> ![tax](https://github.com/dannid2312/fantastic-octo-computing-machine/assets/123451351/5f109719-a6e3-488a-aefe-52f02ff7b008)
- mpg : menunjukkan efisiensi mobil untuk bergerak berapa mile per gallon, untuk menyesuaikan pasar indonesia, dikonversi 1 mpg = 0.425 km per liter <br> ![mpg](https://github.com/dannid2312/fantastic-octo-computing-machine/assets/123451351/cda2ee47-11f1-4388-9575-22bb31bec2b5)
- engineSize : volume mesin dalam satuan Liter atau 1000 cc <br> ![engine](https://github.com/dannid2312/fantastic-octo-computing-machine/assets/123451351/029b3ecd-ef04-4500-a179-ef9e925acf54)

**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data atau exploratory data analysis.

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.

