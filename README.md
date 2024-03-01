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
- transmission : tipe transmisi dari mobil bekas, terdiri dari automatic, manual, semi-auto, dan lainnya <br> ![trans](https://github.com/dannid2312/fantastic-octo-computing-machine/assets/123451351/50e14533-a458-4a7e-9818-66e5a1662ab3)
- fuelType : jenis bahan bakar yang digunakan oleh mobil bekas <br> ![fuel](https://github.com/dannid2312/fantastic-octo-computing-machine/assets/123451351/2bbc1959-6eb3-4d50-9fa7-020063e7fe20)
- brand : perusahaan atau brand yang memproduksi mobil <br> ![brand](https://github.com/dannid2312/fantastic-octo-computing-machine/assets/123451351/5bd6cd37-d67f-47ef-86b7-e34d8d99c3c3)

#### Variabel yang termasuk ke dalam numerical features terdiri diri:
- year : tahun pembuatan mobil, terdapat 2 data outlier yaitu tahun 1970 dan 2060, yang kemudian dihilangkan pada tahap data cleansing  <br> ![year](https://github.com/dannid2312/fantastic-octo-computing-machine/assets/123451351/f652aacc-c24a-4182-a427-b787dac8a1b2)
- price : merupakan kolom target dalam melakukan modelling machine learning yang menyatakan harga dalam mobil bekas dalam satuan GBP, dalam proyek kali ini akan diubah ke IDR dengan nilai konversi 1 GBP = 20000 IDR <br> ![price](https://github.com/dannid2312/fantastic-octo-computing-machine/assets/123451351/02a210a5-ada8-4031-9563-9c2c74a0dc81)
- mileage : jarak yang telah ditempuh oleh mobil bekas dalam satuan mile diubah menjadi satuan km dengan konversi 1 mile = 1.6 km <br> ![mileage](https://github.com/dannid2312/fantastic-octo-computing-machine/assets/123451351/9c2df07b-af5e-4487-9807-b247f3df132a)
- tax : nilai pajak terkait dengan administrasi mobil bekas dalam satuan GBP, dalam proyek kali ini akan diubah ke IDR dengan nilai konversi 1 GBP = 20000 IDR <br> ![tax](https://github.com/dannid2312/fantastic-octo-computing-machine/assets/123451351/5f109719-a6e3-488a-aefe-52f02ff7b008)
- mpg : menunjukkan efisiensi mobil untuk bergerak berapa mile per gallon, untuk menyesuaikan pasar indonesia, dikonversi 1 mpg = 0.425 km per liter <br> ![mpg](https://github.com/dannid2312/fantastic-octo-computing-machine/assets/123451351/cda2ee47-11f1-4388-9575-22bb31bec2b5)
- engineSize : volume mesin dalam satuan Liter atau 1000 cc <br> ![engine](https://github.com/dannid2312/fantastic-octo-computing-machine/assets/123451351/029b3ecd-ef04-4500-a179-ef9e925acf54)

### Perbandingan categorical features terhadap kolom target 'price'
- Perbandingan rata-rata price terhadap model menunjukkan data yang variatif, karena setiap model memiliki penamaan yang berbeda berdasarkan brand masing-masing, sehingga tidak dapat diambil kesimpulan pengaruh model terhadap kolom target price. <br> ![model-price](https://github.com/dannid2312/fantastic-octo-computing-machine/assets/123451351/1fd76878-9542-4df0-85ce-ae4a5b343400)
- Perbandingan rata-rata price terhadap transmission menunjukkan bahwa harga mobil dengan transmisi automatic atau semi-auto memiliki harga lebih tinggi daripada mobil dengan transmisi manual. Hal tersebut sesuai karena spare part untuk membuat transmisi automatic atau semi-auto memang lebih mahal dan canggih dibandingkan transmisi manual. <br> ![price-trans](https://github.com/dannid2312/fantastic-octo-computing-machine/assets/123451351/c1cbae2e-bfbb-4a16-9ee2-43342aedc1da)
- Perbandingan rata-rata price terhadap fuelType menunjukkan bahwa harga mobil yang menggunakan bahan bakar diesel atau hybrid lebih mahal dibandingkan mobil yang menggunakan bahan bakar bensin. <br> ![price-fuel](https://github.com/dannid2312/fantastic-octo-computing-machine/assets/123451351/d7d6240c-486a-41d7-aedb-d95f15bf149b)
- Perbandingan rata-rata price terhadap brand menunjukkan bahwa harga mobil mercedes, bmw, dan audi, cenderung memiliki harga lebih mahal dibandingkan merek mobil lainnya. <br> ![price-brand](https://github.com/dannid2312/fantastic-octo-computing-machine/assets/123451351/604b8a66-997e-4b6b-b83f-08f31cf50ebe)

### Perbandingan numerical features terhadap kolom target 'price'
- Korelasi antara year dengan kolom target price sebesar 0.48 menunjukkan bahwa semakin besar nilai tahun atau semakin muda umur mobil bekas, maka harga mobil bekasnya semakin mahal.
- Korelasi antara mileage dengan kolom target price sebesar -0.39 menunjukkan bahwa semakin besar jarak yang sudah ditempuh, maka harga mobil bekas semakin murah.
- Korelasi antara tax dengan kolom target price sebesar 0.12 menunjukkan korelasi positif namun tidak terlalu signifikan terhadap harga mobil bekas.
- korelasi antara mpg dengan kolom target price sebesar -0.32 menunjukkan bahwa semakin besar atau semakin hemat kebutuhan bahan bakarnya, maka harga mobil bekas semakin mahal.
- Korelasi antar engineSize dengan kolom target price sebesar 0.59 menunjukkan bahwa semakin besar kapasitas mesin, maka semakin mahal harga mobil bekasnya.
<br> ![price-numvar](https://github.com/dannid2312/fantastic-octo-computing-machine/assets/123451351/4c3bd10c-bcaf-482f-a172-4b299d12b65f) <br>

Berdasarkan uraian diatas, disimpulkan bahwa dataset memiliki korelasi yang bersesuaian dengan fakta di lapangan, dimana harga mobil bekas berkorelasi positif dengan tahun pembuatan (year), pajak (tax), dan kapasitas mesin (engineSize), sedangkan jarak tempuh (mileage) dan konsumsi bahan bakar (mpg) memiliki korelasi negatif. Variabel yang berkontribusi besar terhadap harga mobil bekas adalah tahun pembuatan (year) dan kapasitas mesin (engineSize), sedangkan variabel yang memberikan pengaruh paling kecil adalah nilai pajak (tax).

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

