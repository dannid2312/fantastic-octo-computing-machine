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

#### Menghilangkan outlier
Berdasarkan gambaran boxplot pada numerical features diatas, didapatkan bahwa terdapat beberapa data yang bernilai ekstrim atau outlier, sehingga diperlukan langkah untuk menghilangkan outlier tersebut. Metode IQR digunakan untuk mengidentifikasi outlier yang berada di luar Q1 dan Q3. Nilai apa pun yang berada di luar batas ini dianggap sebagai outlier. Outliers yang diidentifikasi oleh boxplot (disebut juga “boxplot outliers”) didefinisikan sebagai data yang nilainya 1.5 QR di atas Q3 atau 1.5 QR di bawah Q1. Setelah outlier dihilangkan, dataset menjadi tersisa 65039 baris dan 10 kolom.
```
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR=Q3-Q1
df=df[~((df<(Q1-1.5*IQR))|(df>(Q3+1.5*IQR))).any(axis=1)]
```

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

Berdasarkan uraian diatas, disimpulkan bahwa dataset memiliki korelasi yang bersesuaian dengan fakta di lapangan, dimana harga mobil bekas berkorelasi positif dengan tahun pembuatan (year), pajak (tax), dan kapasitas mesin (engineSize), sedangkan jarak tempuh (mileage) dan konsumsi bahan bakar (mpg) memiliki korelasi negatif terhadap harga mobil bekas. Variabel yang berkontribusi besar terhadap harga mobil bekas adalah kapasitas mesin (engineSize) kemudian tahun pembuatan (year). Variabel yang memberikan kontribusi paling kecil adalah nilai pajak (tax), sehingga kolom tax akan dihilangkan dari dataset dalam melakukan permodelan.

## Data Preparation
Pada bagian ini akan dilakukan tiga tahap persiapan data, yaitu:

### Encoding fitur kategori
Untuk melakukan proses encoding fitur kategori, salah satu teknik yang umum dilakukan adalah teknik one-hot-encoding. Library scikit-learn menyediakan fungsi ini untuk mendapatkan fitur baru yang sesuai sehingga dapat mewakili variabel kategori. Dataset memiliki empat variabel kategori, yaitu model, transmission, fuelType, dan brand. Proses encoding dilakukan dengan fitur get_dummies. Proses encoding dilakukan untuk mengubah variabel menjadi nilai numerik sehingga lebih mudah diproses oleh model.
```
from sklearn.preprocessing import  OneHotEncoder
for col in categorical_features:
  df = pd.concat([df, pd.get_dummies(df[col], prefix=col)],axis=1)
df.drop(categorical_features, axis=1, inplace=True)
```

### Pembagian dataset dengan fungsi train_test_split dari library sklearn
Sebelum melakukan permodelan, perlu dilakukan pembagian antara dataset untuk dilatih (train) pada model dan dataset untuk menguji (test) performa model. Dalam project ini akan digunakan proporsi pembagian sebesar 80:20 dengan fungsi train_test_split dari sklearn. Pembagian data latih (train) dan data uji (test) perlu dilakukan untuk melakukan transformasi terpisah pada masing-masing dataset. Transformasi terpisah dilakukan agar tidak terjadi kebocoran data uji (test) saat melatih model dengan data latih (train).
```
from sklearn.model_selection import train_test_split
X = df.drop(["price"],axis =1)
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)
```

### Standarisasi
Algoritma machine learning memiliki performa lebih baik dan konvergen lebih cepat ketika dimodelkan pada data dengan skala relatif sama atau mendekati distribusi normal. Proses scaling dan standarisasi membantu untuk membuat fitur data menjadi bentuk yang lebih mudah diolah oleh algoritma. Dalam proyek ini standarisasi dilakukan dengan menggunakan teknik StandarScaler dari library Scikitlearn. StandardScaler melakukan proses standarisasi fitur dengan mengurangkan mean (nilai rata-rata) kemudian membaginya dengan standar deviasi untuk menggeser distribusi. Penerapan standarisasi dengan StandardScaler dilakukan secara terpisah pada masing-masing dataset latih (train) dan dataset uji (test).
```
numerical_features = ['year', 'mileage', 'mpg', 'engineSize']
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train[numerical_features])
X_train[numerical_features] = scaler.transform(X_train.loc[:, numerical_features])
X_train[numerical_features].head()
```

## Modeling
Pada proyek ini, model machine learning yang digunakan adalah dengan tiga algoritma. Performa masing-masing algoritma akan dilakukan evaluasi untuk menentukan algoritma mana yang memberikan hasil prediksi terbaik. Ketiga algoritma yang akan digunakan, terdiri dari:

### K-Nearest Neighbor
Algoritma KNN menggunakan ‘kesamaan fitur’ untuk memprediksi nilai dari setiap data yang baru dengan membandingkan seberapa dekat data baru tersebut dengan data latih. KNN bekerja dengan membandingkan jarak satu sampel ke sampel pelatihan lain dengan memilih sejumlah k tetangga terdekat (dengan k adalah sebuah angka positif). Kelebihan KNN adalah cara kerjanya yang mudah untuk dipahami dan diimplementasikan. KNN biasanya memiliki kinerja lebih baik pada data dengan dimensi rendah. Kekurangan KNN adalah dianggap lemah pada data berdimensi tinggi dan sensitif terhadap outlier.
```
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
knn = KNeighborsRegressor(n_neighbors=10)
knn.fit(X_train, y_train)
```

### Random Forest
Algoritma Random Forest merupakan salah satu model machine learning yang termasuk ke dalam kategori ensemble (group) learning, yaitu model prediksi yang terdiri dari beberapa model dan bekerja secara bersama-sama. Pada model ensemble, setiap model harus membuat prediksi secara independen. Kemudian, prediksi dari setiap model ensemble ini digabungkan untuk membuat prediksi akhir. Random forest berisi beberapa model decision tree yang masing-masing memiliki hyperparameter yang berbeda dan dilatih pada beberapa bagian (subset) data yang berbeda juga. Kelebihan Random Forest adalah kemampuannya dalam mengatasi dataset dengan dimensi yang tinggi serta mampu menangani outlier. Kekurangan dari Random Forest adalah membutuhkan waktu lebih lama untuk pelatihan model dibandingkan KNN karena memiliki model yang lebih kompleks daripada KNN, serta modelnya sulit untuk diinterpretasikan.
```
from sklearn.ensemble import RandomForestRegressor
RF = RandomForestRegressor(n_estimators=50, max_depth=16, random_state=55, n_jobs=-1)
RF.fit(X_train, y_train)
```

### Boosting Algorithm
Algoritma yang menggunakan teknik boosting bekerja dengan membangun model dari data latih. Kemudian ia membuat model kedua yang bertugas memperbaiki kesalahan dari model pertama. Model ditambahkan sampai data latih terprediksi dengan baik atau telah mencapai jumlah maksimum model untuk ditambahkan. Caranya adalah dengan menggabungkan beberapa model sederhana dan dianggap lemah (weak learners) sehingga membentuk suatu model yang kuat (strong ensemble learner). Metode Boosting Algorithm yang digunakan pada proyek ini adalah AdaBoost (adaptive boosting). Kelebihan algoritma ini adalah mampu bekerja pada data berdimensi tinggi maupun data berdimensi rendah, serta mampu menangani model yang overfitting. Kekurangan dari algoritma ini adalah sensitif terhadap outlier, membutuhkan waktu lebih lama untuk pelatihan model, serta modelnya sulit untuk diinterpretasikan.
```
from sklearn.ensemble import AdaBoostRegressor
boosting = AdaBoostRegressor(learning_rate=0.05, random_state=55)
boosting.fit(X_train, y_train)
```

## Evaluation
Matrik yang digunakan untuk mengevaluasi ketiga model yang dikembangkan adalah matrik mean-squared-error (mse) dan matrik accuracy (acc).

### Mean Squared Error (MSE)
MSE adalah metrik yang digunakan untuk mengukur rata-rata kesalahan kuadrat antara nilai prediksi dan nilai aktual. Nilai MSE yang lebih kecil menunjukkan model yang lebih akurat, sehingga MSE bernilai 0 menunjukkan model yang sempurna. Perhitungan MSE dilakukan dengan menghitung selisih antara nilai aktual dan nilai prediksi untuk setiap data, kemudian melakukan fungsi kuadrat pada setiap selisih data. Nilai kuadrat dari setiap selisih data tersebut kemudian dijumlahkan dan dibagi dengan jumlah datanya. Berdasarkan metrik mse, didapatkan nilai error algoritma boosting enam kali lipat lebih besar daripada nilai error model KNN dan Random Forest.  <br> ![mse](https://github.com/dannid2312/fantastic-octo-computing-machine/assets/123451351/cb4113d8-84b9-4d15-b1a6-0e61fbdd8fb4)

### Accuracy (ACC)
Akurasi adalah metrik yang digunakan untuk mengukur proporsi data yang diklasifikasikan dengan benar. Nilai akurasi yang lebih tinggi menunjukkan model yang lebih akurat, sehingga nilai akurasi 100% menunjukkan model yang sempurna. Perhitungan Accuracy dilakukan dengan menghitung jumlah nilai prediksi yang benar dengan nilai aktual, kemudian membagi jumlah prediksi yang benar dengan keseluruh jumlah datanya. Berdasarkan metrik akurasi, didapatkan bahwa nilai akurasi KNN adalah 94.71%, sedikit lebih unggul dibandingkan akurasi Random Forest sebesar 94.25%, sedangkan akurasi algoritma boosting hanya sekitar 70% <br> ![acc](https://github.com/dannid2312/fantastic-octo-computing-machine/assets/123451351/1a58d08a-0a46-47fb-961f-1a4834d2d0d8)

## Hasil Evaluasi
- Data harga mobil bekas yang digunakan bisa menjadi benchmark awal untuk membangun model prediksi harga mobil bekas di pasar Indonesia ke depannya.
- Berdasarkan matriks korelasi, didapatkan bahwa variabel numerik yang paling mempengaruhi harga mobil bekas adalah kapasitas mesin (engineSize) dan tahun pembuatan mobil (year). Semakin besar kapasitas mesin dan semakin muda umur mobilnya, maka semakin mahal harga jual mobil bekasnya.
- Berdasarkan hasil evaluasi, didapatkan bahwa model yang paling baik untuk memprediksi harga mobil bekas bisa menggunakan model KNN maupun model Random Forest yang memiliki akurasi 94%. 

