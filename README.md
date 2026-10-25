# Multi-Dataset Information Retrieval System (CLI-Based)

Proyek UTS Praktikum Penelusuran Informasi. Aplikasi ini menggabungkan lima dataset teks dan menyediakan pencarian dokumen melalui CLI.

## 1. Dataset

Sistem membaca file CSV dengan kolom `judul` dan `konten` dari struktur berikut:

```text
datasets/
├── etd-usk/data.csv
├── etd-ugm/data.csv
├── kompas/data.csv
├── tempo/data.csv
└── mojok/data.csv
```

Dataset yang disertakan berisi **49.484 dokumen**: ETD USK 10.000, ETD UGM 10.000, Kompas 10.000, Tempo 10.000, dan Mojok 9.484.

## 2. Persyaratan

- Python 3.10 atau lebih baru
- pandas
- scikit-learn
- Whoosh

## 3. Instalasi

Buka terminal pada folder proyek, lalu buat virtual environment.

### Windows

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Jika PowerShell memblokir aktivasi environment, jalankan perintah berikut pada Command Prompt:

```bat
.venv\Scripts\activate.bat
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 4. Menjalankan aplikasi

```bash
python ir_system.py
```

Menu CLI yang tersedia:

```text
=== INFORMATION RETRIEVAL SYSTEM ===
[1] Load & Index Dataset
[2] Search Query
[3] Exit
====================================
```

1. Pilih **1** untuk membaca kelima CSV, melakukan preprocessing, membangun indeks Whoosh, dan membentuk matriks Bag of Words menggunakan `CountVectorizer`. Proses ini cukup dilakukan ulang jika dataset berubah atau indeks ingin dibangun ulang.
2. Pilih **2** untuk memasukkan query. Sistem mengambil dokumen yang cocok secara leksikal dari Whoosh, menghitung cosine similarity terhadap vektor BoW query, lalu menampilkan lima hasil teratas.
3. Pilih **3** untuk keluar.

Contoh query uji:

- `pemahaman konsep motivasi peserta didik matematika`
- `prediksi harga apartemen crime rates jakarta`
- `xiaomi mijia smart audio glasses kacamata pintar`
- `pesawat united airlines kelinci mesin`
- `lagu anime semangat`

## 5. Pipeline sistem

1. **Load dataset:** pandas membaca CSV per 500 baris agar pembacaan file bertahap.
2. **Preprocessing:** case folding, tokenisasi Unicode, dan penghapusan stopword Bahasa Indonesia. Kata negasi seperti `tidak`, `bukan`, `belum`, `jangan`, dan `tak` dipertahankan karena dapat mengubah makna.
3. **Indexing:** Whoosh membuat indeks pada field `title` dan `content`. Pencarian memakai `MultifieldParser` dengan grup OR; field judul diberi bobot 2,0 saat pengambilan kandidat.
4. **Bag of Words:** `CountVectorizer` membentuk matriks frekuensi token. Konfigurasi mencakup `min_df=2`, `max_df=0.98`, dan batas maksimum 200.000 fitur.
5. **Ranking:** cosine similarity menghitung kedekatan query dengan setiap dokumen kandidat yang cocok secara leksikal. Lima skor tertinggi ditampilkan.

Rumus cosine similarity:

\[
\operatorname{cosine}(q,d)=\frac{q\cdot d}{\|q\|\,\|d\|}
\]

## 6. Berkas runtime

Setelah indexing berhasil, sistem menyimpan artefak berikut dalam folder `runtime/`:

- `whoosh_index/`: indeks dokumen Whoosh.
- `count_vectorizer.pkl`: model CountVectorizer yang sudah dilatih.
- `document_vectors.pkl`: matriks BoW sparse untuk seluruh dokumen.
- `index_manifest.json`: jumlah dokumen, ukuran matriks, dan metadata proses.

Berkas runtime dapat dibuat ulang melalui menu **[1] Load & Index Dataset**. Jangan mengedit berkas `.pkl` secara manual.

## 7. Menjalankan notebook

Buka `IR_System_Multi_Dataset.ipynb` dari folder proyek. Notebook berisi sumber kode, pemeriksaan awal dataset, dan sel untuk memulai menu CLI. Pastikan working directory notebook berada di folder proyek.

## 8. Catatan metodologis

- Cosine similarity dihitung menggunakan frekuensi token dari Bag of Words, bukan TF-IDF.
- Whoosh dipakai untuk indexing dan mengambil semua dokumen yang cocok dengan istilah query; cosine similarity menentukan urutan akhir hasil.
- Karena tidak menggunakan stemming atau model semantik, variasi imbuhan dan sinonim mungkin tidak selalu terhubung. Hal ini sesuai dengan lingkup minimal yang diwajibkan dalam instruksi UTS.
- Skor cosine mengukur kemiripan leksikal pada representasi BoW, bukan kualitas atau kebenaran isi dokumen.

