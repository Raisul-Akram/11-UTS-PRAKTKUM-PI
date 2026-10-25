"""Sistem Information Retrieval multi-dataset berbasis CLI.

Paket ini menggunakan pandas untuk membaca CSV, Whoosh untuk indeks dan
retrieval kandidat, serta CountVectorizer + cosine similarity untuk reranking.
"""
from __future__ import annotations

import json
import pickle
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from whoosh import index as whoosh_index
    from whoosh.analysis import LowercaseFilter, RegexTokenizer
    from whoosh.fields import ID, STORED, TEXT, Schema
    from whoosh.qparser import MultifieldParser, OrGroup
    WHOOSH_AVAILABLE = True
except ImportError:  # Memungkinkan fungsi preprocessing diuji tanpa Whoosh terpasang.
    whoosh_index = None
    LowercaseFilter = RegexTokenizer = ID = STORED = TEXT = Schema = None
    MultifieldParser = OrGroup = None
    WHOOSH_AVAILABLE = False

try:
    ROOT_DIR = Path(__file__).resolve().parent
except NameError:  # Saat dijalankan sebagai notebook Jupyter.
    ROOT_DIR = Path.cwd()
DATASET_FILES = {
    "etd-usk": ROOT_DIR / "datasets" / "etd-usk" / "data.csv",
    "etd-ugm": ROOT_DIR / "datasets" / "etd-ugm" / "data.csv",
    "kompas": ROOT_DIR / "datasets" / "kompas" / "data.csv",
    "tempo": ROOT_DIR / "datasets" / "tempo" / "data.csv",
    "mojok": ROOT_DIR / "datasets" / "mojok" / "data.csv",
}
RUNTIME_DIR = ROOT_DIR / "runtime"
WHOOSH_DIR = RUNTIME_DIR / "whoosh_index"
VECTORIZER_PATH = RUNTIME_DIR / "count_vectorizer.pkl"
MATRIX_PATH = RUNTIME_DIR / "document_vectors.pkl"
MANIFEST_PATH = RUNTIME_DIR / "index_manifest.json"
MAX_FEATURES = 200_000
MAX_CANDIDATES = None  # None berarti semua kandidat leksikal Whoosh ikut diranking.
TOP_K = 5
CHUNK_SIZE = 500
_RUNTIME_CACHE = None  # Model dan matriks dimuat sekali per sesi pencarian.

# Daftar stopword Bahasa Indonesia yang sengaja konservatif.
# Kata negasi seperti "tidak", "bukan", "belum", dan "jangan" dipertahankan
# karena dapat mengubah makna query.
INDONESIAN_STOPWORDS = set("""
ada adalah adanya adapun agak agaknya agar akan akankah akhirnya aku alah alias
amat amatlah anda andalah antar antara diantaranya apa apaan apabila apakah
apalagi apatah artinya asal asalkan atas atau ataupun awal awalnya bagi
bahkan bahwa bahwasanya beberapa begini beginian beginikah beginilah begitu
begitukah begitulah begitu pun bekerja belakangan belakangnya berapa berapakah
berapalah berapapun berarti berawal berbagai berdatangan beri berikan berikut
berikutnya berjumlah berkali-kali berkata berkehendak berkenaan berlainan
berlalu berlangsung berlebihan bermaksud bermula bersama bersama-sama
bersiap bersiap-siap bertanya bertanya-tanya berturut berturut-turut bertutur
berujar berupa besar betul betulkah biasa biasanya bila bilamana bisa bisakah
boleh bolehkah bolehlah buat bukan bukankah bukanlah bukannya cuma dahulu
dalam dan dapat dari daripada datang dekat demi demikian demikianlah dengan
depan di dia diakhiri diakhirinya dialah diantara diantaranya diberi diberikannya
dibuat dibuatnya didapat didatangkan digunakan diibaratkan diingat diingatkan
diinginkan dijawab dijelaskan dijelaskannya dikarenakan dikatakan dikatakannya
dikerjakan diketahui dikenai dikenal dikenalnya dikira dilakukan dilalui
dilihat dimaksud dimaksudkan dimaksudnya diminta dimintai dimisalkan dimulai
dimulailah dimulainya dimungkinkan dini dipastikan diperbuat diperbuatnya
diperlukan diperlukannya dipersoalkan dipertanyakan dipunyai diri dirinya
disebut disebutkan disebutkannya disini disinilah ditambahkan ditandaskan
ditanya ditanyai ditanyakan ditegaskan ditujukan ditunjuk ditunjuki ditunjukkan
ditunjukkannya ditunjuknya dituturkan dituturkannya diucapkan diucapkannya
diungkapkan dong dua dulu empat enggak enggaknya entah entahlah guna gunakan
hal hampir hanya hanyalah hari harus haruslah harusnya hendak hendaklah hendaknya
hingga ia ialah ibarat ibaratkan ibaratnya ibu ikut ingat ingat-ingat ingin inginkah
inikah inilah itu itukah itulah jadi jadilah jadinya jangan jangankan janganlah jauh
jawab jawabnya jelas jelaslah jelasnya jika jikalau juga jumlah jumlahnya justru
kala kalangan kami kamilah kamu kamulah kan kapan kapankah kapanpun karena
karenanya kasus kata katakan katakanlah katanya ke keadaan kebetulan kecil kedua
keduanya keinginan kelamaan kelihatan kelihatannya kelima keluar kembali kemudian
kemungkinan kemungkinannya kenapa kepada kepadanya keras kesampaian keseluruhan
keseluruhannya keterlaluan ketika khusus khususnya kini kinilah kira kira-kira
kiranya kita kitalah kok kurang lagi lagian lah lain lainnya lalu lama lamanya
lanjut lanjutnya lebih lewat lima luar macam maka makanya makin malah malahan
mampu maupun mana manakala manalagi masa masalah masalahnya masih masihkah
masing masing-masing mau maupun melainkan melakukan melalui melihat melihatnya
memang memastikan memberi memberikan membuat memerlukan memihak meminta
memintakan memisalkan memperbuat mempergunakan memperkirakan memperlihatkan
mempersiapkan mempersoalkan mempertanyakan mempunyai memulai memungkinkan
menaiki menambahkan menandaskan menanti menanti-nanti menanya menanyai
menanyakan mendapat mendapatkan mendatang mendatangi mendatangkan menegaskan
mengakhiri mengapa mengatakan mengatakannya mengenai mengerjakan mengetahui
menggunakan menghendaki mengibaratkan mengibaratkannya mengingat mengingatkan
menginginkan mengira mengucapkan mengucapkannya mengungkapkan menjadi menjawab
menjelaskan menuju menunjuk menunjuki menunjukkan menunjuknya menurut menuturkan
menyampaikan menyangkut menyatakan menyebutkan menyeluruh menyiapkan merasa mereka
merekalah merupakan merujuk meski meskipun meyakini meyakinkan mungkin mungkinkah
nah naik namun nyaris nyatanya oleh olehnya pada padahal padanya pak paling panjang
pantas para pasti pastilah penting pentingnya per pernah pula pun punya rasa rasanya
rata rupanya saat saatnya saja sajalah saling sama sama-sama sambil sampai sampai-sampai
sana sangat sangatlah satu saya sayalah se sebab sebabnya sebagai sebagaimana
sebagainya sebagian sebaik sebaik-baiknya sebaiknya sebaliknya sebanyak sebegitu
sebelum sebelumnya sebenarnya seberapa sebesar sebetulnya sejak sejauh sejumlah
sekadar sekadarnya sekitar sekitarnya sekurang-kurangnya sekurangnya sela selain
selaku selalu selama selamanya selanjutnya seluruh seluruhnya semacam semakin
sementara semisal semisalnya sempat semua semuanya semula sendiri sendirian
sendirinya seolah seolah-olah seorang sepanjang sepantasnya seperlunya seperti
sepertinya sepihak sering seringnya serta sesuai sesudah sesudahnya setelah setempat
setengah seterusnya setiap setiba setibanya setidak-tidaknya setidaknya setinggi
seusai sewaktu siapa siapakah siapapun sini sinilah soal suatu sesuatu sesuatunya
sudah sudahkah sudahlah supaya tadi tadinya tak tambah tampak tampaknya tanpa
tentu tentulah tentunya tegas tegasnya telah tempat tengah tentang tentu tentunya
terasa terbanyak terdahulu terdapat terdiri terhadap teringat terjadilah terjadi
terjadinya terkadang terkira terlalu terlebih termasuk ternyata tersampaikan
tersebut tertentu tertuju terus terutama tetapi tiap tiba tiba-tiba tidak tiga
tinggi toh tunjuk turut tuturnya umumnya ungkap ungkapnya untuk usai utama
utamanya waduh wah wahai waktu waktunya walau walaupun wong yaitu yakni
""".split()) - {"tidak", "bukan", "belum", "jangan", "tak", "enggak", "enggaknya"}
INDONESIAN_STOPWORDS.add("yang")

TOKEN_PATTERN = r"[^\W_]+"


def preprocess_text(text: object) -> str:
    """Case folding, tokenisasi Unicode, dan stopword removal.

    Hasil dikembalikan sebagai token yang dipisahkan spasi agar dapat digunakan
    konsisten oleh Whoosh dan CountVectorizer.
    """
    if text is None:
        return ""
    raw = str(text).casefold()
    tokens = re.findall(TOKEN_PATTERN, raw, flags=re.UNICODE)
    return " ".join(token for token in tokens if token not in INDONESIAN_STOPWORDS)


def make_schema():
    """Membuat skema Whoosh. Teks yang dicari sudah melalui preprocessing."""
    if not WHOOSH_AVAILABLE:
        raise RuntimeError(
            "Library Whoosh belum terpasang. Jalankan: python -m pip install Whoosh"
        )
    analyzer = RegexTokenizer(TOKEN_PATTERN) | LowercaseFilter()
    return Schema(
        doc_id=ID(stored=True, unique=True),
        dataset=ID(stored=True),
        title=TEXT(stored=False, analyzer=analyzer, phrase=False),
        content=TEXT(stored=False, analyzer=analyzer, phrase=False),
        title_display=STORED,
        content_preview=STORED,
    )


def normalize_preview(text: str, limit: int = 420) -> str:
    """Menghasilkan cuplikan konten yang mudah dibaca pada hasil pencarian."""
    cleaned = re.sub(r"\s+", " ", str(text)).strip()
    if len(cleaned) > limit:
        return cleaned[:limit].rstrip() + "…"
    return cleaned or "(Konten kosong)"


def read_dataset_chunks(csv_path: Path):
    """Membaca CSV bertahap agar konsumsi memori saat indexing terkendali."""
    required = {"judul", "konten"}
    for chunk in pd.read_csv(
        csv_path,
        chunksize=CHUNK_SIZE,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8",
    ):
        missing = required.difference(chunk.columns)
        if missing:
            raise ValueError(
                f"Kolom wajib {sorted(missing)} tidak ditemukan pada {csv_path.name}. "
                f"Kolom yang tersedia: {list(chunk.columns)}"
            )
        yield chunk


def index_is_ready() -> bool:
    """Memeriksa apakah indeks Whoosh dan model BoW tersedia lengkap."""
    if not WHOOSH_AVAILABLE:
        return False
    return (
        WHOOSH_DIR.exists()
        and whoosh_index.exists_in(str(WHOOSH_DIR))
        and VECTORIZER_PATH.exists()
        and MATRIX_PATH.exists()
        and MANIFEST_PATH.exists()
    )


def load_and_index_datasets() -> dict:
    """Mengindeks lima CSV dan membangun matriks Bag of Words persisten."""
    global _RUNTIME_CACHE
    _RUNTIME_CACHE = None
    if not WHOOSH_AVAILABLE:
        raise RuntimeError(
            "Whoosh belum tersedia. Instal dependensi dengan: "
            "python -m pip install -r requirements.txt"
        )

    missing_files = [str(path) for path in DATASET_FILES.values() if not path.exists()]
    if missing_files:
        raise FileNotFoundError(
            "Dataset berikut tidak ditemukan:\n- " + "\n- ".join(missing_files)
        )

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    if WHOOSH_DIR.exists():
        shutil.rmtree(WHOOSH_DIR)
    for path in (VECTORIZER_PATH, MATRIX_PATH, MANIFEST_PATH):
        if path.exists():
            path.unlink()

    schema = make_schema()
    ix = whoosh_index.create_in(str(WHOOSH_DIR), schema)
    writer = ix.writer(limitmb=256, procs=1)
    corpus_texts: list[str] = []
    source_counts: dict[str, int] = {name: 0 for name in DATASET_FILES}
    total_docs = 0
    start = time.time()

    print("\nMemuat dataset dan membangun indeks Whoosh...")
    try:
        for dataset_name, csv_path in DATASET_FILES.items():
            print(f"  - Membaca {dataset_name}: {csv_path.relative_to(ROOT_DIR)}")
            for chunk in read_dataset_chunks(csv_path):
                for title_value, content_value in zip(chunk["judul"], chunk["konten"]):
                    title = str(title_value).strip()
                    content = str(content_value).strip()
                    processed_title = preprocess_text(title)
                    processed_content = preprocess_text(content)
                    combined = (processed_title + " " + processed_content).strip()
                    if not combined:
                        combined = "dokumen"

                    writer.add_document(
                        doc_id=str(total_docs),
                        dataset=dataset_name,
                        title=processed_title or "tanpa judul",
                        content=processed_content or "dokumen",
                        title_display=title or "(Tanpa judul)",
                        content_preview=normalize_preview(content),
                    )
                    corpus_texts.append(combined)
                    source_counts[dataset_name] += 1
                    total_docs += 1
            print(f"    Dokumen dibaca: {source_counts[dataset_name]:,}")
        writer.commit()
    except Exception:
        try:
            writer.cancel()
        except Exception:
            pass
        raise

    print("Membangun matriks Bag of Words dengan CountVectorizer...")
    vectorizer = CountVectorizer(
        lowercase=False,
        token_pattern=r"(?u)\b\w+\b",
        min_df=2,
        max_df=0.98,
        max_features=MAX_FEATURES,
    )
    try:
        document_matrix = vectorizer.fit_transform(corpus_texts)
    except ValueError as exc:
        # Fallback hanya untuk korpus kecil/khusus yang gagal setelah pruning.
        print(f"Peringatan konfigurasi awal CountVectorizer: {exc}")
        vectorizer = CountVectorizer(
            lowercase=False,
            token_pattern=r"(?u)\b\w+\b",
            min_df=1,
            max_features=MAX_FEATURES,
        )
        document_matrix = vectorizer.fit_transform(corpus_texts)

    with VECTORIZER_PATH.open("wb") as file:
        pickle.dump(vectorizer, file, protocol=pickle.HIGHEST_PROTOCOL)
    with MATRIX_PATH.open("wb") as file:
        pickle.dump(document_matrix, file, protocol=pickle.HIGHEST_PROTOCOL)

    manifest = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "total_documents": total_docs,
        "source_counts": source_counts,
        "vector_matrix_rows": int(document_matrix.shape[0]),
        "vector_matrix_columns": int(document_matrix.shape[1]),
        "max_features_setting": MAX_FEATURES,
        "max_candidates_per_query": MAX_CANDIDATES,
        "ranking": "cosine similarity on CountVectorizer Bag of Words vectors",
        "candidate_retrieval": "all lexical matches from Whoosh MultifieldParser over title and content",
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    elapsed = time.time() - start
    print("\nIndexing selesai.")
    print(f"Total dokumen      : {total_docs:,}")
    print(f"Jumlah fitur BoW   : {document_matrix.shape[1]:,}")
    print(f"Matriks dokumen    : {document_matrix.shape[0]:,} x {document_matrix.shape[1]:,}")
    print(f"Waktu proses       : {elapsed:.1f} detik")
    print(f"Indeks tersimpan   : {WHOOSH_DIR.relative_to(ROOT_DIR)}")
    print(f"Model tersimpan    : {VECTORIZER_PATH.relative_to(ROOT_DIR)} dan {MATRIX_PATH.relative_to(ROOT_DIR)}")
    return manifest


def search_query(query_text: str, top_k: int = TOP_K, candidate_limit: int | None = MAX_CANDIDATES) -> list[dict]:
    """Mencari kandidat melalui Whoosh lalu mengurutkannya dengan cosine similarity."""
    if not index_is_ready():
        raise RuntimeError("Indeks belum tersedia. Pilih menu [1] Load & Index Dataset terlebih dahulu.")

    cleaned_query = preprocess_text(query_text)
    if not cleaned_query:
        print("Query kosong setelah preprocessing. Masukkan kata kunci yang lebih spesifik.")
        return []

    global _RUNTIME_CACHE
    if _RUNTIME_CACHE is None:
        with VECTORIZER_PATH.open("rb") as file:
            cached_vectorizer = pickle.load(file)
        with MATRIX_PATH.open("rb") as file:
            cached_matrix = pickle.load(file)
        _RUNTIME_CACHE = (cached_vectorizer, cached_matrix)
    vectorizer, document_matrix = _RUNTIME_CACHE

    query_vector = vectorizer.transform([cleaned_query])
    if query_vector.nnz == 0:
        print("Tidak ada istilah query yang masuk ke kosakata BoW. Coba gunakan kata kunci lain.")
        return []

    ix = whoosh_index.open_dir(str(WHOOSH_DIR))
    with ix.searcher() as searcher:
        parser = MultifieldParser(
            ["title", "content"],
            schema=ix.schema,
            fieldboosts={"title": 2.0, "content": 1.0},
            group=OrGroup,
        )
        parsed_query = parser.parse(cleaned_query)
        hits = searcher.search(parsed_query, limit=candidate_limit)
        candidates = []
        for hit in hits:
            try:
                row_id = int(hit["doc_id"])
            except (KeyError, TypeError, ValueError):
                continue
            if 0 <= row_id < document_matrix.shape[0]:
                candidates.append(
                    {
                        "row_id": row_id,
                        "doc_id": hit["doc_id"],
                        "dataset": hit["dataset"],
                        "title": hit["title_display"],
                        "preview": hit["content_preview"],
                        "whoosh_score": float(hit.score),
                    }
                )

    if not candidates:
        print("Whoosh tidak menemukan dokumen yang cocok dengan query tersebut.")
        return []

    row_ids = [candidate["row_id"] for candidate in candidates]
    candidate_matrix = document_matrix[row_ids]
    scores = cosine_similarity(query_vector, candidate_matrix).ravel()
    for candidate, score in zip(candidates, scores):
        candidate["cosine_score"] = float(score)

    ranked = sorted(
        (candidate for candidate in candidates if candidate["cosine_score"] > 0),
        key=lambda item: (item["cosine_score"], item["whoosh_score"]),
        reverse=True,
    )[:top_k]
    return ranked


def display_results(results: list[dict], query: str) -> None:
    print(f"\nHasil pencarian untuk: {query!r}")
    if not results:
        print("Tidak ada hasil dengan skor cosine similarity di atas nol.")
        return
    print("=" * 78)
    for rank, result in enumerate(results, start=1):
        print(f"{rank}. [{result['dataset']}] {result['title']}")
        print(f"   Cosine similarity: {result['cosine_score']:.4f}")
        print(f"   Cuplikan: {result['preview']}")
        print("-" * 78)


def show_menu() -> None:
    print("\n=== INFORMATION RETRIEVAL SYSTEM ===")
    print("[1] Load & Index Dataset")
    print("[2] Search Query")
    print("[3] Exit")
    print("====================================")


def main() -> None:
    """Menjalankan antarmuka CLI sesuai menu yang diwajibkan pada instruksi UTS."""
    print("INFORMATION RETRIEVAL SYSTEM — UTS Praktikum Penelusuran Informasi")
    if not WHOOSH_AVAILABLE:
        print("PERINGATAN: Whoosh belum terpasang di lingkungan ini.")
        print("Instal dependensi dengan: python -m pip install -r requirements.txt")

    while True:
        show_menu()
        choice = input("Pilih menu [1-3]: ").strip()
        if choice == "1":
            try:
                load_and_index_datasets()
            except Exception as exc:
                print(f"Gagal melakukan indexing: {exc}")
        elif choice == "2":
            if not index_is_ready():
                print("Indeks belum siap. Jalankan menu [1] terlebih dahulu.")
                continue
            query_text = input("Masukkan query: ").strip()
            try:
                results = search_query(query_text)
                display_results(results, query_text)
            except Exception as exc:
                print(f"Pencarian gagal: {exc}")
        elif choice == "3":
            print("Sistem ditutup. Terima kasih.")
            break
        else:
            print("Pilihan tidak valid. Masukkan 1, 2, atau 3.")


if __name__ == "__main__":
    main()
