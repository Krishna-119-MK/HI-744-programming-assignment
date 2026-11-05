import json
import re
import sys
from pathlib import Path

import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def clean_text(text, stop_words, lemmatizer):
    words = re.findall(r"[a-z]+", str(text).lower())
    return [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]


def load_records(json_files, stop_words, lemmatizer):
    records = []

    for file_path in json_files:
        with file_path.open("r", encoding="utf-8") as file:
            record = json.load(file)

        words = clean_text(
            record.get("patient", ""),
            stop_words,
            lemmatizer
        )

        records.append({
            "patient_uid": record.get("patient_uid"),
            "patient_id": record.get("patient_id"),
            "cleaned_text": " ".join(words),
            "similar_patients": record.get("similar_patients", "{}")
        })

    return records


def build_tfidf(processed_records, output_folder):
    vectorizer = TfidfVectorizer(
        smooth_idf=True,
        sublinear_tf=True,
        norm="l2"
    )

    texts = [record["cleaned_text"] for record in processed_records]
    tfidf_matrix = vectorizer.fit_transform(texts)

    np.savez_compressed(
        output_folder / "tfidf_matrix.npz",
        data=tfidf_matrix.data,
        indices=tfidf_matrix.indices,
        indptr=tfidf_matrix.indptr,
        shape=tfidf_matrix.shape,
        patient_uids=np.asarray(
            [record["patient_uid"] for record in processed_records],
            dtype=str
        ),
        terms=np.asarray(vectorizer.get_feature_names_out(), dtype=str)
    )

    print("TF-IDF matrix saved.")
    print("Matrix rows and columns:", tfidf_matrix.shape)

    return texts, tfidf_matrix


def add_tfidf_results(processed_records, tfidf_matrix, similarity_results):
    for i, current_record in enumerate(processed_records):
        scores = (tfidf_matrix[i] @ tfidf_matrix.T).toarray().ravel()
        scores[i] = -1

        top_indices = np.argsort(scores)[-5:][::-1]

        similarity_results[current_record["patient_uid"]] = {
            "patient_id": current_record["patient_id"],
            "tfidf": [
                processed_records[index]["patient_uid"]
                for index in top_indices
            ]
        }


def add_bm25_results(processed_records, texts, similarity_results):
    count_vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
    count_matrix = count_vectorizer.fit_transform(texts).tocsr()

    number_of_documents = count_matrix.shape[0]
    document_frequency = np.asarray((count_matrix > 0).sum(axis=0)).ravel()

    idf = np.log(
        1 + (number_of_documents - document_frequency + 0.5)
        / (document_frequency + 0.5)
    )

    document_lengths = np.asarray(count_matrix.sum(axis=1)).ravel()
    average_document_length = document_lengths.mean()

    k1 = 1.5
    b = 0.75

    row_numbers = np.repeat(
        np.arange(number_of_documents),
        np.diff(count_matrix.indptr)
    )
    term_frequencies = count_matrix.data.astype(float)

    length_adjustment = k1 * (
        1 - b + b * document_lengths[row_numbers] / average_document_length
    )

    bm25_weights = idf[count_matrix.indices] * (
        term_frequencies * (k1 + 1)
        / (term_frequencies + length_adjustment)
    )

    bm25_matrix = csr_matrix(
        (
            bm25_weights,
            count_matrix.indices.copy(),
            count_matrix.indptr.copy()
        ),
        shape=count_matrix.shape
    )

    for i, current_record in enumerate(processed_records):
        scores = (count_matrix.getrow(i) @ bm25_matrix.T).toarray().ravel()
        scores[i] = -1

        top_indices = np.argsort(scores)[-5:][::-1]

        similarity_results[current_record["patient_uid"]]["bm25"] = [
            processed_records[index]["patient_uid"]
            for index in top_indices
        ]

        if (i + 1) % 1000 == 0:
            print("BM25 records ranked:", i + 1)


def main():
    if len(sys.argv) > 1:
        data_folder = Path(sys.argv[1])
    else:
        data_folder = Path(__file__).resolve().parent.parent / "data"

    json_files = list(data_folder.glob("*.json"))

    if not json_files:
        print("No JSON files found in:", data_folder)
        return

    output_folder = Path(__file__).resolve().parent.parent / "Output"
    output_folder.mkdir(parents=True, exist_ok=True)

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    processed_records = load_records(
        json_files,
        stop_words,
        lemmatizer
    )

    total_words = sum(
        len(record["cleaned_text"].split())
        for record in processed_records
    )
    average_words = (
        total_words / len(processed_records)
        if processed_records else 0
    )

    print("Records processed:", len(processed_records))
    print(
        "Average words per record after cleaning:",
        round(average_words, 1)
    )

    texts, tfidf_matrix = build_tfidf(processed_records, output_folder)

    similarity_results = {}
    add_tfidf_results(
        processed_records,
        tfidf_matrix,
        similarity_results
    )

    output_file = output_folder / "patient_similarities.json"

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(similarity_results, file, indent=2)

    print("TF-IDF top-five results saved.")
    print("Building BM25 results...")

    add_bm25_results(
        processed_records,
        texts,
        similarity_results
    )

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(similarity_results, file, indent=2)

    print("TF-IDF and BM25 results saved.")


if __name__ == "__main__":
    main()

import evaluate