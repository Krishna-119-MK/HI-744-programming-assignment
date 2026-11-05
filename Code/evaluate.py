import ast
import json
import sys
from pathlib import Path

project_folder = Path(__file__).resolve().parent.parent
if len(sys.argv) > 1:
    data_folder = Path(sys.argv[1])
else:
    data_folder = project_folder / "data"
output_folder = project_folder / "output"

with (output_folder / "patient_similarities.json").open(
    "r", encoding="utf-8"
) as file:
    similarity_results = json.load(file)


def get_relevant_patients(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return set()

    if not isinstance(value, dict):
        return set()

    relevant = set()

    for patient_uid, rating in value.items():
        try:
            if int(rating) in (1, 2):
                relevant.add(str(patient_uid))
        except (TypeError, ValueError):
            pass

    return relevant


performance = {}
precision_totals = {"tfidf": 0, "bm25": 0}
recall_totals = {"tfidf": 0, "bm25": 0}
recall_counts = {"tfidf": 0, "bm25": 0}
excluded_from_recall = 0
overlap_total = 0

for file_path in data_folder.glob("*.json"):
    with file_path.open("r", encoding="utf-8") as file:
        record = json.load(file)

    patient_uid = str(record.get("patient_uid", ""))

    if patient_uid not in similarity_results:
        continue

    relevant = get_relevant_patients(
        record.get("similar_patients", {})
    )
    patient_results = similarity_results[patient_uid]

    performance[patient_uid] = {
        "patient_id": record.get("patient_id")
    }

    for method in ("tfidf", "bm25"):
        retrieved = [
            str(uid) for uid in patient_results.get(method, [])
        ]
        hits = len(set(retrieved) & relevant)

        precision = hits / 5
        recall = hits / len(relevant) if relevant else None

        performance[patient_uid][method] = {
            "precision_5": round(precision, 4),
            "recall_5": round(recall, 4) if recall is not None else None
        }

        precision_totals[method] += precision

        if recall is not None:
            recall_totals[method] += recall
            recall_counts[method] += 1

    if not relevant:
        excluded_from_recall += 1

    tfidf_results = set(patient_results.get("tfidf", []))
    bm25_results = set(patient_results.get("bm25", []))
    overlap_total += len(tfidf_results & bm25_results)

performance_file = output_folder / "performance.json"

with performance_file.open("w", encoding="utf-8") as file:
    json.dump(performance, file, indent=2)

number_evaluated = len(performance)

print("Patients evaluated:", number_evaluated)
print("Patients excluded from recall average:", excluded_from_recall)

for method in ("tfidf", "bm25"):
    average_precision = precision_totals[method] / number_evaluated
    count = recall_counts[method]

    if count > 0:
        average_recall = recall_totals[method] / count
        print(method, "average recall@5:", round(average_recall, 4))
    else:
        print(method, "average recall@5: N/A")

    print(method, "average precision@5:", round(average_precision, 4))

average_overlap = overlap_total / number_evaluated
print("Average top-five overlap:", round(average_overlap, 2), "patients")
print("Per-patient scores saved to performance.json")