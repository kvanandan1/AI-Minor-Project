# ============================================ 
# AI Personalized Learning - Dataset Driven
# ============================================
import pandas as pd
from datasets import Dataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# -------------------------------
# 1. Load Dataset
# -------------------------------
csv_url = "https://huggingface.co/datasets/merve/student_scores/raw/main/dataset.csv"
df = pd.read_csv(csv_url)

# Rename columns (remove spaces for safety)
df = df.rename(columns={
    "math score": "math_score",
    "reading score": "reading_score",
    "writing score": "writing_score"
})

# Add student_id (unique identifier)
df["student_id"] = range(1, len(df) + 1)

# -------------------------------
# 2. Feature Engineering
# -------------------------------
df["average_score"] = (df["math_score"] + df["reading_score"] + df["writing_score"]) / 3

# Subject strength compared to average
df["math_strength"] = df["math_score"] - df["average_score"]
df["reading_strength"] = df["reading_score"] - df["average_score"]
df["writing_strength"] = df["writing_score"] - df["average_score"]

# Performance bands
def categorize(score):
    if score < 60:
        return "Low"
    elif score < 80:
        return "Medium"
    else:
        return "High"

df["math_band"] = df["math_score"].apply(categorize)
df["reading_band"] = df["reading_score"].apply(categorize)
df["writing_band"] = df["writing_score"].apply(categorize)

# Weighted score
df["weighted_score"] = (
    0.4*df["math_score"] + 0.3*df["reading_score"] + 0.3*df["writing_score"]
)

# Skill gaps
df["math_vs_reading_gap"] = abs(df["math_score"] - df["reading_score"])
df["math_vs_writing_gap"] = abs(df["math_score"] - df["writing_score"])
df["reading_vs_writing_gap"] = abs(df["reading_score"] - df["writing_score"])

# Weak subject flags
df["is_math_weak"] = (df["math_score"] < 60).astype(int)
df["is_reading_weak"] = (df["reading_score"] < 60).astype(int)
df["is_writing_weak"] = (df["writing_score"] < 60).astype(int)

# -------------------------------
# 3. Train/Test Split
# -------------------------------
dataset = Dataset.from_pandas(df, preserve_index=False)
dataset_dict = dataset.train_test_split(test_size=0.2, seed=42)

train_df = dataset_dict['train'].to_pandas()
test_df = dataset_dict['test'].to_pandas()

features = [
    "math_score", "reading_score", "writing_score",
    "math_strength", "reading_strength", "writing_strength",
    "weighted_score",
    "math_vs_reading_gap", "math_vs_writing_gap", "reading_vs_writing_gap",
    "is_math_weak", "is_reading_weak", "is_writing_weak"
]

X_train, y_train = train_df[features], train_df["average_score"]
X_test, y_test = test_df[features], test_df["average_score"]

# -------------------------------
# 4. Train Model
# -------------------------------
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"✅ Model trained. MAE on test set: {mae:.2f}")

# -------------------------------
# 5. Personalized Feedback Function
# -------------------------------
def generate_feedback_by_id(student_id: int):
    student_row = df[df["student_id"] == student_id].iloc[0]

    # Prepare features for prediction
    student = pd.DataFrame([student_row[features].to_dict()])
    predicted_score = model.predict(student)[0]

    # Weakness detection
    weaknesses = []
    if student_row["is_math_weak"]:
        weaknesses.append("Math")
    if student_row["is_reading_weak"]:
        weaknesses.append("Reading")
    if student_row["is_writing_weak"]:
        weaknesses.append("Writing")

    feedback = f"Needs improvement in {', '.join(weaknesses)}." if weaknesses else "Strong performance across all subjects."

    # Report
    report = (
        f"Student ID: {student_id}\n"
        f"Predicted Average Score: {predicted_score:.2f}\n"
        f"Performance Bands -> Math: {student_row['math_band']}, "
        f"Reading: {student_row['reading_band']}, "
        f"Writing: {student_row['writing_band']}\n"
        f"Feedback: {feedback}"
    )
    return report

def generate_feedback_for_multiple(student_ids: list):
    reports = []
    for sid in student_ids:
        reports.append(generate_feedback_by_id(sid))
    return "\n\n".join(reports)

# -------------------------------
# 6. Example Usage
# -------------------------------
print("\n🎓 Example: Single Student Report\n")
print(generate_feedback_by_id(10))   # Student with ID 10

print("\n🎓 Example: Multiple Student Reports\n")
print(generate_feedback_for_multiple([5, 15, 25]))
