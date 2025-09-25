# 🎓 AI Personalized Learning - Dataset Driven

This project is a **Machine Learning** application that provides **personalized learning feedback** for students based on their academic scores. The system predicts average scores and gives targeted suggestions to improve weaker subjects.

---

## 📌 Project Overview

The AI Personalized Learning system:
- Loads a student dataset containing math, reading, and writing scores.
- Performs **feature engineering** to create new metrics and performance bands.
- Trains a **Random Forest Regressor** to predict average scores.
- Generates personalized feedback for each student based on their strengths and weaknesses.

---

## 🛠 Features

- **Load Dataset**: Uses a publicly available dataset from Hugging Face.
- **Feature Engineering**: Creates derived features such as:
  - Average scores
  - Subject strengths
  - Performance bands
  - Weighted scores
  - Skill gaps
  - Weak subject flags
- **Model Training**: Trains a `RandomForestRegressor` to predict student performance.
- **Personalized Feedback**: Generates reports for individual students based on predicted results.
- **Batch Feedback**: Supports generating feedback for multiple students at once.

---



