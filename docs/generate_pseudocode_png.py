import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

pseudocode = """
ALGORITHM: Diabetes Prediction using Random Forest
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 1.  LOAD dataset from "kaggle_diabetes.csv"

 2.  RENAME column "DiabetesPedigreeFunction" → "DPF"

 3.  FOR each col IN [Glucose, BloodPressure, SkinThickness, Insulin, BMI]:
         REPLACE 0 with NaN

 4.  IMPUTE missing values:
         Glucose, BloodPressure       ← column mean
         SkinThickness, Insulin, BMI  ← column median

 5.  SPLIT data into X (features) and y (Outcome)
     TRAIN_TEST_SPLIT  (test_size = 0.2, random_state = 0)

 6.  FIT StandardScaler on X_train  → X_train_scaled
     TRANSFORM X_test               → X_test_scaled

 7.  INITIALISE RandomForestClassifier (n_estimators = 20)
     FIT model on X_train_scaled, y_train

 8.  CROSS_VAL_SCORE (cv = 5) → average accuracy

 9.  EVALUATE on X_test_scaled:
         compute confusion_matrix
         compute accuracy_score
         compute classification_report

10.  FUNCTION predict_diabetes(scaler, classifier, features):
         scaled ← scaler.transform(features)
         RETURN classifier.predict(scaled)

11.  FOR each sample IN demo_examples:
         result ← predict_diabetes(scaler, classifier, sample)
         IF result == 1  THEN  PRINT "Has diabetes"
         ELSE                  PRINT "No diabetes"
"""

fig, ax = plt.subplots(figsize=(10, 12))
ax.axis("off")

ax.text(
    0.05, 0.95,
    pseudocode,
    transform=ax.transAxes,
    fontsize=11,
    fontfamily="monospace",
    verticalalignment="top",
    bbox=dict(boxstyle="round,pad=0.8", facecolor="#f0f4ff", edgecolor="#4a6fa5", linewidth=2),
)

plt.tight_layout()
plt.savefig("docs/diabetes_pseudocode.png", dpi=150, bbox_inches="tight", facecolor="white")
print("Saved docs/diabetes_pseudocode.png")
