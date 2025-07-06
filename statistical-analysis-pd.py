import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

adisease = pd.read_csv("animal_disease_dataset.csv", index_col=0)

symptom_relate = pd.melt(adisease, 
                       id_vars=["Disease"], 
                       value_vars=["Symptom 1", "Symptom 2", "Symptom 3"],
                       var_name="Symptom_Position", 
                       value_name="Symptom")


# Chi-Square Test of Independence
# Create a contingency table:
# Rows = symptoms, Columns = diseases, Values = counts
contingency_table = pd.crosstab(symptom_relate["Symptom"], symptom_relate["Disease"])

# Run the chi-square test of independence
chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)

# Print the results
print("Chi-square Statistic:", round(chi2_stat, 2))
print("Degrees of Freedom:", dof)
print("P-value:", round(p_value, 5))


#Association Between Individual Symptoms and Diseases

# Identify all unique symptoms and diseases
unique_symptoms = symptom_relate["Symptom"].unique()
unique_diseases = symptom_relate["Disease"].unique()

# list to store results
significant_pairs = []

# Looping through each symptom–disease pair
for symptom in unique_symptoms:
    for disease in unique_diseases:

        # Create binary columns: symptom_present, disease_present
        df_loop = adisease.copy()
        df_loop["Symptom_Present"] = df_loop[["Symptom 1", "Symptom 2", "Symptom 3"]].isin([symptom]).any(axis=1)
        df_loop["Disease_Present"] = df_loop["Disease"] == disease

        # Create 2x2 contingency table
        contingency = pd.crosstab(df_loop["Symptom_Present"], df_loop["Disease_Present"])

        # Only test if the table is valid (at least 2x2)
        if contingency.shape == (2, 2):
            chi2, p, _, _ = stats.chi2_contingency(contingency)

            # If p < 0.05, consider it significant
            if p < 0.05:
                significant_pairs.append({
                    "Symptom": symptom,
                    "Disease": disease,
                    "P-Value": round(p, 5)
                })

# Step 4: Convert results to DataFrame
significant_disease = pd.DataFrame(significant_pairs)

# Step 5: Sort by lowest p-values
significant_disease = significant_disease.sort_values(by="P-Value")

# Step 6: Display top associations
print(significant_disease.head(10))

#ANOVA for age and temperature against disease

def run_anova(adisease, variable, group_col="Disease"):
    
    temp_df = adisease[[variable, group_col]]
    
    # Group the data by the group_col and extract the numeric variable for each group
    groups = [group[variable].values for _, group in temp_df.groupby(group_col)]

    #Perform one-way ANOVA on the groups
    f_stat, p_val = stats.f_oneway(*groups)

    # Step 4: Print the results
    print(f"ANOVA for {variable}:")
    print("F-statistic:", round(f_stat, 2))
    print("P-value:", round(p_val, 5))
    print()

run_anova(adisease, "Age")
run_anova(adisease, "Temperature")

import statsmodels.stats.multicomp as mc

#further analysis on temperature(Turkey Test)
tukey_tst = adisease[["Temperature", "Disease"]]

tukey_result = mc.pairwise_tukeyhsd(
    endog=tukey_tst["Temperature"],   # numeric variable
    groups=tukey_tst["Disease"],      # group labels
    alpha=0.05                       # significance level
)


print(tukey_result.summary())

tukey_result.plot_simultaneous(figsize=(12, 6))
plt.title("Tukey HSD: Temperature Differences Between Diseases")
plt.xlabel("Mean Difference with 95% CI")
plt.show()

from sklearn.preprocessing import LabelEncoder

# Copy dataset
traindata = adisease.copy()

# Encode categorical columns (Symptom 1–3)
for col in ["Symptom 1", "Symptom 2", "Symptom 3"]:
    revs = LabelEncoder()
    traindata[col] = revs.fit_transform(traindata[col].astype(str))

# Encode target column (Disease)
revd = LabelEncoder()
traindata["Disease"] = revd.fit_transform(traindata["Disease"])


from sklearn.model_selection import train_test_split

# Select features and target
X = traindata[["Symptom 1", "Symptom 2", "Symptom 3", "Age", "Temperature"]]
y = traindata["Disease"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier

# Initialize and train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, classification_report

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=revd.classes_))

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=revd.classes_, yticklabels=revd.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

feature_names = X.columns

# Extract importances
importances = model.feature_importances_

# Create a DataFrame for easy visualization
feat_imp_d = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

# Print sorted importances
print(feat_imp_d)

plt.figure(figsize=(8, 4))
sns.barplot(data=feat_imp_d, x="Importance", y="Feature", hue= "Feature", palette="viridis")
plt.title("Feature Importance in Disease Prediction")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()


