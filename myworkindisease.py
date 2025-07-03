#import panda package to work with table
import pandas as pd 

#read dataset file and Use first column as row index
adisease = pd.read_csv("animal_disease_dataset.csv", index_col=0)

# Checking the first 5 rows to see what the data looks like
print(adisease.head())

# Check column names and data types
print(adisease.info())

# A quick statistical summary of numerical values of the dataset
print(adisease.describe())

# Combine all symptom columns into a single Series and Count how many times each symptom appears
all_symptoms = pd.concat([adisease["Symptom 1"], adisease["Symptom 2"], adisease["Symptom 3"]])

symptom_counts = all_symptoms.value_counts()
print(symptom_counts)

# Count how many times each disease appears
disease_counts = adisease["Disease"].value_counts()

print(disease_counts)

# import data visualization packages
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style of the plot
sns.set(style="whitegrid")

# Set up the figure size
plt.figure(figsize=(12, 6))

# Create the barplot
sns.barplot(x=disease_counts.index, y=disease_counts.values, hue=disease_counts.index, palette="viridis", legend=False)
#add ticks
plt.tick_params(axis="y", which="major", left=True, direction="out", color="black")

# Add labels and title
plt.xlabel("Disease")
plt.ylabel("Number of Cases")
plt.title("Frequency of Diagnosed Diseases")

# Display the plot
plt.tight_layout()
plt.show()

# Group the data by Animal and Disease, then count the number of cases
disease_by_animal = adisease.groupby(["Animal", "Disease"]).size().reset_index(name="Count")

# Check what this table looks like
print(disease_by_animal.head())

# Set up the figure size and style
plt.figure(figsize=(14, 7))
sns.set(style="whitegrid")

# Create a grouped barplot
sns.barplot(data=disease_by_animal, x="Disease", y="Count", hue="Animal", palette="Set2", legend=True)

#add ticks
plt.tick_params(axis="y", which="major", left=True, direction="out", color="black")

# Add titles and labels
plt.title("Disease Frequency by Animal Type")
plt.xlabel("Disease")
plt.ylabel("Number of Cases")

# Display the legend and the plot

plt.tight_layout()
plt.show()


#box plot of age distribution

age_counts = adisease["Age"].value_counts().sort_index()

plt.figure(figsize=(10, 5))
sns.barplot(x=age_counts.index, y=age_counts.values, color="skyblue")

plt.tick_params(axis="y", which="major", left=True, direction="out", color="black")

plt.title("Distribution of Animals by Age")
plt.xlabel("Age (Years)")
plt.ylabel("Number of Animals")
plt.tight_layout()
plt.show()

# Histogram of animal temperatures
plt.figure(figsize=(10, 5))
sns.histplot(adisease["Temperature"], bins=20, kde=True, edgecolor="black", alpha=0.6, color="steelblue")

plt.tick_params(axis="x", which="major", bottom=True, direction="out", color="black")
plt.tick_params(axis="y", which="major", left=True, direction="out", color="black")


plt.title("Distribution of Body Temperature")
plt.xlabel("Temperature (°F)")
plt.ylabel("Number of Animals")
plt.tight_layout()
plt.show()

# Boxplot to compare temperature across diseases
plt.figure(figsize=(10, 5))
sns.boxplot(data=adisease, x="Disease", y="Temperature", hue="Disease", width=0.5, palette="coolwarm")

plt.title("Body Temperature by Disease")
plt.xlabel("Disease")
plt.ylabel("Temperature (°F)")
plt.tight_layout()
plt.show()
