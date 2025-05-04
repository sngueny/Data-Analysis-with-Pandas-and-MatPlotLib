import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for our plots
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 7)

# Task 1: Load and Explore the Dataset
# Load the Titanic dataset
try:
    # Using seaborn's built-in Titanic dataset
    titanic = sns.load_dataset('titanic')
    print("Dataset loaded successfully!")
except Exception as e:
    print(f"Error loading dataset: {e}")

# Display the first 5 rows of the dataset
print("\nFirst 5 rows of the dataset:")
print(titanic.head())

# Explore the structure of the dataset
print("\nDataset information:")
print(titanic.info())

# Check for missing values
print("\nMissing values in each column:")
print(titanic.isnull().sum())

# Clean the dataset by handling missing values
# Fill missing embark_town with the most frequent value
if titanic['embark_town'].isnull().sum() > 0:
    most_common_port = titanic['embark_town'].mode()[0]
    titanic['embark_town'].fillna(most_common_port, inplace=True)

# Fill missing age values with the median age
if titanic['age'].isnull().sum() > 0:
    median_age = titanic['age'].median()
    titanic['age'].fillna(median_age, inplace=True)

# Drop rows with missing 'deck' as it has too many missing values
titanic.drop('deck', axis=1, inplace=True)

# Drop any remaining rows with missing values
titanic.dropna(inplace=True)

print("\nAfter cleaning, missing values:")
print(titanic.isnull().sum())

# Task 2: Basic Data Analysis
# Compute basic statistics of numerical columns
print("\nBasic statistics of numerical columns:")
print(titanic.describe())

# Group by 'class' and compute mean of numerical columns
class_group = titanic.groupby('class')[['age', 'fare', 'survived']].mean()
print("\nMean values grouped by class:")
print(class_group)

# Group by 'sex' and compute mean of numerical columns
sex_group = titanic.groupby('sex')[['age', 'fare', 'survived']].mean()
print("\nMean values grouped by sex:")
print(sex_group)

# Group by 'embark_town' and compute mean of numerical columns
embark_group = titanic.groupby('embark_town')[['age', 'fare', 'survived']].mean()
print("\nMean values grouped by embark_town:")
print(embark_group)

# Task 3: Data Visualization
# 1. Line chart showing average fare by age (binned)
plt.figure(figsize=(12, 6))
age_groups = pd.cut(titanic['age'], bins=range(0, 81, 5))
fare_by_age = titanic.groupby(age_groups)['fare'].mean()

plt.plot(range(0, 80, 5), fare_by_age.values, marker='o', linewidth=2)
plt.title('Average Fare by Age Group', fontsize=16)
plt.xlabel('Age Range', fontsize=12)
plt.ylabel('Average Fare ($)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(range(0, 81, 10))
plt.tight_layout()
plt.savefig('fare_by_age.png')
plt.close()

# 2. Bar chart showing survival rate by class
plt.figure(figsize=(10, 6))
survival_by_class = titanic.groupby('class')['survived'].mean() * 100
sns.barplot(x=survival_by_class.index, y=survival_by_class.values)
plt.title('Survival Rate by Passenger Class', fontsize=16)
plt.xlabel('Passenger Class', fontsize=12)
plt.ylabel('Survival Rate (%)', fontsize=12)
plt.ylim(0, 100)
for i, v in enumerate(survival_by_class.values):
    plt.text(i, v + 2, f"{v:.1f}%", ha='center', fontsize=10)
plt.tight_layout()
plt.savefig('survival_by_class.png')
plt.close()

# 3. Histogram of passenger ages
plt.figure(figsize=(12, 6))
sns.histplot(data=titanic, x='age', bins=30, kde=True)
plt.title('Distribution of Passenger Ages', fontsize=16)
plt.xlabel('Age', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('age_distribution.png')
plt.close()

# 4. Scatter plot of fare vs. age colored by survival
plt.figure(figsize=(12, 8))
sns.scatterplot(data=titanic, x='age', y='fare', hue='survived', 
                palette={0: 'red', 1: 'green'}, alpha=0.7, s=100)
plt.title('Fare vs. Age Colored by Survival', fontsize=16)
plt.xlabel('Age', fontsize=12)
plt.ylabel('Fare ($)', fontsize=12)
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fare_vs_age.png')
plt.close()

# Additional visualization: Survival rate by sex and class
plt.figure(figsize=(12, 6))
sns.barplot(data=titanic, x='class', y='survived', hue='sex')
plt.title('Survival Rate by Class and Sex', fontsize=16)
plt.xlabel('Passenger Class', fontsize=12)
plt.ylabel('Survival Rate', fontsize=12)
plt.legend(title='Sex')
plt.tight_layout()
plt.savefig('survival_by_class_sex.png')
plt.close()

# Print some key insights from our analysis
print("\nKey Insights from the Analysis:")
print(f"1. Overall survival rate: {titanic['survived'].mean()*100:.1f}%")
print(f"2. First class passengers had a {survival_by_class['First']:.1f}% survival rate, " 
      f"compared to only {survival_by_class['Third']:.1f}% for third class.")
print(f"3. The average fare for first class was ${titanic.groupby('class')['fare'].mean()['First']:.2f}, " 
      f"compared to ${titanic.groupby('class')['fare'].mean()['Third']:.2f} for third class.")
print(f"4. Female survival rate: {titanic[titanic['sex'] == 'female']['survived'].mean()*100:.1f}%, " 
      f"Male survival rate: {titanic[titanic['sex'] == 'male']['survived'].mean()*100:.1f}%")
print(f"5. The median age of passengers was {titanic['age'].median():.1f} years.")