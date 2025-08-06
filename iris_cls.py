import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix

df = pd.read_csv("Iris.csv")

if 'Id' in df.columns:
    df = df.drop(columns=['Id'])

print("First 5 rows: ")
print(df.head())
print("\nData summary: ")
print(df.describe())

sns.pairplot(df, hue='Species')
plt.suptitle("Pairplot pf Iris Features",y=1.02)
plt.show()

X = df.drop("Species", axis=1)
y = df["Species"]

X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=0.2,random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nclassification report : \n", classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test,y_pred)
sns.heatmap(conf_matrix, annot= True,cmap="blues", fmt='d', xticklabels=model.classes_,yticklabels=model.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()