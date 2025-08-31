import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sea

#treating the data
df = pd.read_excel("/home/gabriel-cruz/Área de trabalho/projeto(2)_energy+efficiency/data/ENB2012_data.xlsx")

df.columns = df.columns.str.strip()     #removing extra space
df.columns = df.columns.str.lower()     #turns the data to lower

cols = ["compactibilidade relativa","Área da superfície","Área da parede","Área do teto",
        "Altura","Orientação","Área da vidraça","Distribuição da área de vidro",
        "Carga de calor","Carga de resfriamento"]

cols = [col for col in cols if col in df.columns]       #ensuring original columns

#Initial view 
print(df.head())
print(df.describe())
print((df.isnull()).sum())


#Plotting the data with correlation matrix
plt.figure(figsize = (10,15))
sea.heatmap(df.corr(), annot = True, cmap = 'coolwarm')     #heatmap from seaborn
plt.title("Correlation matrix")
plt.show()

#Seeing the relationship among multiple variables in dataset
sea.pairplot(df[cols], height=10)
plt.show()