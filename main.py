import pandas as pd
import sys
import matplotlib.pyplot as plt


# Es guarden les sortides en un arxiu de text
sys.stdout = open('resultat.txt', 'w')

# Es carrega el conjunt de dades de l'arxiu CSV
data = pd.read_csv("sample_wildlife_vehicle_collision_spain.csv", delimiter=';')

# Es mostren les primeres files del dataset per veure l'estructura
print(data.head())

# Informació general del dataset 
print(data.info())


## Quantiat de registres i de variables
# Nombre de registres al conjunt de dades 
num_registres = len(data)
print("Nombre de registres:", num_registres)

# Nombre de variables al conjunt de dades
num_variables = len(data.columns)
print("Nombre de variables:", num_variables)


## Tipus de dades 
# S'exploren les estadistiques descriptives per les variables numèriques
print(data.describe().iloc[:, :10])
print(data.describe().iloc[:, 10:20])
print(data.describe().iloc[:, 20:30])
print(data.describe().iloc[:, 30:40])
print(data.describe().iloc[:, 40:])

# Es conten el nombre de variables numèriques i categòriques
num_numeric_variables = data.select_dtypes(include=['number']).shape[1]
num_categorical_variables = data.select_dtypes(include=['object']).shape[1]

# S'obtenen  els noms de les variables numèriques i categòriques
numeric_variable_names = data.select_dtypes(include=['number']).columns.tolist()
categorical_variable_names = data.select_dtypes(include=['object']).columns.tolist()

# S'imprimeixen els resultats
print("Nombre de variables numèriques:", num_numeric_variables)
print("Nom de les variables numèriques:", numeric_variable_names)
print()
print("Nombre de variables categòriques:", num_categorical_variables)
print("Nom de variables categòriques:", categorical_variable_names)


## Exploració de les variables categòriques 
# Es conten la quantitat de valors unics en cada variable categòrica
variables_categoriques = data.select_dtypes(include=['object']).columns
for variable in variables_categoriques:
    print(f"Variable: {variable}")
    print(data[variable].value_counts())
    print()


## Exploració variables numèriques
# Es visualitza la distribució de les variables numèriques amb histogrames
data.select_dtypes(include=['int64', 'float64']).hist(figsize=(20, 20))
plt.tight_layout(pad=1.0) 
plt.savefig("histogrames.png") 
plt.close() 


## Combinació de dades categòriques y numèriques:
# El nombre total de ferits sense hospitalització per accident per dia de la setmana
ferits_sense_hospitalitzacio_per_dia_setmana = data.groupby('nombre_dia_semana')['total_hl30df'].sum()
print("Nombre total de morts per dia de la setmana:")
print(ferits_sense_hospitalitzacio_per_dia_setmana)







# Restauració del output estàndar a consola 
sys.stdout.close()
sys.stdout = sys.__stdout__
