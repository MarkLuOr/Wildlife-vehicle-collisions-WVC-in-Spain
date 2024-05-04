import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns


# Es guarden les sortides en un arxiu de text
sys.stdout = open('resultat.txt', 'w')

# Es carrega el conjunt de dades de l'arxiu CSV
data = pd.read_csv("sample_wildlife_vehicle_collision_spain.csv", delimiter=';')

# Es mostren les primeres files del dataset per veure l'estructura
print(data.head())

# Informació general del dataset 
print(data.info())

# Variables amb valors Nan
variables_nan = data.columns[data.isnull().any()].tolist()

# Total valors faltants per variables
totals_nuls_per_variable = data[variables_nan].isnull().sum()

print("Variables amb valors Nulls:")
for variable in variables_nan:
    print(f"{variable}: {totals_nuls_per_variable[variable]} valors nulls")


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

# Gràfics de dispersió per variables numèriques
numeric_variables = data.select_dtypes(include=['int64', 'float64']).columns
for i, variable in enumerate(numeric_variables):
    for j, other_variable in enumerate(numeric_variables):
        if i != j:
            plt.scatter(data[variable], data[other_variable])
            plt.xlabel(variable)
            plt.ylabel(other_variable)
            plt.title(f"Scatter Plot: {variable} vs {other_variable}")
            plt.tight_layout(pad=1.0) 
            plt.savefig("gràfic_disperió_numèriques.png") 
            plt.close()

# Es seleccionen variables numèriques representatives en termes de condicions climàtiques
selected_numeric_variables = ['tmed', 'prec', 'sol', 'altitud', 'pendiente', 'imd_total', 'maxspeed']

# Es crea la matriu de dispersió 
sns.pairplot(data[selected_numeric_variables])
plt.savefig("scatter_matrix.png")
plt.close()


# Gràfics de barres apilades per les variables categòriques
categorical_variables = data.select_dtypes(include=['object']).columns
num_cat_vars = len(categorical_variables)
fig, axs = plt.subplots(num_cat_vars, 1, figsize=(15, 5*num_cat_vars))

for i, variable in enumerate(categorical_variables):
    value_counts = data[variable].value_counts()
    
    value_counts.plot(kind='bar', stacked=True, ax=axs[i])
    
    # Es roten les etiquetes de x per facilitar llegibilitat
    axs[i].set_xticklabels(axs[i].get_xticklabels(), rotation=45, ha='right')
    
    axs[i].set_title(f"Stacked Bar Plot for {variable}")
    axs[i].set_xlabel(variable)
    axs[i].set_ylabel("Count")

plt.tight_layout(pad=1.0) 
plt.savefig("barres_apilades_categòriques.png") 
plt.close()


## Combinació de dades categòriques i numèriques:
# El nombre total de ferits sense hospitalització per accident per dia de la setmana
ferits_sense_hospitalitzacio_per_dia_setmana = data.groupby('nombre_dia_semana')['total_hl30df'].sum()
print("Nombre total de ferits sense hospitalització per dia de la setmana:")
print(ferits_sense_hospitalitzacio_per_dia_setmana)

# Total de ferits sense hospitalització per accident per tipus d'animal
total_ferits_animal = data.groupby('nombre_tipo_animal_1f')['total_hl30df'].sum()
print("Total de ferits sense hospitalització per tipus d'animal:")
print(total_ferits_animal)

# Promig de la intensitat mitja diària de tràfic per tipus de carretera
promig_trafic_tipus_carretera = data.groupby('nombre_tipo_via')['imd_total'].mean()
print("Promig de la intensitat mitja diària de tràfic per tipus de carretera:")
print(promig_trafic_tipus_carretera)

# Combinació de variables categòriques: part del dia, mes i província
total_accidents_dia_mes_provincia = data.groupby(['parte_dia', 'nombre_mes', 'nombre_provincia'])['id_num'].count()

# Ordenar els resultats de manera descendent i obtenir el top 10
top_10_accidents = total_accidents_dia_mes_provincia.sort_values(ascending=False).head(10)
print("Top 10 de combinacions de part del dia, mes i província amb més accidents:")
print(top_10_accidents)


## Anàlisis de la correlació
# Matriu de correlació entre variables numèriques
correlation_matrix = data.corr()
print("Matriu de correlació entre variables numèriques:")
print(correlation_matrix)

# Visualizació de la matriu de correlació amb un mapa de calor
plt.figure(figsize=(20, 20))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matriu de Correlació")
plt.tight_layout(pad=1.0) 
plt.savefig("Matriu de Correlació_v_num.png") 
plt.close()


### Anàlisis de casos en concret, per inspeccionar

# Top 10 províncies amb més accidents i animal predominant
top_10_provinciess_accidents = data['nombre_provincia'].value_counts().head(10).index.tolist()

animal_predominant_per_provincia = {}
for provincia in top_10_provinciess_accidents:
    animal_predominant = data[data['nombre_provincia'] == provincia]['nombre_tipo_animal_1f'].mode()[0]
    animal_predominant_per_provincia[provincia] = animal_predominant

print("Top 10 províncies amb més accidents i el seu animal predominant:")
for provincia, animal in animal_predominant_per_provincia.items():
    print(f"Provínca: {provincia}, Animal predominant: {animal}")

# Províncies de Galicia amb més accidents i animal predominant
provincies_galicia = data[data['nombre_ccaa'] == 'Galicia']['nombre_provincia']
top_10_provincies_galicia_accidents = provincies_galicia.value_counts().head(10).index.tolist()

animal_predominant_galicia= {}
for provincia in top_10_provincies_galicia_accidents:
    animal_predominant = data[(data['nombre_ccaa'] == 'Galicia') & (data['nombre_provincia'] == provincia)]['nombre_tipo_animal_1f'].mode()[0]
    animal_predominant_galicia[provincia] = animal_predominant

print("\nTop 10 províncies de Galícia amb més accidents i el seu animal predominant:")
for provincia, animal in animal_predominant_galicia.items():
    print(f"Provincia: {provincia}, Animal predominant: {animal}")


# Restauració del output estàndar a consola 
sys.stdout.close()
sys.stdout = sys.__stdout__
