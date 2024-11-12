import pandas as pd
import random as rn
import sklearn as sk

# 2.1 Lectura y an ́alisis exploratorio de datos

cargar = pd.read_csv(r'C:\Users\ignac\OneDrive\Documentos Online\Código\Diplomado Python\Curso 3\datos\ejemplo_data.csv', sep=',', encoding= 'utf-8')

print(cargar.info())

cargar["ID"] = cargar["ID"].astype(int)
cargar["Activo"] = cargar["Activo"].astype(bool)

# print(cargar.info())
print(cargar.dtypes)

cargar["Unidades"] = cargar["Unidades"].replace('[\$,.]', '', regex=True)
cargar["Unidades"] = pd.to_numeric(cargar["Unidades"])
cargar["Unidades"] = cargar["Unidades"].astype(int)

cargar["2016"] = cargar["2016"].replace('[\$,.]', '', regex=True)
cargar["2016"] = cargar["2016"].astype(float)

# print(cargar.info())
print(cargar.dtypes)

# 2.2 Lectura y an ́alisis exploratorio de datos 2

cargar2 = pd.read_csv(r"C:\Users\ignac\OneDrive\Documentos Online\Código\Diplomado Python\Curso 3\datos\ecommerce_data.csv", sep=",",encoding='ISO-8859-1')

print("Primera impresión: \n")
# print(cargar2.info())
print(cargar2.dtypes)

# Utilizando la función astype transforme el atributo ”InvoiceNo” a entero y el atributo ”Description”
# a string. Vuelva a consultar el estado de las variables.

cargar2["InvoiceNo"] = pd.to_numeric(cargar2["InvoiceNo"], errors="coerce")
cargar2["InvoiceNo"] = cargar2["InvoiceNo"].fillna(0)
cargar2["InvoiceNo"] = cargar2["InvoiceNo"].astype(int)

cargar2["Description"] = cargar2["Description"].astype(str)

print("\nSegunda impresión: \n")
# print(cargar2.info())
print(cargar2.dtypes)

# Convierta el atributo ”Quantity” a entero y ”UnitPrice” a flotante.

cargar2["Quantity"] = cargar2["Quantity"].astype(int)
cargar2["UnitPrice"] = cargar2["UnitPrice"].astype(float)

# La columna ”InvoiceDate” contiene un string que representa ”fecha-hora”, separe la columna en
# dos columnas que representen cada atributo por separado.

cargar2[["InvoiceFecha", "InvoiceHora"]] = cargar2["InvoiceDate"].str.split(" ", expand=True)
print("Columnas creadas sin problemas.")

# Añada una nueva columna que represente el monto total para cada boleta.

cargar2["MontoTotal"] = cargar2["Quantity"] * cargar2["UnitPrice"]
print("Columna creada sin problemas.")

# Exporte la base de datos procesada en formato ”.csv”

cargar2.to_csv(r"C:\Users\ignac\OneDrive\Documentos Online\Código\Diplomado Python\Curso 3\datos\ecomm_mod.csv", index=False, encoding="ISO-8859-1")
print("Archivo creado sin problemas.")

# # 2.3 Estadísticas Descriptivas

nacionalidades = ["Chileno", "Argentino", "Brasileño", "Colombiano", "Mexicano", "Canadiense", "Español", "Francés", "Italiano", "Alemán"]
coloresPiel = ["Blanco", "Pálido", "Moreno", "Negro"]

personas = {}

for x in range(1, 51):
    altura = round(rn.uniform(1.5, 2.0), 2)
    coeficienteIntelectual = rn.randint(85, 130)
    peso = round(rn.uniform(50, 100),0)
    colorPiel = rn.choice(coloresPiel)
    nacionalidad = rn.choice(nacionalidades)
    
    personas[f"persona_{x}"] = [altura, coeficienteIntelectual, peso, colorPiel, nacionalidad]

# Muestra el diccionario
# for x,y in personas.items():
#     print(f"{x} : {y}")

# Transformar el diccionario a dataframe de pandas

dfPersonas = pd.DataFrame.from_dict(personas, orient='index', columns=["Altura","Coeficiente Intelectual", "Peso", "Color de Piel", "Nacionalidad"])

# Medidas tendencia central y dispersión

medidasTendenciaCentral = dfPersonas[["Altura","Coeficiente Intelectual", "Peso"]].agg(["mean", "median"])
medidasTendenciaCentral.loc["mean"] = round(medidasTendenciaCentral.loc["mean"],2)

print(medidasTendenciaCentral)

medidasDispersion = dfPersonas[["Altura","Coeficiente Intelectual", "Peso"]].agg(["std", "var"])
print(medidasDispersion)

# 2.4. Transformación e imputación de datos

# Cargar las bases de datos de nombre ”ratings data.csv” y ”books data.csv

ratings = pd.read_csv(r"C:\Users\ignac\OneDrive\Documentos Online\Código\Diplomado Python\Curso 3\datos\ratings_data.csv", sep=",",encoding='ISO-8859-1')
books = pd.read_csv(r"C:\Users\ignac\OneDrive\Documentos Online\Código\Diplomado Python\Curso 3\datos\books_data.csv", sep=";",encoding='ISO-8859-1')
print("Archivos cargados correctamente.")

# Utilizando "ratings data.csv" genere un diagnóstico de números perdidos. Luego impute los valores
# de acuerdo a la media y de acuerdo a otro criterio seleccionado por usted. Explore las opciones de
# imputación del método fillna() de Pandas.

# Si entiendo bien la indicación, hay instrucciones ocultas o implícitas: los valores a to_numeric, 
# reemplazar los que creo valores perdidos con NaN (por ejemplo, los ceros) y finalmente 
# reemplazar por la media. No me queda claro la necesidad de importar sklearn, 
# quizás ahí hay algo que lo hace todo más fácil.
