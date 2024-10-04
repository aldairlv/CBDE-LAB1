import chromadb
from datasets import load_dataset
from chromadb.utils import embedding_functions
from time import time
import warnings
import numpy as np

# Esto es porque me genera un warning de un tokkenizador que no es necesario, de esta manera no aparece
warnings.simplefilter("ignore", category=FutureWarning)


def load_our_dataset():
    """ Carga nuestro dataset """
    print("Cargando el Book Corpus dataset")
    return load_dataset("williamkgao/bookcorpus100mb")


if __name__ == '__main__':
    # configuramos chroma
    chroma_client = chromadb.PersistentClient(path="./ChromaDB")

    # La default embedding function es la all-MiniLM-L6-v2
    default_ef = embedding_functions.DefaultEmbeddingFunction()

    # Obtenemos la coleccion en la base de datos
    collection_euclidean = chroma_client.get_collection(name='bookCorpus', embedding_function=default_ef)
    collection_cosine = chroma_client.get_collection(name='bookCorpusCosine', embedding_function=default_ef)
    print(f"Se han obtenido un total de {collection_euclidean.count()} sentencias de la coleccion.")

    individual_times = []
    start_time_euclidean = time()
    for i in range(0, 10):
        sentence = collection_euclidean.get(ids=[f"id_{i}"])
        sentencia = sentence["documents"][0]
        individual_initial_time = time()
        result = collection_euclidean.query(query_texts =[sentencia], n_results=3) #el primer resultado es la misma sentencia
        end_individual_time = time()
        print(f"Las sentencias mas parecidas a {sentencia}, usando el metodo Euclidiano son:")
        print(f"1. ID: {result['ids'][0][1]}, Sentencia: {result['documents'][0][1]}, Distancia: {result['distances'][0][1]}")
        print(f"2. ID: {result['ids'][0][2]}, Sentencia: {result['documents'][0][2]}, Distancia: {result['distances'][0][2]}")
        print("\n")
        individual_times.append(end_individual_time - individual_initial_time)

    end_time_euclidean = time()

    if individual_times:
        min_time = min(individual_times)
        max_time = max(individual_times)
        avg_time = np.mean(individual_times)
        std_dev_time = np.std(individual_times)

        print(f"\nTiempo minimo de cálculo Euclidiana: {min_time:.6f} segundos")
        print(f"Tiempo maximo de cálculo Euclidiana: {max_time:.6f} segundos")
        print(f"Tiempo promedio de cálculo Euclidiana: {avg_time:.6f} segundos")
        print(f"Desviacion estandar de cálculo Euclidiana: {std_dev_time:.6f} segundos")

        print(f"Tiempo total de obtener los 10 primeros resultados: {end_time_euclidean - start_time_euclidean:.6f} segundos")

    individual_times_cosine = []
    start_time_cosine = time()
    for i in range(0, 10):
        sentence = collection_euclidean.get(ids=[f"id_{i}"])
        sentencia = sentence["documents"][0]
        individual_initial_time = time()
        result = collection_cosine.query(query_texts=[sentencia],
                                            n_results=3)  # el primer resultado es la misma sentencia
        end_individual_time = time()
        print(f"Las sentencias mas parecidas a {sentencia}, usando el metodo del Coseno son:")
        print(
            f"1. ID: {result['ids'][0][1]}, Sentencia: {result['documents'][0][1]}, Distancia: {result['distances'][0][1]}")
        print(
            f"2. ID: {result['ids'][0][2]}, Sentencia: {result['documents'][0][2]}, Distancia: {result['distances'][0][2]}")
        print("\n")
        individual_times_cosine.append(end_individual_time - individual_initial_time)

    end_time_cosine = time()

    if individual_times_cosine:
        min_time = min(individual_times_cosine)
        max_time = max(individual_times_cosine)
        avg_time = np.mean(individual_times_cosine)
        std_dev_time = np.std(individual_times_cosine)

        print(f"\nTiempo minimo de cálculo Euclidiana: {min_time:.6f} segundos")
        print(f"Tiempo maximo de cálculo Euclidiana: {max_time:.6f} segundos")
        print(f"Tiempo promedio de cálculo Euclidiana: {avg_time:.6f} segundos")
        print(f"Desviacion estandar de cálculo Euclidiana: {std_dev_time:.6f} segundos")

        print(
            f"Tiempo total de obtener los 10 primeros resultados: {end_time_cosine - start_time_cosine:.6f} segundos")




