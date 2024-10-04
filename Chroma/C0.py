import chromadb
from datasets import load_dataset
from time import time
import numpy as np

def load_our_dataset():
    """ Carga nuestro dataset """
    print("Cargando el Book Corpus dataset")
    return load_dataset("williamkgao/bookcorpus100mb")

if __name__ == '__main__':
    # Cargar el dataset
    dataset = load_our_dataset()

    #configuramos chroma
    chroma_client = chromadb.PersistentClient(path="./ChromaDB")

    # Crear la coleccion en la base de datos
    #chroma_client.reset() # Borramos la base de datos si ya existe
    collection = chroma_client.create_collection(name='bookCorpus')
    collection_cosine = chroma_client.create_collection(name='bookCorpusCosine', metadata={"hnsw:space": "cosine"})

    print("Se ha creado la bd correctamente.")

    # Almacenamos los tiempos de inserción
    individual_insertion_times = []
    ini_total_insertion_time = time()  # Inicio del tiempo total de insercion

    #hacemos las inserciones en batch porque sino tarda mucho (mayor el batch_size, menor el tiempo de insercion)
    batch_size = 1000
    for i in range(0, len(dataset['train']['text'][:20000]), batch_size):
        ini_individual_time = time()
        batch = dataset['train']['text'][i:i + batch_size]
        ids = [f"id_{j}" for j in range(i, i + len(batch))]
        collection.add(documents=batch, ids=ids)
        end_individual_time = time()
        individual_insertion_times.append(end_individual_time - ini_individual_time)

    end_total_insertion_time = time()  # Fin del tiempo total de inserción
    print("Se han insertado correctamente 20,000 sentencias.")

    if individual_insertion_times:
        min = min(individual_insertion_times)
        max = max(individual_insertion_times)
        avg = np.mean(individual_insertion_times)
        std_dev = np.std(individual_insertion_times)

        print(f"Tiempo mínimo de inserción: {min:.6f} segundos")
        print(f"Tiempo máximo de inserción: {max:.6f} segundos")
        print(f"Tiempo promedio de inserción: {avg:.6f} segundos")
        print(f"Desviación estándar de inserción: {std_dev:.6f} segundos")
        print(f"Tiempo total de inserción: {end_total_insertion_time - ini_total_insertion_time:.6f} segundos")

        individual_insertion_times_cosine = []
        ini_total_insertion_time = time()  # Inicio del tiempo total de insercion

        print("Insertando elementos en la coleccion con metrica de calculo de distancia coseno...")
        # hacemos las inserciones en batch porque sino tarda mucho (mayor el batch_size, menor el tiempo de insercion)
        for i in range(0, len(dataset['train']['text'][:20000]), batch_size):
            ini_individual_time = time()
            batch = dataset['train']['text'][i:i + batch_size]
            ids = [f"id_{j}" for j in range(i, i + len(batch))]
            collection_cosine.add(documents=batch, ids=ids)
            end_individual_time = time()
            individual_insertion_times_cosine.append(end_individual_time - ini_individual_time)
        end_total_insertion_time = time()  # Fin del tiempo total de inserción
        print("Se han insertado correctamente 20,000 sentencias.")

        if individual_insertion_times_cosine:
            #Esta raro porque de otra manera no me funciona
            #ordena los tiempos de insercion de menor a mayor
            individual_insertion_times_cosine.sort()
            avg_cos = np.mean(individual_insertion_times_cosine)
            std_dev_cos = np.std(individual_insertion_times_cosine)

            print(f"Tiempo mínimo de inserción: {individual_insertion_times_cosine[0]:.6f} segundos")
            print(f"Tiempo máximo de inserción: {individual_insertion_times_cosine[19]:.6f} segundos")
            print(f"Tiempo promedio de inserción: {avg_cos:.6f} segundos")
            print(f"Desviación estándar de inserción: {std_dev_cos:.6f} segundos")
            print(f"Tiempo total de inserción: {end_total_insertion_time - ini_total_insertion_time:.6f} segundos")



