import psycopg2
from config import load_config
from datasets import load_dataset
import time
import numpy as np

def load_our_dataset():
    """ Load our dataset """
    print("Loading Book Corpus dataset")
    return load_dataset("williamkgao/bookcorpus100mb")


def connect(config):
    """ Connect to the PostgreSQL database server """
    try:
        with psycopg2.connect(**config) as conn:
            print('Connected to the PostgreSQL server.')
            return conn
    except (psycopg2.DatabaseError, Exception) as error:
        print(error)


if __name__ == '__main__':
    # Configuracion de la conexion con la base de datos
    config = load_config()
    conn = connect(config)

    # Obtenemos los datos de los 10 primeros registros de nuestra tabla
    cur = conn.cursor()

    select_query = '''
    SELECT * FROM bookCorpus LIMIT 10;
    '''
    cur.execute(select_query)
    sentences = [record for record in cur.fetchall()]

    # Buscamos los embeddings de las sentencias que acabamos de obtener
    select_query = '''
    SELECT * FROM bookCorpusEmbeddings WHERE sentence_id = %s;
    '''
    embeddings = []
    for sentence in sentences:
        cur.execute(select_query, (sentence[0],))
        embeddings.append(cur.fetchone())

    # Almacenamos todos los embeddings de al bd en un array
    select_query = '''
    SELECT * FROM bookCorpusEmbeddings;
    '''
    cur.execute(select_query)
    all_embeddings = [record for record in cur.fetchall()]

    # almacenamos la lista total de sentencias para poder compararlas despues
    select_query = '''
    SELECT * FROM bookCorpus;
    '''
    cur.execute(select_query)
    all_sentences = [record for record in cur.fetchall()]


    print("Calculando las distancias euclidianas...")

    euclidean_times = []  # Tiempos

    total_euclidean_start_time = time.time()  # Inicio total Euclidiana

    for find_embedding in embeddings:
        start_time = time.time()  # Inicio individual
        distances = []
        id_find_embedding = int(find_embedding[0])
        find_embedding = np.array(find_embedding[2])

        for search_embedding in all_embeddings:
            id_search_embedding = int(search_embedding[0])
            if id_find_embedding != id_search_embedding:
                search_embedding = np.array(search_embedding[2])
                euclidean = np.sqrt(np.sum(np.square(find_embedding - search_embedding)))
                save = [id_search_embedding, euclidean]
                distances.append(save)

        distances.sort(key=lambda x: x[1])  # Ordenar por distancia (mas pequeña a mas grande)
        end_time = time.time()  # Finalización individual
        euclidean_times.append(end_time - start_time)  # Guardar el tiempo individual

        # Imprimir las 2 distancias más pequeñas
        print(f"\nLas 2 distancias más pequeñas para la sentencia '{sentences[id_find_embedding - 1][1]}' son:")
        print(
            f"ID Sentencia: {distances[0][0]}, Distancia: {distances[0][1]:.4f}, Sentencia: {all_sentences[distances[0][0] - 1][1]}")
        print(
            f"ID Sentencia: {distances[1][0]}, Distancia: {distances[1][1]:.4f}, Sentencia: {all_sentences[distances[1][0] - 1][1]}")

    # Calcular estadísticas de los tiempos euclidianos
    if euclidean_times:
        min_time = min(euclidean_times)
        max_time = max(euclidean_times)
        avg_time = np.mean(euclidean_times)
        std_dev_time = np.std(euclidean_times)

        print(f"\nTiempo minimo de cálculo Euclidiana: {min_time:.6f} segundos")
        print(f"Tiempo maximo de cálculo Euclidiana: {max_time:.6f} segundos")
        print(f"Tiempo promedio de cálculo Euclidiana: {avg_time:.6f} segundos")
        print(f"Desviacion estandar de cálculo Euclidiana: {std_dev_time:.6f} segundos")

        total_euclidean_end_time = time.time()  # Tiempo de finalización total Euclidiana
        total_euclidean_time = total_euclidean_end_time - total_euclidean_start_time  # Tiempo total Euclidiana
        print(f"\nTiempo total de calculo Euclidiana: {total_euclidean_time:.6f} segundos")


    print("\n\n\n")

    # --- Calcular las distancias Manhattan ---
    print("Calculando las distancias Manhattan...")

    manhattan_times = []

    total_manhattan_start_time = time.time()  # Inicio total Manhattan

    for find_embedding in embeddings:
        start_time = time.time()  # Inicio individual
        distances = []
        id_find_embedding = int(find_embedding[0])
        find_embedding = np.array(find_embedding[2])

        for search_embedding in all_embeddings:
            id_search_embedding = int(search_embedding[0])
            if id_find_embedding != id_search_embedding:
                search_embedding = np.array(search_embedding[2])
                manhattan = np.sum(np.abs(find_embedding - search_embedding))
                save = [id_search_embedding, manhattan]
                distances.append(save)

        distances.sort(key=lambda x: x[1])  # Ordenar por distancia (mas pequeña a mas grande)
        end_time = time.time()  # Finalización individual
        manhattan_times.append(end_time - start_time)  # Guardar tiempo individual

        # Imprimir las 2 distancias más pequeñas
        print(f"\nLas 2 distancias más pequeñas para la sentencia '{sentences[id_find_embedding - 1][1]}' son:")
        print(
            f"ID Sentencia: {distances[0][0]}, Distancia: {distances[0][1]:.4f}, Sentencia: {all_sentences[distances[0][0] - 1][1]}")
        print(
            f"ID Sentencia: {distances[1][0]}, Distancia: {distances[1][1]:.4f}, Sentencia: {all_sentences[distances[1][0] - 1][1]}")

    # Calcular estadisticas de los tiempos Manhattan
    if manhattan_times:
        min_time = min(manhattan_times)
        max_time = max(manhattan_times)
        avg_time = np.mean(manhattan_times)
        std_dev_time = np.std(manhattan_times)

        print(f"\nTiempo minimo de cálculo Manhattan: {min_time:.6f} segundos")
        print(f"Tiempo maximo de cálculo Manhattan: {max_time:.6f} segundos")
        print(f"Tiempo promedio de cálculo Manhattan: {avg_time:.6f} segundos")
        print(f"Desviacion estándar de cálculo Manhattan: {std_dev_time:.6f} segundos")

        total_manhattan_end_time = time.time()  # Tiempo de finalización total Manhattan
        total_manhattan_time = total_manhattan_end_time - total_manhattan_start_time  # Tiempo total Manhattan
        print(f"\nTiempo total de calculo Manhattan: {total_manhattan_time:.6f} segundos")

    # Cerrar conexion
    cur.close()
    conn.close()
