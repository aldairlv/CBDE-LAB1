import psycopg2
from config import load_config
import time
import numpy as np

def connect(config):
    """ Connect to the PostgreSQL database server """
    try:
        with psycopg2.connect(**config) as conn:
            print('Connected to the PostgreSQL server.')
            return conn
    except (psycopg2.DatabaseError, Exception) as error:
        print(error)

def calculate_and_print_statistics(parcial_times, total_time, name_method):
    if parcial_times:
        min_time = min(parcial_times)
        max_time = max(parcial_times)
        avg_time = np.mean(parcial_times)
        std_dev_time = np.std(parcial_times)

        print(f"\nTiempo minimo de cálculo {name_method}: {min_time:.6f} segundos")
        print(f"Tiempo maximo de cálculo {name_method}: {max_time:.6f} segundos")
        print(f"Tiempo promedio de cálculo {name_method}: {avg_time:.6f} segundos")
        print(f"Desviacion estandar de cálculo {name_method}: {std_dev_time:.6f} segundos")
    
    print(f"\nTiempo total de calculo {name_method}: {total_time:.6f} segundos")

def two_min_dist(sentences,id_find_embedding,distances,all_sentences):
    print(f"\nLas 2 distancias más pequeñas para la sentencia '{sentences[id_find_embedding - 1][1]}' son:")
    print(
        f"ID Sentencia: {distances[0][0]}, Distancia: {distances[0][1]:.4f}, Sentencia: {all_sentences[distances[0][0] - 1][1]}")
    print(
        f"ID Sentencia: {distances[1][0]}, Distancia: {distances[1][1]:.4f}, Sentencia: {all_sentences[distances[1][0] - 1][1]}")

def get_data(limit,cur):
    select_query = f'''
    SELECT * FROM bookCorpus LIMIT {limit};
    '''
    cur.execute(select_query)
    return [record for record in cur.fetchall()]

def get_sentences_embeddings(sentences):
    select_query = '''
    SELECT * FROM bookCorpusEmbeddings WHERE sentence_id = %s;
    '''
    embeddings = []
    for sentence in sentences:
        cur.execute(select_query, (sentence[0],))
        embeddings.append(cur.fetchone())
    return embeddings

def get_all_embeddings(embeddings):
    select_query = '''
    SELECT * FROM bookCorpusEmbeddings;
    '''
    cur.execute(select_query)
    return [record for record in cur.fetchall()]

def get_all_sentences():
    select_query = '''
    SELECT * FROM bookCorpus;
    '''
    cur.execute(select_query)
    return [record for record in cur.fetchall()]

def calcular_distancia(method,embeddings,all_embeddings,sentences,all_sentences):
    times = []  # Tiempos
    total_start_time = time.time()  # Inicio total Euclidiana

    for find_embedding in embeddings:
        start_time = time.time()  # Inicio individual
        distances = []
        id_find_embedding = int(find_embedding[0])
        find_embedding = np.array(find_embedding[2])

        for search_embedding in all_embeddings:
            id_search_embedding = int(search_embedding[0])
            if id_find_embedding != id_search_embedding:
                search_embedding = np.array(search_embedding[2])
                if method == "euclidian":
                    dist = np.sqrt(np.sum(np.square(find_embedding - search_embedding)))
                elif method == "cosine":
                    dist = np.dot(find_embedding, search_embedding) / (np.linalg.norm(find_embedding) * np.linalg.norm(search_embedding))
                save = [id_search_embedding, dist]
                distances.append(save)

        if method == "euclidian":
            distances.sort(key=lambda x: x[1])  # Ordenar por distancia (mas pequeña a mas grande)
        elif method == "cosine":
            distances.sort(key=lambda x: x[1], reverse=True)  # Ordenar por distancia (mas pequeña a mas grande)
        end_time = time.time()  # Finalización individual
        times.append(end_time - start_time)  # Guardar el tiempo individual

        # Imprimir las 2 distancias más pequeñas
        two_min_dist(sentences,id_find_embedding,distances,all_sentences)

    total_end_time = time.time()  # Tiempo de finalización total Euclidiana
    total_time = total_end_time - total_start_time  # Tiempo total Euclidiana
    return times, total_time

if __name__ == '__main__':
    # Configuracion de la conexion con la base de datos
    config = load_config()
    conn = connect(config)

    # Obtenemos un cursor para trabajar sobre la base de datos
    cur = conn.cursor()

    # Obtenemos los datos de los 10 primeros registros de nuestra tabla
    limit = 10
    sentences = get_data(limit,cur)

    # Buscamos los embeddings de las sentencias que acabamos de obtener
    embeddings = get_sentences_embeddings(sentences)

    # Almacenamos todos los embeddings de al bd en un array
    all_embeddings = get_all_embeddings(embeddings)
    
    # Almacenamos la lista total de sentencias para poder compararlas despues
    all_sentences = get_all_sentences()

    print("Calculando las distancias euclidianas...")
    euclidean_times, total_euclidean_time = calcular_distancia("euclidian",embeddings,all_embeddings,sentences,all_sentences)

    # Calcular estadísticas de los tiempos euclidianos
    calculate_and_print_statistics(euclidean_times, total_euclidean_time,"Euclidiana")

    print("\n\n\n")

    # --- Calcular las distancias Coseno ---
    print("Calculando las distancias Coseno...")
    cosine_times, total_cosine_time = calcular_distancia("cosine",embeddings,all_embeddings,sentences,all_sentences)
    
    # Calcular estadisticas de los tiempos Coseno
    calculate_and_print_statistics(cosine_times, total_cosine_time,"Coseno")
   
    # Cerrar conexion
    cur.close()
    conn.close()
