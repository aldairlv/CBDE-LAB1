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


if __name__ == '__main__':
    # Configuracion de la conexion con la base de datos
    config = load_config()
    conn = connect(config)

    # Obtenemos los datos de los 10 primeros registros de nuestra tabla
    cur = conn.cursor()

    sentences = []
    select_query = '''
    SELECT * FROM bookCorpusPG WHERE id = %s;
    '''
    #buscamos los elementos del 1 al 10
    for i in range(1, 11):
        cur.execute(select_query, (i,))
        sentences.append(cur.fetchone())

    print("Calculando las distancias euclidianas...")

    euclidean_times = []  # Tiempos

    total_euclidean_start_time = time.time()  # Inicio total Euclidiana

    for find_embedding in sentences:
        start_time = time.time()  # Inicio individual
        select_query = ''' 
        SELECT id, text, embedding <-> (SELECT embedding FROM bookCorpusPG WHERE id = %s) AS distance 
        FROM bookCorpusPG 
        WHERE id != %s 
        ORDER BY distance 
        LIMIT 2;
        '''
        cur.execute(select_query, (find_embedding[0], find_embedding[0]))
        distances = [record for record in cur.fetchall()]
        end_time = time.time()  # Finalización individual
        euclidean_times.append(end_time - start_time)  # Guardar el tiempo individual

        # Imprimir las 2 distancias más pequeñas junto con la distancia calculada
        print(f"\nLas 2 distancias más pequeñas para la sentencia con id = {find_embedding[0]}: '{find_embedding[1]}' son:")
        print(
            f"ID Sentencia: {distances[0][0]}, Sentencia: {distances[0][1]}, Distancia: {distances[0][2]:.4f}")
        print(
            f"ID Sentencia: {distances[1][0]}, Sentencia: {distances[1][1]}, Distancia: {distances[1][2]:.4f}")

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

    print("Calculando las distancias con la metrica del coseno...")

    cosine_times = []  # Tiempos
    total_cosine_start_time = time.time()  # Inicio total Euclidiana
    for find_embedding in sentences:
        start_time = time.time()  # Inicio individual
        select_query = ''' 
        SELECT id, text, embedding <=> (SELECT embedding FROM bookCorpusPG WHERE id = %s) AS distance 
        FROM bookCorpusPG 
        WHERE id != %s 
        ORDER BY distance 
        LIMIT 2;
        '''
        cur.execute(select_query, (find_embedding[0], find_embedding[0]))
        distances = [record for record in cur.fetchall()]
        end_time = time.time()  # Finalización individual
        cosine_times.append(end_time - start_time)  # Guardar el tiempo individual

        # Imprimir las 2 distancias más pequeñas junto con la distancia calculada
        print(f"\nLas 2 distancias más pequeñas para la sentencia con id = {find_embedding[0]}: '{find_embedding[1]}' son:")
        print(
            f"ID Sentencia: {distances[0][0]}, Sentencia: {distances[0][1]}, Distancia: {distances[0][2]:.4f}")
        print(
            f"ID Sentencia: {distances[1][0]}, Sentencia: {distances[1][1]}, Distancia: {distances[1][2]:.4f}")

    total_cosine_end_time = time.time()  # Tiempo de finalización total  Coseno

    # Calcular estadísticas de los tiempos coseno
    if cosine_times:
        min_time = min(cosine_times)
        max_time = max(cosine_times)
        avg_time = np.mean(cosine_times)
        std_dev_time = np.std(cosine_times)

        print(f"\nTiempo minimo de cálculo Coseno: {min_time:.6f} segundos")
        print(f"Tiempo maximo de cálculo Coseno: {max_time:.6f} segundos")
        print(f"Tiempo promedio de cálculo Coseno: {avg_time:.6f} segundos")
        print(f"Desviacion estandar de cálculo Coseno: {std_dev_time:.6f} segundos")

        total_cosine_time = total_cosine_end_time - total_cosine_start_time  # Tiempo total Coseno
        print(f"\nTiempo total de calculo Coseno: {total_cosine_time:.6f} segundos")

    # Cerrar conexion
    cur.close()
    conn.close()