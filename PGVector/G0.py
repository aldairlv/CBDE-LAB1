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
    # Configuración de la conexión con la base de datos
    config = load_config()
    conn = connect(config)

    # Cargar el dataset
    dataset = load_our_dataset()

    # Crear la tabla en la base de datos
    cur = conn.cursor()

    #drop de la tabla por si ya existe y tiene datos
    drop_table_query = '''
    DROP TABLE bookCorpusPG CASCADE;
    '''

    cur.execute(drop_table_query)
    conn.commit()

    #Proseguimos con la creacion de la tabla
    create_table_query = '''
    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE TABLE IF NOT EXISTS bookCorpusPG (
        id SERIAL PRIMARY KEY,
        text TEXT,
        embedding vector(384)
    );
    '''
    cur.execute(create_table_query)
    conn.commit()
    print("Tabla bookCorpus creada corerctamente")

    # Almacenamos los tiempos de inserción
    insertion_times = []

    total_insertion_start_time = time.time()  # Inicio de las inserciones

    # Insertar los datos en PostgreSQL
    for row in dataset['train']['text'][:20000]:
        insert_query = '''
                        INSERT INTO bookCorpusPG (text)
                        VALUES (%s);
                        '''
        start_time = time.time()  # Tiempo de inicio de la inserción de un elemento
        cur.execute(insert_query, (row,))
        conn.commit()
        end_time = time.time()  # Tiempo de finalización

        insertion_times.append(end_time - start_time)  # Guardamos el tiempo de inserción

    total_insertion_end_time = time.time()  # Tiempo de finalización de todas las inserciones

    print("Se han insertado los 20.000 primeros elementos del dataset en la base de datos")

    # Cerrar la conexión
    cur.close()
    conn.close()

    # Calculamos las estadísticas
    if insertion_times:
        min = min(insertion_times)
        max = max(insertion_times)
        avg = np.mean(insertion_times)
        std_dev = np.std(insertion_times)

        print(f"Tiempo mínimo de inserción: {min:.6f} segundos")
        print(f"Tiempo máximo de inserción: {max:.6f} segundos")
        print(f"Tiempo promedio de inserción: {avg:.6f} segundos")
        print(f"Desviación estándar de inserción: {std_dev:.6f} segundos")

    # Calcular y mostrar el tiempo total de inserciones
    total_insertion_time = total_insertion_end_time - total_insertion_start_time
    print(f"Tiempo total de todas las inserciones: {total_insertion_time:.6f} segundos")