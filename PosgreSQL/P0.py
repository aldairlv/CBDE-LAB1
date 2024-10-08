import time
import psycopg2
import numpy as np
from config import load_config
from datasets import load_dataset

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

def drop_table(cur,conn):
    drop_table_query = '''DROP TABLE IF EXISTS bookCorpus CASCADE;'''
    cur.execute(drop_table_query)
    conn.commit()

def create_table(cur,conn):
    create_table_query = '''
    CREATE TABLE IF NOT EXISTS bookCorpus (
        id SERIAL PRIMARY KEY,
        text TEXT
    );
    '''
    cur.execute(create_table_query)
    conn.commit()
    print("Tabla bookCorpus creada correctamente")

def calculate_and_print_statistics(insertion_times, total_time):
    """ Calculamos las estadísticas"""
    if insertion_times:
        min_time = min(insertion_times)
        max_time = max(insertion_times)
        avg_time = np.mean(insertion_times)
        std_dev = np.std(insertion_times)


        print(f"Tiempo mínimo de inserción: {min_time:.6f} segundos")
        print(f"Tiempo máximo de inserción: {max_time:.6f} segundos")
        print(f"Tiempo promedio de inserción: {avg_time:.6f} segundos")
        print(f"Desviación estándar de inserción: {std_dev:.6f} segundos")

    # Calcular y mostrar el tiempo total de inserciones
    print(f"Tiempo total de todas las inserciones: {total_time:.6f} segundos")

def insert_data(dataset,limit):
    insertion_times = []
    total_insertion_start_time = time.time()  # Inicio de las inserciones
    # Insertar los datos en PostgreSQL
    for row in dataset['train']['text'][:limit]:
        insert_query = '''INSERT INTO bookCorpus (text) VALUES (%s);'''
        start_time = time.time()  # Tiempo de inicio de la inserción de un elemento
        cur.execute(insert_query, (row,))
        conn.commit()
        end_time = time.time()  # Tiempo de finalización

        insertion_times.append(end_time - start_time)  # Guardamos el tiempo de inserción

    total_insertion_end_time = time.time()  # Tiempo de finalización de todas las inserciones
    total_insertion_time = total_insertion_end_time - total_insertion_start_time
    print("Se han insertado los 20.000 primeros elementos del dataset en la base de datos")
    return insertion_times, total_insertion_time

def insert_data_in_batches(dataset, limit, batch_size):
    insert_query = '''INSERT INTO bookCorpus (text) VALUES (%s);'''
    insertion_times = []
    total_insertion_start_time = time.time()  # Inicio de inserciones

    # Crear los datos en memoria fuera del bucle principal
    data = dataset['train']['text'][:limit]

    # Iterar sobre los datos con el tamaño de lote
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]  # Obtener un lote de frases
        start_time = time.time()  # Tiempo de inicio de la inserción
        # Realizar la inserción del lote completo con executemany
        cur.executemany(insert_query, [(sentence,) for sentence in batch])
        end_time = time.time()  # Tiempo de finalización de la inserción de este lote

        # Almacenar tiempo de inserción del lote
        insertion_times.append(end_time - start_time)

    # Confirmar los cambios en la base de datos después de todos los lotes
    conn.commit()
    total_insertion_end_time = time.time()  # Tiempo de finalización de todas las inserciones
    total_insertion_time = total_insertion_end_time - total_insertion_start_time

    return insertion_times, total_insertion_time

if __name__ == '__main__':
    # Configuración de la conexión con la base de datos
    config = load_config()
    conn = connect(config)

    # Cargar el dataset
    dataset = load_our_dataset()

    # Crear cursor para modificar la base de datos
    cur = conn.cursor()

    #drop de la tabla por si ya existe y tiene datos
    drop_table(cur,conn)

    #Proseguimos con la creacion de la tabla
    create_table(cur,conn)

    # Almacenamos los tiempos de inserción
    insertion_times = []
    limit = 20000 # Solo los 20.000 primeros elementos del dataset

    # Inserción una por una
    insertion_times, total_insertion_time = insert_data(dataset,limit)
   
    # Inserción por batches
    #batch_size = 200
    #insertion_times, total_insertion_time = insert_data_in_batches(dataset,limit,batch_size)
    
    # Calculamos las estadísticas
    calculate_and_print_statistics(insertion_times, total_insertion_time)

    # Cerrar la conexión
    cur.close()
    conn.close()
