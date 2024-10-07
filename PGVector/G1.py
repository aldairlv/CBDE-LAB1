import psycopg2
from config import load_config
from sentence_transformers import SentenceTransformer
import time
import numpy as np
import warnings

# Ignorar el warning innecesario
warnings.simplefilter("ignore", category=FutureWarning)

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

    # Obtenemos todos los datos de nuestra tabla
    cur = conn.cursor()

    select_query = '''
    SELECT * FROM bookCorpusPG;
    '''
    cur.execute(select_query)
    records = cur.fetchall()
    #escrivimos por pantalla el numero de registros
    print(f"Numero de registros: {len(records)}")
    # Generar embeddings
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print("Generando los embeddings...")
    embedding_start_time = time.time()  # Tiempo de inicio antes de generar los para embeddings
    sentences = [record[1] for record in records]
    embeddings = model.encode(sentences)
    embedding_end_time = time.time()  # Tiempo de finalización de la generacion de los embeddings
    print(f"Tiempo total de la generacion de los embeddings: {embedding_end_time - embedding_start_time:.6f} segundos")

    # Actualizar los embeddings en la base de datos
    embedding_insertion_times = []

    start_insertion_time = time.time()  # Inicio de las inserciones
    for i, embedding in enumerate(embeddings):
        record_id = records[i][0]  # Obtener el id correspondiente al registro
        update_embedding_query = '''
        UPDATE bookCorpusPG
        SET embedding = %s
        WHERE id = %s;
        '''

        start_time = time.time()  # Inicio de la actualizacion del embedding
        cur.execute(update_embedding_query, (embedding.tolist(), record_id))
        conn.commit()
        end_time = time.time()  #Final de la actualizacion del embedding

        embedding_insertion_times.append(end_time - start_time)

    end_insertion_time = time.time()  # Final de todas las inserciones
    print("Se han actualizado todos los embeddings en la base de datos")

    # Cerrar la conexión
    cur.close()
    conn.close()

    # Calcular estadísticas para embeddings
    if embedding_insertion_times:
        min_embedding_time = min(embedding_insertion_times)
        max_embedding_time = max(embedding_insertion_times)
        avg_embedding_time = np.mean(embedding_insertion_times)
        std_dev_embedding_time = np.std(embedding_insertion_times)

        print(f"Tiempo mínimo de inserción de embeddings: {min_embedding_time:.6f} segundos")
        print(f"Tiempo máximo de inserción de embeddings: {max_embedding_time:.6f} segundos")
        print(f"Tiempo promedio de inserción de embeddings: {avg_embedding_time:.6f} segundos")
        print(f"Desviación estándar de inserción de embeddings: {std_dev_embedding_time:.6f} segundos")
        print(f"Tiempo total de inserción de embeddings: {end_insertion_time - start_insertion_time:.6f} segundos")