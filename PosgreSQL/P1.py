import psycopg2
from config import load_config
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import time
import numpy as np
import warnings

#Esto es porque me genera un warning de un tokkenizador que no es necesario, de esta manera no aparece
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
    SELECT * FROM bookCorpus;
    '''
    cur.execute(select_query)
    sentences = [record[1] for record in cur.fetchall()]

    # Generar embeddings
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print("Generando los embeddings...")
    embedding_start_time = time.time()  # Tiempo de inicio antes de generar los para embeddings
    embeddings = model.encode(sentences)
    embedding_end_time = time.time()  # Tiempo de finalización de la generacion de los embeddings
    print(f"Tiempo total de la generacion de los embeddings: {embedding_end_time - embedding_start_time:.6f} segundos")

    #hacemos drop de la tabla por si ya existe y tiene datos
    drop_table_query = '''
    DROP TABLE IF EXISTS bookCorpusEmbeddings;
    '''
    cur.execute(drop_table_query)
    conn.commit()

    #generamos la nueva tabla donde guardaremos los embeddings
    create_embedded_table_query = '''
    CREATE TABLE IF NOT EXISTS bookCorpusEmbeddings (
        id SERIAL PRIMARY KEY,
        sentence_id INT REFERENCES bookCorpus(id),
        embedding REAL[]
    );
    '''

    cur.execute(create_embedded_table_query)
    conn.commit()

    # Insertar los datos en PostgreSQL
    embedding_insertion_times = []

    start_insertion_time = time.time()  # Inicio de las inserciones
    for i, embedding in enumerate(embeddings):
        insert_embedding_query = '''
            INSERT INTO bookCorpusEmbeddings (sentence_id, embedding)
            VALUES (%s, %s);
            '''

        start_time = time.time()  # Inicio de la insercion del embedding
        cur.execute(insert_embedding_query, (i + 1, embedding.tolist()))
        conn.commit()
        end_time = time.time()  #Final de la insercion del embedding

        embedding_insertion_times.append(end_time - start_time)

    end_insertion_time = time.time()  # Final de todas las inserciones
    print("Se han insertado todos los embeddings en la base de datos")

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




