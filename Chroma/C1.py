import chromadb
from datasets import load_dataset
from chromadb.utils import embedding_functions
from time import time
import warnings

#Esto es porque me genera un warning de un tokkenizador que no es necesario, de esta manera no aparece
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
    initial_time = time()
    collection = chroma_client.get_collection(name='bookCorpus', embedding_function=default_ef)
    collection_cosine = chroma_client.get_collection(name='bookCorpusCosine', embedding_function=default_ef)
    end_time = time()
    print(f"Tiempo total de obtener la coleccion con embeddings: {end_time - initial_time:.6f} segundos")
    print(f"Se han obtenido un total de {collection.count()} sentencias de la coleccion.")
    print(f"Se han obtenido un total de {collection_cosine.count()} sentencias de la coleccion con metrica de calculo de distancia coseno.")

    # Escrivimos por pantalla las 10 primeras sentencais, para demostrar que estas ya tienen los embeddings
    print(collection.peek())
