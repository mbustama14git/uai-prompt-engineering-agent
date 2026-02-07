import numpy as np
from openai import OpenAI
from redis import Redis
from redis.commands.search.query import Query
import config

client = OpenAI(api_key=config.gpt_key)

VECTOR_FIELD_NAME = 'content_vector'

def find_vector_in_redis(query):
    url = "redis://{}:{}@{}:{}/{}".format(
        config.redis_username,
        config.redis_password,
        config.redis_host,
        config.redis_port,
        config.redis_db
    )

    r = Redis.from_url(url=url)
    top_k = 2

    # Crear embedding con la sintaxis correcta
    embedding_response = client.embeddings.create(
        input=query,
        model="text-embedding-ada-002"  # Parámetro corregido
    )
    
    # Acceder al embedding correctamente
    embedded_query = np.array(
        embedding_response.data[0].embedding,  # Sintaxis corregida
        dtype=np.float32
    ).tobytes()

    # Preparar la query
    q = Query(f'*=>[KNN {top_k} @{VECTOR_FIELD_NAME} $vec_param AS vector_score]'
            ).sort_by('vector_score').paging(0, top_k).return_fields(
                'filename', 'text_chunk', 'text_chunk_index', 'content'
            ).dialect(2)
    
    params_dict = {"vec_param": embedded_query}

    # Ejecutar la consulta
    results = r.ft(config.redis_index).search(q, query_params=params_dict)
    
    return results.docs