from bd.chroma_store import search, count

print("Docs en Chroma:", count())

queries = [
    "¿Qué variables afectan la potencia del molino SAG?",
    "¿Cómo influye el pH en la flotación de cobre?",
    "¿Qué significa un aumento de torque en un espesador?"
]

for q in queries:
    print("\n======================")
    print("QUERY:", q)
    hits = search(q, top_k=3)
    for i, h in enumerate(hits, 1):
        print(f"\n--- HIT {i} ---")
        print("id:", h["id"])
        print("distance:", h["distance"])
        print("source:", h["metadata"].get("source"))
        print("preview:", h["text"][:220].replace("\n", " "))
