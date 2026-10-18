from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# LOAD EMBEDDING MODEL
# --------------------------------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")


# --------------------------------------------------
# MEMORIES
# --------------------------------------------------

TEST_MEMORIES = [
    {
        "mode": "classroom",
        "text": "Maths test is tomorrow at 3 PM.",
        "memory_type": "exam",
    },
    {
        "mode": "meeting",
        "text": "Team meeting is tomorrow at 2 PM.",
        "memory_type": "meeting",
    },
    {
        "mode": "hospital",
        "text": "Doctor appointment is tomorrow at 4 PM.",
        "memory_type": "appointment",
    },
    {
        "mode": "classroom",
        "text": "Teacher explained convolutional neural networks.",
        "memory_type": "lesson",
    },
    {
        "mode": "classroom",
        "text": "The professor explained deep learning concepts.",
        "memory_type": "lesson",
    },
    {
        "mode": "classroom",
        "text": "Assignment on neural networks is due next week.",
        "memory_type": "assignment",
    }
]


# --------------------------------------------------
# CREATE MEMORY EMBEDDINGS
# --------------------------------------------------

memory_texts = [
    memory["text"]
    for memory in TEST_MEMORIES
]

memory_embeddings = model.encode(memory_texts)


# --------------------------------------------------
# SEMANTIC SEARCH FUNCTION
# --------------------------------------------------

def semantic_search(query, current_mode, top_k=3):

    # --------------------------------------------------
    # FILTER MEMORIES BY CURRENT MODE
    # --------------------------------------------------

    filtered_indexes = [
        i
        for i, memory in enumerate(TEST_MEMORIES)
        if memory["mode"] == current_mode
    ]

    # --------------------------------------------------
    # NO MEMORIES FOR THIS MODE
    # --------------------------------------------------

    if not filtered_indexes:
        return []

    # --------------------------------------------------
    # CREATE QUERY EMBEDDING
    # --------------------------------------------------

    query_embedding = model.encode([query])

    # --------------------------------------------------
    # GET ONLY CURRENT MODE EMBEDDINGS
    # --------------------------------------------------

    mode_embeddings = memory_embeddings[filtered_indexes]

    # --------------------------------------------------
    # CALCULATE COSINE SIMILARITY
    # --------------------------------------------------

    similarities = cosine_similarity(
        query_embedding,
        mode_embeddings
    )[0]

    # --------------------------------------------------
    # RANK RESULTS
    # --------------------------------------------------

    ranked_positions = similarities.argsort()[::-1]

    results = []

    for position in ranked_positions[:top_k]:

        original_index = filtered_indexes[position]

        results.append({
            "memory": TEST_MEMORIES[original_index],
            "score": float(similarities[position])
        })

    return results


# --------------------------------------------------
# DISPLAY SEARCH RESULTS
# --------------------------------------------------

def run_test(query, current_mode):

    print("\n" + "=" * 60)
    print("MODE-FILTERED SEMANTIC SEARCH")
    print("=" * 60)

    print("\nCurrent mode:")
    print(current_mode)

    print("\nQuestion:")
    print(query)

    results = semantic_search(
        query,
        current_mode
    )

    print("\nRetrieved memories:")

    if not results:
        print("No memories found.")
        return

    for i, result in enumerate(results, start=1):

        memory = result["memory"]

        print(f"\n{i}. {memory['text']}")
        print("   Mode:", memory["mode"])
        print("   Type:", memory["memory_type"])
        print("   Similarity:", round(result["score"], 4))


# --------------------------------------------------
# TEST 1 — CLASSROOM
# --------------------------------------------------

run_test(
    query="What did we learn about neural network training?",
    current_mode="classroom"
)


# --------------------------------------------------
# TEST 2 — HOSPITAL
# --------------------------------------------------

run_test(
    query="What did we learn about neural network training?",
    current_mode="hospital"
)