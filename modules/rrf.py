"""Reciprocal rank fusion across query variation result lists."""
from collections import defaultdict

def reciprocal_rank_fusion(chunk_lists, k=60):
    # print("\n" + "=" * 60)
    # print("APPLYING QUERY-VARIATION RRF")
    # print(f"Fusing {len(chunk_lists)} hybrid result lists (k={k})")
    # print("=" * 60)
 
    rrf_scores = defaultdict(float)
    all_unique_chunks = {}
    chunk_id_map = {}
    chunk_counter = 1
 
    for query_idx, chunks in enumerate(chunk_lists, 1):
        # print(f"\nQuery {query_idx} contributes {len(chunks)} docs:")
        for position, chunk in enumerate(chunks, 1):
            content = chunk.page_content
            if content not in chunk_id_map:
                chunk_id_map[content] = f"Chunk_{chunk_counter}"
                chunk_counter += 1
            all_unique_chunks[content] = chunk
            score = 1 / (k + position)
            rrf_scores[content] += score
            # print(f"  Rank {position}: {chunk_id_map[content]} +{score:.4f} "
                  # f"(total={rrf_scores[content]:.4f}) | {content[:60]}...")
 
    sorted_chunks = sorted(
        [(all_unique_chunks[c], s) for c, s in rrf_scores.items()],
        key=lambda x: x[1],
        reverse=True,
    )
    # print(f"\n✅ Query-variation RRF complete: {len(sorted_chunks)} unique chunks ranked.")
    return sorted_chunks
