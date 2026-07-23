import json
import logging
import random
from collections import defaultdict
from statistics import mean

import torch

logger = logging.getLogger(__name__)

class Data:

    def __init__(self, pages_path: str, relevant_path: str, cosine_sim_rank_path: str | None = None, device: torch.device | None = None):
        self.pages_path = pages_path
        self.relevant_path = relevant_path
        self.cosine_sim_rank = None
        self.cosine_sim_rank_path = cosine_sim_rank_path
        self.device = device if device is not None else torch.device("cpu")
        self.cosine_sim_rank_wb = {}

    def load_pages(self) -> None:
        with open(self.pages_path, 'r', encoding='utf-8') as f:
            self.pages = json.load(f)
            self.pages_ids = [page['page_id'] for page in self.pages]

    def load_relevant(self) -> None:
        with open(self.relevant_path, 'r', encoding='utf-8') as f:
            self.relevant = json.load(f)
            self.query_ids = [query['query_id'] for query in self.relevant]

    def load_cosine_sim(self) -> None:
        with open(self.cosine_sim_rank_path, 'r', encoding='utf-8') as f:
            self.cosine_sim_rank = json.load(f)
            logger.info(f"Number of entries: {len(self.cosine_sim_rank) if isinstance(self.cosine_sim_rank, dict) else 'N/A'}")
            self.cosine_sim_rank = {k: v['relevant_chunks'] for k, v in self.cosine_sim_rank.items() if v}

    def get_ranked_with_prev_chunks_from_query_id(self, query_id: int) -> list[int]:
        ranked_chunks = self.cosine_sim_rank[str(query_id)]
        addtional_chunks = []
        for n in ranked_chunks:
            prev = n - 1
            if prev not in ranked_chunks and prev != -1:
                addtional_chunks.append(prev)
        ranked_chunks.extend(addtional_chunks)
        return ranked_chunks

    def get_avg_sim(self, query_id: int) -> float:
        with open(self.cosine_sim_rank_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            return json_data[str(query_id)]["avg_similarity"]

    def get_page_chunks_dict(self, page_id: str) -> dict[int, torch.Tensor]:
        page = next((page for page in self.pages if page["page_id"] == page_id), None)
        if page:
            page_obj = {chunk["chunk_id"]: torch.tensor(chunk["embedding"], device=self.device) for chunk in page.get("chunks", [])}
            return page_obj
        return {}

    def get_query_obj_from_id(self, query_id: int) -> dict | None:
        return next((query for query in self.relevant if query["query_id"] == query_id), None)

    def get_query_ids_by_difficulty(self) -> tuple[list[int], list[int]]:
        fair_query_ids = []
        difficult_query_ids = []
        for q in self.query_ids:
            query = self.get_query_obj_from_id(q)
            relevant_chunks = query.get("relevant_chunks")
            ranked_chunks = self.cosine_sim_rank[str(q)]
            if any(x in relevant_chunks for x in ranked_chunks):
                fair_query_ids.append(q)
            else:
                difficult_query_ids.append(q)
        logger.info(f"Count of fair queries: {len(fair_query_ids)}")
        return fair_query_ids, difficult_query_ids

    def split_query_ids(self, query_ids: list[int], first_split_ratio: float) -> tuple[list[int], list[int]]:

        random.seed(1)

        first_set = []
        second_set = []
        pages = {}

        # map page_id -> list of query_ids
        for q in query_ids:
            query = self.get_query_obj_from_id(q)
            page_id = query.get("page_id")
            if page_id not in pages:
                pages[page_id] = []
            pages[page_id].append(q)

        # determine number of pages for the first split
        page_ids = sorted(pages.keys())
        split_index = int(len(page_ids) * first_split_ratio)

        first_page_ids = set(page_ids[:split_index])
        second_page_ids = set(page_ids[split_index:])

        # assign query_ids according to page split
        for pid in first_page_ids:
            first_set.extend(pages[pid])
        for pid in second_page_ids:
            second_set.extend(pages[pid])

        first_set = sorted(first_set)
        random.shuffle(first_set)
        second_set = sorted(second_set)
        random.shuffle(second_set)

        return first_set, second_set

    def balanced_split_query_ids(self, query_ids: list[int], first_split_ratio: float) -> tuple[list[int], list[int]]:
        first_set = []
        second_set = []
        pages = {}
        query_score = {}

        # map page_id -> list of query_ids
        for q in query_ids:
            query = self.get_query_obj_from_id(q)
            page_id = query.get("page_id")
            if page_id not in pages:
                pages[page_id] = []
            pages[page_id].append(q)

            relevant_chunks = query.get("relevant_chunks")
            ranked_chunks = self.get_ranked_with_prev_chunks_from_query_id(q)

            score = 0
            for i, elem in enumerate(ranked_chunks):
                if elem in relevant_chunks:
                    score += 1 / (i + 1)

            query_score[q] = score

        # 1. Compute mean score per page
        page_mean = {
            page_id: mean(query_score[q] for q in queries)
            for page_id, queries in pages.items()
        }

        # 2. Sort by mean score and assign classes
        # Use page_id as tiebreaker for deterministic sorting
        sorted_pages = sorted(page_mean.items(), key=lambda x: (x[1], x[0]))
        n = len(sorted_pages)
        third = n // 3

        page_class = {}
        for i, (page_id, _) in enumerate(sorted_pages):
            if i < third:
                cls = 'low'
            elif i < 2 * third:
                cls = 'medium'
            else:
                cls = 'high'
            page_class[page_id] = cls

        # 3. Group by class
        class_groups = defaultdict(list)
        for pid, cls in page_class.items():
            class_groups[cls].append(pid)

        # 4. Split within each class
        first_page_ids = []
        second_page_ids = []

        for cls in ['low', 'medium', 'high']:  # enforce class order
            # Sort by mean score, then by page_id for deterministic ordering
            ids_sorted = sorted(class_groups[cls], key=lambda pid: (page_mean[pid], pid))
            split_index = int(len(ids_sorted) * first_split_ratio)
            first_page_ids.extend(ids_sorted[:split_index])
            second_page_ids.extend(ids_sorted[split_index:])

        # 5. Order the final sets by mean score (FIXED INDENTATION)
        first_page_ids = sorted(first_page_ids, key=lambda pid: (page_mean[pid], pid))
        second_page_ids = sorted(second_page_ids, key=lambda pid: (page_mean[pid], pid))

        # assign query_ids according to page split, ordered by score within each page
        for pid in first_page_ids:
            # Sort queries within each page by score (ascending)
            page_queries = sorted(pages[pid], key=lambda q: query_score[q])
            first_set.extend(page_queries)
        for pid in second_page_ids:
            # Sort queries within each page by score (ascending)
            page_queries = sorted(pages[pid], key=lambda q: query_score[q])
            second_set.extend(page_queries)

        return first_set, second_set

    def get_full_set(self) -> dict[str, list[int]]:

        pages = {}

        # map page_id -> list of query_ids
        for q in self.query_ids:
            query = self.get_query_obj_from_id(q)
            page_id = query.get("page_id")
            if page_id not in pages:
                pages[page_id] = []
            pages[page_id].append(q)

        return pages
