import json
import torch

class Data:

    def __init__(self, pages_path, relevant_path, pages_doub_even_path, pages_doub_odd_path, cosine_sim_rank_path = None):
        self.pages_path = pages_path
        self.relevant_path = relevant_path
        self.cosine_sim_rank_path = cosine_sim_rank_path
        self.pages_doub_even_path = pages_doub_even_path
        self.pages_doub_odd_path = pages_doub_odd_path
        self.device = torch.device("mps")
        self.cosine_sim_rank_wb = {}

    def load_pages(self):
        with open(self.pages_path, 'r', encoding='utf-8') as f:
            self.pages = json.load(f)
            self.pages_ids = [page['page_id'] for page in self.pages]

    def load_pages_even(self):
        with open(self.pages_doub_even_path, 'r', encoding='utf-8') as f:
            self.pages_even = json.load(f)
            self.pages_even_ids = [page['page_id'] for page in self.pages_even]

    def load_pages_odd(self):
        with open(self.pages_doub_odd_path, 'r', encoding='utf-8') as f:
            self.pages_odd = json.load(f)
            self.pages_odd_ids = [page['page_id'] for page in self.pages_odd]

    def load_relevant(self):
        with open(self.relevant_path, 'r', encoding='utf-8') as f:
            self.relevant = json.load(f)

            # remove objects with empty "relevant_chunks" (2/535 were found)
            # self.relevant = [
            #     query for query in self.relevant 
            #     if query.get("relevant_chunks")
            # ]

            self.query_ids = [query['query_id'] for query in self.relevant]

    def load_cosine_sim(self):
        with open(self.cosine_sim_rank_path, 'r', encoding='utf-8') as f:
            self.cosine_sim_rank = json.load(f)
            self.cosine_sim_rank = {k: v['relevant_chunks'] for k, v in self.cosine_sim_rank.items() if v}

    def get_ranked_with_prev_chunks_from_query_id(self, query_id):
        ranked_chunks = self.cosine_sim_rank[str(query_id)]
        addtional_chunks = []
        for n in ranked_chunks:
            prev = n - 1
            if prev not in ranked_chunks and prev != -1:
                addtional_chunks.append(prev)
        ranked_chunks.extend(addtional_chunks)
        return ranked_chunks

    def get_page_chunks_dict(self, page_id):
        page = next((page for page in self.pages if page["page_id"] == page_id), None)
        page_even = next((page for page in self.pages_even if page["page_id"] == page_id), None)
        page_odd = next((page for page in self.pages_odd if page["page_id"] == page_id), None)
        if page and page_even and page_odd:
            page_obj = {chunk["chunk_id"]: torch.tensor(chunk["embedding"], device=self.device) for chunk in page.get("chunks", [])}
            page_obj_even = {chunk["chunk_id"]: torch.tensor(chunk["embedding"], device=self.device) for chunk in page_even.get("chunks", [])}
            page_obj_odd = {chunk["chunk_id"]: torch.tensor(chunk["embedding"], device=self.device) for chunk in page_odd.get("chunks", [])}
            return page_obj, page_obj_even, page_obj_odd
        return {}

    def get_query_obj_from_id(self, query_id):
        return next((query for query in self.relevant if query["query_id"] == query_id), None)
    
    def get_query_ids_by_difficulty(self):
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
        print(f"Count of fair queries: {len(fair_query_ids)}")
        return fair_query_ids, difficult_query_ids
    
    def split_query_ids(self, query_ids, first_split_ratio):
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

        return first_set, second_set

        

        

