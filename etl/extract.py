import json
import logging
import os
from collections import defaultdict

from config import get_config
from trec_car.read_data import iter_outlines, iter_pages, iter_paragraphs

logger = logging.getLogger(__name__)


def parse_qrels(qrels_path):
    query_to_para = defaultdict(list)
    with open(qrels_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                query_id, _, para_id = parts[:3]
                query_to_para[query_id].append(para_id)
    return query_to_para


def get_query_strings(outlines_path):
    queryid_to_querytext = {}
    queryid_to_pageid = {}
    with open(outlines_path, 'rb') as f:
        for page in iter_outlines(f):
            page_title = page.page_name
            page_id = page.page_id
            for section_path in page.flat_headings_list():
                heading_names = [sec.heading for sec in section_path]
                query_id = page_id + '/' + '/'.join(heading_names)
                if heading_names:
                    query_text = page_title + " / " + " / ".join(heading_names)
                else:
                    query_text = page_title
                queryid_to_querytext[query_id] = query_text
                queryid_to_pageid[query_id] = page_id
    return queryid_to_querytext, queryid_to_pageid


def get_paragraph_id_to_text_map(paragraphs_cbor_path):
    """Returns a dict mapping paragraph IDs (SHA hashes) to their full text."""
    para_id_to_text = {}
    with open(paragraphs_cbor_path, 'rb') as f:
        for paragraph in iter_paragraphs(f):
            para_id_to_text[paragraph.para_id] = paragraph.get_text()
    return para_id_to_text


def get_full_page_texts(pages_path, paras_path):
    """Extract full Wikipedia page texts by concatenating all unique paragraphs from each page."""
    paraid_to_text = get_paragraph_id_to_text_map(paras_path)

    pageid_to_fulltext = {}

    with open(pages_path, 'rb') as f:
        for page in iter_pages(f):
            page_id = page.page_id
            page_title = page.page_name

            seen_para_ids = set()

            def collect_paragraph_ids(item):
                if hasattr(item, 'paragraph') and item.paragraph:
                    para_id = item.paragraph.para_id
                    seen_para_ids.add(para_id)
                if hasattr(item, 'children'):
                    for child in item.children:
                        collect_paragraph_ids(child)

            for section_path in page.flat_headings_list():
                for section in section_path:
                    collect_paragraph_ids(section)

            page_texts = [page_title]

            for para_id in seen_para_ids:
                if para_id in paraid_to_text:
                    para_text = paraid_to_text[para_id].strip()
                    if para_text:
                        page_texts.append(para_text)

            pageid_to_fulltext[page_id] = '\n\n'.join(page_texts)

    return pageid_to_fulltext


def process_fold(fold_idx, all_pages, all_queries, dataset, raw_dir, folds = False):
    base_dir = f"{raw_dir}/benchmarkY1-{dataset}"
    if folds and dataset == 'train':
        paras = f"{base_dir}/fold-{fold_idx}-{dataset}.pages.cbor-paragraphs.cbor"
        outlines = f"{base_dir}/fold-{fold_idx}-{dataset}.pages.cbor-outlines.cbor"
        pages = f"{base_dir}/fold-{fold_idx}-{dataset}.pages.cbor"
    else:
        paras = f"{base_dir}/{dataset}.pages.cbor-paragraphs.cbor"
        outlines = f"{base_dir}/{dataset}.pages.cbor-outlines.cbor"
        pages = f"{base_dir}/{dataset}.pages.cbor"

    qrels_path = outlines.replace("outlines.cbor", "hierarchical.qrels")

    logger.info(f"Processing fold {fold_idx}")

    queryid_to_paras = parse_qrels(qrels_path)
    queryid_to_querytext, queryid_to_pageid = get_query_strings(outlines)

    logger.info("Loading paragraph texts...")
    paraid_to_text = get_paragraph_id_to_text_map(paras)
    logger.info(f"Loaded {len(paraid_to_text)} paragraph texts")

    logger.info("Loading full page texts...")
    pageid_to_fulltext = get_full_page_texts(pages, paras)
    logger.info(f"Loaded {len(pageid_to_fulltext)} full page texts")

    # Add page entries
    for page_id, full_text in pageid_to_fulltext.items():
        page_entry = {
            "page_id": page_id,
            "full_page_text": full_text
        }
        all_pages.append(page_entry)

    # Add query entries
    for query_id, para_ids in queryid_to_paras.items():
        query_text = queryid_to_querytext.get(query_id, None)
        if not query_text:
            continue

        page_id = queryid_to_pageid.get(query_id, None)
        if not page_id:
            continue

        relevant_paragraphs = []
        for para_id in para_ids:
            para_text = paraid_to_text.get(para_id, "")
            if para_text.strip():
                relevant_paragraphs.append(para_text)

        query_entry = {
            "query": query_text,
            "page_id": page_id,
            "relevant_paragraphs": relevant_paragraphs
        }
        all_queries.append(query_entry)

def extract(dataset, cfg=None):
    if cfg is None:
        cfg = get_config()

    raw_dir = cfg.data.raw_dir
    extract_dir = cfg.data.extract_dir

    all_pages = []
    all_queries = []
    os.makedirs(extract_dir, exist_ok=True)

    process_fold(0, all_pages, all_queries, dataset, raw_dir, False)

    # Write combined pages.jsonl
    pages_output = f"{extract_dir}/pages_{dataset}.jsonl"
    with open(pages_output, 'w', encoding='utf-8') as pages_file:
        for entry in all_pages:
            pages_file.write(json.dumps(entry) + "\n")
    logger.info(f"Exported {len(all_pages)} total pages to {pages_output}")

    # Write combined relevant_paragraphs.jsonl
    queries_output = f"{extract_dir}/relevant_paragraphs_{dataset}.jsonl"
    with open(queries_output, 'w', encoding='utf-8') as queries_file:
        for entry in all_queries:
            queries_file.write(json.dumps(entry) + "\n")
    logger.info(f"Exported {len(all_queries)} total queries to {queries_output}")
