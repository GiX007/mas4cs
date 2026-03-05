"""
Data Loading and Preprocessing Module.

This module handles MultiWOZ 2.2 dataset operations:
- Loading raw dataset from HuggingFace
- Processing dialogues into MAS4CS format
- Domain filtering (hotel, restaurant only)
- MultiWOZ schema constants and processing
"""
from .data_loader import load_multiwoz, load_split_data
from .data_loader_v2 import load_split, filter_by_domains, load_all_dialogues
from .preprocess_dataset import (
    select_dialogue_sample, transform_dialogue, transform_dataset, filter_dataset_domains,
    run_preprocessing_pipeline, extract_schema, create_domain_summary,
)
from .dataset_constants import (
    VALID_DOMAINS, VALID_INTENTS, BOOKING_INTENTS, INFO_INTENTS, VALID_ACTION_TYPES, GENERAL_ACTS,
    SLOTS_BY_DOMAIN, BOOKING_REQUIRED_SLOTS, SLOT_VALUE_NORMALIZATION, INFORMABLE_SLOTS, BOOKING_SLOTS
)
from .extraction_hf import extract_ground_truth_intent, extract_slots_from_frames, normalize_slot_value
from .extraction import extract_gt_intent, extract_gt_slots, extract_dialogue_acts,  extract_booking

__all__ = [
    # Core data loading
    "load_multiwoz",
    "load_split_data",

    # V2 loading and filtering
    "load_split",
    "filter_by_domains",
    "load_all_dialogues",

    # # Preprocessing functions
    "select_dialogue_sample",
    "transform_dialogue",
    "transform_dataset",
    "filter_dataset_domains",
    "run_preprocessing_pipeline",
    "extract_schema",
    "create_domain_summary",

    # Schema constants
    "VALID_DOMAINS",
    "VALID_INTENTS",
    "BOOKING_INTENTS",
    "INFO_INTENTS",
    "VALID_ACTION_TYPES",
    "GENERAL_ACTS",
    "SLOTS_BY_DOMAIN",
    "BOOKING_REQUIRED_SLOTS",
    "SLOT_VALUE_NORMALIZATION",
    "INFORMABLE_SLOTS",
    "BOOKING_SLOTS",

    # Extraction functions
    "extract_ground_truth_intent",
    "extract_slots_from_frames",
    "normalize_slot_value",

    "extract_gt_intent",
    "extract_gt_slots",
    "extract_dialogue_acts",
    "extract_booking",
]
