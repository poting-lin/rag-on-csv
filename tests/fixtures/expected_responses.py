"""
Expected responses for common queries in CSV QA Bot integration tests.
"""

# Define expected responses for specific queries
EXPECTED_RESPONSES = {
    "sound_measurements": {
        "all_records_festival": {
            "contains": ["Night festival"],
            "not_contains": ["Morning bird sounds", "Rush hour traffic"],
            "min_matches": 2  # Expecting at least 2 records with "festival" in Notes
        },
        "high_decibel": {
            "contains": ["70.2"],
            "min_matches": 1  # Expecting at least 1 record with DecibelsA > 70
        },
        "urban_park_location": {
            "contains": ["Urban Park"],
            "min_matches": 9  # Expecting 9 records with Location: Urban Park
        }
    },
    "lab_data": {
        "optical_equipment": {
            "contains": ["Optical Microscope"],
            "min_matches": 1
        }
    },
    "bookshop": {
        "fiction_books": {
            "contains": ["Fiction"],
            "min_matches": 3
        }
    }
}
