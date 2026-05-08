filter_fact_json_schema = {
    "type": "object",
    "properties": {
        "fact": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {
                    "type": "string"  # All items in the inner array must be strings
                },
                "minItems": 3,
                "maxItems": 3,
                "additionalItems": False  # Block extra items
            },
        }
    },
    "required": ["fact"]
}

lkg_keyword_json_schema = {
    "type": "object",
    "properties": {
        "keywords": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "minItems": 1,
        }
    },
    "required": ["keywords"]
}

ATLAS_SCHEMA = {
    "entity_relation": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "Head": {
                    "type": "string",
                    "description": "The head entity in the relation, must be a non-empty string."
                },
                "Relation": {
                    "type": "string",
                    "description": "The relation between the head and tail entities, must be a non-empty string."
                },
                "Tail": {
                    "type": "string",
                    "description": "The tail entity in the relation, must be a non-empty string."
                }
            },
            "required": ["Head", "Relation", "Tail"],
            "additionalProperties": False
        },
        "description": "An array of entity-relation triples, where each triple consists of a head entity, a relation, and a tail entity."
    },
    "event_entity": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "Event": {
                    "type": "string",
                    "description": "A simple sentence describing the event, must be a non-empty string."
                },
                "Entity": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "description": "An entity related to the event, must be a non-empty string."
                    },
                    "description": "An array of entities related to the event, must not be empty."
                }
            },
            "required": ["Event", "Entity"],
            "additionalProperties": False
        },
        "description": "An array of event-entity pairs, where each pair consists of an event and an array of entities related to that event."
    },
    "event_relation": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "Head": {
                    "type": "string",
                    "description": "A simple sentence describing the first event, must be a non-empty string."
                },
                "Relation": {
                    "type": "string",
                    "description": "The relation between the two events, must be a non-empty string."
                },
                "Tail": {
                    "type": "string",
                    "description": "A simple sentence describing the second event, must be a non-empty string."
                }
            },
            "required": ["Head", "Relation", "Tail"],
            "additionalProperties": False
        },
        "description": "An array of event-relation triples, where each triple consists of a head event, a relation, and a tail event."
    }
}