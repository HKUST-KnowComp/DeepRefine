import json
from typing import List, Any
import json_repair
import jsonschema

def normalize_key(key):
    return key.strip().lower()

# recover function can be fix_triple_extraction_response, fix_filter_triplets
def validate_output(output_str, **kwargs):
    schema = kwargs.get("schema")
    fix_function = kwargs.get("fix_function", None)
    allow_empty = kwargs.get("allow_empty", True)
    if fix_function:
        parsed_data = fix_function(output_str, **kwargs)  
    jsonschema.validate(instance=parsed_data, schema=schema)
    if not allow_empty and (not parsed_data or len(parsed_data) == 0):
        raise ValueError("Parsed data is empty after validation.")
    return json.dumps(parsed_data, ensure_ascii=False)

def fix_filter_triplets(data: str, **kwargs) -> dict:
    data = json_repair.loads(data)
    processed_facts = []
    def find_triplet(element: Any) -> List[str] | None:
        # Base case: a valid triplet
        if isinstance(element, list) and len(element) == 3 and all(isinstance(item, str) for item in element):
            return element
        # Recursive case: dig deeper into nested lists
        elif isinstance(element, list):
            for sub_element in element:
                result = find_triplet(sub_element)
                if result:
                    return result
        return None

    for item in data.get("fact", []):
        triplet = find_triplet(item)
        if triplet:
            processed_facts.append(triplet)

    return {"fact": processed_facts}

def fix_triple_extraction_response(response: str, **kwargs) -> str:
    """Attempt to fix and validate JSON response based on the prompt type."""
    result_schema = kwargs.get("schema")
    assert "items" in result_schema, "Schema must define 'items' for triple_extraction."
    required_keys = result_schema['items'].get("required", [])
    # Extract the JSON list from the response
    # raise error if prompt_type is not provided
    json_start_token = response.find("[")
    if json_start_token == -1:
        # add [ at the start
        response = "[" + response.strip() + "]"
    parsed_objects = json_repair.loads(response)
    if len(parsed_objects) == 0:
        return []
    # Define required keys for each prompt type
    
    corrected_data = []
    seen_triples = set()
    for idx, item in enumerate(parsed_objects):
        if not isinstance(item, dict):
            print(parsed_objects)
            print(f"Item {idx} must be a JSON object. Problematic item: {item}")
            continue
        
        # Correct the keys
        corrected_item = {}
        for key, value in item.items():
            norm_key = normalize_key(key)
            matching_expected_keys = [exp_key for exp_key in required_keys if normalize_key(exp_key) in norm_key]
            if len(matching_expected_keys) == 1:
                corrected_key = matching_expected_keys[0]
                corrected_item[corrected_key] = value
            else:
                corrected_item[key] = value
        
        # Check for missing keys in corrected_item
        missing = required_keys - corrected_item.keys()
        if missing:
            print(f"Item {idx} missing required keys: {missing}. Problematic item: {item}")
            continue
        
        # Validate and correct the values according to the schema
        for key in required_keys:
            # since it a array of items, here for example it loop through Head
            required_type = result_schema['items']['properties'].get(key, {}).get("type")
            if required_type == "string":
                if not isinstance(corrected_item[key], str) or not corrected_item[key].strip():
                    # convert to str for empty values
                    if corrected_item[key] is None:
                        continue
                    corrected_item[key] = str(corrected_item[key]).strip()
                    print(f"Fixed item {idx} {key} to string: {corrected_item[key]}")
            if required_type == "array":
                if not isinstance(corrected_item[key], list) or not corrected_item[key]:
                    print(f"Item {idx} {key} must be a non-empty array. Problematic item: {corrected_item}")
                    continue
                else:
                    corrected_item[key] = [str(item).strip() for item in corrected_item[key] if isinstance(item, str)]
    
        triple_tuple = tuple((k, str(v)) for k, v in corrected_item.items())
        if triple_tuple in seen_triples:
            # print(f"Item {idx} is a duplicate triple: {corrected_item}")
            continue
        else:
            seen_triples.add(triple_tuple)
            corrected_data.append(corrected_item)

    if not corrected_data:
        return []
    
    return corrected_data

def fix_lkg_keywords(data: str, **kwargs) -> dict:
    """
    Extract and flatten keywords into a list of strings, filtering invalid types.
    """
    data = json_repair.loads(data)
    processed_keywords = []
    
    def collect_strings(element: Any) -> None:
        if isinstance(element, str):
            if len(element) <= 200:  # Filter out keywords longer than 100 characters
                processed_keywords.append(element)
        elif isinstance(element, list):
            for item in element:
                collect_strings(item)
    
    # Start processing from the root "keywords" field
    collect_strings(data.get("keywords", []))
    
    return {"keywords": processed_keywords}

