import json
import re
from evalap.clients import LlmClient, split_think_answer
from deepdiff import DeepDiff
from . import metric_registry



def parse_json_from_response(response: str) -> dict | None:
    """
    Extract and parse JSON from an LLM response.
    Handles responses that may contain extra text or markdown formatting.
    """
    if (type(response) is dict):
        return response

    # Try to parse the response directly
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON in the response using regex
    # Look for content between curly braces
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    
    # Try to find JSON in markdown code blocks
    code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1))
        except json.JSONDecodeError:
            pass
    
    return None


def count_total_values(data):
    """
    Recursively count all values in a nested data structure.
    This includes:
    - All leaf values (strings, numbers, booleans, None)
    - Each item in arrays/lists
    - Each key-value pair in objects/dictionaries
    """
    if data is None:
        return 1
    elif isinstance(data, (str, int, float, bool)):
        return 1
    elif isinstance(data, list):
        return sum(count_total_values(item) for item in data)
    elif isinstance(data, dict):
        return sum(count_total_values(value) for value in data.values())
    else:
        return 1  # For any other type


def calculate_similarity_score(expected_data, extracted_data, diff):
    """
    Calculate similarity score based on total number of values rather than just keys.
    
    Args:
        expected_data: Ground truth data structure
        extracted_data: LLM extracted data structure  
        diff: DeepDiff result between expected and extracted data
    
    Returns:
        float: Score between 0 and 1, where 1 is perfect match
    """
    if expected_data is None and extracted_data is None:
        return 1.0
    if expected_data is None or extracted_data is None:
        return 0.0
    
    if not diff:
        return 1.0
    
    # Count total values in expected data
    total_expected_values = count_total_values(expected_data)
    
    # Count the number of differences
    diff_count = 0
    
    # Count value changes
    diff_count += len(diff.get('values_changed', {}))
    
    # Count type changes
    diff_count += len(diff.get('type_changes', {}))
    
    # Count missing items (in expected but not in extracted)
    # DeepDiff returns SetOrdered objects, not dicts, so iterate directly
    if 'dictionary_item_removed' in diff:
        for removed_path in diff['dictionary_item_removed']:
            # Extract the actual removed value from expected_data using the path
            # For simplicity, count each removed key as 1 difference
            diff_count += 1
    
    if 'iterable_item_removed' in diff:
        for removed_item in diff['iterable_item_removed'].values():
            diff_count += count_total_values(removed_item)
    
    # Count extra items (in extracted but not in expected)  
    if 'dictionary_item_added' in diff:
        for added_path in diff['dictionary_item_added']:
            # Count each added key as 1 difference
            diff_count += 1
    
    if 'iterable_item_added' in diff:
        for added_item in diff['iterable_item_added'].values():
            diff_count += count_total_values(added_item)
    
    # Calculate score
    if total_expected_values == 0:
        return 1.0 if diff_count == 0 else 0.0
    
    score = max(0.0, 1.0 - (diff_count / total_expected_values))
    return score


def calculate_field_level_scores(expected_data, extracted_data):
    """
    Calculate similarity scores for each top-level key in the JSON structure.
    
    Args:
        expected_data: Ground truth data structure (dict)
        extracted_data: LLM extracted data structure (dict)
    
    Returns:
        dict: Dictionary mapping each key to its similarity score
    """
    if not isinstance(expected_data, dict) or not isinstance(extracted_data, dict):
        return {}
    
    key_scores = {}
    all_keys = set(expected_data.keys()) | set(extracted_data.keys())
    
    for key in all_keys:
        expected_value = expected_data.get(key)
        extracted_value = extracted_data.get(key)
        
        # Calculate diff for this specific key
        key_diff = DeepDiff(expected_value, extracted_value, ignore_order=True)
        
        # Calculate score for this key
        key_score = calculate_similarity_score(expected_value, extracted_value, key_diff)
        key_scores[key] = key_score
    
    return key_scores


@metric_registry.register(
    name="llm_structured_output",
    description="Compare json_schema LLM output with provided ground truth JSON (output_true) and compute an overall score and key level accuracy scores",
    metric_type="llm",  # Mark as LLM type since it uses LLM for extraction
    require=["output", "output_true", "query"],
)
def llm_structured_output_metric(output, output_true, **kwargs):
    # Parse the expected output (ground truth)
    try:
        expected_data = parse_json_from_response(output_true)
    except json.JSONDecodeError as e:
        return 0.0, json.dumps({
            "error": f"Failed to parse ground truth JSON: {str(e)}",
            "output_true": output_true[:500]
        }), output
    
    extracted_data = parse_json_from_response(output)
    # Handle extraction failures
    if extracted_data is None:
        observation = {
            "score": 0.0,
            "error": "Failed to extract valid JSON from LLM response",
            # "llm_response": (answer if answer else llm_response)[:500],
            "expected_data": expected_data
        }
        return 0.0, json.dumps(observation, indent=2), output
    
    # Calculate similarity scores
    diff = DeepDiff(expected_data, extracted_data)
    field_scores = calculate_field_level_scores(expected_data, extracted_data)
    overall_score = sum(field_scores.values()) / len(field_scores)

    # Build detailed observation
    observation = {
        "score": overall_score,
        "extracted_data": extracted_data,
        "expected_data": expected_data,
        "field_scores": field_scores,
        # "extraction_details": {
        #     "model": getattr(judge_model, "name", "unknown"),
        #     "temperature": sampling_params.get("temperature", "unknown"),
        #     "prompt_type": getattr(judge_model, "aliased_name", "unknown")
        # }
    }
    
    
    return overall_score, json.dumps(observation, indent=2), output