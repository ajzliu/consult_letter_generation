from consult_letter import create_consult_letter
from openai_chat import chat_content

import os
import json
import pytest

TEST_DATA_DIR = './test_data'

def load_test_data(directory):
    """
    Loads test cases from JSON files in the given directory.

    Returns: a list containing all of the test data cases.
    """
    test_data = []
    # Iterate over all test data in test_data folder
    for filename in os.listdir(directory):
        if not filename.endswith('.json'):
            continue
        
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r') as file:
            data = json.load(file)
            test_data.append(pytest.param(data, id=filename))
    return test_data

# Run the test for every test case in the test_data folder.
@pytest.mark.parametrize("test_data", load_test_data(TEST_DATA_DIR))
def test_create_consult_letter(test_data):
    consult_letter = create_consult_letter(
        user_info=test_data['test_case']['user_info'],
        specialty=test_data['test_case']['specialty'],
        note_date=test_data['test_case']['note_date'],
        note_content=test_data['test_case']['note_content'],
    )

    result = chat_content(
        messages=[
            {
                "role": "system",
                "content": f"You are a professional medical assistant of {test_data['test_case']['specialty']}, your job is to verify the content of a consult letter based on the original data in JSON format.",
            },
            {
                "role": "user",
                "content": f"""
                    The consult letter is as following, delimited by ```:
                    ```
                    {consult_letter}
                    ```
                """,
            },
            {
                "role": "user",
                "content": f"Follow these test points when you verify the consult letter with the original data (equivalent meaning in the letter is OK):\n" + test_data['grading_criteria'],
            },
            {
                "role": "user",
                "content": "Write me PASS **ONLY** if the consult letter is correct according to the test points, and FAIL with reason if not",
            },
        ]
    )

    assert result.upper() == "PASS"