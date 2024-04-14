from openai_chat import chat_content

import json, tqdm, os, random
from datetime import datetime


def generate_random_date_string(start_year=2020, end_year=datetime.now().year):
    """
    Generates a random date string in the format YYYY-MM-DD.

    Parameters:
    start_year (int): The earliest possible year.
    end_year (int): The latest possible year.

    Returns:
    str: A date string in the format YYYY-MM-DD.
    """
    # Generate a random year, month, and day
    year = random.randint(start_year, end_year)
    month = random.randint(1, 12)
    
    # Determine the last day of the month
    if month in [1, 3, 5, 7, 8, 10, 12]:
        day = random.randint(1, 31)
    elif month in [4, 6, 9, 11]:
        day = random.randint(1, 30)
    else:  # February, taking into account leap years
        if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
            day = random.randint(1, 29)
        else:
            day = random.randint(1, 28)
    
    # Return the date string
    return f"{year:04d}-{month:02d}-{day:02d}"

def generate_test_case():
    """
    Generates a test case using GPT. Assumes use of a turbo model capable of
    the `response_format` field.

    Returns:
    A tuple containing the specialty and the generated test case and criteria
    in JSON format.
    """
    
    # GPT only likes to generate dermatology test cases, so we randomly pick
    # from a list of specialties and provide it to GPT instead.
    medical_specialties = [
        "Allergy and Immunology",
        "Anesthesiology",
        "Dermatology",
        "Diagnostic Radiology",
        "Emergency Medicine",
        "Medical Genetics",
        "Neurology",
        "Nuclear Medicine",
        "Obstetrics and Gynecology",
        "Ophthalmology",
        "Pathology",
        "Physical Medicine and Rehabilitation",
        "Psychiatry",
        "Radiation Oncology",
        "Surgery",
        "Urology",
        "Cardiology",
        "Endocrinology",
        "Gastroenterology",
        "Hematology",
        "Infectious Disease",
        "Nephrology",
        "Oncology",
        "Pulmonology",
        "Rheumatology",
        "Sports Medicine"
    ]
    random_specialty = random.choice(medical_specialties)

    # The model really likes to pick the exact same date (likely the end of the
    # date of training data?), so we also generate a random date
    random_date = generate_random_date_string()

    system_prompt = '''
        You are a QA tester for a medical AI assistant company. You are tasked with creating test data to evaluate the AI assistant for converting consultation notes to a SOAP note email. The data should be in JSON format, with the following fields:
        - `user_info`: A dictionary containing the bio of the specialist doctor, including name and email. For example,
        ```JSON
        {
            "name": "Dr. John Doe",
            "email": "drjohndoe@clinic.com"
        }
        ```
        - **`specialty`**: A string representing the specialty of the doctor (e.g., "Otolaryngology").
        - **`note_content`**: A dictionary with sections like "Chief Complaint," "History of Present Illness," etc., containing the consultation notes, e.g.
        ```JSON
        {
            "Chief Complaint": "The patient is a 29-year-old G2P1 at 20 weeks gestation who presents for a routine prenatal visit.",
            "History of Present Illness": "\\n• Left-sided ear pain\\n• No drainage noted...",
            ...
        }
        ```
        - **`note_date`**: A String representing the date of the SOAP note in ISO format (e.g., "2022-01-01").

        Two complete examples of test data are below:
        ```JSON
        {
            "user_info": {"name": "Dr. John Doe", "email": "drjohndoe@clinic.com"},
            "specialty": "Obstetrics & Gynecology (ObGyn)",
            "note_date": "2022/01/01",
            "note_content": {
                "Patient Name": "Jane",
                "Patient Age": None,
                "Gender": "female",
                "Chief Complaint": "OB consultation for pregnancy management with planned repeat cesarean section.",
                "History of Present Illness": None,
                "Past Medical History": "The patient had COVID-19 in 2021, after which she experienced heart pain, but subsequent evaluations by her family doctor and hospital visits confirmed that everything was okay.",
                "Past Surgical History": "The patient had a cesarean section in 2019 and an abortion due to a fetal health issue.",
                "Family History": None,
                "Social History": "Jane is employed part-time as a banker, working two to three days per week. She and her spouse reside in a non-specified location without nearby family support. However, they have a local friend network. Postpartum, Jane's mother will assist, and they intend to employ a babysitter for two months.",
                "Obstetric History": "The patient is currently pregnant with her third child. She has had one previous live birth via cesarean section and one abortion due to fetal health issues. Her first child was born slightly premature at approximately 37 weeks, weighing 2.5 kilograms.",
                "The Review of Systems": "The patient reports no asthma, heart problems, seizures, or migraines. She has experienced chest pain post-COVID-19 but has been evaluated and found to be in good health. She is currently active, engaging in pregnancy yoga once a week and walking when she feels able.",
                "Current Medications": None,
                "Allergies": "The patient is allergic to minocycline.",
                "Vital Signs": None,
                "Physical Examination": None,
                "Investigations": None,
                "Problem": "1. Previous cesarean section (654.21)",
                "Differential Diagnosis": None,
                "Plan": "• Scheduled repeat cesarean section at 39 weeks gestation\n• Instructed patient to present to City Medical Center for emergency cesarean section if labor begins prior to scheduled date\n• Advised patient to walk daily for 20 to 30 minutes to improve blood pressure and baby's health\n• Arranged follow-up appointment in three weeks, with subsequent visits every two weeks, then weekly as due date nears",
                "Surgery Discussion": "• Purpose of the Surgery: The purpose of the repeat cesarean section is to safely deliver the baby, given the patient's previous cesarean delivery and her choice for a planned cesarean this time.\n• Risks and Complications: The risks of cesarean section include bleeding, infection, or injury to the bladder or bowel. These risks are small but not zero.\n• Anesthesia: Spinal anesthesia will be used during the procedure, which will prevent pain but allow the patient to be awake.\n• Alternatives: N/A",
            }
        }
        ```

        ```JSON
        {
            "user_info": { 
                "name": "Dr. John Doe", 
                "email": "drjohndoe@clinic.com", 
            },
            "specialty": "Otolaryngology",
            "note_content": {
                "Patient Name": "Betty", 
                "Chief Complaint": "Ear pain", 
                "History of Present Illness": "\n• Left-sided ear pain\n• No drainage noted\n• Intermittent hearing loss reported\n• Pain worsens with chewing\n• Inconsistent use of mouthpiece for teeth clenching\n• Pain relief when lying on contralateral side", 
                "Social History": "\n• Occasional Reactive use for allergies\n• Allergy to salt", "The Review of Systems": "\n• Intermittent hearing loss\n• No swallowing issues\n• No nasal congestion\n• Allergies present, takes Reactive occasionally", 
                "Current Medications": "\n• Reactive for allergies", "Allergies": "\n• Allergic to salt", 
                "Physical Examination": "\n• Right ear canal clear\n• Right tympanic membrane intact\n• Right ear space aerated\n• Left ear canal normal\n• Left eardrum normal, no fluid or infection\n• Nose patent\n• Paranasal sinuses normal\n• Oral cavity clear\n• Tonsils absent\n• Good dentition\n• Pain along pterygoid muscles\n• Heart and lungs clear\n• No neck tenderness or lymphadenopathy", 
                "Assessment and Plan": "Problem 1:\nEar pain\nDDx:\n• Temporomandibular joint disorder: Likely given the jaw pain, history of teeth clenching, and normal ear examination.\nPlan:\n- Ordered audiogram to check hearing\n- Advised to see dentist for temporomandibular joint evaluation\n- Recommended ibuprofen for pain\n- Suggested soft foods diet\n- Avoid chewing gum, hard candies, hard fruits, ice, and nuts\n- Follow-up if symptoms persist" 
            },
            "note_date": "2022-01-01"
        }
        ```

        You will generate test cases in the form of a JSON object with the following format:
        - `test_case`: contains the test data itself, following the format specified above.
        - `grading_criteria`: contains a single string with a bullet pointed list of grading criteria to grade the consultation letter generated from the test data with.
            - One of the criteria must be that the letter should have the correct doctor's name and email, but mention specifically what the name and email is based on the test case.
            - Another of the criteria should be that the letter mentions the correct patient's name and the correct date of the visit, again replacing with the specific name and date in the test case.
            - Also include a bullet point mentioning that the letter should not include a subject line or salutation, e.g. "Dear colleague,".
            - Lastly, include a few bullet points that check for the presence of specific, important details that be mentioned in the letter.

        Here is an example of a `grading_criteria` string:
        ```
        "- The letter shall have the doctor's name "John Doe"\n- The letter shall mention patient name as Jane, and the encounter happened at 2022/01/01.\n- The letter shall mention the patient had COVID-19 in 2021 with subsequent heart pain but found okay.\n- The letter shall mention the patient had a cesarean section in 2019 and an abortion due to a fetal health issue.\n- The letter shall mention the patient is allergic to minocycline."
        ```

        You must answer in JSON format using the format described with the `test_case` and `grading_criteria`.
    '''
    
    user_prompt = f'Please generate a test case for a doctor in the {random_specialty} specialty for a vist on {random_date}'

    # Sometimes this still generates malformed or excessively long data, so errors still do occur.
    test_case_data = chat_content(
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature = 1,
        max_tokens = 2048,
        response_format={"type": "json_object"},
    )

    print(test_case_data)

    # In order to use the response_format field, you must use GPT-4 Turbo or GPT-3.5 Turbo.
    return random_specialty, json.loads(test_case_data)

def to_snake_case(input_str):
    """
    Converts a given string to snake_case.

    Parameters:
    input_str (str): The string to be converted.

    Returns:
    str: The converted string in snake_case.
    """
    # Convert the string to lowercase and replace spaces with underscores
    snake_case_str = input_str.lower().replace(" ", "_")
    return snake_case_str

if __name__ == "__main__":
    folder_name = "test_data"

    # Generate 30 different test cases
    for i in tqdm.tqdm(range(5)):
        # Generate a test case
        specialty, data = generate_test_case()

        # Reformat the specialty field for file naming
        specialty = to_snake_case(specialty)
    
        # Ensure the test_data exists
        os.makedirs(folder_name, exist_ok=True)
        
        # Start with a file number of 0
        file_number = 0
        file_path = f"{folder_name}/{specialty}_{file_number}.json"
        
        # Check if the file exists and find an unused name
        while os.path.exists(file_path):
            file_number += 1
            file_path = f"{folder_name}/{specialty}_{file_number}.json"
        
        # Write the test_case data to the new file
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
        
        print(f"Data written to {file_path}")