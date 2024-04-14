"""
Your task is to implement `create_consult_letter` function to generate a consult letter based on the SOAP note.

The input parameters are:
- user_info: a dictionary contains the bio of the doctor, such as
    {
        "name": "Dr. John Doe", # the name of the doctor
        "email": "drjohndoe@clinic.com", # the email of the doctor
    }
- specialty: a string represents the specialty of the doctor, such as "Obstetrics and Gynecology"
- note_content: a dictionary contains the content of the SOAP note, where the key is the section name and the value is the content of the section, such as
    {
        "Chief Complaint": "The patient is a 34-year-old G2P1 at 38 weeks gestation who presents for a routine prenatal visit.",
        "History of Present Illness": "The patient is a 34-year-old G2P1 at 38 weeks gestation who presents for a routine prenatal visit.",
        ...
    }
- note_date: a string represents the date of the SOAP note, such as "2022-01-01"
"""

import json
from typing import Optional

from openai_chat import chat_content


def create_consult_letter(
    user_info: dict, specialty: str, note_content: dict[str, Optional[str]], note_date: str
) -> str:
    # System prompt
    prompt = f'''
    You are a professional medical assistant for a {specialty} specialist tasked with generating an email-style consultation letter in response to the referring family doctor.

    You will be provided information about the consultation in JSON format. The JSON data will contain the following fields:
    - `user_info`: a dictionary containing the bio of the specialist doctor, including name and email. You will use this dictionary to write the specialist's sign-off of the note.
    - `note_content`: a dictionary containing information about the consultation. This dictionary will contain sections like "Chief Complaint", "Physical Examination", and "Assessment and Plan". You will use the information in this dictionary to write the letter, and you must only include information provided to you in the JSON data.

    You will write the letter in SOAP note format. You may separate each section or the sections themselves using paragraphs, but you will never include section headings. Never include a subject line or saluation (e.g. "Dear colleague,"). The letter must also be in complete sentences.

    The consultation was on {note_date}. You must mention this consultation date in your letter.

    You will end the letter with a sign-off from the doctor, including the information provided in the `user_data` entry of the JSON data and the doctor's specialty ({specialty}). You will only generate the SOAP note and the specialist's sign off.
    '''

    # Combines the user_info and note_content into a single JSON object as a string
    data = json.dumps({"user_info": user_info, "note_content": note_content})

    # Generates the letter using the system prompt and the user_info and note_content as the user prompt
    return chat_content(
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": data},
        ],
        temperature = 0.1,
        max_tokens = 512,
    )
