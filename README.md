# Empathia.AI Prompt Engineer Interview Task

## Summary

I have developed a function to generate a consult letter in response to a referring family doctor, provided information about the doctor, the specialty, the SOAP note content, and the date. Additionally, I also outline a framework for regression testing and generating large sets of test cases for testing purposes using the OpenAI API.

## Running the Program
1. Install dependencies by running `pip install -r requirements.txt` from the project directory.
2. Run tests by running either `python -m pytest ./consult_letter/test_consult_letter_generated.py` for the generated test cases or `python -m pytest ./consult_letter/test_consult_letter.py` for the provided sample test case.

## Developing the Program
### A few brief notes
1. Even though the original called for the `gpt-4-1106-preview` model, I've switched over to using the `gpt-4-turbo-2024-04-09` model for the additional `response_format` functionality of turbo models that I use to generate test data.
2. ChatGPT was used to speed up implementation by generating code in some areas.

### Consult Letter Generation

I implemented the `create_consult_letter` function by simply using a system prompt to tell GPT-4 how to parse the input data (`user_info` and `note_content`), and then sending it the input data as a user prompt. The `user_info` and `note_content` was formatted in JSON, and the `note_date` and `specialty` were integrated into the system prompt manually because the data structure is much more consistent compared to `user_info` and `note_content`, which do not have clear specifications on their contents (e.g. how will it handle a case where `user_info` also has a phone number?).

Originally, the system prompt was much longer and went into great detail about what a SOAP note was and what information it should contain, but I saw [this article](https://kenkantzer.com/lessons-after-a-half-billion-gpt-tokens/) on Hacker News that "found that not enumerating an exact list or instructions in the prompt produced better results, if that thing was already common knowledge." The original system prompt gave me good results, but after cutting out almost all of the detailed instructions and only keeping the instruction to `write the letter in SOAP note format`, I (qualitatively) got equally good results with fewer tokens, so I just decided to keep the shorter prompt.

Here is the system prompt in its entirety:

```
You are a professional medical assistant for a {specialty} specialist tasked with generating an email-style consultation letter in response to the referring family doctor.

You will be provided information about the consultation in JSON format. The JSON data will contain the following fields:
- `user_info`: a dictionary containing the bio of the specialist doctor, including name and email. You will use this dictionary to write the specialist's sign-off of the note.
- `note_content`: a dictionary containing information about the consultation. This dictionary will contain sections like "Chief Complaint", "Physical Examination", and "Assessment and Plan". You will use the information in this dictionary to write the letter, and you must only include information provided to you in the JSON data.

You will write the letter in SOAP note format. You may separate each section or the sections themselves using paragraphs, but you will never include section headings. Never include a subject line or saluation (e.g. "Dear colleague,"). The letter must also be in complete sentences.

The consultation was on {note_date}. You must mention this consultation date in your letter.

You will end the letter with a sign-off from the doctor, including the information provided in the `user_data` entry of the JSON data and the doctor's specialty ({specialty}). You will only generate the SOAP note and the specialist's sign off.
```

The prompt might be improved further by replacing the line breaks with actual `\n` characters, but this will need further testing.

### Testing Framework

After doing a little bit of research, I found that one of the most common ways to test LLM applications was to benchmark it against a set of data (potentially generated via an LLM as well) consistently, and grade the results (also potentially via an LLM), so that's what I did here.

#### Generating test cases

`data_generator.py` generates random test cases and criteria to judge the resulting consult letter using GPT-4 and stores them in the `test_data` folder in JSON format. They are named by the specialty of the test case itself (e.g. OB-GYN) and a number specifying which number test case it is from that specialty.

Each test case in JSON contains the following:
- `test_case`: stores the test data itself (the inputs `user_info`, `specialty`, `note_content`, and `note_date`).
-  `grading_criteria`: stores a string containing the grading criteria to judge a consult letter based on the test case from.

I came up with the following grading criteria for GPT-4 to generate for each test case:
- The letter should have the correct doctor's name and email, but mention specifically what the name and email is based on the test case.
- The letter mentions the correct patient's name and the correct date of the visit, again replacing with the specific name and date in the test case.
- The letter should not include a subject line or salutation, e.g. "Dear colleague,".
- A few bullet points that check for the presence of specific, important details that be mentioned in the letter.

This can easily be edited in the prompt itself, but I picked these based on the example test given. 

The reason why I chose to generate the grading criteria along with the test case itself was that GPT-4 wasn't particularly good at judging the consult letters based on a general set of criteria (e.g. does the letter contain all of the information in the `note_content`?). My original approach that did this provided the generated letter along with the test case data and the general set of criteria, and when I dug deeper, GPT-4 was just hallucinating the PASS or FAIL. One particularly bad example of this was when it started the message with a FAIL, but when asked for a reason, it reasoned through each criterion in the general list and then came to the conclusion that it shouldn't have been a FAIL and should be changed to PASS. You can see the old test data in the `old_test_data` directory.

Here is the complete system prompt used to generate a test case and its grading criteria:

 ```
You are a QA tester for a medical AI assistant company. You are tasked with creating test data to evaluate the AI assistant for converting consultation notes to a SOAP note email. The data should be in JSON format, with the following fields:
- `user_info`: A dictionary containing the bio of the specialist doctor, including name and email. For example,
```JSON
{
    "name": "Dr. John Doe",
    "email": "drjohndoe@clinic.com"
}
` ``
- **`specialty`**: A string representing the specialty of the doctor (e.g., "Otolaryngology").
- **`note_content`**: A dictionary with sections like "Chief Complaint," "History of Present Illness," etc., containing the consultation notes, e.g.
```JSON
{
    "Chief Complaint": "The patient is a 29-year-old G2P1 at 20 weeks gestation who presents for a routine prenatal visit.",
    "History of Present Illness": "\\n• Left-sided ear pain\\n• No drainage noted...",
    ...
}
` ``
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
` ``

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
` ``

You will generate test cases in the form of a JSON object with the following format:
- `test_case`: contains the test data itself, following the format specified above.
- `grading_criteria`: contains a single string with a bullet pointed list of grading criteria to grade the consultation letter generated from the test data with.
    - One of the criteria must be that the letter should have the correct doctor's name and email, but mention specifically what the name and email is based on the test case.
    - Another of the criteria should be that the letter mentions the correct patient's name and the correct date of the visit, again replacing with the specific name and date in the test case.
    - Also include a bullet point mentioning that the letter should not include a subject line or salutation, e.g. "Dear colleague,".
    - Lastly, include a few bullet points that check for the presence of specific, important details that be mentioned in the letter.

Here is an example of a `grading_criteria` string:
\`\`\`
"- The letter shall have the doctor's name "John Doe"\n- The letter shall mention patient name as Jane, and the encounter happened at 2022/01/01.\n- The letter shall mention the patient had COVID-19 in 2021 with subsequent heart pain but found okay.\n- The letter shall mention the patient had a cesarean section in 2019 and an abortion due to a fetal health issue.\n- The letter shall mention the patient is allergic to minocycline."
\`\`\`

You must answer in JSON format using the format described with the `test_case` and `grading_criteria`.
```

I didn't test a zero-shot version of this prompt because I believed that it needed examples to be able to closely follow the format and style I wanted for the test cases.

Here is the user prompt:
```
Please generate a test case for a doctor in the {random_specialty} specialty for a vist on {random_date}
```

I randomly generate the specialty and date for the test case because GPT-4 does not seem to generate these randomly; it biases towards test cases in cardiology and dermatology especially, and it always ends up picking the same date or some time similar to it. 

#### Testing using the generated test cases

The `test_consult_letter_generated.py` file is very similar to the `test_consult_letter.py` test case, but it just runs it for every single file in the `test_data` folder and unpacks the data in the file correctly.

`test_data` is the loaded dictionary of a test case from file, and here is the complete message data provided to GPT-4:

```
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
```

#### Further Extensions

Further complexity can be added to the generated test cases by randomly choose some fields to be missing data and observe how the prompt handles it, choosing details to check for in a more rigorous way, or using more realistic examples of what the data will look like.