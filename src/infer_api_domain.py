import pandas as pd
import json
from openai import OpenAI
from tqdm import tqdm
from guideline_and_clinical_trials import guideline, clinical_trials


class PromptGenerator:
    def __init__(self, guideline: str, clinical_trials: str, inspection: str, diagnosis: str, case_history: str, examination: str, vol1: str, vol2: str, vol3: str):
        self.guideline = guideline
        self.clinical_trials = clinical_trials
        self.inspection = inspection
        self.diagnosis = diagnosis
        self.case_history = case_history
        self.examination = examination
        self.vol1 = vol1
        self.vol2 = vol2
        self.vol3 = vol3

    def generate_patient_prompt(self) -> str:
        patient_prompt_template = """
        
        You are a medical agent specializing in explaining cerebral hemorrhage-related disease conditions and treatment plans in a gentle, clear, and empathetic manner to both patients and their families.
        Based on the following available data (medical history, physical examinations, laboratory test results, CT reports, and segmentation results), you must strictly reference the "2022 Guideline for the Management of Patients With Spontaneous Intracerebral Hemorrhage" from the American Heart Association/American Stroke Association, as well as findings from major clinical trials including ENRICH, INTERACT3, SWITCH, and ANNEXA-I, to provide a synchronized communication and explanation to both the patient and their family.
        Specific Requirements:
        1. Start by addressing the patient directly, using simple, warm language to briefly explain what has happened and the main direction of treatment, helping the patient understand and feel reassured.
        2. Then, address the family members, providing a more detailed explanation of the need for further examinations, the rationale behind treatment choices, potential risks, and rehabilitation expectations, so that the family can better support decision-making.
        3. Throughout the communication, use language that is easy for non-medical individuals to understand; if medical terms must be used, provide a brief and clear explanation.
        4. Maintain a tone that is scientific, authoritative, and positively encouraging.
        5. Base all explanations strictly on the available data; do not fabricate or assume information. If uncertainties exist, state them transparently.
        6. Avoid using absolute expressions such as "100%" or "definitely"; instead, prefer phrasing like "likely," "tends to," or "based on current evidence."
        Suggested Output Structure:
        1. Communication paragraph directed toward the patient
        2. Supplementary explanation paragraph directed toward the family
        3. Summary and encouragement paragraph

        2022 Guideline for the Management of Patients With Spontaneous Intracerebral Hemorrhage: A Guideline From the American Heart Association/American Stroke Association 
        {guideline}

        Clinical Trials:
        {clinical_trials}
        
        The patient's available information is as follows:
        Imaging findings: {inspection} 
        Impression: {diagnosis} 
        Medical history: {case_history} 
        Laboratory Tests: {examination} 

        CT Image Segmentation Results of Cerebral Hemorrhage：
        Intraparenchymal hemorrhage: {vol1}
        Intraventricular hemorrhage: {vol2} 
        Perihematomal edema: {vol3}

        """
        return patient_prompt_template.format(
            guideline=self.guideline,
            clinical_trials=self.clinical_trials,
            inspection=self.inspection,
            diagnosis=self.diagnosis,
            case_history=self.case_history,
            examination=self.examination,
            vol1=self.vol1,
            vol2=self.vol2,
            vol3=self.vol3
        )

    def generate_doctor_prompt(self) -> str:
        doctor_prompt_template = """
        As a medical agent specializing in the diagnosis and treatment of cerebral hemorrhage, your task is to provide precise, evidence-based recommendations using only the available data: medical history, physical examinations, laboratory tests, CT reports, and CT segmentation results. No additional clinical information will be accessible.
        When formulating treatment strategies, strictly reference the "2022 Guideline for the Management of Patients With Spontaneous Intracerebral Hemorrhage" from the American Heart Association/American Stroke Association, as well as findings from major clinical trials including ENRICH, INTERACT3, SWITCH, and ANNEXA-I.
        Request for Recommendations:
        1. Additional Testing Recommendations:
        Identify any further diagnostic tests that are necessary for a comprehensive assessment of the patient's condition.
        2. Treatment Recommendations:
        Provide preliminary treatment suggestions — including pharmacological management, surgical intervention, or other measures — tailored to the patient's specific circumstances, and in strict accordance with the referenced guidelines and clinical trial data.
        Additional Requirements:
        1. All recommendations must rigorously adhere to the specified guideline and trial evidence.
        2. Individual patient differences must be carefully considered to ensure that the recommendations are personalized and adaptable.
        3. The recommendations should aim to optimize diagnostic efficiency and therapeutic outcomes.
        4. It is critical to validate all available data during the management process. If patient data significantly deviate from guideline standards — for example, a hematoma volume exceeding the specified 30–80 mL range (e.g., 80 mL) — avoid uncritical application of guideline-based classifications. Conduct a critical appraisal before making recommendations. 
        5. Clearly indicate the source of each recommendation, specifying whether it is based on a particular guideline or a clinical trial result.
        
        2022 Guideline for the Management of Patients With Spontaneous Intracerebral Hemorrhage: A Guideline From the American Heart Association/American Stroke Association 
        {guideline}

        Clinical Trials:
        {clinical_trials}

        The available information for the patient is as follows:
        Imaging findings: {inspection} 
        Impression: {diagnosis} 
        Medical history: {case_history} 
        Laboratory Tests: {examination} 

        CT Image Segmentation Results of Cerebral Hemorrhage：
        Intraparenchymal hemorrhage: {vol1}
        Intraventricular hemorrhage: {vol2} 
        Perihematomal edema: {vol3}
        """

        return doctor_prompt_template.format(
            guideline=self.guideline,
            clinical_trials=self.clinical_trials,
            inspection=self.inspection,
            diagnosis=self.diagnosis,
            case_history=self.case_history,
            examination=self.examination,
            vol1=self.vol1,
            vol2=self.vol2,
            vol3=self.vol3
        )


def process_data(file_path: str) -> pd.DataFrame:
    df = pd.read_excel(file_path, sheet_name=0)
    inspection_column = df['Imaging findings']
    diagnosis_column = df['Impression']
    case_history_column = df['Medical history']
    examination_column = df['Laboratory Tests']
    vol1_column = df['label_1_volume_mL']
    vol2_column = df['label_2_volume_mL']
    vol3_column = df['label_3_volume_mL']

    prompts = []
    for i in range(len(df)):
        vol1 = f"{round(vol1_column[i], 2)} mL" if vol1_column[i] != 0.0 else "N/A"
        vol2 = f"{round(vol2_column[i], 2)} mL" if vol2_column[i] != 0.0 else "N/A"
        vol3 = f"{round(vol3_column[i], 2)} mL" if vol3_column[i] != 0.0 else "N/A"

        generator = PromptGenerator(
            guideline,
            clinical_trials,
            inspection_column[i],
            diagnosis_column[i],
            case_history_column[i],
            examination_column[i],
            vol1,
            vol2,
            vol3
        )

        prompts.append({
            "patient_prompt": generator.generate_patient_prompt(),
            "doctor_prompt": generator.generate_doctor_prompt()
        })

    return pd.DataFrame(prompts)


def main():

    file_path = '/home/pc/lyy/ICH-agent/Test/translated_predict_vol_final.xlsx'  

    prompts_df = process_data(file_path)

    patient_results = []
    doctor_results = []
    
    openai_api_key = "EMPTY"


    openai_api_base = "http://localhost:8000/v1"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    for _, row in tqdm(prompts_df.iterrows(), total=prompts_df.shape[0], desc="Processing Prompts"):
        
        chat_response = client.chat.completions.create(
        model="/devdata/llm_weights/Qwen3-30B-A3B",
        messages=[
            {"role": "user", "content": row["patient_prompt"]},
        ],
        max_tokens=32768,
        temperature=0.6,
        top_p=0.95,
        extra_body={
            "top_k": 20, 
            "chat_template_kwargs": {"enable_thinking": True},
        },
        )
        patient_result = {}
        
        patient_result['reasoning_content'] = chat_response.choices[0].message.reasoning_content
        patient_result['content'] = chat_response.choices[0].message.content
        chat_response = client.chat.completions.create(
        model="/devdata/llm_weights/Qwen3-30B-A3B",
        messages=[
            {"role": "user", "content": row["doctor_prompt"]},
        ],
        max_tokens=32768,
        temperature=0.6,
        top_p=0.95,
        extra_body={
            "top_k": 20, 
            "chat_template_kwargs": {"enable_thinking": True},
        },
        )
        doctor_result = {}
        doctor_result['reasoning_content'] = chat_response.choices[0].message.reasoning_content
        doctor_result['content'] = chat_response.choices[0].message.content

        patient_results.append(patient_result)
        doctor_results.append(doctor_result)

    root_path="/home/pc/lyy/ICH-agent/agent/Qwen3-30B-A3B_result/eng"
    with open(f"{root_path}/patient_results.json", "w") as file:
        json.dump(patient_results, file, ensure_ascii=False, indent=4)
    with open(f"{root_path}/doctor_results.json", "w") as file:
        json.dump(doctor_results, file, ensure_ascii=False, indent=4)

    patient_df = pd.DataFrame(patient_results)
    doctor_df = pd.DataFrame(doctor_results)
    patient_df.to_excel(f'{root_path}/patient_results.xlsx', index=False)
    doctor_df.to_excel(f'{root_path}/doctor_results.xlsx', index=False)
    print("Results saved to patient_results.xlsx and doctor_results.xlsx")



if __name__ == "__main__":
    main()