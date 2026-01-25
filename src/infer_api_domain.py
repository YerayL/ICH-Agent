import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from guideline_and_clinical_trials import clinical_trials, guideline


COLUMN_OPTIONS: Dict[str, List[str]] = {
    "inspection": ["Imaging findings", "检查所见"],
    "diagnosis": ["Impression", "诊断结论"],
    "case_history": ["Medical history", "病历"],
    "examination": ["Laboratory Tests", "检验"],
    "vol1": ["label_1_volume_mL"],
    "vol2": ["label_2_volume_mL"],
    "vol3": ["label_3_volume_mL"],
}


@dataclass
class InferenceConfig:
    data_path: Path
    model: str
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    output_dir: Path = Path("./Qwen3-30B-A3B_result/eng")
    sheet_index: int = 0
    max_tokens: int = 32768
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    limit_rows: Optional[int] = None


class PromptGenerator:
    def __init__(
        self,
        guideline_text: str,
        clinical_trials_text: str,
        inspection: str,
        diagnosis: str,
        case_history: str,
        examination: str,
        vol1: str,
        vol2: str,
        vol3: str,
    ) -> None:
        self.guideline = guideline_text
        self.clinical_trials = clinical_trials_text
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

        CT Image Segmentation Results of Cerebral Hemorrhage:
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
            vol3=self.vol3,
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

        CT Image Segmentation Results of Cerebral Hemorrhage:
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
            vol3=self.vol3,
        )


def _pick_column(df: pd.DataFrame, options: Iterable[str], logical_name: str) -> pd.Series:
    for col in options:
        if col in df.columns:
            return df[col]
    available = ", ".join(df.columns)
    raise KeyError(f"Missing column for {logical_name}. Expected one of {options}; available columns: {available}")


def _format_volume(value: float) -> str:
    if value == 0.0:
        return "N/A"
    return f"{round(value, 2)} mL"


def build_prompt_dataframe(file_path: Path, sheet_index: int, limit_rows: Optional[int]) -> pd.DataFrame:
    df = pd.read_excel(file_path, sheet_name=sheet_index)
    if limit_rows is not None:
        df = df.head(limit_rows)

    inspection = _pick_column(df, COLUMN_OPTIONS["inspection"], "inspection")
    diagnosis = _pick_column(df, COLUMN_OPTIONS["diagnosis"], "diagnosis")
    case_history = _pick_column(df, COLUMN_OPTIONS["case_history"], "case_history")
    examination = _pick_column(df, COLUMN_OPTIONS["examination"], "examination")
    vol1 = _pick_column(df, COLUMN_OPTIONS["vol1"], "vol1")
    vol2 = _pick_column(df, COLUMN_OPTIONS["vol2"], "vol2")
    vol3 = _pick_column(df, COLUMN_OPTIONS["vol3"], "vol3")

    prompts: List[Dict[str, str]] = []
    for idx in range(len(df)):
        generator = PromptGenerator(
            guideline_text=guideline,
            clinical_trials_text=clinical_trials,
            inspection=str(inspection[idx]),
            diagnosis=str(diagnosis[idx]),
            case_history=str(case_history[idx]),
            examination=str(examination[idx]),
            vol1=_format_volume(float(vol1[idx])),
            vol2=_format_volume(float(vol2[idx])),
            vol3=_format_volume(float(vol3[idx])),
        )
        prompts.append(
            {
                "patient_prompt": generator.generate_patient_prompt(),
                "doctor_prompt": generator.generate_doctor_prompt(),
            }
        )

    return pd.DataFrame(prompts)


def _extract_message_choice(chat_response) -> Dict[str, Optional[str]]:
    message = chat_response.choices[0].message
    return {
        "reasoning_content": getattr(message, "reasoning_content", None),
        "content": message.content,
    }


def run_inference(config: InferenceConfig) -> None:
    prompts_df = build_prompt_dataframe(config.data_path, config.sheet_index, config.limit_rows)

    client = OpenAI(api_key=config.api_key, base_url=config.base_url)

    patient_results: List[Dict[str, Optional[str]]] = []
    doctor_results: List[Dict[str, Optional[str]]] = []

    for _, row in tqdm(prompts_df.iterrows(), total=prompts_df.shape[0], desc="Processing prompts"):
        patient_response = client.chat.completions.create(
            model=config.model,
            messages=[{"role": "user", "content": row["patient_prompt"]}],
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            extra_body={"top_k": config.top_k, "chat_template_kwargs": {"enable_thinking": True}},
        )
        patient_results.append(_extract_message_choice(patient_response))

        doctor_response = client.chat.completions.create(
            model=config.model,
            messages=[{"role": "user", "content": row["doctor_prompt"]}],
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            extra_body={"top_k": config.top_k, "chat_template_kwargs": {"enable_thinking": True}},
        )
        doctor_results.append(_extract_message_choice(doctor_response))

    config.output_dir.mkdir(parents=True, exist_ok=True)
    patient_json = config.output_dir / "patient_results.json"
    doctor_json = config.output_dir / "doctor_results.json"
    patient_excel = config.output_dir / "patient_results.xlsx"
    doctor_excel = config.output_dir / "doctor_results.xlsx"

    with patient_json.open("w", encoding="utf-8") as file:
        json.dump(patient_results, file, ensure_ascii=False, indent=4)
    with doctor_json.open("w", encoding="utf-8") as file:
        json.dump(doctor_results, file, ensure_ascii=False, indent=4)

    pd.DataFrame(patient_results).to_excel(patient_excel, index=False)
    pd.DataFrame(doctor_results).to_excel(doctor_excel, index=False)

    print(f"Saved patient results to {patient_excel}")
    print(f"Saved doctor results to {doctor_excel}")


def parse_args() -> InferenceConfig:
    parser = argparse.ArgumentParser(description="Batch infer domain-enhanced Qwen3-30B-A3B for ICH prompts")
    parser.add_argument("data_path", type=Path, help="Path to the Excel file containing patient data")
    parser.add_argument("--model", default="/devdata/llm_weights/Qwen3-30B-A3B", help="Model identifier served by vLLM/OpenAI API")
    parser.add_argument("--base-url", default="http://localhost:8000/v1", help="OpenAI-compatible base URL")
    parser.add_argument("--api-key", default="EMPTY", help="API key for the inference endpoint")
    parser.add_argument("--output-dir", type=Path, default=Path("./Qwen3-30B-A3B_result/eng"), help="Directory to save outputs")
    parser.add_argument("--sheet-index", type=int, default=0, help="Excel sheet index to load")
    parser.add_argument("--max-tokens", type=int, default=32768, help="Maximum tokens per completion")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p nucleus sampling")
    parser.add_argument("--top-k", type=int, default=20, help="Top-k sampling")
    parser.add_argument("--limit-rows", type=int, default=None, help="Optional limit on number of rows to process")
    args = parser.parse_args()

    return InferenceConfig(
        data_path=args.data_path,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        output_dir=args.output_dir,
        sheet_index=args.sheet_index,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        limit_rows=args.limit_rows,
    )


def main() -> None:
    config = parse_args()
    run_inference(config)


if __name__ == "__main__":
    main()