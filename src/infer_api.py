import pandas as pd
import json
from openai import OpenAI
from tqdm import tqdm

guideline = """
DIAGNOSIS AND ASSESSMENT
Physical Examination and Laboratory Assessment
In patients with spontaneous ICH, focused history, physical examination, and routine laboratory work and tests on hospital admission (eg, complete blood count, prothrombin time/ international normalized ratio [INR]/partial thromboplastin time, creatinine/estimated glomerular filtration rate, glucose, cardiac troponin and ECG, toxicology screen, and inflammatory markers) should be performed to help identify the type of hemorrhage, active medical issues, and risk of unfavorable outcomes.

Neuroimaging for ICH Diagnosis and Acute Course
In patients presenting with stroke-like symptoms, rapid neuroimaging with CT or MRI is recommended to confirm the diagnosis of spontaneous ICH.
In patients with spontaneous ICH and/or IVH, serial head CT can be useful within the first 24 hours after symptom onset to evaluate for hemorrhage expansion.
In patients with spontaneous ICH and/or IVH and with low GCS score or ND, serial head CT can be useful to evaluate for hemorrhage expansion, development of hydrocephalus, brain swelling, or herniation.
In patients with spontaneous ICH, CT angiography (CTA) within the first few hours of ICH onset may be reasonable to identify patients at risk for subsequent HE.
In patients with spontaneous ICH, using noncontrast computed tomography (NCCT) markers of HE to identify patients at risk for HE may be reasonable.

Diagnostic Assessment for ICH Pathogenesis
In patients with lobar spontaneous ICH and age <70 years, deep/posterior fossa spontaneous ICH and age <45 years, or deep/ posterior fossa and age 45 to 70 years without history of hypertension, acute CTA plus consideration of venography is recommended to exclude macrovascular causes or cerebral venous thrombosis.
In patients with spontaneous IVH and no detectable parenchymal hemorrhage, catheter intra-arterial digital subtraction angiography (DSA) is recommended to exclude a macrovascular cause.
In patients with spontaneous ICH and a CTA or magnetic resonance angiography (MRA) suggestive of a macrovascular cause, catheter intra-arterial DSA should be performed as soon as possible to confirm and manage underlying intracranial vascular malformations.
In patients with (a) lobar spontaneous ICH and age <70 years, (b) deep/posterior fossa ICH and age <45 years, or (c) deep/posterior fossa and age 45 to 70 years without history of hypertension and negative noninvasive imaging (CTA±venography and MRI/MRA), catheter intra-arterial DSA is reasonable to exclude a macrovascular cause.
In patients with spontaneous ICH with a negative CTA/venography, it is reasonable to perform MRI and MRA to establish a nonmacrovascular cause of ICH (such as CAA, deep perforating vasculopathy, cavernous malformation, or malignancy).
In patients with spontaneous ICH who undergo CT or MRI at admission, CTA plus consideration of venography or MRA plus consideration of venography performed acutely can be useful to exclude macrovascular causes or cerebral venous thrombosis.
In patients with spontaneous ICH and a negative catheter intra-arterial DSA and no clear microvascular diagnosis or other defined structural lesion, it may be reasonable to perform a repeat catheter intra-arterial DSA 3 to 6 months after ICH onset to identify a previously obscured vascular lesion.

MEDICAL AND NEUROINTENSIVE TREATMENT FOR ICH
In patients with spontaneous ICH requiring acute BP lowering, careful titration to ensure continuous smooth and sustained control of BP, avoiding peaks and large variability in SBP, can be beneficial for improving functional outcomes.
In patients with spontaneous ICH in whom acute BP lowering is considered, initiating treatment within 2 hours of ICH onset and reaching target within 1 hour can be beneficial to reduce the risk of HE and improve functional outcome.
In patients with spontaneous ICH of mild to moderate severity presenting with SBP between 150 and 220 mmHg, acute lowering of SBP to a target of 140 mmHg with the goal of maintaining in the range of 130 to 150 mmHg is safe and may be reasonable for improving functional outcomes.
In patients with spontaneous ICH presenting with large or severe ICH or those requiring surgical decompression, the safety and efficacy of intensive BP lowering are not well established.
In patients with spontaneous ICH of mild to moderate severity presenting with SBP >150 mmHg, acute lowering of SBP to <130 mmHg is potentially harmful.

Inpatient Care Setting
In patients with spontaneous ICH, provision of care in a specialized inpatient (eg, stroke) unit with a multidisciplinary team is recommended to improve outcomes and reduce mortality.
In patients with spontaneous ICH, provision of care at centers that can provide the full range of high-acuity care and expertise is recommended to improve outcomes.
In patients with spontaneous ICH and clinical hydrocephalus, transfer to centers with neurosurgical capabilities for definitive hydrocephalus management (eg, EVD placement and monitoring) is recommended to reduce mortality.
In patients with spontaneous ICH, care delivery that includes multidisciplinary teams trained in neurological assessment is recommended to improve outcomes.
In hospitalized patients with spontaneous ICH who require hospital transfer but do not have adequate airway protection, cannot support adequate gas exchange, and/or do not have a stable hemodynamic profile, appropriate life-sustaining therapies should be initiated before transportation to prevent acute medical decompensation in transport.
In patients with spontaneous ICH without indications for ICU admission at presentation, initial provision of care in a stroke unit compared with a general ward is reasonable to reduce mortality and improve outcomes.
In patients with moderate to severe spontaneous ICH, IVH, hydrocephalus, or infratentorial location, provision of care in a neuro-specific ICU compared with a general ICU is reasonable to improve outcomes and reduce mortality.
In patients with IVH or infratentorial ICH location, transfer to centers with neurosurgical capabilities might be reasonable to improve outcomes.
In patients with larger supratentorial ICH, transfer to centers with neurosurgical capabilities may be reasonable to improve outcomes.

Prevention and Management of Acute Medical Complications
In patients with spontaneous ICH, the use of standardized protocols and/or order sets is recommended to reduce disability and mortality.
In patients with spontaneous ICH, a formal dysphagia screening protocol should be implemented before initiation of oral intake to reduce disability and the risk of pneumonia.
In patients with spontaneous ICH, continuous cardiac monitoring for the first 24 to 72 hours of admission is reasonable to monitor for cardiac arrhythmias and new cardiac ischemia.
In patients with spontaneous ICH, diagnostic laboratory and radiographic testing for infection on admission and throughout the hospital course is reasonable to improve outcomes.

Thromboprophylaxis and Treatment of Thrombosis
In nonambulatory patients with spontaneous ICH, intermittent pneumatic compression (IPC) starting on the day of diagnosis is recommended for VTE (DVT and pulmonary embolism [PE]) prophylaxis.
In nonambulatory patients with spontaneous ICH, low-dose UFH or LMWH can be useful to reduce the risk for PE.
In nonambulatory patients with spontaneous ICH, initiating low-dose UFH or LMWH prophylaxis at 24 to 48 hours from ICH onset may be reasonable to optimize the benefits of preventing thrombosis relative to the risk of HE.
In nonambulatory patients with spontaneous ICH, graduated compression stockings of knee-high or thigh-high length alone are not beneficial for VTE prophylaxis.
For patients with acute spontaneous ICH and proximal DVT who are not yet candidates for anticoagulation, the temporary use of a retrievable filter is reasonable as a bridge until anticoagulation can be initiated.
For patients with acute spontaneous ICH and proximal DVT or PE, delaying treatment with UFH or LMWH for 1 to 2 weeks after the onset of ICH might be considered.

Nursing Care
In patients with spontaneous ICH, frequent neurological assessments (including GCS) should be performed by ED nurses in the early hyperacute phase of care to assess change in status, neurological examination, or level of consciousness.
In patients with spontaneous ICH, frequent neurological assessments in the ICU and stroke unit are reasonable for up to 72 hours of admission to detect early.
In patients with spontaneous ICH, specialized nurse stroke competencies can be effective in improving outcome and mortality.

Glucose Management
In patients with spontaneous ICH, monitoring serum glucose is recommended to reduce the risk of hyperglycemia and hypoglycemia.
In patients with spontaneous ICH, treating hypoglycemia (<40–60 mg/d, <2.2–3.3 mmol/L) is recommended to reduce mortality.
In patients with spontaneous ICH, treating moderate to severe hyperglycemia (>180– 200 mg/dL, >10.0–11.1 mmol/L) is reasonable to improve outcomes.

Temperature Management
In patients with spontaneous ICH, pharmacologically treating an elevated temperature may be reasonable to improve functional outcomes.
In patients with spontaneous ICH, the usefulness of therapeutic hypothermia (<35°C/95°F) to decrease peri-ICH edema is unclear.

Seizures and Antiseizure Drugs
In patients with spontaneous ICH, impaired consciousness, and confirmed electrographic seizures, antiseizure drugs should be administered to reduce morbidity.
In patients with spontaneous ICH and clinical seizures, antiseizure drugs are recommended to improve functional outcomes and prevent brain injury from prolonged recurrent seizures.
In patients with spontaneous ICH and unexplained abnormal or fluctuating mental status or suspicion of seizures, continuous electroencephalography (≥24 hours) is reasonable to diagnose electrographic seizures and epileptiform discharges.
In patients with spontaneous ICH without evidence of seizures, prophylactic antiseizure medication is not beneficial to improve functional outcomes, long-term seizure control, or mortality.

Neuroinvasive Monitoring, ICP, and Edema Treatment
In patients with spontaneous ICH or IVH and hydrocephalus that is contributing to decreased level of consciousness, ventricular drainage should be performed to reduce mortality.
In patients with moderate to severe spontaneous ICH or IVH with a reduced level of consciousness, ICP monitoring and treatment might be considered to reduce mortality and improve outcomes.
In patients with spontaneous ICH, the efficacy of early prophylactic hyperosmolar therapy for improving outcomes is not well established.
In patients with spontaneous ICH, bolus hyperosmolar therapy may be considered for transiently reducing ICP.
In patients with spontaneous ICH, corticosteroids should not be administered for treatment of elevated ICP.

SURGICAL INTERVENTIONS
Hematoma Evacuation
MIS Evacuation of ICH
For patients with supratentorial ICH of >20- to 30-mL volume with GCS scores in the moderate range (5–12), minimally invasive hematoma evacuation with endoscopic or stereotactic aspiration with or without thrombolytic use can be useful to reduce mortality compared with medical management alone.
For patients with supratentorial ICH of >20- to 30-mL volume with GCS scores in the moderate range (5–12) being considered for hematoma evacuation, it may be reasonable to select minimally invasive hematoma evacuation over conventional craniotomy to improve functional outcomes.
For patients with supratentorial ICH of >20- to 30-mL volume with GCS scores in the moderate range (5–12), the effectiveness of minimally invasive hematoma evacuation with endoscopic or stereotactic aspiration with or without thrombolytic use to improve functional outcomes is uncertain.

MIS Evacuation of IVH
For patients with spontaneous ICH, large IVH, and impaired level of consciousness, EVD is recommended in preference to medical management alone to reduce mortality.
For patients with a GCS score >3 and primary IVH or IVH extension from spontaneous supratentorial ICH of <30-mL volume requiring EVD, minimally invasive IVH evacuation with EVD plus thrombolytic is safe and is reasonable compared with EVD alone to reduce mortality.
For patients with a GCS score >3 and primary IVH or IVH extension from spontaneous supratentorial ICH of <30-mL volume requiring EVD, the effectiveness of minimally invasive IVH evacuation with EVD plus thrombolytic use to improve functional outcomes is uncertain.
For patients with severe spontaneous ICH‚ large IVH, and impaired level of consciousness, the efficacy of EVD for improving functional outcomes is not well established.
For patients with spontaneous supratentorial ICH of <30-mL volume and IVH requiring EVD, the usefulness of minimally invasive IVH evacuation with neuroendoscopy plus EVD, with or without thrombolytic, to improve functional outcomes and reduce permanent shunt dependence is uncertain.

Craniotomy for Supratentorial Hemorrhage
For most patients with spontaneous supratentorial ICH of moderate or greater severity, the usefulness of craniotomy for hemorrhage evacuation to improve functional outcomes or mortality is uncertain.
In patients with supratentorial ICH who are deteriorating, craniotomy for hematoma evacuation might be considered as a lifesaving measure.

Craniotomy for Posterior Fossa Hemorrhage
For patients with cerebellar ICH who are deteriorating neurologically, have brainstem compression and/or hydrocephalus from ventricular obstruction, or have cerebellar ICH volume ≥15 mL, immediate surgical removal of the hemorrhage with or without EVD is recommended in preference to medical management alone to reduce mortality.

Craniectomy for ICH
In patients with supratentorial ICH who are in a coma, have large hematomas with significant midline shift, or have elevated ICP refractory to medical management, decompressive craniectomy with or without hematoma evacuation may be considered to reduce mortality.
In patients with supratentorial ICH who are in a coma, have large hematomas with significant midline shift, or have elevated ICP refractory to medical management, effectiveness of decompressive craniectomy with or without hematoma evacuation to improve functional outcomes is uncertain.

Outcome Prediction
In patients with spontaneous ICH, administering a baseline measure of overall hemorrhage severity is recommended as part of the initial evaluation to provide an overall measure of clinical severity.
In patients with spontaneous ICH, a baseline severity score might be reasonable to provide a general framework for communication with the patient and their caregivers.
In patients with spontaneous ICH, a baseline severity score should not be used as the sole basis for forecasting individual prognosis or limiting life-sustaining treatment.

Decisions to Limit Life-Sustaining Treatment
In patients with spontaneous ICH who do not have preexisting documented requests for life sustaining therapy limitations, aggressive care, including postponement of new DNAR orders or withdrawal of medical support until at least the second full day of hospitalization, is reasonable to decrease mortality and improve functional outcome.
In patients with spontaneous ICH who are unable to fully participate in medical decision making, use of a shared decision-making model between surrogates and physicians is reasonable to optimize the alignment of care with patient wishes and surrogate satisfaction.
In patients with spontaneous ICH who have DNAR status, limiting other medical and surgical interventions, unless explicitly specified by the patient or surrogate, is associated with increased patient mortality.

Rehabilitation and Recovery
In patients with spontaneous ICH, multidisciplinary rehabilitation, including regular team meetings and discharge planning, should be performed to improve functional outcome and reduce morbidity and mortality.
In patients with spontaneous ICH with mild to moderate severity, early supported discharge is beneficial to increase the likelihood of patients living at home at 3 months.
In patients with spontaneous ICH with moderate severity, early rehabilitation beginning 24 to 48 hours after onset (including ADL training, stretching, functional task training) may be considered to improve functional outcome and reduce mortality.
In patients with spontaneous ICH without depression, fluoxetine therapy is not effective to enhance poststroke functional status.
In patients with spontaneous ICH, very early and intense mobilization within the first 24 hours is associated with lower likelihood of good recovery.

Neurobehavioral Complications
In patients with spontaneous ICH and moderate to severe depression, appropriate evidence-based treatments including psychotherapy and pharmacotherapy are useful to reduce symptoms of depression.
In patients with spontaneous ICH, administration of depression and anxiety screening tools in the postacute period is recommended to identify patients with poststroke depression and anxiety.
In patients with spontaneous ICH, administration of a cognitive screening tool in the postacute period is useful to identify patients with cognitive impairment and dementia. 
In patients with spontaneous ICH and cognitive impairment, referral for cognitive therapy is reasonable to improve cognitive outcomes.
In patients with spontaneous ICH and preexisting or new mood disorders requiring pharmacotherapy, continuation or initiation of SSRIs after ICH can be beneficial for the treatment of mood disorders.
In patients with spontaneous ICH and cognitive impairment, treatment with cholinesterase inhibitors or memantine might be considered to improve cognitive outcomes.

Prognostication of Future ICH Risk
In patients with spontaneous ICH in whom the risk for recurrent ICH may facilitate prognostication or management decisions, it is reasonable to incorporate the following risk factors for ICH recurrence into decision-making: (a) lobar location of the initial ICH; (b) older age; (c) presence, number, and lobar location of microbleeds on MRI; (d) presence of disseminated cortical superficial siderosis on MRI; (e) poorly controlled hypertension; (f) Asian or Black race; and (g) presence of apolipoprotein E ε2 or ε4 alleles.

BP Management
In patients with spontaneous ICH, BP control is recommended to prevent hemorrhage recurrence.
In patients with spontaneous ICH, it is reasonable to lower BP to an SBP of 130 mmHg and diastolic BP (DBP) of 80 mmHg for longterm management to prevent hemorrhage recurrence.

Management of Antithrombotic Agents
In patients with spontaneous ICH and conditions placing them at high risk of thromboembolic events, for example, a mechanical valve or LVAD, early resumption of anticoagulation to prevent thromboembolic complications is reasonable.
In patients with spontaneous ICH with an indication for antiplatelet therapy, resumption of antiplatelet therapy may be reasonable for the prevention of thromboembolic events based on consideration of benefit and risk.
In patients with nonvalvular atrial fibrillation (AF) and spontaneous ICH, the resumption of anticoagulation to prevent thromboembolic events and reduce all-cause mortality may be considered based on weighing benefit and risk.
In patients with AF and spontaneous ICH in whom the decision is made to restart anticoagulation, initiation of anticoagulation ≈7 to 8 weeks after ICH may be considered after weighing specific patient characteristics to optimize the balance of risks and benefits.
In patients with AF and spontaneous ICH deemed ineligible for anticoagulation, left atrial appendage closure may be considered to reduce the risk of thromboembolic events.

Management of Other Medications
In patients with spontaneous ICH and an established indication for statin pharmacotherapy, the risks and benefits of statin therapy on ICH outcomes and recurrence relative to overall prevention of cardiovascular events are uncertain.
In patients with spontaneous ICH, regular long-term use of nonsteroidal anti-inflammatory drugs (NSAIDs) is potentially harmful because of the increased risk of ICH.

Lifestyle Modifications/Patient and Caregiver Education
In patients with spontaneous ICH, lifestyle modification is reasonable to reduce BP.
In patients with spontaneous ICH, avoiding heavy alcohol consumption is reasonable to reduce hypertension and risk of ICH recurrence.
In patients with spontaneous ICH, lifestyle modification, including supervised training and counseling, may be reasonable to improve functional recovery.
In patients with spontaneous ICH, psychosocial education for the caregiver can be beneficial to increase patients’ activity level and participation and/or quality of life.
In patients with spontaneous ICH, practical support and training for the caregiver are reasonable to improve patients’ standing balance.
"""
clinical_trials = """
1.	Following the ENRICH trial, the minimally invasive trans-sulcal parafascicular surgery procedure performed within 24 hours of ictus is recommended specifically for lobar ICH with a hematoma volume more than 30 mL but strictly less than 80 mL. The procedure aims to reduce the hematoma volume to <15 mL in patients aged 18–80 years without significant premorbid disability.

2.	Following the INTERACT3 trial, Implementation of a care bundle protocol for intensive blood pressure lowering and other management algorithms for physiological control within several hours of the onset of symptoms resulted in improved functional outcome for patients with acute intracerebral haemorrhage.
The care bundle protocol included the early intensive lowering of systolic blood pressure (target <140 mm Hg), strict glucose control (target 6·1-7·8 mmol/L in those without diabetes and 7·8-10·0 mmol/L in those with diabetes), antipyrexia treatment (target body temperature ≤37·5°C), and rapid reversal of warfarin-related anticoagulation (target international normalised ratio <1·5) within 1 h of treatment, in patients where these variables were abnormal.

3.	The INTERACT4 study suggests prehospital blood-pressure reduction did not improve functional outcomes in a cohort of patients with undifferentiated acute stroke, of whom 46.5% subsequently received a diagnosis of hemorrhagic stroke. 

4.	The SWITCH study suggests that decompressive craniectomy without clot evacuation might benefit patients aged 18–75 years with 30–100 mL basal ganglia or thalamic ICH.

5.	ANNEXA-I study suggests among patients with intracerebral hemorrhage who were receiving factor Xa inhibitors, andexanet resulted in better control of hematoma expansion than usual care but was associated with thrombotic events, including ischemic stroke. 
"""



class PromptGenerator:
    """
    生成患者和医生的提示文本。
    """
    def __init__(self, guideline: str, clinical_trials: str, inspection: str, diagnosis: str, case_history: str, examination: str, vol1: str, vol2: str, vol3: str):
        """
        初始化PromptGenerator类。

        :param inspection: 检查所见
        :param diagnosis: 诊断结论
        :param case_history: 病历
        :param examination: 检验
        :param vol1: 脑实质出血体积
        :param vol2: 脑室积血体积
        :param vol3: 血肿周围水肿体积
        """
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
        """
        生成面向患者的提示文本。

        :return: 面向患者的提示文本
        """
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

        检查所见：{inspection} 

        诊断结论：{diagnosis} 

        病历：{case_history} 

        检验：{examination} 

        脑出血CT影像分割结果：脑实质出血{vol1}，脑室积血{vol2}，血肿周围水肿{vol3}。


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
        """
        生成面向医生的提示文本。

        :return: 面向医生的提示文本
        """
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

        检查所见：{inspection} 

        诊断结论：{diagnosis} 

        病历：{case_history} 

        检验：{examination} 

        脑出血CT影像分割结果：脑实质出血{vol1}，脑室积血{vol2}，血肿周围水肿{vol3}。
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
    """
    处理数据。

    :param file_path: 数据文件路径
    :return: 处理后的数据
    """
    df = pd.read_excel(file_path, sheet_name=0)
    inspection_column = df['检查所见']
    diagnosis_column = df['诊断结论']
    case_history_column = df['病历']
    examination_column = df['检验']
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

    file_path = '/path/to/ICH-agent/Test/predict_vol.xlsx' 

    prompts_df = process_data(file_path)

    patient_results = []
    doctor_results = []
    
    # Set OpenAI's API key and API base to use vLLM's API server.
    openai_api_key = "EMPTY"


    openai_api_base = "http://localhost:8000/v1"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    for _, row in tqdm(prompts_df.iterrows(), total=prompts_df.shape[0], desc="Processing Prompts"):
        chat_response = client.chat.completions.create(
        model="/path/to/Qwen3-30B-A3B",
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
        model="/path/to/Qwen3-30B-A3B",
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

    with open("/path/to/ICH-agent/agent/Qwen3-30B-A3B_result/patient_results.json", "w") as file:
        json.dump(patient_results, file, ensure_ascii=False, indent=4)
    with open("/path/to/ICH-agent/agent/Qwen3-30B-A3B_result/doctor_results.json", "w") as file:
        json.dump(doctor_results, file, ensure_ascii=False, indent=4)

    patient_df = pd.DataFrame(patient_results)
    doctor_df = pd.DataFrame(doctor_results)
    patient_df.to_csv('/path/to/lyy/ICH-agent/agent/Qwen3-30B-A3B_result/patient_results.csv', index=False)
    doctor_df.to_csv('/path/to/lyy/ICH-agent/agent/Qwen3-30B-A3B_result/doctor_results.csv', index=False)
    print("Results saved to patient_results.csv and doctor_results.csv")


if __name__ == "__main__":
    main()