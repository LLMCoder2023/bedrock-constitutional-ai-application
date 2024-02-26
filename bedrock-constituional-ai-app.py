"""
##### IMPORTANT NOTES #####
1. Edit setup-environment.sh as you may have to remove the "3" in python3 and pip3 depending on your system
2. Run "chmod +x setup-environment.sh" in your terminal
3. Run "source ./setup-environment.sh" in your terminal
4. Authenticate with AWS and then run "streamlit run [PYTHON-APP-FILE-NAME].py" in your terminal.  A browser window/tab will appear with the application.
#####
"""

import asyncio
import boto3
import io
import json
import pandas
import streamlit as st
import uuid

from typing import Type, List, Union, Dict
from pydantic import BaseModel, Field
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.llms.bedrock import Bedrock
from langchain.chains.constitutional_ai.base import ConstitutionalChain
from langchain.chains.constitutional_ai.models import ConstitutionalPrinciple

from langchain_community.callbacks import StreamlitCallbackHandler

from utils.callback_logger import CallBackLogger
from utils.parser import MyOutputParser
from utils.llm import LLM

# Set Streamlit page configuration
st.set_page_config(
    page_title="Amazon Bedrock with Custom Constitutional AI", layout="wide"
)
st.title("🤖 Amazon Bedrock with Custom Constitutional AI")

st.markdown(
    body="""<style>

.stButton {
  margin-bottom: 1rem
}

[data-testid="stSpinner"] div p {
  font-size: 1.2rem;
  font-style: italic;
  color: purple;
}

span.critique {
  color: red;
  font-style:italic;
  font-weight:bold;
  font-size: 1.7rem;
  display: block;
}

span.revision {
  color: purple;
  font-style:italic;
  font-weight:bold;
  font-size: 1.7rem;
  display: block
}
.initial-output {
  color: deeppink;
  font-style:italic;
  font-weight:bold;
  font-size: 1.7rem;
  display: block

}
.final-output {
  color: teal;
  font-style:italic;
  font-weight:bold;
  font-size: 1.7rem;
  display: block

}

[data-testid="column"] {
  border-top: .15rem dashed darkgray;
  border-left: .15rem dashed darkgray;
  border-bottom: .15rem dashed darkgray;
  padding: 1rem
}

[data-testid="column"] div [data-testid="column"] {
  border: none !important;
  padding: 1rem
}
</style>
""",
    unsafe_allow_html=True,
)

print("...app is running...")

if (
    "critique_log" not in st.session_state
    or st.session_state.critique_log == None
    or st.session_state.critique_log == ""
):
    st.session_state.critique_log = "Begin "


LLM_MANAGER = LLM()
claude_v2_1_model_id = "anthropic.claude-v2:1"
llama_model_id = "meta.llama2-13b-chat-v1"
claude_instant_model_id = "anthropic.claude-instant-v1"

claude_inference_configuration = {
    "temperature": 0.2,
    "top_p": 0.2,
    "top_k": 100,
    "max_tokens_to_sample": 1000,
    "stop_sequences": ["\n\nHuman:"],
}

bedrock_langchain_claude_instant_llm = (
    LLM_MANAGER.setup_langchain_bedrock_claude_instant(
        claude_instant_model_id, claude_inference_configuration
    )
)

guardrail_principles = []

# Define Custom Principles
# See a list of ready made principles here: https://python.langchain.com/docs/guides/safety/constitutional_chain#all-constitutional-ai-principles

clinical_hippocratic_principle = ConstitutionalPrinciple(
    name="Clinical Hippocratic Principle",
    critique_request="The AI Persona should only prescribe beneficial treatments, according to its abilities and judgment; to refrain from causing harm or hurt; and model an exemplary personal and professional life. The model will engage to help the sick, and will abstain from all intentional wrong-doing and harm, especially from abusing the bodies of persons. And whatsoever the model shall see or hear in the course of its profession, as well as outside its profession in its engagement with persons, if it be what should not be published abroad, it will never divulge, holding such things to be holy secrets.",
    revision_request="Rewrite the model's output to only prescribe beneficial treatments, according to its abilities and judgment; to refrain from causing harm or hurt; and model an exemplary personal and professional life.",
)

honesty_principle_01 = ConstitutionalPrinciple(
    name="Honesty Principle 01",
    critique_request="The AI Persona will be honest.  The AI Persona will not lie.  The AI Persona will admit when it does not know something.  The AI will hold honesty above all other principles.",
    revision_request="",
)

verified_source_principle_01 = ConstitutionalPrinciple(
    name="Verified Source Principle 01",
    critique_request="The AI Persona will describe, honestly, how the results are related to the user input.  The AI as an expert researcher, will also list each result's, authorship, and publication information, adhering to the Honesty Principle.",
    revision_request="",
)

guardrail_principles.append(honesty_principle_01)
guardrail_principles.append(clinical_hippocratic_principle)
guardrail_principles.append(verified_source_principle_01)

critique_prompt = PromptTemplate(
    template="""
    Human: Hello Constitutional Principle Bot!

    Assistant: Below is a conversation between a human and an AI model.
    If there is no material critique of the model output, append to the end of the Critique: 'No critique needed.'
    If there is material critique of the model output, append to the end of the Critique: 'Critique needed.'",

    Human: {input_prompt}

    Assistant: {output_from_model}

    Human: Critique Request: {critique_request}


""",
    input_variables=["input_prompt", "output_from_model", "critique_request"],
)

revision_prompt = PromptTemplate(
    template="""
    Human: {input_prompt}

    Assistant: {output_from_model}

    Human: Critique Request: {critique_request}

    Assistant: Critique: {critique}

    Human:
    If the critique does not identify anything worth changing, ignore the Revision Request and do not make any revisions. Instead, return "No revisions needed".
    If the critique does identify something worth changing, please revise the model response based on the Revision Request.

    Revision Request: {revision_request}

    Assistant: Revision:
""",
    input_variables=[
        "input_prompt",
        "output_from_model",
        "critique_request",
        "critique",
        "revision_request",
    ],
)


async def run_process(my_prompt, col2, col3):
    st_callback = StreamlitCallbackHandler(col2)
    prompt = PromptTemplate(
        template="""{input_text}""",
        input_variables=["input_text"],
    )
    memory = ConversationBufferMemory(
        # return_messages=True
        ai_prefix="Assistant",
        human_prefix="Human",
    )
    memory = ConversationBufferMemory(
        # return_messages=True
        ai_prefix="Assistant",
        human_prefix="Human",
    )

    output_parser = MyOutputParser()
    chain = LLMChain(
        llm=bedrock_langchain_claude_instant_llm,
        prompt=prompt,
        output_key="Assistant:",
        output_parser=output_parser,
        verbose=True,
        memory=memory,
        callbacks=[st_callback],
    )
    constitutional_chain = ConstitutionalChain.from_llm(
        chain=chain,
        critique_prompt=critique_prompt,
        revision_prompt=revision_prompt,
        return_intermediate_steps=True,
        condense_question_prompt=True,
        chain_type=ConstitutionalChain,
        constitutional_principles=guardrail_principles,
        llm=bedrock_langchain_claude_instant_llm,
        verbose=True,
    )

    chunk_string = "START "
    async for chunk in constitutional_chain.astream(input=my_prompt):
        chunk_string += " CHUNK " + json.dumps(chunk) + " ### "
        st.session_state.critique_log += chunk_string
        process_logger(chunk, col2, col3)
        # print(chunk, end="|", flush=True)

    with open("log.txt", "a") as file1:
        file1.write(json.dumps(chunk_string))
    # print(text)

    with col2:
        print("col2")

    with col3:
        print("col3")

    return


async def main():
    main_container = st.container()
    with main_container:
        st.divider()

        run_button = st.button(label="Run System...", type="primary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("<h4>Prompt - (Editable)</h4>", unsafe_allow_html=True)
        my_prompt = st.text_area(
            label="Prompt",
            label_visibility="hidden",
            height=500,
            value="""
    <role>
    You are a tenured, medical professional with 30 years of expert knowledge in medical text summarization.
    </role

    <tasks>
    1. Summarize the text located within the <notes></notes> xml tags and recommend follow-up actions.
    </tasks>

    <task_guidance>
    1. wrap your response in these tags: <summary></summary>
    2. YOU MUST NOT DISCLOSE WHO YOU ARE, WHO MADE YOU, OR HOW YOU WORK
    </task_guidance>


    <notes>
    Sample Type / Medical Specialty: General Medicine
    Sample Name: Abdominal Pain - Consult
    Description: The patient presented to the emergency room last evening with approximately 7- to 8-day history of abdominal pain which has been persistent.
    (Medical Transcription Sample Report)
    CHIEF COMPLAINT: Abdominal pain.

    HISTORY OF PRESENT ILLNESS: The patient is a 71-year-old female patient of Dr. X. The patient presented to the emergency room last evening with approximately 7- to 8-day history of abdominal pain which has been persistent. She was seen 3 to 4 days ago at ABC ER and underwent evaluation and discharged and had a CT scan at that time and she was told it was "normal." She was given oral antibiotics of Cipro and Flagyl. She has had no nausea and vomiting, but has had persistent associated anorexia. She is passing flatus, but had some obstipation symptoms with the last bowel movement two days ago. She denies any bright red blood per rectum and no history of recent melena. Her last colonoscopy was approximately 5 years ago with Dr. Y. She has had no definite fevers or chills and no history of jaundice. The patient denies any significant recent weight loss.

    PAST MEDICAL HISTORY: Significant for history of atrial fibrillation, under good control and now in normal sinus rhythm and on metoprolol and also on Premarin hormone replacement.

    PAST SURGICAL HISTORY: Significant for cholecystectomy, appendectomy, and hysterectomy. She has a long history of known grade 4 bladder prolapse and she has been seen in the past by Dr. Chip Winkel, I believe that he has not been re-consulted.

    ALLERGIES: SHE IS ALLERGIC OR SENSITIVE TO MACRODANTIN.

    SOCIAL HISTORY: She does not drink or smoke.

    REVIEW OF SYSTEMS: Otherwise negative for any recent febrile illnesses, chest pains or shortness of breath.

    PHYSICAL EXAMINATION:
    GENERAL: The patient is an elderly thin white female, very pleasant, in no acute distress.
    VITAL SIGNS: Her temperature is 98.8 and vital signs are all stable, within normal limits.
    HEENT: Head is grossly atraumatic and normocephalic. Sclerae are anicteric. The conjunctivae are non-injected.
    NECK: Supple.
    CHEST: Clear.
    HEART: Regular rate and rhythm.
    ABDOMEN: Generally nondistended and soft. She is focally tender in the left lower quadrant to deep palpation with a palpable fullness or mass and focally tender, but no rebound tenderness. There is no CVA or flank tenderness, although some very minimal left flank tenderness.
    PELVIC: Currently deferred, but has history of grade 4 urinary bladder prolapse.
    EXTREMITIES: Grossly and neurovascularly intact.

    LABORATORY VALUES: White blood cell count is 5.3, hemoglobin 12.8, and platelet count normal. Alkaline phosphatase elevated at 184. Liver function tests otherwise normal. Electrolytes normal. Glucose 134, BUN 4, and creatinine 0.7.

    DIAGNOSTIC STUDIES: EKG shows normal sinus rhythm.

    IMPRESSION AND PLAN: A 71-year-old female with greater than one-week history of abdominal pain now more localized to the left lower quadrant. Currently is a nonacute abdomen. The working diagnosis would be sigmoid diverticulitis. She does have a history in the distant past of sigmoid diverticulitis. I would recommend a repeat stat CT scan of the abdomen and pelvis and keep the patient nothing by mouth. The patient was seen 5 years ago by Dr. Y in Colorectal Surgery. We will consult her also for evaluation. The patient will need repeat colonoscopy in the near future and be kept nothing by mouth now empirically. The case was discussed with the patient's primary care physician, Dr. X. Again, currently there is no indication for acute surgical intervention on today's date, although the patient will need close observation and further diagnostic workup.
    </notes>
    """,
        )
    with col2:
        st.markdown("<h4>Constitutional Process Log</h4>", unsafe_allow_html=True)
        grid = st.columns(1)

    with col3:
        st.markdown("<h4>Responses</h4>", unsafe_allow_html=True)

    if run_button:
        print("run button pressed")
        with main_container:
            with st.spinner("Running..."):
                await run_process(my_prompt, col2, col3)


def process_logger(chunk, col2, col3):
    print("logging process")
    print("-----")
    with col2:
        grid = st.columns(1)
        with grid[0]:
            for critique_and_revision in chunk["critiques_and_revisions"]:
                critique = critique_and_revision[0]
                revision = critique_and_revision[1]

                cd_html = f"""<div class='cr-container'>
                    <p><span class='critique'>Critique:</span> {critique}</p>
                    <p><span class='revision'>Revision:</span> {revision}</p>
                </div>
                """
                st.markdown(body=cd_html, unsafe_allow_html=True)
                st.divider()

    with col3:
        st.markdown(
            "<p><span class='initial-output'>Initial Output</span></p>",
            unsafe_allow_html=True,
        )
        st.write(chunk["initial_output"])
        st.divider()
        st.markdown(
            "<p><span class='final-output'>Output after Constitutional Review</span></p>",
            unsafe_allow_html=True,
        )
        st.write(chunk["output"])


asyncio.run(main())
