import streamlit as st
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv
from typing import TypedDict
import logging

from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# -------------------- Setup --------------------
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -------------------- State --------------------
class ResumeState(TypedDict):
    resume_text: str
    parsed_profile: str
    evaluation: str
    job_roles: str
    skill_gaps: str
    action_plan: str

# -------------------- PDF Reader --------------------
def get_resume_text(pdf_file) -> str:
    logging.info("Extracting text from uploaded PDF.")
    text = ""
    reader = PdfReader(pdf_file)
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted
    return text

# -------------------- Agent 1: Resume Parser (SYSTEMATIC OUTPUT) --------------------
def resume_parser_agent(state: ResumeState):
    prompt = PromptTemplate.from_template("""
You are an expert resume analyst.

Read the resume carefully and extract information.
The output MUST be clearly structured, readable, and systematic.

STRICT FORMATTING RULES:
- Do NOT use JSON, tables, or symbols
- Do NOT merge sections into one paragraph
- Each section must start on a new line
- Use clear headings exactly as shown
- Use plain text only

Resume:
{resume_text}

FORMAT THE OUTPUT EXACTLY LIKE THIS:

Name:
<full name or Not Mentioned>

Years of Experience:
<clearly describe experience level>

Skills:
<group skills logically with line breaks>

Tools & Technologies:
<group tools logically with line breaks>

Education:
<degree, institution, duration>

Projects:
<project name and 1–2 line description>

Domain Focus:
<primary career or technical focus>
""")

    response = (prompt | llm).invoke(
        {"resume_text": state["resume_text"]}
    ).content

    return {"parsed_profile": response}

# -------------------- Agent 2: Resume Evaluator (DETAILED) --------------------
def resume_evaluator_agent(state: ResumeState):
    prompt = PromptTemplate.from_template("""
You are a senior recruiter and ATS specialist.

Based on the resume profile below, provide a deep evaluation.

Profile:
{parsed_profile}

Explain in detail:
Overall resume strengths with reasoning
Weak areas and why they matter in hiring
Resume quality score out of 10 with justification
ATS optimization issues and improvement suggestions
How this resume would perform in real hiring pipelines
""")

    response = (prompt | llm).invoke(
        {"parsed_profile": state["parsed_profile"]}
    ).content

    return {"evaluation": response}

# -------------------- Agent 3: Job Role Matcher (DETAILED) --------------------
def job_role_agent(state: ResumeState):
    prompt = PromptTemplate.from_template("""
You are a career advisor.

Based on the resume profile below, suggest the most suitable job roles.

Profile:
{parsed_profile}

For each role, explain:
Job title
Expected seniority level
Why this role matches the candidate’s skills
What companies typically expect for this role
Career growth potential
""")

    response = (prompt | llm).invoke(
        {"parsed_profile": state["parsed_profile"]}
    ).content

    return {"job_roles": response}

# -------------------- Agent 4: Skill Gap Analyzer (DETAILED) --------------------
def skill_gap_agent(state: ResumeState):
    prompt = PromptTemplate.from_template("""
You are an industry skill-gap analyst.

Compare the candidate profile with current industry standards.

Profile:
{parsed_profile}

Target Job Roles:
{job_roles}

Provide a detailed gap analysis:
Technical skills missing or weak
Why these skills are important in real jobs
Tools, frameworks, and platforms to learn
Recommended certifications with reasoning
""")

    response = (prompt | llm).invoke({
        "parsed_profile": state["parsed_profile"],
        "job_roles": state["job_roles"]
    }).content

    return {"skill_gaps": response}

# -------------------- Agent 5: Career Coach (DETAILED) --------------------
def career_coach_agent(state: ResumeState):
    prompt = PromptTemplate.from_template("""
You are an experienced career coach.

Based on the evaluation and skill gaps below, create a realistic career action plan.

Evaluation:
{evaluation}

Skill Gaps:
{skill_gaps}

Create a detailed 30-60-90 day plan explaining:
Resume improvements and why they matter
Learning roadmap with focus areas
Project ideas aligned with job roles
Job application and interview preparation strategy
""")

    response = (prompt | llm).invoke({
        "evaluation": state["evaluation"],
        "skill_gaps": state["skill_gaps"]
    }).content

    return {"action_plan": response}

# -------------------- LangGraph --------------------
graph = StateGraph(ResumeState)

graph.add_node("parser", resume_parser_agent)
graph.add_node("evaluator", resume_evaluator_agent)
graph.add_node("job_roles", job_role_agent)
graph.add_node("skill_gap", skill_gap_agent)
graph.add_node("career_coach", career_coach_agent)

graph.set_entry_point("parser")

graph.add_edge("parser", "evaluator")
graph.add_edge("evaluator", "job_roles")
graph.add_edge("job_roles", "skill_gap")
graph.add_edge("skill_gap", "career_coach")
graph.add_edge("career_coach", END)

resume_graph = graph.compile()

# -------------------- Streamlit UI --------------------
def main():
    st.set_page_config("AI Resume Screener", layout="wide")
    st.title("🧠 Multi-Agent AI Resume Screener")

    uploaded_file = st.file_uploader(
        "Upload Resume (PDF)",
        type=["pdf"]
    )

    if uploaded_file:
        with st.spinner("Running detailed multi-agent analysis..."):
            resume_text = get_resume_text(uploaded_file)
            final_state = resume_graph.invoke(
                {"resume_text": resume_text}
            )

        st.success("Detailed Analysis Complete")

        st.header("📌 Parsed Resume Profile (Detailed)")
        st.write(final_state["parsed_profile"])

        st.header("📊 Resume Evaluation (Recruiter View)")
        st.write(final_state["evaluation"])

        st.header("🎯 Job Role Recommendations (Detailed)")
        st.write(final_state["job_roles"])

        st.header("🧩 Skill Gap Analysis (Industry Comparison)")
        st.write(final_state["skill_gaps"])

        st.header("🚀 30-60-90 Day Career Action Plan")
        st.write(final_state["action_plan"])

if __name__ == "__main__":
    main()
