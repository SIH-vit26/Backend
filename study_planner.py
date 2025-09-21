from typing import TypedDict,List
from pathlib import Path
from langgraph.graph import StateGraph,END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import json
import pandas as pd
import os
from dotenv import load_dotenv
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from datetime import datetime
from report import generate_pdf_report
load_dotenv()


class StudentDetails(TypedDict):
    comments:str
    subjects:List[str]
    weaktopics:List[str]
    WeeklyStudyHours:int
    StudyPlan:str


class StudyPlan():
    def __init__(self):
        self.llm=ChatOpenAI(model="gpt-3.5-turbo",temperature=0,max_retries=3)
    
    def input_parser(self,text:str):
        message="You are an expert academic counsellor. Provided with the student details, " \
        "Extract the following details in JSON format:\n1. comments: A brief comment on the student's " \
        "subjects: A list of subjects the student is enrolled in and weak at. Do not include the subjects he is good fair or stromg at.\n3. " \
        "weaktopics: A list of topics the student is struggling with.\n4. " \
        "WeeklyStudyHours: Recommended number of study hours per week.\n" \
        "Ensure the output is a valid JSON object with keys 'comments', 'subjects', 'weaktopics', and 'WeeklyStudyHours'.\n\nStudent Details:\n" \
        "The input recived is:\n"+text+"\n\n" \
        "NOTE: If any of the information is missing, assign default values as follows:\n" \
        "comments: '', subjects: [], weaktopics: [], WeeklyStudyHours: 0\n"

        response = self.llm.invoke(message)
        raw_output = response.content

        try:
            parsed = json.loads(raw_output)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON output: {raw_output}")

        # Assign with defaults if keys are missing
        student_details: StudentDetails = {
            "comments": parsed.get("comments", ""),
            "subjects": parsed.get("subjects", []),
            "weaktopics": parsed.get("weaktopics", []),
            "WeeklyStudyHours": parsed.get("WeeklyStudyHours", 0),
            "StudyPlan": ""  # Will be filled later
        }

        self.student_details = student_details
        return student_details

    def knowledge_integration(self) -> dict:
        kb_folder = Path.cwd() / os.getenv("FOLDER_PATH")
        json_data = {}
        
        if not kb_folder.exists():
            return {"error": f"Folder not found at {kb_folder}"}
        
        json_files = list(kb_folder.glob("*.json"))
        
        if not json_files:
            return {"error": "No JSON files found in the folder"}
        
        for json_file in sorted(json_files):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                json_data[json_file.stem] = data
            except Exception as e:
                json_data[json_file.stem] = {"error": f"Failed to read file: {str(e)}"}
        
        return json_data
        

def study_plan_generator(student_details: StudentDetails) -> dict:
    llm=ChatOpenAI(model="gpt-3.5-turbo",temperature=0,max_retries=3)
    
    if not student_details:
        student_details={}
        return {"error": "No student details provided. Cannot create a study plan."}
    
    knowledge_context=StudyPlan().knowledge_integration()

    print("Knowledge Context Retrieved:")
    print(knowledge_context)
    print("*"*50)

    if not knowledge_context or "error" in knowledge_context:
        return {"error": "Knowledge base is unavailable. Cannot create a study plan."}
    
    message = f"""
        You are an expert academic counselor and study planner. Your task is to generate a **personalized 4-week study plan** 
        for the student based strictly on the provided NCERT knowledge base and the student's details.

        ---

        ## Knowledge Base Structure
        The knowledge base is organized in JSON format with the following hierarchy:
        - Each subject (Chemistry, Mathematics, Physics) is represented as an object.
        - Each subject contains one or more **books**.
        - Each book contains a list of **units/chapters**.
        - Each unit or chapter has:
        - A number (unitNumber/chapterNumber).
        - A title (unitTitle/chapterTitle).
        - A list of **topics** under it.

        ⚠️ Important: You must only use **topics, chapters, and units exactly as they appear in this knowledge base**. 
        Do not invent new topics or alter names.

        ---

        ## Student Details
        - Comments: {student_details.get("comments")}
        - Subjects: {", ".join(student_details.get("subjects", []))}
        - Weak Topics: {", ".join(student_details.get("weaktopics", []))}
        - Weekly Study Hours: {student_details.get("WeeklyStudyHours")} hours

        ---

        ## NCERT Knowledge Base
        {knowledge_context}

        ---

        ## Instructions for Generating the Study Plan
        1. **Strictly select chapters and topics from the NCERT knowledge base above.**
        - Do not invent topics or modify names.
        - Respect the chapter/unit organization in the KB.
        - First give importance to the subjects the student is weak a based on the Weak Topics. 
        2. **Prioritize weak topics** from the student's details and ensure they appear multiple times in the schedule.
        3. Distribute the total {student_details.get("WeeklyStudyHours")} study hours across 7 days per week, for 4 weeks (28 days).
        - Divide into daily sessions with clear timeslots (e.g., 6:00–7:00 PM).
        - Include breaks after 1–2 hours.
        4. Maintain **balance**:
        - Reinforce weak topics frequently.
        - Cover all subjects fairly so the student progresses across the syllabus.
        5. Output Format:
        - Week by Week → Day by Day.
        - Each entry must include:
            - Day, Time Slot, Subject, Book, Unit/Chapter Title, Topic, Activity/Focus.
        6. End with a **monthly summary**:
        - Chapters and topics covered.
        - Expected learning outcomes (e.g., mastery of weak topics, balanced coverage of subjects).

        ---

        ## Example Output Format
        Week 1  
        - **Day 1 (Monday)**  
        - 6:00–7:00 PM → Mathematics → Book: NCERT Mathematics Part 1 → Unit 2: Inverse Trigonometric Functions → Topic: Properties of Inverse Trigonometric Functions → Practice problem-solving  
        - 7:15–8:15 PM → Physics → Book: NCERT Physics Part 1 → Chapter 1: Electric Charges and Fields → Topic: Coulomb's Law → Numerical practice  

        Week 2 … continue in the same structure up to Week 4.

        ---

        Now generate the detailed 4-week study plan strictly using the chapters and topics from the provided NCERT knowledge base.
        """


    
    response = llm.invoke(message)
    study_plan = response.content
    
    return {"StudyPlan": study_plan}

    
# study=StudyPlan()
# student_details=study.input_parser("The student, Rohan, has been attending classes " \
# "regularly but his performance in Mathematics and Physics has been below average. " \
# "Teachers have specifically pointed out that he struggles with Trigonometry and Semiconductor Electronics. " \
# "In Chemistry and Computer Science, he performs fairly well. " \
# "Considering his current learning pace, mentors recommend around 18–20 hours of study per week, " \
# "focusing more on problem-solving practice in Math and Physics")

# # print("Student Details Parsed:")
# print(student_details)
# print("*"*50)

# print("Generating Study Plan:")

student_graph=StateGraph(StudentDetails)
student_graph.add_node("StudyPlanGeneration",study_plan_generator)
student_graph.set_entry_point("StudyPlanGeneration")
student_graph.add_edge("StudyPlanGeneration",END)
result=student_graph.compile()

# final_output=result.invoke(student_details)

# print(final_output["StudyPlan"])

# Generate PDF Report
#generate_pdf_report(final_output["StudyPlan"])