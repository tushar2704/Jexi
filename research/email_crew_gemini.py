##© 2024 Tushar Aggarwal. All rights reserved.(https://tushar-aggarwal.com)
##Jexi by [Towards-GenAI] (https://github.com/Towards-GenAI)
##################################################################################################
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from crewai import Agent, Task, Crew, Process
from langchain_community.tools import DuckDuckGoSearchRun
import google.generativeai as genai
from dotenv import load_dotenv
import logging
##################################################################################################
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

if google_api_key:
    logger.info("Google API Key loaded successfully.")
else:
    logger.error("Failed to load Google API Key.")
##################################################################################################
llm = ChatGoogleGenerativeAI(
    model="gemini-pro", 
    verbose=True,
    temperature=0.8,
    google_api_key=google_api_key
)
##################################################################################################
tool_search = DuckDuckGoSearchRun()
##################################################################################################
email_author = Agent(
    role='Professional Email Author',
    goal='Craft concise and engaging emails to sell medical products.',
    backstory='Experienced in writing impactful medical sales emails.',
    verbose=True,
    memory=True,
    allow_delegation=False,
    llm=llm,
    tools=[tool_search]
)

marketing_strategist = Agent(
    role='Marketing Strategist',
    goal='Lead the team in creating effective cold emails',
    backstory='A seasoned Chief Marketing Officer with a keen eye for standout medical marketing content.',
    verbose=True,
    memory=True,
    allow_delegation=True,
    llm=llm
)

content_specialist = Agent(
    role='Content Specialist', 
    goal='Critique and refine email content',
    backstory='A professional copywriter with a wealth of experience in persuasive medical writing.',
    verbose=True,
    memory=True,
    allow_delegation=False,
    llm=llm
)
##################################################################################################
mail_task = Task(
    description='''1. Generate two distinct variations of a cold email promoting a Medical testing solution. 
    2. Evaluate the written emails for their effectiveness and engagement.
    3. Scrutinize the emails for grammatical correctness and clarity.
    4. Adjust the emails to align with best practices for cold outreach. Consider the feedback 
    provided to the marketing_strategist.
    5. Revise the emails based on all feedback, creating two final versions.''',
    agent=marketing_strategist,  
    save_output_as_file=True,
    expected_output="Two final versions of the cold email."  # Add this line
)
##################################################################################################
email_crew = Crew(
    agents=[email_author, marketing_strategist, content_specialist],
    tasks=[mail_task],
    verbose=True,
    process=Process.sequential,
    parse_output_as_pydantic=True
)
##################################################################################################
print("Crew: Working on Email Task")
emails_output = email_crew.kickoff()
##################################################################################################
