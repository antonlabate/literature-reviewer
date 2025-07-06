import os
from dotenv import load_dotenv
import yaml

from crewai import Agent, Task, Crew
from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, List, Union

load_dotenv()
os.environ['OPENAI_MODEL_NAME'] = "gpt-4.1-2025-04-14"#"gpt-4.1-mini-2025-04-14"

class Paper(BaseModel):
    """
    Pydantic model representing a hotel with basic information.
    """
    title: str = Field(description="Paper title", min_length=1)
    summary: str = Field(description="Paper summary", min_length=1)  # Use gt=0 instead of min_length for int
    pdfUrl: Optional[str] = Field(description="The URL to paper PDF", min_length=1)
    authors: List[str] = Field(description="List of authors", min_length=1)
    published: str = Field(description="The date of publication")

    class Config:
        # Enable validation on assignment
        validate_assignment = True

# Define file paths for YAML configurations
files = {
    'agents': 'agents_config/agents.yaml',
    'tasks': 'agents_config/tasks_arxiv.yaml'
}

# Load configurations from YAML files
configs = {}
for config_type, file_path in files.items():
    with open(file_path, 'r') as file:
        configs[config_type] = yaml.safe_load(file)

server_params=StdioServerParameters(
    command="python3",
    args=["servers/arxiv_search_server.py"],
    env={"UV_PYTHON": "3.11", **os.environ},
)

with MCPServerAdapter(server_params) as mcp_tools:
    print(f"Available tools: {[tool.name for tool in mcp_tools]}")
    research_theme = "Fine-tuning LoRA adapters for enhancing model's comprehension of low-resource languages."

    literature_researcher_agent = Agent(
        #config=configs["agents"]["literature_researcher"]
        role = "literature researcher",
        goal = f"Provide a comprehensive and rich list of suggestions of great and diverse articles that are closely related to the user's research theme {research_theme}.",
        backstory= f"""You are a senior Research Scientist with vast experience with article writing and literature review of scientific papers. 
    With a keen eye for detail, you are able to analyze vast lists of papers and hand-pick the best ones,
    that present relevant findings and are particular relevant to your research and project.
    Now, your expertise is asked to help the user by selecting and suggesting the best articles related to his research. """,
        tools = mcp_tools,
        reasoning=True,
        verbose=True
    )

    literature_review_task = Task(
        config=configs["tasks"]["review_literature"],
        agent=literature_researcher_agent
    )

    research_writer_agent = Agent(
        #config=configs["agents"]["literature_researcher"]
        role = "Scientific research writer",
        goal = f"Write a complete, comprehensive and rich 'Related Works' section for your article to be submitted to NeurIPS, respecting the conference style to increase your chances of acceptance.",
        backstory= f"""You are a senior Research Scientist with vast experience with article writing and literature review of scientific papers. 
    You have published several articles in top AI conferences such as Neurips, AAAI and ACL. 
    With a keen eye for detail and great expression capability, you are able to analyze and effectilvely communicate your ideas through your research papers,
    with clear and rich descriptions. You are an expert in synthesizing research literature in great literature reviews. 
    Now, you are writing a paper to submit to NeurIPS. You have already selected the best related works and just have to write the 'Related works' section.""",
        reasoning=True,
        verbose=True
    )

    related_works_writing_task = Task(
        config=configs["tasks"]["related_works_writing"],
        agent=research_writer_agent,
        context=[literature_review_task]
    )

    crew = Crew(
        agents=[literature_researcher_agent, research_writer_agent],
        tasks=[literature_review_task, related_works_writing_task],
        verbose=True
    )

    result = crew.kickoff(inputs={"research_theme": "Fine-tuning LoRA adapters for enhancing model's comprehension of low-resource languages."})
    print(f"Literature review result: {result}.")
