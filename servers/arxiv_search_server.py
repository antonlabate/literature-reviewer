import arxiv
import json
import os

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, List, Union

from mcp.server.fastmcp import FastMCP

import io
import requests
from PyPDF2 import PdfReader


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

# Initialize FastMCP server
mcp = FastMCP("research_arxiv")

@mcp.tool()
def search_papers(topic: str, max_results: int = 5) -> List[str]:
    """
    Search for papers on arXiv based on a topic and store their information.
    
    Args:
        topic: The topic to search for
        max_results: Maximum number of results to retrieve (default: 5)
        
    Returns:
        String of papers found in the search. Each paper has title, authors, summary, link and publication date
    """
    
    # Use arxiv to find the papers 
    client = arxiv.Client()

    # Search for the most relevant articles matching the queried topic
    search = arxiv.Search(
        query = topic,
        max_results = max_results,
        sort_by = arxiv.SortCriterion.Relevance
    )

    papers_found = client.results(search)

    # Process each paper and add to papers_info  
    papers = []
    for paper in papers_found:

        paper_info = Paper(
            title=paper.title,
            authors= [author.name for author in paper.authors],
            summary=paper.summary,
            pdfUrl=paper.pdf_url,
            published=str(paper.published.date())
        )
        papers.append(paper_info)
    
    papers = "\n".join([paper.model_dump_json(serialize_as_any=True, indent=4) for paper in papers])

    return papers

@mcp.tool()
def read_pdf(url: str) -> List[str]:
    """
    Reads the PDF associated to the provided URL.
    
    Args:
        url: The URL of the PDF to be read
        
    Returns:
        The text of PDF's first page. This can be used to get a sense and a brief summary of what is the PDF content.
    """
    if(url[-1]=="\\"):
        url = url[:-2]
    response = requests.get(url=url)
    on_fly_mem_obj = io.BytesIO(response.content)

    try:
        pdf_file = PdfReader(on_fly_mem_obj)
        page = pdf_file.pages[0].extract_text()
    
        return page
    
    except:
        return "Could not open PDF file."


if __name__ == "__main__":
    mcp.run(transport='stdio') 