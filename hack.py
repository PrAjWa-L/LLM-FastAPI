import os
import time
import requests
import faiss
import re
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# ✅ Configure Gemini API
GEMINI_API_KEY = "AIzaSyDWiXZueejp1eE61y5FeQf-S4U2EM3U0rQ"
genai.configure(api_key=GEMINI_API_KEY)

# ✅ Initialize Embedding Model & FAISS
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_dim = embed_model.get_sentence_embedding_dimension()

index = faiss.IndexFlatL2(embedding_dim)
job_metadata = []

# ✅ FastAPI Setup with CORS (for frontend integration if needed)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Selenium Setup
def setup_selenium():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

    # ✅ Use a relative path for ChromeDriver
    chrome_driver_path = os.path.join(os.getcwd(), "chromedriver.exe")

    service = Service(chrome_driver_path)
    driver = webdriver.Chrome(service=service, options=chrome_options)

    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
        "source": "Object.defineProperty(navigator, 'webdriver', { get: () => undefined })"
    })
    
    return driver

# ✅ Web Scraping Function
def scrape_jobs_from_naukri(query: str, location: str = "", num_jobs: int = 10):
    driver = setup_selenium()
    
    query_param = query.replace(" ", "+")
    location_param = location.replace(" ", "+")
    search_url = f"https://www.naukri.com/jobs?k={query_param}&l={location_param}"
    
    driver.get(search_url)
    time.sleep(5)

    jobs = []
    wait = WebDriverWait(driver, 20)
    job_cards = driver.find_elements(By.XPATH, "//div[contains(@class, 'jobtuple')]")[:num_jobs]

    for card in job_cards:
        try:
            title = card.find_element(By.XPATH, ".//a[contains(@class, 'title')]").text or "N/A"
            company = card.find_element(By.XPATH, ".//a[contains(@class, 'comp-name')]").text or "N/A"
            location_text = card.find_element(By.XPATH, ".//span[contains(@class, 'loc-wrap')]//span").text or "N/A"
            exp = card.find_element(By.XPATH, ".//span[contains(@class, 'expwdth')]").text or "N/A"
            
            try:
                raw_skills = card.find_element(By.XPATH, ".//ul[contains(@class, 'tags-gt')]").text
                skills = re.sub(r"([a-z])([A-Z])", r"\1 \2", raw_skills)
            except:
                skills = "N/A"

            jobs.append({
                "title": title,
                "company": company,
                "location": location_text,
                "experience": exp,
                "skills": skills
            })

        except Exception as e:
            print(f"Error extracting job details: {e}")
            continue

    driver.quit()
    return jobs

# ✅ Add Jobs to FAISS Index
def add_jobs_to_index(jobs):
    global index, job_metadata

    job_texts = []
    new_metadata = []

    for job in jobs:
        combined = f"Title: {job['title']}. Company: {job['company']}. Location: {job['location']}. Experience: {job['experience']}. Skills: {job['skills']}"
        
        # Prevent duplicate embeddings
        if combined not in job_texts:
            job_texts.append(combined)
            new_metadata.append(job)

    embeddings = embed_model.encode(job_texts, convert_to_numpy=True)
    
    if len(embeddings) > 0:
        index.add(embeddings)
        job_metadata.extend(new_metadata)

# ✅ Retrieve Relevant Jobs
def retrieve_relevant_jobs(query: str, top_k: int = 10):
    query_embedding = embed_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    
    results = [job_metadata[idx] for idx in indices[0] if idx < len(job_metadata)]
    return results

# ✅ Construct LLM Prompt
def construct_prompt(query: str, retrieved_jobs):
    if not retrieved_jobs:
        return f"No relevant job listings found for '{query}'. Try refining your search."

    context = "\n".join([
        f"{i+1}. {job['title']} at {job['company']} ({job['location']}) - Exp: {job['experience']}, Skills: {job['skills']}"
        for i, job in enumerate(retrieved_jobs)
    ])

    prompt = (
        f"Job Search Query: {query}\n\n"
        f"Matching Job Listings:\n{context}\n\n"
        "Generate a professional summary including industry insights, required qualifications, and potential career opportunities."
    )
    
    return prompt

# ✅ Call Gemini LLM
def call_gemini_llm(prompt: str):
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        return response.text if response else "No response from LLM."
    except Exception as e:
        return f"Error generating response from Gemini: {e}"

# ✅ FastAPI Request Model
class QueryRequest(BaseModel):
    query: str
    location: str = ""

# ✅ FastAPI Endpoint
@app.post("/retrieve_jobs")
def retrieve_jobs_endpoint(request: QueryRequest):
    try:
        # Scrape jobs
        scraped_jobs = scrape_jobs_from_naukri(request.query, request.location)
        if not scraped_jobs:
            raise HTTPException(status_code=404, detail="No jobs found.")

        # Add to FAISS
        add_jobs_to_index(scraped_jobs)

        # Retrieve similar jobs
        relevant_jobs = retrieve_relevant_jobs(request.query)

        # Generate LLM response
        prompt = construct_prompt(request.query, relevant_jobs)
        llm_response = call_gemini_llm(prompt)

        return {
            "query": request.query,
            "location": request.location,
            "retrieved_jobs": relevant_jobs,
            "gemini_response": llm_response
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
