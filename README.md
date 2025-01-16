
# Data Steward Agent 

This repository contains a Python-based system as a starting point for creating a data steward agent for data governance tools

## Features
- Manage governance artifacts 
- Search
- Modular and extensible design for adding new tools and agents.

## Requirements
- Python 3.8+
- OpenAI API Key
- IBM Knowledge catalog API Key and URL

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/ijgitsh/DataStewardAgent_IKC.git
   cd DataStewardAgent_IKC

2- Install the requirements
   ```bash
   pip install -r requirements.txt
   ```
3-Set environment variables
   ```bash
   os.environ["OPENAI_API_KEY"] = ""
   os.environ["GOVERNANCE_API_KEY"] = ""
   os.environ["GOVERNANCE_API_URL"] = ""
   ```
## Run
4- run the script   
   ```bash
    python datastewardagent.py
