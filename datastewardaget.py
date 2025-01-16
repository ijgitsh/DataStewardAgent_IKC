import os
import requests
from openai import OpenAI


from crewai import Agent, Task, Crew, Process
from crewai_tools import tool

# Set environment variables
os.environ["OPENAI_API_KEY"] = ""
os.environ["GOVERNANCE_API_KEY"] = ""
os.environ["GOVERNANCE_API_URL"] = ""

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOVERNANCE_API_KEY = os.getenv("GOVERNANCE_API_KEY")
GOVERNANCE_API_URL = os.getenv("GOVERNANCE_API_URL")

client = OpenAI(api_key=OPENAI_API_KEY)


@tool
def execute_lucene_query(query: str, is_simple: bool = True, limit: int = 100, role: str = "viewer") -> str:
    """
    Execute a Lucene syntax query to search for assets and artifacts.

    Args:
        query (str): The search query in Lucene syntax.
        is_simple (bool): Whether to use simple query syntax. Default is True.
        limit (int): The maximum number of results to return. Default is 100.
        role (str): Role access for governance artifacts. Default is "viewer".

    Returns:
        str: JSON response containing search results or an error message.
    """
    headers = {
        "Authorization": f"Bearer {GOVERNANCE_API_KEY}",
        "Run-as-Tenant": "999",
        "Content-Type": "application/json",
    }

    params = {
        "query": query,
        "isSimple": is_simple,
        "limit": limit,
        "role": role,
        "auth_scope": "all"
    }

    response = requests.get(f"{GOVERNANCE_API_URL}/v3/search", headers=headers, params=params)

    if response.status_code == 200:
        return f"Search results:\n{response.json()}"
    else:
        return f"Error: {response.status_code} - {response.text}"


@tool
def create_category(category_name: str) -> str:
    """
    Create a category in the governance system.

    Args:
        api_url (str): The base URL for the governance API.
        api_key (str): The API key for authentication.
        category_name (str): The name of the category to create.

    Returns:
        str: Success message or error message.
    """
    headers = {"Authorization": f"Bearer {GOVERNANCE_API_KEY}", "Content-Type": "application/json"}
    payload = {"name": category_name}
    response = requests.post(f"{GOVERNANCE_API_URL}/v3/categories", json=payload, headers=headers)
    return f"Category '{category_name}' created successfully." if response.status_code == 201 else f"Error: {response.text}"


@tool
def delete_category(category_id: str) -> str:
    """
    Delete a category from the governance system.

    Args:
        api_url (str): The base URL for the governance API.
        api_key (str): The API key for authentication.
        category_id (str): The ID of the category to delete.

    Returns:
        str: Success message or error message.
    """
    headers = {"Authorization": f"Bearer {GOVERNANCE_API_KEY}"}
    response = requests.delete(f"{GOVERNANCE_API_URL}/v3/categories/{category_id}", headers=headers)
    return "Category deleted successfully." if response.status_code == 204 else f"Error: {response.text}"

'''
@tool
def find_category(category_id: str) -> str:
    """
    Find a category by name or ID.

    Args:
        api_url (str): The base URL for the governance API.
        api_key (str): The API key for authentication.
        query (str): The search query (e.g., category name).

    Returns:
        str: List of matching categories or error message.
    """
    headers = {"Authorization": f"Bearer {GOVERNANCE_API_KEY}"}
    response = requests.get(f"{GOVERNANCE_API_URL}/v3/categories/{category_id}", headers=headers)
    if response.status_code == 200:
        categories = response.json()
        return f"Found categories: {categories}" if categories else "No categories found."
    return f"Error: {response.text}"



@tool
def import_categories(api_url: str, api_key: str, file_path: str) -> str:
    headers = {"Authorization": f"Bearer {api_key}"}
    with open(file_path, "rb") as file:
        files = {"file": file}
        response = requests.post(f"{api_url}/categories/import", files=files, headers=headers)
    return "Categories imported successfully." if response.status_code == 200 else f"Error: {response.text}"


@tool
def export_categories(api_url: str, api_key: str, export_params: dict) -> str:
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.get(f"{api_url}/categories/export", params=export_params, headers=headers)
    if response.status_code == 200:
        file_path = "exported_categories.json"
        with open(file_path, "wb") as file:
            file.write(response.content)
        return f"Categories exported successfully to {file_path}."
    return f"Error: {response.text}"


# Tools for Business Terms
@tool
def create_business_term(api_url: str, api_key: str, term_name: str, description: str) -> str:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"name": term_name, "description": description, "type": "business_term"}
    response = requests.post(f"{api_url}/business_terms", json=payload, headers=headers)
    return f"Business term '{term_name}' created successfully." if response.status_code == 201 else f"Error: {response.text}"


@tool
def delete_business_term(api_url: str, api_key: str, term_id: str) -> str:
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.delete(f"{api_url}/business_terms/{term_id}", headers=headers)
    return "Business term deleted successfully." if response.status_code == 204 else f"Error: {response.text}"


@tool
def find_business_term(api_url: str, api_key: str, query: str) -> str:
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.get(f"{api_url}/business_terms?search={query}", headers=headers)
    if response.status_code == 200:
        terms = response.json()
        return f"Found business terms: {terms}" if terms else "No business terms found."
    return f"Error: {response.text}"


@tool
def import_business_terms(api_url: str, api_key: str, file_path: str) -> str:
    headers = {"Authorization": f"Bearer {api_key}"}
    with open(file_path, "rb") as file:
        files = {"file": file}
        response = requests.post(f"{api_url}/business_terms/import", files=files, headers=headers)
    return "Business terms imported successfully." if response.status_code == 200 else f"Error: {response.text}"


@tool
def export_business_terms(api_url: str, api_key: str, export_params: dict) -> str:
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.get(f"{api_url}/business_terms/export", params=export_params, headers=headers)
    if response.status_code == 200:
        file_path = "exported_business_terms.json"
        with open(file_path, "wb") as file:
            file.write(response.content)
        return f"Business terms exported successfully to {file_path}."
    return f"Error: {response.text}"
'''

# Agents
category_manager = Agent(
    role="Category Manager",
    goal="Manage categories, including creation, deletion, and organization.",
    backstory="An expert in handling category hierarchies and ensuring proper organization.",
    verbose=True,
    tools=[create_category, delete_category],
)

business_term_manager = Agent(
    role="Business Term Manager",
    goal="Manage business terms, including creation, deletion, and retrieval.",
    backstory="An experienced manager of business terms, focusing on governance and compliance.",
    verbose=True,
    tools=[],
)

lucene_search_agent = Agent(
    role="Search Manager",
    goal="Execute complex queries to search for governance artifacts.",
    backstory="A powerful agent capable of searching across large datasets with sophisticated query capabilities.",
    verbose=True,
    tools=[execute_lucene_query],
)


# Send User Input to LLM for Interpretation
def infer_actions_with_llm(user_input: str) -> dict:
    """
    Interpret the user input using LLM and return inferred actions.
    """
    prompt = f"""
    Interpret the following input and infer the actions required to manage governance artifacts (categories, business terms, etc.).
    Output a JSON structure describing the artifact type and required actions.

    Input: "{user_input}"

    Example Output:
    {{
        "artifact_type": "business_term",
        "actions": [
            {{
                "type": "create",
                "name": "Confidentiality",
                "description": "This term defines data privacy and confidentiality."
            }},
            {{
                "type": "delete",
                "id": "12345"
            }}
        ]
    }}

    or 

    {{
        "artifact_type":"category",
        "actions":[
        {{
             {{"type": "search",
             "query": ""query": {{"match": {{"Finance" }}}}"
             }}
        }}
        ]
    }}
    """

     
    try:
        response = client.chat.completions.create(model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for structuring user inputs."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1500,
        temperature=0.7)
        # Parse and return the structured response
        return eval(response.choices[0].message.content.strip())
    except Exception as e:
        raise ValueError(f"Error in LLM inference: {e}")

'''
# Generate Tasks from Actions
def generate_tasks_from_actions(actions: list, artifact_type: str) -> list:
    """
    Create tasks dynamically based on inferred actions.
    """
    tasks = []
    for action in actions:
        if artifact_type == "category":
            # Map category actions to tools
            if action["type"] == "create":
                task = Task(
                    description=f"Create category '{action['name']}'.",
                    expected_output="Confirmation of category creation by returning the json from the request",
                    tools=[create_category],
                    agent=category_manager,
                )
            elif action["type"] == "delete":
                task = Task(
                    description=f"Delete category with ID '{action['id']}'.",
                    expected_output="Confirmation of category deletion.",
                    tools=[delete_category],
                    agent=category_manager,
                )
            tasks.append(task)

        elif artifact_type == "business_term":
            # Map business term actions to tools
            if action["type"] == "create":
                task = Task(
                    description=f"Create business term '{action['name']}' with description '{action['description']}'.",
                    expected_output="Confirmation of business term creation.",
                    tools=[create_business_term],
                    agent=business_term_manager,
                )
            elif action["type"] == "delete":
                task = Task(
                    description=f"Delete business term with ID '{action['id']}'.",
                    expected_output="Confirmation of business term deletion.",
                    tools=[delete_business_term],
                    agent=business_term_manager,
                )
            tasks.append(task)
    return tasks
'''
def generate_tasks_from_actions(actions: list, artifact_type: str) -> list:
    """
    Create tasks dynamically based on inferred actions.
    """
    tasks = []
    for action in actions:
        if artifact_type == "category":
            # Map category actions to tools
            if action["type"] == "create":
                task = Task(
                    description=f"Create category '{action['name']}'.",
                    expected_output="Confirmation of category creation by returning the JSON from the request.",
                    tools=[create_category],
                    agent=category_manager,
                )
            elif action["type"] == "delete":
                task = Task(
                    description=f"Delete category with ID '{action['id']}'.",
                    expected_output="Confirmation of category deletion.",
                    tools=[delete_category],
                    agent=category_manager,
                )
            elif action["type"] == "search":
                task = Task(
                    description=f"Search for categories with query '{action['query']}'.",
                    expected_output="Search results containing matching categories.",
                    tools=[execute_lucene_query],
                    agent=category_manager,
                )
            tasks.append(task)

        elif artifact_type == "business_term":
            # Map business term actions to tools
            if action["type"] == "create":
                task = Task(
                    description=f"Create business term '{action['name']}' with description '{action['description']}'.",
                    expected_output="Confirmation of business term creation.",
                    tools=[create_business_term],
                    agent=business_term_manager,
                )
            elif action["type"] == "delete":
                task = Task(
                    description=f"Delete business term with ID '{action['id']}'.",
                    expected_output="Confirmation of business term deletion.",
                    tools=[delete_business_term],
                    agent=business_term_manager,
                )
            elif action["type"] == "search":
                task = Task(
                    description=f"Search for business terms with query '{action['query']}'.",
                    expected_output="Search results containing matching business terms.",
                    tools=[execute_lucene_query],
                    agent=business_term_manager,
                )
            tasks.append(task)

        elif artifact_type == "search":
            # General search action (for all artifact types)
            task = Task(
                description=f"Perform a search with query '{action['query']}'.",
                expected_output="Search results containing matching artifacts.",
                tools=[execute_lucene_query],
                agent=business_term_manager if "business_term" in action.get("artifact_type", "") else category_manager,
            )
            tasks.append(task)

    return tasks


# Main Function
def main():
    user_inputs = [
        "Create 3 categories : finance, IT and HR",
        "view information about a finance category "
       ]

    for user_input in user_inputs:
        print(f"Processing request: {user_input}")

        # Step 1: Interpret input
        inferred_data = infer_actions_with_llm(user_input)
        artifact_type = inferred_data.get("artifact_type")
        actions = inferred_data.get("actions", [])
        print(actions)

        # Step 2: Generate tasks
        tasks = generate_tasks_from_actions(actions, artifact_type)

        # Step 3: Create and kick off the crew
        crew = Crew(
            agents=[category_manager,business_term_manager,lucene_search_agent],
            tasks=tasks,
            process=Process.sequential,
        )
        result = crew.kickoff(inputs={})
        print(f"Execution Result:\n{result}")


if __name__ == "__main__":
    main()
