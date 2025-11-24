
## **Image Research Agent**

The **Image Research Agent** analyzes images and performs multi-agent research on image components using **Granite 4 Tiny-H (`ibm/granite4:tiny-h`)** with the **CrewAI** framework.

### 🔹 Key Features:

* **Image-based multi-agent research** using CrewAI.
* **Granite 4 Tiny-H** powers low-latency orchestration and tool calls; pair with a vision backend of your choice.
* Identifies objects, retrieves related research articles, and provides historical backgrounds.
* Demonstrates a **different agentic workflow** from the Retrieval Agent.

### **Image Researcher in Action:**

![alt-text](docs/images/image_explainer_example_1.png)

### **Architecture:**

![alt text](docs/images/image_explainer_agent.png)

---



## **4. Import the Agent Python Script into Open WebUI**

1. Open `http://localhost:8080/` and log into Open WebUI.
2. Admin panel → **Functions** → **+** to add.
3. Name it (e.g., “Granite RAG Agent” or “Image Research Agent”).
4. Paste the relevant Python script:

   * `granite_autogen_rag.py` (Retrieval Agent)
   * `image_researcher_granite_crewai.py` (Image Research Agent)
5. Save and enable the function.
6. Adjust settings (inference endpoint, search API, **model ID**) via the gear icon.

⚠️ If you see OpenTelemetry errors while importing `image_researcher_granite_crewai.py`, see [this issue](https://github.com/ibm-granite-community/granite-retrieval-agent/issues/25).




# ⚙️ Configuration Parameters

## **Image Research Agent**

| Parameter                | Description                               | Default Value            |
| ------------------------ | ----------------------------------------- | ------------------------ |
| task_model_id            | Primary model for task execution          | `ibm/granite4:tiny-h`    |
| vision_model_id          | Vision model for image analysis           | *(set as needed)*        |
| openai_api_url           | API endpoint for OpenAI-style model calls | `http://localhost:11434` |
| openai_api_key           | API key for authentication                | `ollama`                 |
| vision_api_url           | Endpoint for vision-related tasks         | `http://localhost:11434` |
| model_temperature        | Controls response randomness              | `0`                      |
| max_research_categories  | Number of categories to research          | `4`                      |
| max_research_iterations  | Iterations for refining research results  | `6`                      |
| include_knowledge_search | Option to include knowledge base search   | `False`                  |
| run_parallel_tasks       | Run tasks concurrently                    | `False`                  |

# 🚀 Usage

## **Image Research Agent**

* Upload an image to initiate research.
* Prompt with specifics to refine focus.