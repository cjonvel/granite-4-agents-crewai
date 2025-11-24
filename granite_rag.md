
## **Granite Retrieval Agent**

The **Granite Retrieval Agent** is an **Agentic RAG (Retrieval Augmented Generation) system** designed for querying both local documents and web retrieval sources. It uses multi-agent task planning, adaptive execution, and tool calling via **Granite 4 (`ibm/granite4:latest`)**.

### 🔹 Key Features:

* General agentic RAG for document and web retrieval using **Autogen/AG2**.
* Uses **Granite 4 (ibm/granite4:latest)** as the primary language model.
* Integrates with [Open WebUI Functions](https://docs.openwebui.com/features/plugin/functions/) for interaction via a chat UI.
* **Optimized for local execution** (e.g., tested on MacBook Pro M3 Max with 64 GB RAM).

### **Retrieval Agent in Action:**

![The Agent in action](docs/images/GraniteAgentDemo.gif)

### **Architecture:**

![alt text](docs/images/agent_arch.png)

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

## **Granite Retrieval Agent**

| Parameter         | Description                               | Default Value               |
| ----------------- | ----------------------------------------- | --------------------------- |
| task_model_id     | Primary model for task execution          | `ibm/granite4:latest`       |
| vision_model_id   | Vision model for image analysis           | *(set as needed)*           |
| openai_api_url    | API endpoint for OpenAI-style model calls | `http://localhost:11434`    |
| openai_api_key    | API key for authentication                | `ollama`                    |
| vision_api_url    | Endpoint for vision-related tasks         | `http://localhost:11434/v1` |
| model_temperature | Controls response randomness              | `0`                         |
| max_plan_steps    | Maximum steps in agent planning           | `6`                         |


# 🚀 Usage

## **1 Load Documents into Open WebUI**

1. In Open WebUI, navigate to `Workspace` → `Knowledge`.
2. Click `+` to create a new collection.
3. Upload documents for the **Granite Retrieval Agent** to query.


## **Granite Retrieval Agent (AG2-based RAG)**

* Queries **local documents** and **web sources**.
* Performs **multi-agent task planning** and **adaptive execution**.

**Examples**

```text
Study my meeting notes to figure out the technological capabilities of the projects I’m involved in. Then, search the internet for other open-source projects with similar features.
```
