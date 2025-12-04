## **Granite Retrieval Agent (AG2-based RAG)**

The **Granite Retrieval Agent (AG2-based RAG)** is an **Agentic RAG (Retrieval Augmented Generation) system** designed for querying both local documents and web retrieval sources. It uses multi-agent task planning, adaptive execution, and tool calling via **Granite 4 (`ibm/granite4:latest`)**.

- Queries **local documents** and **web sources**.
- Performs **multi-agent task planning** and **adaptive execution**.

Artificial intelligence (AI) agents are generative AI (gen AI) systems or programs capable of autonomously designing and executing task workflows using available tools. Can you build agentic workflows without needing extremely large, costly large language models (LLMs)? **The answer is yes.** In this tutorial, we will demonstrate how to build a multi agent RAG system locally.

## Agentic RAG overview

Retrieval-Augmented Generation (RAG) is an effective way of providing an LLM with additional datasets from various data sources without the need for expensive fine-tuning. Similarly, agentic RAG leverages an AI agent’s ability to plan and execute subtasks along with the retrieval of relevant information to supplement an LLM's knowledge base. This allows for the optimization and greater scalability of RAG applications.

The future of agentic RAG is multi agent RAG, where several specialized agents collaborate to achieve optimal latency and efficiency. We will demonstrate this using a small, efficient model like Granite 3.2 and combine it with a modular agent architecture. We will use multiple specialized "mini agents" that collaborate to achieve tasks through adaptive planning and tool calling. Like humans, a team of agents, or a multi agent system, often outperforms the heroic efforts of an individual, especially when they have clearly defined roles and effective communication.

## Multi agent architecture: When collaboration beats competition

Our Granite Retrieval Agent relies on a modular architecture in which each agent has a specialized role. Much like humans, agents perform best when they have targeted instructions and just enough context to make an informed decision. Too much extraneous information, such as an unfiltered chat history, can create a “needle in the haystack” problem, where it becomes increasingly difficult to decipher signal from noise.

In this agentic AI architecture, the agents work together sequentially to achieve the goal. Here is how the generative AI system is organized:

![system_schema](images/g1.png)

- **Planner Agent**: Creates the initial high-level plan, once, in the beginning of the workflow. For example, if a user asks, “What are comparable open source projects to the ones my team is using?” then the agent will put together a step-by-step plan that may look something like this: “1. Search team documents for open source technologies. 2. Search the web for similar open source projects to the ones found in step 1.” If any of these steps fail or provide insufficient results, the steps can be later adapted by the Reflection Agent.

- **Research Assistant**: The Research Assistant is the workhorse of the system. It takes in and executes instructions such as, “Search team documents for open source technologies.” For step 1 of the plan, it uses the initial instruction from the Planner Agent. For subsequent steps, it also receives curated context from the outcomes of previous steps. For example, if tasked with “Search the web for similar open source projects,” it will also receive the output from the previous document search step. Depending on the instruction, the Research Assistant can use tools like web search or document search, or both, to fulfill its task.

- **Summarizer Agent**: The Summarizer Agent condenses the Research Assistant’s findings into a concise, relevant response. For example, if the Research Assistant finds detailed meeting notes stating, “We discussed the release of Tool X that uses Tool Y underneath,” then the Summarizer Agent extracts only the relevant snippets such as, "Tool Y is being used," and reformulates it to directly answer the original instruction. This may seem like a small detail, but it can help give higher quality results and keep the model on task, especially as one step builds upon the output of another step.

- **Critic Agent**: The Critic Agent is responsible for deciding whether the output of the previous step satisfactorily fulfilled the instruction it was given. It receives two pieces of information: the single step instruction that was just executed and the output of that instruction from Summarizer Agent. Having a Critic Agent weigh in on the conversation brings clarity around whether the goal was achieved, which is needed for the planning of the next step.

- **Reflection Agent**: The reflection agent is our executive decision maker. It decides what step to take next, whether that is encroaching onto the next planned step, pivoting course to make up for mishaps or giving the thumbs up that the goal has been completed. Much like a real-life CEO, it performs its best decision making when it has a clear goal in mind and is presented with concise findings on the progress that has or has not been made to reach that goal. The output of the Reflection Agent is either the next step to take or the instructions to terminate if the goal has been reached. We present the Reflection Agent with the following items:

  - The goal.
  - The original plan.
  - The last step executed.
  - The output of the Summarizer and Critic Agents from the last step.
  - A concise sequence of previously executed instructions (just the instructions, not their output).
  - Presenting these items in a structured format makes it clear to our decision maker what has been done so that it can decide what needs to happen next.

- **Report Generator**: Once the goal is achieved, the Report Generator synthesizes all findings into a cohesive output that directly answers the original query. While each step in the process generates targeted outputs, the Report Generator ties everything together into a final report.

## Leveraging open source tools

For beginners, it can be difficult to build an agentic AI application from scratch. Hence, we will use a set of open source tools.

The following architecture diagram illustrates how the Granite Retrieval Agent integrates multiple tools for agentic RAG.

![alt text](images/g2-bis.png)

- **Open WebUI**: The user interacts with the system through an intuitive chat interface hosted in Open WebUI. This interface acts as the primary point for submitting queries (such as “Fetch me the latest news articles pertaining to my project notes”) and viewing the outputs.

- **Python-based** agent (AG2 Framework): At the core of the system is a Python-based agent built using AutoGen (AG2). This agent coordinates the workflow by breaking down tasks and dynamically calling tools to execute steps. The agent has access to two primary tools:

  - **Document search tool**: Fetches relevant information from a vector database containing uploaded project notes or documents stored as embeddings. This vector search leverages the built-in documental retrieval APIs inside of Open WebUI, rather than setting up an entirely separate data store.
  - **Web search tool**: Performs web-based searches to gather external knowledge and real-time information. In this case, we are using SearXNG as our metasearch engine.

- **Optional Ollama**: You can download IBM Granite 4 H Tiny LLM that will serve as the language model powering the system. It is hosted locally using Ollama, ensuring fast inference, cost efficiency and data privacy.

- **watsonx.ai on IBM Cloud**: thanks to the api key that the instructor will share with you, use Granite 4 H Small to test the power of a enterprise grade agentic LLM, either through openai REST API or watsonx.ai REST api

---

## **Develop the agent**

1. Open the [granite_autogen_rag.py file](granite_autogen_rag.py) in your IDE.

2. Read carefully the section **Assistant prompts** at the beginning of the file to understand the system prompt for each agent.

3. Have a look to the **Pipe** class definition and variables. Those are variables that you will be able to set values in Open WebUI directly and allow you the input your api key or change model name.
   Default is setup for **ollama** as follow:

```py
class Pipe:
    class Valves(BaseModel):
        TASK_MODEL_ID: str = Field(default="ibm/granite4:latest")
        VISION_MODEL_ID: str = Field(default="granite3.2-vision:2b")
        OPENAI_API_URL: str = Field(default="http://localhost:11434")
        OPENAI_API_KEY: str = Field(default="ollama")
        VISION_API_URL: str = Field(default="http://localhost:11434")
        MODEL_TEMPERATURE: float = Field(default=0)
        MAX_PLAN_STEPS: int = Field(default=6)
```

If you want to use **watsonx.ai** rather than ollam, change the values to:

```py
class Pipe:
    class Valves(BaseModel):
        TASK_MODEL_ID: str = Field(default="ibm/granite-4-h-small")
        VISION_MODEL_ID: str = Field(default="meta-llama/llama-3-2-11b-vision-instruct")
        OPENAI_API_URL: str = Field(default="https://ca-tor.ml.cloud.ibm.com/ml/gateway/v1")
        OPENAI_API_KEY: str = Field(default="")
        VISION_API_URL: str = Field(default="https://ca-tor.ml.cloud.ibm.com/ml/gateway/v1")
        MODEL_TEMPERATURE: float = Field(default=0)
        MAX_PLAN_STEPS: int = Field(default=6)
```

4. Review class definition for `Plan`, `CriticDecision`, `Step` and `SearchQueries`. Those classes define the output format of each agent in our architecture.

5. Edit the `base_llm_config` object. The default for **ollama** is:

```py
        base_llm_config = {
            "model": default_model,
            "client_host": base_url,
            "api_type": "ollama",
            "temperature": model_temp,
        }
```

Change it if needed for **watsonx.ai** to:

```py
        base_llm_config = {
            "model": default_model,
            "base_url": base_url,
            "api_type" : "openai",
            "api_key": api_key,
            "temperature": model_temp
        }
```

6. Review the `llm_configs` object (same for ollama and watsonx.ai):

```py
        llm_configs = {
            "ollama_llm_config": {**base_llm_config, "config_list": [{**base_llm_config}]},
            "planner_llm_config": {**base_llm_config, "config_list": [{**base_llm_config, "response_format": Plan}]},
            "critic_llm_config": {**base_llm_config, "config_list": [{**base_llm_config, "response_format": CriticDecision}]},
            "reflection_llm_config": {**base_llm_config, "config_list": [{**base_llm_config, "response_format": Step}]},
            "search_query_llm_config": {**base_llm_config, "config_list": [{**base_llm_config, "response_format": SearchQueries}]},
            "vision_llm_config": {
                "config_list": [
                    {
                        "model": vision_model,
                        "base_url": vision_url,
                        "api_type": "openai",
                        "api_key": api_key
                    }
                ]
            },
```

Each agent will use the base llm for its execution, but each one has a different format for the output, as per the class defined in step 4.

7. In the next section of the file, we will define our agents. See how llm configuration and prompt are associated for each one. Each agent is of type [autogen.ConversableAgent](https://docs.ag2.ai/latest/docs/api-reference/autogen/ConversableAgent/)

```py
### Agents
        # Generic LLM completion, used for servicing Open WebUI originated requests
        generic_assistant = ConversableAgent(
            name="Generic_Assistant",
            llm_config=llm_configs["ollama_llm_config"],
            human_input_mode="NEVER",
        )

        # Vision Assistant
        vision_assistant = ConversableAgent(
            name="Vision_Assistant",
            llm_config=llm_configs["vision_llm_config"],
            human_input_mode="NEVER",
        )

        # Provides the initial high level plan
        planner = ConversableAgent(
            name="Planner",
            system_message=PLANNER_MESSAGE,
            llm_config=llm_configs["planner_llm_config"],
            human_input_mode="NEVER",
        )
```

8. Next, review the `Tool Definitions` section, for example:

   ```py
       @assistant.register_for_llm(
               name="personal_knowledge_search",
               description="Searches personal documents according to a given query",
           )
   ```

   We are defining 2 tools that our agents can use:

   - _web_search_: use the configured search engine
   - _personal_knowledge_search_: allow to search in documents provided by the user as local knowledge (collection of documents)

9. Review `Begin Agentic Workflow` section:

- starts with the input sentence from the user in the chat
- if images are detected in the input, use vision_assistant to describe the image
- create the plan using the `user_proxy` to chat with the assistant
- loop for each step with tjhe expected loop research -> step critic -> goal critic -> reflection
- when no more steps are planned exit and compile the answer

## **Retrieval Agent in Action:**

Review this short video as an example execution.

![The Agent in action](docs/images/GraniteAgentDemo.gif)

## **Test your agent in **

### **1. Import the Agent Python Script into Open WebUI**

1. Open `http://localhost:8080/` and log into Open WebUI.
2. Admin panel → **Functions** → **+ New Function** to add.
3. Name it `Granite RAG Agent` and add a description to your function, eg "Queries local documents and web sources to perform the task asked by the user with plan, critic and reflect steps"
4. Paste the relevant Python script:
   - `granite_autogen_rag.py` (Retrieval Agent)
5. **Save** and **Confirm** the import
6. **Enable** the function by clicking the toggle button
7. Adjust settings by clicking the **wheel** ⚙️ icon:

   | Parameter         | Description                               | Default Value                                                               |
   | ----------------- | ----------------------------------------- | --------------------------------------------------------------------------- |
   | task_model_id     | Primary model for task execution          | `ibm/granite4:latest` or `ibm/granite-4-h-small`                            |
   | vision_model_id   | Vision model for image analysis           | `granite3.2-vision:2b` or `meta-llama/llama-3-2-11b-vision-instruct`        |
   | openai_api_url    | API endpoint for OpenAI-style model calls | `http://localhost:11434` or `https://ca-tor.ml.cloud.ibm.com/ml/gateway/v1` |
   | openai_api_key    | API key for authentication                | `ollama` or IBM cloud api key                                               |
   | vision_api_url    | Endpoint for vision-related tasks         | `http://localhost:11434` or `https://ca-tor.ml.cloud.ibm.com/ml/gateway/v1` |
   | model_temperature | Controls response randomness              | `0`                                                                         |
   | max_plan_steps    | Maximum steps in agent planning           | `6`                                                                         |

Ensure all settings are ok.

If you are using watsonx.ai on IBM Cloud, copy the api key shared by the instructors and click the **Open Api Key** entry to add your key:
![alt text](images/g3-valves.png)

### **2 Load Documents into Open WebUI**

As an example, we will use a pdf document detailing [IBM Granite 4.0: A Hybrid LLM for Healthcare AI](https://intuitionlabs.ai/pdfs/ibm-granite-4-0-a-hybrid-llm-for-healthcare-ai.pdf) as the knowledge source. Download the file and save it to your computer. We will then use it in Open WebUI.

1. In Open WebUI, navigate to `Workspace` → `Knowledge`.
2. Click `+ New Knowledge` to create a new collection.
3. Give a **name** and **description** of this collection, and click **Create knowledge**
4. Click the **+** button on th right to add content and upload documents for the **Granite Retrieval Agent** to query.

### **3 Test in the chat **

1. In Open WebUI, navigate to the home page.
2. Select your `Granite RAG Agent` in the list.
3. Click the + icon and select `Attach knowledge`, select your created collection
4. Type your query

Wait for the answer. Review the logs in the terminal where you start your Open WebUI server, to see what's happening!

![alt text](images/g4-response.png)

Feel free to deep dive to learn more on granite 4!

![alt text](images/g5-follow.png)

**Examples**

```text
Study my meeting notes to figure out the technological capabilities of the projects I’m involved in. Then, search the internet for other open-source projects with similar features.
```
