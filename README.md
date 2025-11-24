# Granite Retrieval and Image Research Agents


## Lab introdction

During this lab we will develop 2 agents that are using IBM Granite 4 to plan and execute:
- Granite Retrieval Agent
- Image Resarch Agent  

Here are the technologies/frameworks we will use for development and execution:
- python as programming language
- [CrewAI](https://www.crewai.com/?utm_source=ibm_developer&utm_content=in_content_link&utm_id=tutorials_awb-build-agentic-rag-system-granite) as the agentic framework
- [Open WebUI](https://docs.openwebui.com/category/-web-search?utm_source=ibm_developer&utm_content=in_content_link&utm_id=tutorials_awb-build-agentic-rag-system-granite) as a local workbench to load and what with the agents
- an optional Web Search engine in Open WebUI, thanks to Open WebUI you can configure the engine of your choice (Tavily, SearXNG,  )
- Granite 4 and llama models running on watsonx.ai on IBM cloud

## Why Granite 4 for these agents?

Granite 4 introduces a **hybrid Mamba-2/Transformer** architecture (with MoE variants) that targets **lower memory use and faster inference**, making it a strong fit for agentic RAG and function-calling workflows. It uses **>70% lower memory** and **~2× faster inference** vs. comparable models, which helps these agents run locally or on modest GPUs with lower cost and latency. Models are **Apache-2.0 licensed**, **ISO 42001 certified**, and cryptographically signed for governance and security. 

**Tiny-H (7B total / ~1B active)** is optimized for **low-latency, small-footprint deployments**—ideal for the Image Researcher’s quick tool calls and orchestration steps. The family emphasizes **instruction following, tool calling, RAG, JSON output, multilingual dialog, and code (incl. FIM)**, aligning with both agents’ needs.

---




# 🔑 Prepare your environment

* **Common Installation Instructions**: The setup for **Open WebUI** remains the same for both agents.

## **1. Set up your python environment**                    

You need to have a python environment ready and an IDE to work easily!

## **2. Install Open WebUI**

```bash
pip install open-webui
open-webui serve
```

## **3. Optional: Set Up Web Search in Open WebUI**

* **Flexible Web Search**: Agents use the Open WebUI search API, integrating the search engine of your choice by following the  [Configuration guide](https://docs.openwebui.com/category/web-search).
 
 


# 📚 Agents development

Here are the links for each agent tutorial, you can run in any order you want. Use the link in the latest column to open the tutorial.

| Feature                 | Description                                                          | Models Used                                | Code Link                                                                  |  Link                                                                                                                                                                         |
| ----------------------- | -------------------------------------------------------------------- | ------------------------------------------ | -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Granite Retrieval Agent | General Agentic RAG for document and web retrieval using Autogen/AG2 | **Granite-4.0-H-Small (ibm-granite/granite-4.0-h-small:latest)** [Granite 4 h small on HuggingFace](https://huggingface.co/ibm-granite/granite-4.0-h-small)        | [granite_autogen_rag.py](./granite_autogen_rag.py)                         | [Build a multi-agent RAG system with Granite locally](./granite_rag.md)                                                      |
| Image Research Agent    | Image-based multi-agent research using CrewAI with Granite Vision    | **Granite-4.0-H-Small (ibm-granite/granite-4.0-h-small:latest)** [Granite 4 h small on HuggingFace](https://huggingface.co/ibm-granite/granite-4.0-h-small)   | [image_researcher_granite_crewai.py](./image_researcher_granite_crewai.py) | [Build an AI research agent for image analysis with Granite 3.2 Reasoning and Vision models](image_researcher.md) |

---




## **6. Configure Web Search in Open WebUI**

To set up a search provider (e.g., SearXNG), follow [this guide](https://docs.openwebui.com/tutorials/web-search/searxng#4-gui-configuration).

---

 