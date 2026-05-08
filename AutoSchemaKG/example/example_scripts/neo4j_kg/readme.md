# Neo4j Knowledge Graph API

This directory contains examples for hosting and querying knowledge graphs stored in Neo4j databases via OpenAI-compatible API endpoints.

## Files

- **`atlas_api_server_demo.py`**: General-purpose API server demo that sets up an OpenAI-compatible endpoint for querying a Neo4j knowledge graph with LLM-powered retrieval.

- **`atlas_api_server_demo_wiki.py`**: API server demo specifically configured for Wikipedia knowledge graph hosted in Neo4j.

- **`atlas_api_server_demo_pes2o.py`**: API server demo specifically configured for PES2O (Papers with Entities, Subjects, and Ontologies) knowledge graph.

- **`atlas_api_server_demo_cc.py`**: API server demo specifically configured for CommonCrawl knowledge graph with customized retrieval settings.

- **`atlas_api_client_demo.py`**: Client example that demonstrates how to query multiple knowledge graph API servers (Wiki, PES2O, CC) using the OpenAI client interface.


## Quick Start

### Server Setup

1. Configure your Neo4j connection and API keys in `config.ini`
2. Start a knowledge graph API server:
```bash
python atlas_api_server_demo_wiki.py    # For Wikipedia KG
# or
python atlas_api_server_demo_cc.py      # For CommonCrawl KG
# or
python atlas_api_server_demo_pes2o.py   # For PES2O KG
```

### Client Usage

Query the running API servers:
```bash
python atlas_api_client_demo.py
```

The client will send queries to all configured knowledge graph servers and return enriched responses.

