# Week 7 - LLM Structured Output

A FastAPI application that uses Gemini function calling to extract structured candidate information from unstructured text.

## Features

- LLM-powered candidate information extraction
- Structured output using function calling
- Pydantic validation
- Retry and graceful error handling
- Token usage tracking
- Latency monitoring
- Estimated cost tracking
- Simple web UI for testing the extraction flow

## Technologies

- Python
- FastAPI
- Gemini API
- Pydantic
- Jinja2
- HTML/CSS/JavaScript

## API Endpoints

- `GET /` - Web interface
- `POST /extract` - Extract candidate information
- `GET /metrics` - View LLM usage and cost metrics
- `GET /docs` - FastAPI Swagger documentation

## Example Input

```text
Jane Doe is a Product Designer with 4 years of experience.
She is skilled in Figma, UX research and prototyping.