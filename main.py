import os
import time
import logging
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ValidationError
from google import genai
from google.genai import types


# ==========================================
# LOAD ENVIRONMENT VARIABLES
# ==========================================

load_dotenv()


# ==========================================
# FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="Week 7 LLM Structured Output API"
)


# ==========================================
# UI CONFIGURATION
# ==========================================

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)

templates = Jinja2Templates(
    directory="templates"
)


# ==========================================
# GEMINI CLIENT
# ==========================================

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options=types.HttpOptions(
        timeout=30000
    )
)


# ==========================================
# LOGGING
# ==========================================

logging.basicConfig(
    filename="llm_usage.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)


# ==========================================
# REQUEST / RESPONSE SCHEMAS
# ==========================================

class ExtractionRequest(BaseModel):
    text: str


class CandidateProfile(BaseModel):
    name: str
    role: str
    years_of_experience: int
    skills: List[str]


# ==========================================
# USAGE TRACKING
# ==========================================

total_requests = 0
total_input_tokens = 0
total_output_tokens = 0
total_cost = 0.0


# Gemini 3.8 Flash pricing
INPUT_COST_PER_MILLION = 0.75
OUTPUT_COST_PER_MILLION = 3.75


# ==========================================
# FUNCTION DECLARATION
# ==========================================

extract_candidate_function = types.FunctionDeclaration(
    name="extract_candidate",
    description=(
        "Extract candidate information from unstructured text."
    ),
    parameters_json_schema={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The candidate's full name."
            },
            "role": {
                "type": "string",
                "description": "The candidate's current or stated professional role."
            },
            "years_of_experience": {
                "type": "integer",
                "description": "The candidate's years of professional experience."
            },
            "skills": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "A list of skills supported by the text."
            }
        },
        "required": [
            "name",
            "role",
            "years_of_experience",
            "skills"
        ]
    }
)


extract_candidate_tool = types.Tool(
    function_declarations=[
        extract_candidate_function
    ]
)


# ==========================================
# HOME / UI
# ==========================================

@app.get(
    "/",
    response_class=HTMLResponse
)
def home(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html"
    )


# ==========================================
# EXTRACTION ENDPOINT
# ==========================================

@app.post(
    "/extract",
    response_model=CandidateProfile
)
def extract_candidate(request: ExtractionRequest):

    global total_requests
    global total_input_tokens
    global total_output_tokens
    global total_cost

    start_time = time.perf_counter()

    system_prompt = """
You are a candidate information extraction assistant.

Your task is to extract candidate information
from the provided text.

Use the extract_candidate function.

Rules:
- Extract only information supported by the text.
- Do not invent information.
- Return the candidate's full name.
- Return their current or stated professional role.
- Return their years of experience as a whole number.
- Return their skills as a list.
"""

    user_prompt = f"""
Extract the candidate information from this text:

{request.text}
"""

    response = None
    result = None
    last_error = None

    # ======================================
    # TRY TWICE
    # ======================================

    for attempt in range(2):

        try:

            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    tools=[extract_candidate_tool],
                    tool_config=types.ToolConfig(
                        function_calling_config=types.FunctionCallingConfig(
                            mode="ANY",
                            allowed_function_names=[
                                "extract_candidate"
                            ]
                        )
                    )
                )
            )

            # ------------------------------
            # TOKEN USAGE
            # ------------------------------

            usage = response.usage_metadata

            input_tokens = (
                usage.prompt_token_count or 0
            )

            output_tokens = (
                usage.candidates_token_count or 0
            )

            total_tokens = (
                usage.total_token_count or 0
            )


            # ------------------------------
            # REQUIRE FUNCTION CALL
            # ------------------------------

            function_calls = response.function_calls

            if not function_calls:

                raise ValueError(
                    "The LLM did not return the expected function call."
                )


            function_call = function_calls[0]


            if function_call.name != "extract_candidate":

                raise ValueError(
                    "Unexpected function returned by the LLM."
                )


            # ------------------------------
            # VALIDATE FUNCTION ARGUMENTS
            # ------------------------------

            try:

                result = CandidateProfile.model_validate(
                    function_call.args
                )

            except ValidationError as error:

                raise ValueError(
                    f"Malformed function arguments: {error}"
                )


            # Successful extraction
            break


        except Exception as error:

            last_error = error

            logging.warning(
                f"LLM attempt {attempt + 1} failed: {error}"
            )

            # Retry once
            if attempt == 0:
                time.sleep(2)


    # ======================================
    # BOTH ATTEMPTS FAILED
    # ======================================

    if result is None:

        latency = (
            time.perf_counter() - start_time
        )

        logging.error(
            f"LLM request failed after retry. "
            f"Latency={latency:.2f}s "
            f"Error={last_error}"
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "The LLM service is temporarily unavailable "
                "or returned an invalid response. "
                "Please try again."
            )
        )


    # ======================================
    # COST CALCULATION
    # ======================================

    input_cost = (
        input_tokens / 1_000_000
    ) * INPUT_COST_PER_MILLION

    output_cost = (
        output_tokens / 1_000_000
    ) * OUTPUT_COST_PER_MILLION

    request_cost = (
        input_cost + output_cost
    )


    # ======================================
    # LATENCY
    # ======================================

    latency = (
        time.perf_counter() - start_time
    )


    # ======================================
    # UPDATE METRICS
    # ======================================

    total_requests += 1

    total_input_tokens += input_tokens

    total_output_tokens += output_tokens

    total_cost += request_cost


    estimated_cost_per_100 = (
        total_cost / total_requests
    ) * 100


    # ======================================
    # LOG SUCCESSFUL REQUEST
    # ======================================

    logging.info(
        f"Request={total_requests} "
        f"InputTokens={input_tokens} "
        f"OutputTokens={output_tokens} "
        f"TotalTokens={total_tokens} "
        f"Latency={latency:.2f}s "
        f"Cost=${request_cost:.8f} "
        f"EstimatedCostPer100="
        f"${estimated_cost_per_100:.6f}"
    )


    return result


# ==========================================
# METRICS ENDPOINT
# ==========================================

@app.get("/metrics")
def get_metrics():

    if total_requests == 0:

        estimated_cost_per_100 = 0

    else:

        estimated_cost_per_100 = (
            total_cost / total_requests
        ) * 100


    return {
        "total_requests": total_requests,

        "total_input_tokens":
            total_input_tokens,

        "total_output_tokens":
            total_output_tokens,

        "total_tokens":
            (
                total_input_tokens
                + total_output_tokens
            ),

        "estimated_total_cost_usd":
            round(total_cost, 8),

        "estimated_cost_per_100_requests_usd":
            round(
                estimated_cost_per_100,
                6
            )
    }