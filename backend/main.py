import boto3
import json
from fastapi import FastAPI
from agents import Agent, Runner, function_tool
from agent.prompt import SYSTEM_PROMPT
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()
s3_client = boto3.client("s3")

BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

@function_tool
def get_ndvi_data_from_s3(region: str, year: int):
    s3_key = f"ndvi_results/{region}/{year}.json" # This might be subject to change based on how team implements S3
    try:
        obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=s3_key)
        data = json.loads(obj["Body"].read())
        return data
    except s3_client.exceptions.NoSuchKey:
        return {"error": f"No NDVI data found for {region} {year}"}
    except Exception as e:
        return {"error": str(e)}

@app.get("/overview")
async def get_ai_overview(region: str, year: int):
    instructions = SYSTEM_PROMPT.format(input_region=region, input_year=year)

    ai_overview_agent = Agent(
        name="AI-Overview",
        instructions=instructions,
        tools=[get_ndvi_data_from_s3]
    )

    # Runner.run requires an input string, even if the tool provides real data
    user_input = f"Generate NDVI overview for {region} {year}"
    result = await Runner.run(ai_overview_agent, user_input)

    return {"overview": result.final_output}