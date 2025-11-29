import json
import os
import uuid
import boto3
import logging
from typing import List, Dict, Any
from botocore.exceptions import BotoCoreError, ClientError
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth

# --- Logging Setup ---
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# --- Configuration ---
# Use environment variables for flexibility
REGION = os.getenv("LEX_REGION", os.environ.get("AWS_REGION", "us-east-1"))
LEX_BOT_ID = os.environ.get("LEX_BOT_ID")
LEX_BOT_ALIAS_ID = os.environ.get("LEX_BOT_ALIAS_ID")
LEX_LOCALE_ID = os.environ.get("LEX_LOCALE_ID", "en_US")
AOSS_INDEX = os.environ.get("AOSS_INDEX", "photos")
AOSS_HOST = os.environ.get("AOSS_HOST")  # e.g., "search-photos-xxxx.us-east-1.aoss.amazonaws.com"

# --- Client Initialization ---
try:
    # 1. Credentials for OpenSearch
    credentials = boto3.Session().get_credentials()
    # Note: Use 'aoss' for Serverless, 'es' for standard OpenSearch
    auth = AWSV4SignerAuth(credentials, REGION, 'aoss')

    # 2. OpenSearch Client
    aoss_client = OpenSearch(
        hosts=[{"host": AOSS_HOST, "port": 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
    )

    # 3. AWS Clients
    lex_client = boto3.client("lexv2-runtime", region_name=REGION)
    s3_client = boto3.client("s3")
    
    logger.info("Clients initialized successfully.")

except Exception as e:
    logger.error(f"Failed to initialize clients: {e}")
    raise e


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Main entry point for the Search Lambda.
    """
    logger.info(f"Received event: {json.dumps(event)}")

    # 1. Extract Query from API Gateway Event
    query_params = event.get("queryStringParameters")
    if not query_params:
        logger.warning("No queryStringParameters found in event.")
        return _response(400, "No query parameters found. Please provide '?q=keywords'")

    text = query_params.get("q")
    if not text:
        logger.warning("Parameter 'q' is missing.")
        return _response(400, "Missing 'q' parameter")

    logger.info(f"Processing search query: '{text}'")

    # 2. Get Keywords from Lex (Disambiguation)
    # We generate a unique session ID for every request to keep Lex context fresh
    session_id = str(uuid.uuid4())
    keywords = get_slot_values(text, session_id)

    # Fallback: If Lex fails to find keywords (e.g., bot not trained well), use raw text
    if not keywords:
        logger.info("Lex found no keywords. Falling back to raw text search.")
        keywords = [text]
    else:
        logger.info(f"Lex extracted keywords: {keywords}")

    # 3. Query OpenSearch
    try:
        image_objs = aoss_query(keywords, limit=10)
        logger.info(f"OpenSearch returned {len(image_objs)} hits.")
    except Exception as e:
        logger.error(f"OpenSearch query failed: {e}", exc_info=True)
        return _response(500, "Internal Search Error")

    # 4. Process Results & Generate Presigned URLs
    results = []
    seen_keys = set()  # Deduplication Set

    for image_obj in image_objs:
        bucket = image_obj.get("bucket")
        key = image_obj.get("objectKey")
        labels = image_obj.get("labels", [])

        # Validation: Ensure bucket and key exist
        if not bucket or not key:
            logger.warning(f"Skipping invalid result object: {image_obj}")
            continue

        # Deduplication: Check if we already processed this photo
        if key in seen_keys:
            logger.debug(f"Duplicate found for key '{key}', skipping.")
            continue
        
        seen_keys.add(key)

        # Generate Presigned URL
        url = generate_presigned_url(
            s3_client,
            "get_object",
            {"Bucket": bucket, "Key": key},
            300  # URL valid for 5 minutes
        )

        if url:
            results.append({
                "url": url,
                "labels": labels
            })

    logger.info(f"Returning {len(results)} unique results to client.")

    # 5. Return Success Response
    return _response(200, {"results": results})


def get_slot_values(text: str, session_id: str) -> List[str]:
    """
    Calls Lex V2 to interpret the user's text and extract keywords (slots).
    """
    try:
        logger.info(f"Calling Lex V2 with session_id: {session_id}")
        lex_resp = lex_client.recognize_text(
            botId=LEX_BOT_ID,
            botAliasId=LEX_BOT_ALIAS_ID,
            localeId=LEX_LOCALE_ID,
            sessionId=session_id,
            text=text,
        )

        # Navigate the deep JSON response safely
        session_state = lex_resp.get("sessionState", {})
        intent = session_state.get("intent", {})
        slots = intent.get("slots", {})

        if not slots:
            logger.info("Lex response contained no slots.")
            return []

        values = []
        for slot_name, slot_data in slots.items():
            if slot_data:
                value_obj = slot_data.get("value", {})
                
                # Logic: Prefer Resolved values (synonyms mapped to master term),
                # then Interpreted values (what Lex thinks it heard),
                # then Original values (what user actually typed).
                if value_obj.get("resolvedValues"):
                    logger.debug(f"Using resolved values for slot '{slot_name}'")
                    values.extend(value_obj["resolvedValues"])
                elif value_obj.get("interpretedValue"):
                    logger.debug(f"Using interpreted value for slot '{slot_name}'")
                    values.append(value_obj["interpretedValue"])
                elif value_obj.get("originalValue"):
                    logger.debug(f"Using original value for slot '{slot_name}'")
                    values.append(value_obj["originalValue"])

        return values

    except ClientError as e:
        logger.error(f"Lex ClientError: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error calling Lex: {e}")
        return []


def aoss_query(keywords: List[str], limit: int) -> List[Dict]:
    """
    Constructs and executes the OpenSearch query.
    """
    # Join keywords with space. 'OR' operator handles the logic.
    search_string = " ".join(keywords)
    logger.info(f"Executing OpenSearch query for: '{search_string}'")

    query = {
        "size": limit,
        "query": {
            "match": {
                "labels": {
                    "query": search_string,
                    "operator": "or"  # Returns photo if ANY keyword matches
                }
            }
        }
    }

    resp = aoss_client.search(index=AOSS_INDEX, body=query)
    
    # Extract the "_source" (the original JSON document we stored) from hits
    hits = resp.get("hits", {}).get("hits", [])
    return [hit["_source"] for hit in hits]


def generate_presigned_url(client, client_method, method_parameters, expires_in):
    """
    Generates a temporary public URL for a private S3 object.
    """
    try:
        url = client.generate_presigned_url(
            ClientMethod=client_method,
            Params=method_parameters,
            ExpiresIn=expires_in
        )
        return url
    except ClientError as e:
        logger.error(f"Failed to generate presigned URL: {e}")
        return None


def _response(code, body_content):
    """
    Helper to format the API Gateway response with CORS headers.
    """
    return {
        "statusCode": code,
        "headers": {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
            "Content-Type": "application/json"
        },
        "body": json.dumps(body_content)
    }