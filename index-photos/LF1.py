import boto3
import uuid
import logging
import json
from typing import List
import datetime as dt
import os
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth

logging = logging.getLogger()

s3_resource = boto3.resource('s3')
s3_client = boto3.client('s3')
rekognition = boto3.client('rekognition')


region = 'us-east-1'
service = 'aoss'
credentials = boto3.Session().get_credentials()
auth = AWSV4SignerAuth(credentials, region, service)
aoss_client = OpenSearch(
    hosts=[{"host": os.getenv("AOSS_HOST"), "port": 443}],
    http_auth=auth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection,
)


index = os.getenv("AOSS_INDEX", "photos")

def lambda_handler(event, context):
    try:
        record = event["Records"][0]
        bucket_name = record["s3"]["bucket"]["name"]
        object_key = record["s3"]["object"]["key"]
        timestamp = record["eventTime"]

        # read image
        s3_object = s3_resource.Object(bucket_name, object_key)
        image_bytes = s3_object.get()['Body'].read()

        # Get custom and rekognition labels for the image
        custom_labels = getCustomLabels(bucket_name, object_key)
        rekognition_labels = getLabels(bucket_name, object_key)

        labels = (custom_labels + rekognition_labels) if custom_labels else rekognition_labels

        logging.info(f"Merged labels: {labels}")

        indexData(bucket_name, object_key, timestamp, labels)

        return {
            "statusCode": 200,
            "body": json.dumps("Indexed successfully")
        }

    except Exception as e:
        logging.error(f"Error processing object: {str(e)}")
        raise
        return {
            "statusCode": 500,
            "body": json.dumps("Indexing failed")
        }


def indexData(bucket: str, key: str, timestamp: str, labels: List[str]):
    """index the image with the labels"""
    # Add a document to the index.
    document = { 
                "objectKey": key, 
                "bucket": bucket, 
                "createdTimestamp": timestamp, 
                "labels": labels,
            }
    try:
        response = aoss_client.index(
            index=index,
            body=document,
            id=str(uuid.uuid1()),
        )
        logging.info(f"Indexed data: {response}")
        return response
    except Exception as e:
        logging.error(f"Error indexing data: {str(e)}")
        raise

def getLabels(bucket: str, key: str):
    """Get labels from Rekognition"""
    try:
        response = rekognition.detect_labels(
            Image={"S3Object": {"Bucket": bucket, "Name": key}, limit=10}
        )
        logging.info(f"Rekognition response: {response}")
        rk_labels = [label["Name"] for label in response["Labels"]]
        return rk_labels
    except Exception as e:
        logging.error(f"Error getting labels: {str(e)}")
        raise

def getCustomLabels(bucket: str, key: str):
    """Get custom labels from S3"""
    try:
        response = s3_client.head_object(
            Bucket=bucket,
            Key=key
        )
        logging.info(f"S3 response: {response}")
        metadata = response.get("Metadata", {})
        custom_raw = metadata.get("customlabels", "")
        custom_labels = [x.strip() for x in custom_raw.split(",")] if custom_raw else []
        return custom_labels
    except Exception as e:
        logging.error(f"Error getting custom labels: {str(e)}")
        raise
