import os
import json
import sys
import boto3
import shutil
from UPDAPSACPY import UPCAPSACRunOneFile
import updapsutil as utl


def cleanup_files(path):
    """Remove a file or directory."""
    try:
        if os.path.isdir(path):  # Check if it's a directory
            shutil.rmtree(path)
            print(f"Cleaned up directory: {path}")
        elif os.path.isfile(path):  # Check if it's a file
            os.remove(path)
            print(f"Cleaned up file: {path}")
        else:
            print(f"Path does not exist: {path}")
    except Exception as e:
        print(f"Error cleaning up {path}: {e}")


def lambda_handler(event, context=None):
    print("Starting script...")

    # Parse JSON input
    json_input_path = event.get("JSON_INPUT_PATH")
    json_output_path = event.get("JSON_OUTPUT_PATH")
    climatic_file_name = event.get("CLIMATIC_FILE_NAME")
    climatic_file_s3_folder = event.get("CLIMATIC_FILE_S3_FOLDER")
    username = event.get("USERNAME")

    # Use /tmp for writable operations
    local_climatic_folder = "/tmp/climatic_data"
    local_climatic_file = os.path.join(local_climatic_folder, climatic_file_name)

    jsonfld, jsonfile = os.path.split(json_input_path)

    jsonfilenoext = os.path.splitext(jsonfile)[0]

    StatusTxtFile = jsonfilenoext + '_status.txt'
    ErrorTxtFile = jsonfilenoext + '_error.txt'

    # Delete old run files...
    fld_out, json_output_file = os.path.split(json_output_path)

    utl.delete_file(fld_out, StatusTxtFile)
    utl.delete_file(fld_out, ErrorTxtFile)
    utl.delete_file(fld_out, json_output_file)

    try:

        # Create the local folder for climatic files
        os.makedirs(local_climatic_folder, exist_ok=True)

        # Initialize S3 client
        s3 = boto3.client('s3')

        # Download the JSON input file from S3
        bucket1 = json_input_path.split('/')[2]
        inps = utl.loadjson(jsonfld, jsonfile, bucket1, username)

        # Download the climatic data file from S3
        print(f"Downloading climatic data file from {climatic_file_s3_folder}/{climatic_file_name}")
        bucket2 = climatic_file_s3_folder.split('/')[2]
        key2 = '/'.join(climatic_file_s3_folder.split('/')[3:] + [climatic_file_name])
        filename2 = local_climatic_file
        print(f'Bucket: {bucket2} \nKey: {key2} \nFilename: {filename2}')

        s3.download_file(
            Bucket=bucket2,
            Key=key2,
            Filename=filename2
        )

        # Process the file using UPCAPSACRunOneFile

        result = UPCAPSACRunOneFile(inps, jsonfld, jsonfile, fld_out, json_output_file,
                                    StatusTxtFile, ErrorTxtFile,
                                    local_climatic_file)

        print("Processing completed.")

        # Return the results
        return {
            "statusCode": 200,
            "body": json.dumps(result)
        }

    except Exception as e:
        print(f"Error: {e}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }

    finally:
        # Cleanup all temporary files and folders
        cleanup_files(local_climatic_folder)  # climate directory
        # cleanup_files(local_json_file)  # input file
        # cleanup_files(f"/tmp/{StatusTxtFile}")
        # cleanup_files(f"/tmp/{ErrorTxtFile}")


if __name__ == "__main__":
    payload = {
        "JSON_INPUT_PATH": "s3://meapa2-io/meminkutay/MEAPA_sample_input_1yr.json",
        "JSON_OUTPUT_PATH": "s3://meapa2-io/meminkutay/MEAPA_sample_input_1yr-output.json",
        "CLIMATIC_FILE_NAME": "CFM-118838.csv.gz",
        "CLIMATIC_FILE_S3_FOLDER": "s3://meapaweather/HCM-CFM",
        "USERNAME": "meminkutay"
    }

    # payload = {
    #     "JSON_INPUT_PATH": "s3://meapa2-io/meminkutay/CE431.json",
    #     "JSON_OUTPUT_PATH": "s3://meapa2-io/meminkutay/CE431-output.json",
    #     "CLIMATIC_FILE_NAME": "CFM-118838.csv.gz",
    #     "CLIMATIC_FILE_S3_FOLDER": "s3://meapaweather/HCM-CFM",
    #     "USERNAME": "meminkutay"
    # }

    lambda_handler(payload, None)
