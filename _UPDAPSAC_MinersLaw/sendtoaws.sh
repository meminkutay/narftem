#!/bin/sh

set -e  # Exit immediately if a command exits with a non-zero status

echo "--------- Build the Docker image"
docker build -t meapa2-app .
#docker build --platform linux/amd64 -t meapa2-app .

echo "--------- Tag the image"
docker tag meapa2-app 604671695045.dkr.ecr.us-east-2.amazonaws.com/meapa2-app:latest

echo  "--------- Push to ECR"
aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin 604671695045.dkr.ecr.us-east-2.amazonaws.com
#
#aws ecr create-repository --repository-name meapa2-app

docker push 604671695045.dkr.ecr.us-east-2.amazonaws.com/meapa2-app:latest

echo "--------- Update the Lambda function"
aws lambda update-function-code --function-name meapa2-app-lambda --image-uri 604671695045.dkr.ecr.us-east-2.amazonaws.com/meapa2-app:latest