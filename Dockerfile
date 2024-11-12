# Use Amazon Linux base image for Lambda
FROM public.ecr.aws/lambda/python:3.8

# Install OS dependencies
RUN yum -y install gcc mysql-devel

# Install required Python packages
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Copy the Lambda function code
COPY lambda_function.py . 

# Define handler for Lambda
CMD ["lambda_function.lambda_handler"]  
