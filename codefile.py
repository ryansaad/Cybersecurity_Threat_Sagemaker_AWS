# Load & Explore the Dataset in SageMaker
# Note : change the S3 bucket name to your named S3 bucket 

import boto3
import pandas as pd
import io

# Setup S3 client
#****************************************************
s3_client = boto3.client('s3')

# Download the file into memory
response = s3_client.get_object(Bucket='cybersecurity-ml-data1', Key='raw-data/UNSW_NB15_training-set.csv')

# Read it into pandas
df = pd.read_csv(io.BytesIO(response['Body'].read()))

# Explore
print(df.shape)
print(df.columns)
print(df.head())
print(df['label'].value_counts())

# Clean and Normalize the Data
#**************************************************
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 1. Drop columns that aren't useful for training
df = df.drop(columns=['id', 'attack_cat'])

# 2. Handle categorical columns using one-hot encoding
categorical_cols = ['proto', 'service', 'state']
df = pd.get_dummies(df, columns=categorical_cols)

# 3. Normalize numerical columns
# First, identify numerical columns (excluding the label)
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_cols.remove('label')

scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

print(df.shape)
print(df.columns)
print(df.head())
print(df['label'].value_counts()) 



# Feature Engineering 
#******************************************************************
# Add feature: byte_ratio
df['byte_ratio'] = df['sbytes'] / (df['dbytes'] + 1)

# Add feature: is_common_port (based on ct_dst_sport_ltm which can hint at dest port)
df['is_common_port'] = df['ct_dst_sport_ltm'].isin([80, 443, 22]).astype(int)

# Add feature: flow_intensity = (spkts + dpkts) / dur
df['flow_intensity'] = (df['spkts'] + df['dpkts']) / (df['dur'] + 1e-6)


 # Save the Preprocessed Data to S3

import sagemaker
from sagemaker import get_execution_role

# Create SageMaker session and define bucket
#**************************************************************************
session = sagemaker.Session()
bucket = 'cybersecurity-ml-data1'  # Replace with your actual S3 bucket name
processed_prefix = 'processed-data'      # Folder in S3 to store processed files

# Save preprocessed data locally
df.to_csv('preprocessed_data.csv', index=False)

# Upload to S3 inside the 'processed-data/' folder
s3_path = session.upload_data(
    path='preprocessed_data.csv',
    bucket=bucket,
    key_prefix=processed_prefix
)

print(f"Preprocessed data uploaded to: {s3_path}")



#Load Preprocessed Data from S3
#***************************************************************
import pandas as pd
import boto3
import sagemaker

# Set up session and bucket
session = sagemaker.Session()
bucket = 'cybersecurity-ml-data1'
processed_prefix = 'processed-data'

# Download preprocessed data from S3
s3 = boto3.client('s3')
file_name = 'preprocessed_data.csv'
s3.download_file(bucket, f'{processed_prefix}/{file_name}', file_name)

# Load into pandas
df = pd.read_csv(file_name)
df.head()

#Split Data into Train/Test Sets

from sklearn.model_selection import train_test_split
from sklearn.datasets import dump_svmlight_file
import pandas as pd

# Load data
df = pd.read_csv('preprocessed_data.csv')

X = df.drop(columns=['label'])
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# CSV for inspection
train_df = pd.concat([y_train, X_train], axis=1)
test_df = pd.concat([y_test, X_test], axis=1)
train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)

# LIBSVM for SageMaker
dump_svmlight_file(X_train, y_train, 'train.libsvm', zero_based=True)
dump_svmlight_file(X_test, y_test, 'test.libsvm', zero_based=True)


#Upload Training and Test Data to S3
#*****************************************************************************

import sagemaker

session = sagemaker.Session()
bucket = 'cybersecurity-ml-data1'
train_prefix = 'xgboost-data/train'
test_prefix = 'xgboost-data/test'

train_input = session.upload_data('train.libsvm', bucket=bucket, key_prefix=train_prefix)
test_input = session.upload_data('test.libsvm', bucket=bucket, key_prefix=test_prefix)

print(f"Training data: {train_input}")
print(f"Testing data: {test_input}")


#Set Up the XGBoost Training Job
#*****************************************************************************888

from sagemaker import image_uris
from sagemaker.estimator import Estimator

xgboost_image_uri = image_uris.retrieve("xgboost", region=session.boto_region_name, version="1.3-1")

xgb = Estimator(
    image_uri=xgboost_image_uri,
    role=sagemaker.get_execution_role(),
    instance_count=1,
    instance_type='ml.m5.large',
    output_path=f's3://cybersecurity-ml-data1/xgboost-model-output',
    sagemaker_session=session
)

xgb.set_hyperparameters(
    objective='binary:logistic',
    num_round=100,
    max_depth=5,
    eta=0.2,
    gamma=4,
    min_child_weight=6,
    subsample=0.8,
    verbosity=1
)

#Train the Model
#*********************************************************************
# Train using data channels
xgb.fit({'train': train_input, 'validation': test_input})

Evaluate Model Performance
pip install xgboost
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report

# Load and convert data
train_data = pd.read_csv('train.csv', header=None, dtype=str)
test_data = pd.read_csv('test.csv', header=None, dtype=str)

# Convert all columns to numeric
train_data = train_data.apply(pd.to_numeric, errors='coerce')
test_data = test_data.apply(pd.to_numeric, errors='coerce')

# Drop any rows with NaNs
train_data = train_data.dropna()
test_data = test_data.dropna()

# Split into features (X) and labels (y)
X_train = train_data.iloc[:, 1:]
y_train = train_data.iloc[:, 0]
X_test = test_data.iloc[:, 1:]
y_test = test_data.iloc[:, 0]

# Convert to DMatrix format
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test)

# Set parameters and train the model
params = {
    "objective": "binary:logistic",
    "max_depth": 5,
    "eta": 0.2,
    "gamma": 4,
    "min_child_weight": 6,
    "subsample": 0.8,
    "verbosity": 1
}

model = xgb.train(params=params, dtrain=dtrain, num_boost_round=100)

# Predict
y_pred_prob = model.predict(dtest)
y_pred = [1 if p > 0.5 else 0 for p in y_pred_prob]

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


# Create a SageMaker Model from the Trained Model Artifact
#********************************************************************************
import boto3
from sagemaker import image_uris

sagemaker_client = boto3.client("sagemaker")
region = "us-east-1"
bucket_name = "cybersecurity-ml-data1"
model_artifact = f"s3://cybersecurity-ml-data1/xgboost-model-output/sagemaker-xgboost-2025-05-21-08-48-39-127/output/model.tar.gz"
model_name = "cybersecurity-threat-xgboost"

# Get XGBoost image URI
image_uri = image_uris.retrieve("xgboost", region=region, version="1.3-1")

# Use actual IAM Role ARN
execution_role = "arn:aws:iam::252366102382:role/SageMakerCybersecurityRole"

# Register the model
response = sagemaker_client.create_model(
    ModelName=model_name,
    PrimaryContainer={
        "Image": image_uri,
        "ModelDataUrl": model_artifact
    },
    ExecutionRoleArn=execution_role
)

print(f"Model {model_name} registered successfully in SageMaker!")

#Deploy the Model as a SageMaker Endpoint
#************************************************************

# Define model name if not already defined
model_name = "cybersecurity-threat-xgboost"

# Define endpoint configuration
endpoint_config_name = "cybersecurity-threat-config"

sagemaker_client.create_endpoint_config(
    EndpointConfigName=endpoint_config_name,
    ProductionVariants=[
        {
            "VariantName": "DefaultVariant",
            "ModelName": model_name,
            "InstanceType": "ml.m5.large",
            "InitialInstanceCount": 1,
            "InitialVariantWeight": 1
        }
    ]
)

# Deploy endpoint
endpoint_name = "cybersecurity-threat-endpoint"

sagemaker_client.create_endpoint(
    EndpointName=endpoint_name,
    EndpointConfigName=endpoint_config_name
)

print(f"Endpoint '{endpoint_name}' is being deployed. This may take a few minutes...")

#Test the Deployed Endpoint

import boto3
import numpy as np

runtime_client = boto3.client("sagemaker-runtime")

# Sample input in CSV format
sample_input = "0.5,0.3,0.8,0.2,0.1,0.6,0.9,0.4"

# Invoke the endpoint
response = runtime_client.invoke_endpoint(
    EndpointName="cybersecurity-threat-endpoint",  # or use endpoint_name if defined
    ContentType="text/csv",
    Body=sample_input
)

# Get prediction from response
result = response["Body"].read().decode("utf-8")
prediction_score = float(result.strip())

# Interpret prediction
predicted_label = "THREAT" if prediction_score > 0.5 else "SAFE"

print(f"Prediction: {predicted_label}")


#Create a SageMaker Pipeline Definition
#********************************************************************************************

# SageMaker Pipeline
import boto3
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep
from sagemaker.workflow.parameters import ParameterString
from sagemaker.inputs import TrainingInput
from sagemaker import image_uris

# Setup
session = sagemaker.Session()
role = sagemaker.get_execution_role()
bucket = 'cybersecurity-ml-data1'

# Parameters
training_instance_type = ParameterString(
    name="TrainingInstanceType", 
    default_value="ml.m5.large"
)

# Use built-in XGBoost (same as your step 3.3)
xgb_estimator = sagemaker.estimator.Estimator(
    image_uri=image_uris.retrieve("xgboost", session.boto_region_name, version="1.3-1"),
    role=role,
    instance_count=1,
    instance_type=training_instance_type,
    output_path=f's3://{bucket}/pipeline-model-output/',
    hyperparameters={
        'objective': 'binary:logistic',
        'num_round': 100,
        'max_depth': 5,
        'eta': 0.2,
        'gamma': 4,
        'min_child_weight': 6,
        'subsample': 0.8,
        'verbosity': 1
    }
)

# Training Step
step_train = TrainingStep(
    name="TrainCybersecurityModel",
    estimator=xgb_estimator,
    inputs={
        "train": TrainingInput(
            s3_data=f's3://{bucket}/xgboost-data/train/train.libsvm',
            content_type="text/libsvm"
        ),
        "validation": TrainingInput(
            s3_data=f's3://{bucket}/xgboost-data/test/test.libsvm',
            content_type="text/libsvm"
        )
    }
)

# Create Pipeline
pipeline = Pipeline(
    name="simple-cybersecurity-pipeline",
    parameters=[training_instance_type],
    steps=[step_train],
    sagemaker_session=session,
)

# Run Pipeline
def run_pipeline():
    pipeline.upsert(role_arn=role)
    print("Pipeline created successfully!")
    
    execution = pipeline.start()
    print(f"Pipeline execution started: {execution.arn}")
    return execution

print("Automated pipeline ready! Run: execution = run_pipeline()")

#Trigger the Pipeline Execution
execution = run_pipeline()
print("SageMaker Pipeline Execution Started!")

status = execution.describe()['PipelineExecutionStatus']
print(f"Pipeline Status: {status}")



#Automate Retraining with AWS EventBridge
#**********************************************************************************************
import json
import boto3

def lambda_handler(event, context):
# Initialize SageMaker client
    sagemaker_client = boto3.client("sagemaker")

    try:
# Start your pipeline
        response = sagemaker_client.start_pipeline_execution(
            PipelineName="simple-cybersecurity-pipeline"# Your pipeline name
        )

        print(f"Pipeline started: {response['PipelineExecutionArn']}")

        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": "Pipeline execution started successfully",
                "executionArn": response['PipelineExecutionArn']
            })
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps(f"Error starting pipeline: {str(e)}")
        }




#Test the Setup
#*****************************************************************************
# Test automation
import boto3

s3_client = boto3.client('s3')

# Upload a test file to trigger the pipeline
test_content = "This is test data for pipeline automation"
s3_client.put_object(
    Bucket='cybersecurity-ml-data1',
    Key='new-data/test-trigger.txt',
    Body=test_content
)

print("Test file uploaded! Check Lambda logs to see if pipeline triggered.")
