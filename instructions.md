# Step-by-Step Project Instructions

Follow this guide to build, deploy, and automate the Cybersecurity Threat Detection System on AWS.
Note : for Jupyter Notebook code check the code_file
---

## 1. Preprocess Data and Feature Engineering

### Step 1.1: Create an IAM Role for SageMaker
1.  Navigate to the **IAM Console** in AWS.
2.  Go to **Roles** -> **Create role**.
3.  **Trusted entity type**: Select **AWS service**.
4.  **Use case**: Choose **SageMaker**.
5.  Attach the following managed policies: `AmazonSageMakerFullAccess` and `AmazonS3FullAccess`.
6.  Name the role `SageMakerCybersecurityRole` and create it.

### Step 1.2: Set Up SageMaker Notebook Instance
1.  Go to the **Amazon SageMaker** console.
2.  Navigate to **Notebook** -> **Notebook instances** -> **Create notebook instance**.
3.  **Notebook instance name**: `cybersecurity-notebook`
4.  **Instance type**: `ml.t2.medium` (This is eligible for the AWS Free Tier).
5.  **IAM role**: Select the `SageMakerCybersecurityRole` you created.
6.  Click **Create notebook instance**. Wait for the status to become **InService**, then click **Open Jupyter**.

### Step 1.3: Download and Prepare the Dataset
1.  Download the public UNSW-NB15 dataset from [this link](https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/UNSW_NB15_training-set.csv).
2.  In the **S3 Console**, create a new bucket (e.g., `cybersecurity-ml-data-yourname`). make sure its unique.
3.  Inside the bucket, create a folder named `raw-data`.
4.  Upload the `UNSW_NB15_training-set.csv` file to the `raw-data/` folder.

### Step 1.4: Preprocess in a Jupyter Notebook
1.  In your Jupyter environment, create a new notebook (`data_preprocessing.ipynb`).
2.  **Load the data** from S3 into a pandas DataFrame.
3.  **Clean and Normalize**:
    * Drop unhelpful columns (`id`, `attack_cat`).
    * One-hot encode categorical columns (`proto`, `service`, `state`).
    * Scale numerical features using `sklearn.preprocessing.StandardScaler`.
4.  **Feature Engineering**:
    * Create new features like `byte_ratio` and `flow_intensity` to help the model identify threats.
5.  **Save the Processed Data**:
    * Save the final, cleaned DataFrame to a new CSV file (`preprocessed_data.csv`).
    * Upload this file to your S3 bucket under a new folder named `processed-data/`.

## 2. Training and Testing the Model

### Step 2.1: Prepare Data for XGBoost
1.  In your notebook, load the `preprocessed_data.csv` file from the previous step.
2.  Split the data into training (80%) and testing (20%) sets.
3.  Convert the datasets to the **LIBSVM** format required by SageMaker's XGBoost algorithm. The label must be the first column.
4.  Upload `train.libsvm` and `test.libsvm` to a new folder in your S3 bucket named `xgboost-data/`.

### Step 2.2: Configure and Run the SageMaker Training Job
1.  Define a SageMaker `Estimator` for the built-in XGBoost algorithm.
2.  Specify the official XGBoost container image URI.
3.  Set the instance type (`ml.m5.large`), IAM role, and S3 output path for the model artifacts.
4.  Define the model's hyperparameters (`objective='binary:logistic'`, `num_round=100`, etc.).
5.  Start the training job by calling the `.fit()` method, pointing to your training and validation data in S3. SageMaker will automatically provision infrastructure and run the job.

### Step 2.3: Evaluate the Model
1.  Once training is complete, install the `xgboost` library in your notebook (`!pip install xgboost`).
2.  Retrain the model locally within the notebook using the same parameters to get an accessible model object.
3.  Use the model to make predictions on your test set.
4.  Calculate the **accuracy** and print a **classification report** to evaluate performance.

---

## 3. Deploy and Serve the Model

### Step 3.1: Create a SageMaker Model
1.  Register the trained model artifact from S3 as a formal SageMaker Model object. You will need the S3 path to the `model.tar.gz` file created by the training job and the ARN of your `SageMakerCybersecurityRole`.

### Step 3.2: Deploy the Model to an Endpoint
1.  Define an **Endpoint Configuration**, specifying the instance type (e.g., `ml.m5.large`) for hosting the model.
2.  Use the configuration to **Create an Endpoint**. SageMaker will deploy your model as a real-time, scalable API. This may take several minutes.

### Step 3.3: Test the Endpoint
1.  Use the `sagemaker-runtime` client in your notebook to invoke the endpoint.
2.  Send a sample payload in CSV format representing a single network event.
3.  Receive the prediction score from the endpoint and interpret it (e.g., > 0.5 is a "THREAT").

---

## 4. Automating with SageMaker Pipelines

### Step 4.1: Define the Pipeline
1.  Using the SageMaker Python SDK, define a `Pipeline` object.
2.  Create a `TrainingStep` that uses the XGBoost estimator you configured earlier. This makes the training process a reusable step.
3.  Upsert and start the pipeline manually from the notebook to ensure it works correctly.

### Step 4.2: Create an Automated Trigger
1.  **Create a Lambda Function**:
    * Create a simple Lambda function (`trigger-cybersecurity-pipeline`) that uses `boto3` to start an execution of your SageMaker Pipeline.
    * Grant this Lambda function permissions to start SageMaker pipelines by attaching the `AmazonSageMakerFullAccess` policy to its execution role.
2.  **Create an EventBridge Rule**:
    * Set up a rule that listens for new objects being created in a specific S3 prefix (e.g., `new-data/`).
    * Set the **target** of this rule to be the Lambda function you just created.

### Step 4.3: Test the Automation
1.  Ensure your S3 bucket has event notifications enabled to communicate with EventBridge.
2.  Upload a dummy file to the `new-data/` folder in your S3 bucket.
3.  This action will trigger the entire chain: S3 -> EventBridge -> Lambda -> SageMaker Pipeline.
4.  Verify a new pipeline execution has started in the SageMaker console.

---

## 5. Project Clean-Up

To avoid ongoing AWS costs, delete all created resources in the following order:
1.  **SageMaker Endpoint, Model, and Endpoint Configuration**.
2.  **SageMaker Notebook Instance** (Stop it first).
3.  **S3 Bucket** (Empty it first, then delete).
4.  **CloudWatch Log Groups** associated with SageMaker.
5.  **Lambda Function** and **EventBridge Rule**.
6.  **IAM Role** (`SageMakerCybersecurityRole`).
