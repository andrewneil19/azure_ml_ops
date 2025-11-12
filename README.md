# Telecom Customer Churn Prediction - ML Engineering Project

Machine learning engineering project focused on model development, experiment tracking, and deployment for customer churn prediction using Azure Machine Learning and MLflow.

> **Project Context:** This was completed as part of a collaborative MLOps project. This repository showcases my individual contributions to the machine learning engineering and deployment components. Additional pipeline automation and application development were completed by my project partner.

## 📊 Project Overview

This project demonstrates core ML engineering skills through building, optimizing, and deploying a machine learning model for predicting telecom customer churn. My work focused on the data management, model training, experiment tracking, and deployment phases of the ML lifecycle.

**Business Problem:** Predict which telecom customers are likely to churn to enable proactive retention strategies.

## 🎯 My Contributions

### Data Management & Versioning
- **Multi-Version Data Assets:** Created and managed two versions of telecom churn dataset in Azure ML
  - Version 1: 2,192 customer records for baseline
  - Version 2: 3,333 customer records for expanded training
- **Data Quality Validation:** Implemented checks for missing values, duplicates, and class imbalance
- **Feature Engineering:** Designed `ServiceCallsPerWeek` metric as customer satisfaction proxy

### Model Development & Optimization
- **Initial Model Training:** Built Random Forest Classifier with MLflow experiment tracking
  - Baseline performance: 88.5% accuracy, 80.0% recall, 66.9% F1 score
- **Hyperparameter Optimization:** Iteratively tuned model parameters
  - Increased n_estimators: 75 → 150 trees
  - Increased max_depth: 5 → 10 levels
  - Optimized performance: 92.7% accuracy, 72.7% F1 score (+4.2% accuracy improvement)
- **Model Evaluation:** Logged comprehensive metrics including confusion matrices and classification reports

### Experiment Tracking
- **MLflow Integration:** Comprehensive parameter and metric logging
  - Logged: hyperparameters, performance metrics, confusion matrices
  - Tracked: data quality metrics, feature engineering details, train/test split info
- **Model Registry:** Registered both model versions with metadata for version control
- **Reproducibility:** Documented random seeds, environment specs, and training configurations

### Model Deployment
- **Scoring Script Development:** Created inference script with proper input validation
- **Azure ML Endpoint:** Deployed model to endpoint 
- **A/B Testing Setup:** Configured traffic routing (70% optimized model, 30% baseline)
- **API Testing:** Validated endpoint with sample predictions and documented request/response format

## Technologies Used

| Category | Technologies |
|----------|-------------|
| **Cloud Platform** | Microsoft Azure Machine Learning |
| **ML Framework** | scikit-learn (Random Forest Classifier) |
| **Experiment Tracking** | MLflow, Azure ML Experiments |
| **Deployment** | Azure ML Managed Endpoints |
| **Languages** | Python 3.10 |
| **Key Libraries** | pandas, numpy, scikit-learn, joblib, azureml-sdk, azureml-mlflow |
| **Tools** | Jupyter Notebooks, Azure ML Studio |

**Note:** This repository contains only the components I developed and documented (Steps 1-4, 7 of the original project). The complete group project included additional components (pipeline automation, Flask API, Streamlit UI) developed by my project partner.

### Data Setup

The project uses the telecom churn dataset (V1 and V2 versions):
- **Version 1:** 2,192 customer records
- **Version 2:** 3,333 customer records (expanded dataset)

Datasets are managed as Azure ML Data Assets with version tracking.

## Model Development

### Feature Engineering

Custom feature created for improved prediction:
```python
# ServiceCallsPerWeek: Customer satisfaction proxy
df['ServiceCallsPerWeek'] = df['CustServCalls'] / df['AccountWeeks']
```

### Model Architecture

**Random Forest Classifier** - Selected for:
- Handling mixed data types (binary, discrete, continuous)
- Built-in feature importance
- Robustness to outliers
- Class imbalance handling via `class_weight='balanced'`

### Initial Model (Version 1)
```python
RandomForestClassifier(
    n_estimators=75,
    max_depth=5,
    class_weight='balanced',
    random_state=12345
)
```
**Performance:** Accuracy: 88.5%, Recall: 80.0%, Precision: 57.4%

### Optimized Model (Version 2)
```python
RandomForestClassifier(
    n_estimators=150,  # Increased ensemble diversity
    max_depth=10,      # Deeper trees for complex patterns
    class_weight='balanced',
    random_state=12345
)
```
**Performance:** Accuracy: 92.7%, Recall: 66.9%, Precision: 79.5%

## 📈 My ML Engineering Workflow

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│   Data      │────>│  Feature     │────>│   Model     │────>│  Experiment  │
│  Versioning │     │ Engineering  │     │  Training   │     │   Tracking   │
└─────────────┘     └──────────────┘     └─────────────┘     └──────────────┘
                                                                      │
                                                                      ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│   A/B       │<────│  Deployment  │<────│   Model     │<────│    Model     │
│  Testing    │     │  (REST API)  │     │  Registry   │     │  Evaluation  │
└─────────────┘     └──────────────┘     └─────────────┘     └──────────────┘
```

This workflow represents the components I developed:
1. **Data Versioning** - Created V1 and V2 data assets in Azure ML
2. **Feature Engineering** - Added ServiceCallsPerWeek metric
3. **Model Training** - Developed and optimized Random Forest classifier  
4. **Experiment Tracking** - Logged all parameters and metrics with MLflow
5. **Model Evaluation** - Analyzed performance across multiple metrics
6. **Model Registry** - Registered both model versions
7. **Deployment** - Created scoring script and deployed to Azure ML endpoint
8. **A/B Testing** - Configured traffic routing between model versions

## Model Deployment

### Azure ML Managed Endpoint

**Endpoint Configuration:**
- **Compute:** Managed compute instance (Standard_F2s_v2)
- **Authentication:** Key-based authentication
- **Traffic Split:** 
  - Deployment 1 (Optimized Model): 70%
  - Deployment 2 (Baseline Model): 30%

### REST API Usage

**Request Format:**
```json
{
  "data": [
    {
      "AccountWeeks": 128,
      "ContractRenewal": 1,
      "DataPlan": 1,
      "DataUsage": 2.7,
      "CustServCalls": 1,
      "DayMins": 265.1,
      "DayCalls": 110,
      "MonthlyCharge": 89.0,
      "OverageFee": 9.87,
      "RoamMins": 10.0,
      "ServiceCallsPerWeek": 5.0
    }
  ]
}
```

**Response Format:**
```json
{
  "predictedOutcomes": [0],  // 0 = No Churn, 1 = Churn
  "inputFeatures": { ... }
}
```

## 📊 Experiment Tracking with MLflow

**Logged Parameters:**
- Model hyperparameters (n_estimators, max_depth, class_weight)
- Feature engineering details
- Train/test split ratio

**Logged Metrics:**
- Accuracy, Precision, Recall, F1 Score
- Confusion Matrix
- Class distribution (churn rate: 85.5%, no-churn rate: 14.5%)
- Data quality metrics (null count, duplicate count)

**Logged Artifacts:**
- Trained model (pickle format)
- Confusion matrix visualization
- Training/test datasets

## 🎛️ A/B Testing Configuration

I configured the deployment for A/B testing to enable model version comparison:

<img width="485" height="164" alt="image" src="https://github.com/user-attachments/assets/c43ea31e-d206-448a-8908-8965cf44a1d6" />


**Purpose:**
- Validate optimized model performance with real traffic
- Compare V1 vs V2 model predictions in production
- Enable safe rollback if needed
- Provide data for performance analysis

## 📊 Model Performance - Confusion Matrix

### Confusion Matrix (Test Set)

|  | Predicted: No Churn | Predicted: Churn |
|--|---------------------|------------------|
| **Actual: No Churn** | 769 | 86 |
| **Actual: Churn** | 29 | 116 |

### Model Improvements

**V1 → V2 Optimization:**
- Accuracy: +4.2 percentage points (88.5% → 92.7%)
- F1 Score: +5.8 percentage points (66.9% → 72.7%)
- Precision: +22.0 percentage points (57.4% → 79.5%)

## 💡 Key Technical Achievements

### ML Engineering Skills Demonstrated
- ✅ **Data Versioning:** Multi-version dataset management in Azure ML with quality validation
- ✅ **Feature Engineering:** Created domain-relevant features for improved predictions
- ✅ **Experiment Tracking:** Comprehensive MLflow logging of parameters, metrics, and artifacts
- ✅ **Model Optimization:** Iterative hyperparameter tuning with measurable improvements (+4.2% accuracy)
- ✅ **Model Registry:** Version control with automated registration and metadata
- ✅ **Cloud Deployment:** Azure ML managed endpoint with authentication
- ✅ **A/B Testing:** Traffic routing configuration for model comparison
- ✅ **Reproducibility:** Documented environments, random seeds, and configurations

### Production-Ready Practices
- Proper train/test splits with stratification for imbalanced data
- Confusion matrix analysis beyond simple accuracy metrics
- Scoring script with input validation and error handling
- Environment specification for dependency management
- Comprehensive documentation of modeling decisions

## 🔗 Additional Project Components

The complete group project included additional MLOps infrastructure components:

### Completed by Team
- **Automated Pipeline Orchestration:** End-to-end pipeline using Azure ML `mldesigner` library for modular, reusable workflows
- **REST API Development:** Local Flask API for model inference with GET/POST endpoints
- **Streamlit UI:** Interactive web application for user-friendly predictions

These components demonstrate the full MLOps lifecycle when combined with my ML engineering and deployment work.

## 🚀 Potential Enhancements

Additional improvements that could extend this work:

**Infrastructure & Automation:**
- Automated retraining triggers based on data drift detection
- CI/CD pipeline integration with GitHub Actions or Azure DevOps
- Scheduled pipeline execution for regular model updates

**Monitoring & Observability:**
- Real-time model performance dashboard
- Data drift monitoring with alerting
- Prediction logging and analysis infrastructure

**Advanced ML Techniques:**
- Feature importance analysis with SHAP values
- Ensemble methods combining multiple model types
- Online learning for incremental model updates
- Feature store for centralized feature management

**Production Optimization:**
- Model quantization for faster inference
- Batch prediction endpoints for efficiency
- Multi-armed bandit for dynamic A/B testing
- Advanced model explainability for business stakeholders

## 🎓 Skills Demonstrated

**Machine Learning Engineering:**
- Model development and training with scikit-learn
- Hyperparameter tuning and optimization
- Feature engineering and data preprocessing
- Model evaluation with multiple metrics
- Handling imbalanced classification problems

**MLOps & Cloud Platforms:**
- Azure Machine Learning workspace and compute
- MLflow experiment tracking and logging
- Model registry and version control
- Cloud model deployment and endpoint management
- A/B testing configuration

**Software Engineering:**
- Python development with ML stack (pandas, numpy, scikit-learn)
- Jupyter notebook development
- Environment management (conda)
- Version control best practices
- Technical documentation

## Project Context

Completed as part of a collaborative MLOps project for advanced coursework. My contributions focused on the ML engineering pipeline: data management, model training, experiment tracking, and deployment. This work demonstrates skills directly applicable to ML Engineer and MLOps Engineer roles.

## 📄 License

This project is licensed under the MIT License.

---

**Note:** This repository showcases my individual contributions to a collaborative MLOps project. The work demonstrates ML engineering skills in model development, experiment tracking, and cloud deployment - core competencies for ML Engineer and MLOps Engineer roles.
