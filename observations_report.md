# CLA 4 Report
## ML Model Deployment Strategies and Model Serving using FastAPI

**Student Name:** ____________________  
**Roll Number:** ____________________  
**Dataset:** Titanic – Machine Learning from Disaster (Kaggle)

### Objective
To train a classification model, evaluate its accuracy, demonstrate online, batch, and streaming inference, and expose the trained pipeline through a FastAPI REST API.

### Methodology
The Kaggle `train.csv` dataset is loaded and the selected passenger features are preprocessed. Missing numerical values are filled using the median, missing categorical values using the most frequent value, and categorical features are one-hot encoded. A Random Forest classifier is trained on a stratified training split and evaluated on a held-out test split. The complete preprocessing and classifier pipeline is saved as `model.pkl`.

### Evaluation
**Actual test accuracy:** Fill this from `outputs/evaluation.txt` after running `python train_model.py`.

### Inference Strategies
- **Online inference:** Loads the saved pipeline and predicts one held-out passenger record.
- **Batch inference:** Predicts 20 held-out passenger records together and writes `outputs/batch_predictions.csv`.
- **Streaming inference:** Processes 10 held-out records one at a time, with a one-second delay between records.
- **FastAPI serving:** Provides `GET /health` to check service status and `POST /predict` to return a survival prediction. Pydantic validates incoming passenger data.

### Observations
1. Online inference processes one record per request, while batch inference processes multiple records in one operation.
2. Streaming inference handles records sequentially as they arrive; batch inference processes a collected group of records together.
3. FastAPI provides HTTP endpoints, request validation support through Pydantic, and interactive API testing through Swagger UI.
4. Reusing one saved pipeline ensures consistent preprocessing and model behavior across scripts and the API, and avoids retraining for every prediction.

### Conclusion
The assignment demonstrates the lifecycle of a machine learning model from training and evaluation to multiple inference patterns and REST API serving. Complete this report with the measured accuracy and attach screenshots from the actual runs before submission.
