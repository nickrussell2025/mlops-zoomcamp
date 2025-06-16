FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

COPY predict.py .

RUN pip install pandas pyarrow scikit-learn==1.5.0

ENTRYPOINT ["python", "predict.py"]