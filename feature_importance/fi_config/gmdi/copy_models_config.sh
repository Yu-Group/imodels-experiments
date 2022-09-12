echo "Regression DGPs..."
for f in $(find . -type d -name "*_dgp" -o -name "*_regression" ! -name "*_logistic_dgp" ! -name "*_classification" ! -name "*-"); do
    echo "$f"
    cp models_regression.py "$f"/models.py
done

echo "Classification DGPs..."
for f in $(find . -type d -name "*_logistic_dgp" -o -name "*_classification" ! -name "*-"); do
    echo "$f"
    cp models_classification.py "$f"/models.py
done
