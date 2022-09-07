echo "Regression DGPs..."
for f in $(find . -type d -name "*_dgp" ! -name "*_logistic_dgp"); do
    echo "$f"
    cp models_regression.py "$f"/models.py
done

echo "Classification DGPs..."
for f in $(find . -type d -name "*_logistic_dgp"); do
    echo "$f"
    cp models_regression.py "$f"/models.py
done
