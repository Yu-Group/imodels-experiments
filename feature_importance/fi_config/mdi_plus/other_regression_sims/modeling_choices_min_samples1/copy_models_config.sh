echo "Modeling Choices Regression DGPs..."
for f in $(find . -type d -name "*_dgp"); do
    echo "$f"
    cp models.py "$f"/models.py
done
