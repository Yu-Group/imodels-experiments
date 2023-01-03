echo "Regression DGPs..."
for f in $(find . -type d -name "*_dgp" -o -name "*regression" ! -name "*logistic_dgp" ! -name "*classification" ! -name "*robust_dgp" ! -name "*-"); do
    echo "$f"
    cp models_regression.py "$f"/models.py
done

echo "Classification DGPs..."
for f in $(find . -type d -name "*logistic_dgp" -o -name "*classification" ! -name "*-"); do
    echo "$f"
    cp models_classification.py "$f"/models.py
done

echo "Robust DGPs..."
for f in $(find . -type d -name "*robust_dgp" ! -name "*-"); do
    echo "$f"
    cp models_robust.py "$f"/models.py
done

echo "Bias Sims..."
f=mdi_bias_sims/entropy_sims/linear_dgp
echo "$f"
cp models_bias.py "$f"/models.py
f=mdi_bias_sims/entropy_sims/logistic_dgp
echo "$f"
cp models_bias.py "$f"/models.py
f=mdi_bias_sims/correlation_sims/normal_block_cor_partial_linear_lss_dgp
echo "$f"
cp models_bias.py "$f"/models.py

