import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
from copulas.multivariate import GaussianMultivariate
from scipy.stats import gaussian_kde, norm
import warnings
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ----------------------------
# Load data
# ----------------------------
df = pd.read_excel(r'E:\0workspace\Guosai-2025\filtered-ref-converted.xlsx')
df_mid = df[(df['孕妇BMI'] >= 27) & (df['孕妇BMI'] <= 38)].copy()

print(f"Total samples in BMI 27-38: {len(df_mid)}")


# ----------------------------
# 3-variable Copula modeling
# ----------------------------
def fit_3d_copula(df_sub):
    """Fit 3-variable (BMI, Gestational Week, Y concentration) Copula model"""
    if len(df_sub) < 10:
        return None, None, None

    bmi_sub = df_sub['孕妇BMI'].values
    t_sub = df_sub['孕周'].values
    y_sub = df_sub['Y染色体浓度'].values

    # Marginal distribution transformation
    ecdf_bmi = ECDF(bmi_sub)
    ecdf_t = ECDF(t_sub)
    ecdf_y = ECDF(y_sub)

    u_bmi = ecdf_bmi(bmi_sub)
    u_t = ecdf_t(t_sub)
    u_y = ecdf_y(y_sub)

    data_uv = pd.DataFrame({'u_bmi': u_bmi, 'u_t': u_t, 'u_y': u_y})

    # Fit Gaussian Copula
    copula_model = GaussianMultivariate()
    copula_model.fit(data_uv)

    return copula_model, (ecdf_bmi, ecdf_t, ecdf_y), (bmi_sub, t_sub, y_sub)


# ----------------------------
# Conditional probability calculation
# ----------------------------
def calculate_conditional_probability(copula_model, ecdfs, target_bmi, target_week, threshold=0.04):
    """Calculate probability of Y concentration exceeding threshold given BMI and gestational week"""
    ecdf_bmi, ecdf_t, ecdf_y = ecdfs
    bmi_sub, t_sub, y_sub = data_orig

    # Convert to Copula space
    u_bmi_target = ecdf_bmi(target_bmi)
    u_t_target = ecdf_t(target_week)

    # Generate conditional samples from Copula
    n_samples = 5000
    conditional_samples = []

    for _ in range(n_samples):
        # Generate 3-variable sample
        sample = copula_model.sample(1)
        u_bmi_sample = sample['u_bmi'].values[0]
        u_t_sample = sample['u_t'].values[0]
        u_y_sample = sample['u_y'].values[0]

        # If BMI and gestational week are close to target values, accept the sample
        if (abs(u_bmi_sample - u_bmi_target) < 0.1 and
                abs(u_t_sample - u_t_target) < 0.1):
            conditional_samples.append(u_y_sample)

    if not conditional_samples:
        return 0.0

    # Convert back to Y concentration space
    y_conditional = np.quantile(y_sub, conditional_samples)

    # Calculate probability of exceeding threshold
    prob_above_threshold = np.mean(y_conditional >= threshold)

    return prob_above_threshold


# ----------------------------
# Main analysis
# ----------------------------
# Define BMI intervals
intervals = [(27, 31.5), (31.5, 33.5), (33.5, 38)]
midpoints = [(low + high) / 2 for (low, high) in intervals]

# Gestational week range (10-25 weeks)
weeks = np.linspace(10, 25, 30)
threshold = 0.04

# Store results
results = {}

for i, (low, high) in enumerate(intervals):
    df_sub = df_mid[(df_mid['孕妇BMI'] >= low) & (df_mid['孕妇BMI'] <= high)]

    if len(df_sub) < 10:
        print(f"Interval {low}-{high} has too few samples ({len(df_sub)}), skipping")
        continue

    print(f"Processing BMI interval {low}-{high}, sample count: {len(df_sub)}")

    # Fit Copula model
    copula_model, ecdfs, data_orig = fit_3d_copula(df_sub)
    if copula_model is None:
        continue

    # Calculate probability for each gestational week
    probabilities = []
    bmi_midpoint = midpoints[i]

    for week in weeks:
        prob = calculate_conditional_probability(copula_model, ecdfs, bmi_midpoint, week, threshold)
        probabilities.append(prob)
        print(f"  BMI {bmi_midpoint:.1f}, Week {week:.1f}: Probability = {prob:.3f}")

    # Find optimal timing (earliest week with probability ≥95%)
    best_week = None
    for week, prob in zip(weeks, probabilities):
        if prob >= 0.95:
            best_week = week
            break

    # If not reaching 95%, find week with highest probability
    if best_week is None:
        best_week = weeks[np.argmax(probabilities)]

    results[f'BMI_{low}_{high}'] = {
        'best_week': best_week,
        'max_prob': max(probabilities),
        'probabilities': probabilities,
        'bmi_midpoint': bmi_midpoint
    }

# ----------------------------
# Visualization
# ----------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Probability curves
for interval_key, result in results.items():
    ax1.plot(weeks, result['probabilities'],
             label=f'{interval_key} (Optimal: {result["best_week"]:.1f} weeks)',
             linewidth=2, marker='o')

    # Mark optimal timing
    ax1.axvline(x=result['best_week'], linestyle='--', alpha=0.5)

ax1.axhline(y=0.95, color='r', linestyle='--', label='95% Accuracy Threshold')
ax1.axvline(x=12, color='g', linestyle='--', label='12 Weeks Risk Boundary')
ax1.set_xlabel('Gestational Week')
ax1.set_ylabel('Probability of Y Concentration ≥4%')
ax1.set_title('Optimal NIPT Timing by BMI Interval')
ax1.legend()
ax1.grid(True)

# 3D visualization
ax2 = fig.add_subplot(122, projection='3d')

for interval_key, result in results.items():
    bmi_val = result['bmi_midpoint']
    probs = result['probabilities']

    # Create 3D surface
    X, Y = np.meshgrid([bmi_val], weeks)
    Z = np.array(probs).reshape(1, -1)

    ax2.plot_surface(X, Y, Z, alpha=0.6, label=interval_key)

ax2.set_xlabel('BMI')
ax2.set_ylabel('Gestational Week')
ax2.set_zlabel('Accuracy Probability')
ax2.set_title('BMI-Week-Accuracy 3D Relationship')

plt.tight_layout()
plt.show()

# ----------------------------
# Output optimal timing recommendations
# ----------------------------
print("\n=== Optimal NIPT Testing Timing Recommendations ===")
print("Based on Copula model and risk-accuracy trade-off:")
print("-" * 60)

for interval_key, result in results.items():
    best_week = result['best_week']
    max_prob = result['max_prob']
    risk_level = "Low risk" if best_week <= 12 else "Medium risk" if best_week <= 27 else "High risk"

    print(f"{interval_key}:")
    print(f"  Optimal timing: {best_week:.1f} weeks ({risk_level})")
    print(f"  Expected accuracy: {max_prob * 100:.1f}%")

    if best_week > 12:
        print(f"  ⚠️ Note: Exceeds 12 weeks, increased risk, recommend earlier testing")
    print()

# Comprehensive recommendations
print("Comprehensive Recommendations:")
print("1. Complete testing before 12 weeks to minimize risk")
print("2. If extremely high accuracy (≥95%) is required, can postpone slightly but not beyond 25 weeks")
print("3. Higher BMI may require slightly later testing to ensure accuracy")