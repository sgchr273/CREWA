import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal

np.random.seed(42)

# Parameters
n_id = 300
n_ood = 200

# ID distribution: centered at origin, smaller variance along v_perp
# v_k direction (kept): high variance
# v_perp direction (perpendicular): low variance
id_vk = np.random.normal(0, 1.5, n_id)          # high variance
id_vperp = np.random.normal(0, 0.3, n_id)       # low variance

# OOD distribution: shifted and more spread in v_perp
ood_vk = np.random.normal(0.5, 1.2, n_ood)      # slightly shifted
ood_vperp = np.random.normal(0, 0.8, n_ood)     # much higher variance

# Create figure
fig, ax = plt.subplots(figsize=(10, 8))

# Draw ellipse representing ID covariance structure
# major axis: v_k direction, minor axis: v_perp direction
ellipse = Ellipse(
    xy=(0, 0),
    width=4 * 1.5,    # 4 sigma along v_k
    height=4 * 0.3,   # 4 sigma along v_perp
    angle=0,
    linewidth=2,
    edgecolor='gray',
    facecolor='lightgray',
    alpha=0.2,
    label='ID Covariance Structure'
)
ax.add_patch(ellipse)

# Plot ID points
ax.scatter(id_vk, id_vperp, c='#2E86AB', s=60, alpha=0.7, 
           label=f'ID samples (n={n_id})', edgecolors='black', linewidth=0.5)

# Plot OOD points
ax.scatter(ood_vk, ood_vperp, c='#A23B72', s=60, alpha=0.7, 
           label=f'OOD samples (n={n_ood})', marker='^', edgecolors='black', linewidth=0.5)

# Draw axes labels with mathematical notation
ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.3)
ax.axvline(x=0, color='k', linestyle='--', linewidth=0.5, alpha=0.3)

ax.set_xlabel(r'$\mathbf{v}_k$ direction (Kept Subspace, high variance)', fontsize=12, fontweight='bold')
ax.set_ylabel(r'$\mathbf{v}_{\perp}$ direction (Perpendicular Subspace, low variance)', 
              fontsize=12, fontweight='bold')
ax.set_title('ID vs OOD Distribution across Kept and Perpendicular Subspaces', 
             fontsize=14, fontweight='bold', pad=20)

ax.legend(loc='upper right', fontsize=11, framealpha=0.95)
ax.grid(True, alpha=0.3)
ax.set_xlim(-4, 4)
ax.set_ylim(-3, 3)

# Add annotations
ax.text(0.02, 0.98, 'ID: concentrated along $\mathbf{v}_k$, tight along $\mathbf{v}_{\perp}$', 
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='#2E86AB', alpha=0.2))
ax.text(0.02, 0.88, 'OOD: spreads into $\mathbf{v}_{\perp}$ directions', 
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='#A23B72', alpha=0.2))

plt.tight_layout()
plt.savefig('id_ood_subspace_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

print("Figure saved as 'id_ood_subspace_visualization.png'")