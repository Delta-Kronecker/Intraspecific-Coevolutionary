import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import re
import os

# ====================================================
# Configuration
# ====================================================
output_dir_html = "interactive_plots"
output_dir_png = "static_plots"
os.makedirs(output_dir_html, exist_ok=True)
os.makedirs(output_dir_png, exist_ok=True)

plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

# Load the CSV
df = pd.read_csv("results.csv")
print(f"Data loaded. Rows: {len(df)}, Columns: {len(df.columns)}")

# ====================================================
# Identify loci and phenotype columns
# ====================================================
loci = set()
for col in df.columns:
    if col.startswith('epsilon_'):
        loci.add(col[len('epsilon_'):])
loci = sorted(list(loci))
num_loci = len(loci)
print(f"Detected {num_loci} loci: {loci}")

pheno_cols = [col for col in df.columns if col.startswith('f_') and re.match(r'^[A-Za-z]+$', col[2:])]
print(f"Detected {len(pheno_cols)} phenotype columns: {pheno_cols}")

p_cols = [f'p_{l}' for l in loci]
alpha_cols = [f'alpha_{l}' for l in loci]
mu_cols = [f'mu_bar_{l}' for l in loci]
mu0_cols = [f'mu0_{l}' for l in loci]

# ====================================================
# Downsampling for performance
# ====================================================
max_points = 2000
if len(df) > max_points:
    step = len(df) // max_points
    df_sample = df.iloc[::step].copy()
    print(f"Downsampled from {len(df)} to {len(df_sample)} points for interactivity.")
else:
    df_sample = df.copy()

# ====================================================
# Helper functions
# ====================================================
def save_html(fig, filename):
    path = os.path.join(output_dir_html, filename)
    fig.write_html(path)
    print(f"Saved HTML: {path}")

def save_png(fig, filename):
    path = os.path.join(output_dir_png, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved PNG: {path}")

def count_dominant(phen):
    return sum(1 for c in phen if c.isupper())

# ====================================================
# 1. Allele Frequencies Over Time
# ====================================================
# HTML
fig = px.line(df_sample, x='Generation', y=p_cols,
              title='Dominant Allele Frequencies Over Time',
              labels={'value': 'Allele Frequency', 'variable': 'Locus'})
fig.update_layout(hovermode='x unified')
save_html(fig, '01_allele_frequencies.html')

# PNG
fig, ax = plt.subplots()
for locus in loci:
    ax.plot(df['Generation'], df[f'p_{locus}'], label=f'p_{locus}')
ax.set_xlabel('Generation'); ax.set_ylabel('Allele frequency')
ax.set_title('Dominant Allele Frequencies Over Time')
ax.legend(); ax.grid(True)
save_png(fig, '01_allele_frequencies.png')

# ====================================================
# 2. Mean Phenotype Performance (mu_bar)
# ====================================================
fig = px.line(df_sample, x='Generation', y=mu_cols,
              title='Mean Phenotype Performance Over Time',
              labels={'value': 'Mean Performance (μ̄)', 'variable': 'Locus'})
fig.update_layout(hovermode='x unified')
save_html(fig, '02_mean_performance.html')

fig, ax = plt.subplots()
for locus in loci:
    ax.plot(df['Generation'], df[f'mu_bar_{locus}'], label=f'mu_bar_{locus}')
ax.set_xlabel('Generation'); ax.set_ylabel('Mean performance')
ax.set_title('Mean Phenotype Performance Over Time')
ax.legend()
save_png(fig, '02_mean_performance.png')

# ====================================================
# 3. Selection Pressure (alpha)
# ====================================================
fig = px.line(df_sample, x='Generation', y=alpha_cols,
              title='Selection Pressure Over Time',
              labels={'value': 'Selection Pressure (α)', 'variable': 'Locus'})
fig.update_layout(hovermode='x unified')
save_html(fig, '03_selection_pressure.html')

fig, ax = plt.subplots()
for locus in loci:
    ax.plot(df['Generation'], df[f'alpha_{locus}'], label=f'α_{locus}')
ax.set_xlabel('Generation'); ax.set_ylabel('Selection pressure (α)')
ax.set_title('Selection Pressure Over Time')
ax.legend()
save_png(fig, '03_selection_pressure.png')


# ====================================================
# 4. Population Size
# ====================================================
import matplotlib.ticker as ticker

fig = px.line(df_sample, x='Generation', y='N_Total',
              title='Population Size Over Time',
              labels={'N_Total': 'Total Population (N)'})
fig.update_layout(hovermode='x unified')
save_html(fig, '04_population_size.html')

fig, ax = plt.subplots()
ax.plot(df['Generation'], df['N_Total'], color='black')
ax.set_xlabel('Generation')
ax.set_ylabel('Total Population (N)')
ax.set_title('Population Size Over Time')

ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x):,}'))

save_png(fig, '04_population_size.png')


# ====================================================
# 5. Mean Fitness (C_bar)
# ====================================================
fig = px.line(df_sample, x='Generation', y='C_bar',
              title='Mean Population Fitness Over Time',
              labels={'C_bar': 'Mean Fitness (C̄)'})
fig.update_layout(hovermode='x unified')
save_html(fig, '05_mean_fitness.html')

fig, ax = plt.subplots()
ax.plot(df['Generation'], df['C_bar'], color='green')
ax.set_xlabel('Generation'); ax.set_ylabel('Mean Fitness (C̄)')
ax.set_title('Mean Population Fitness Over Time')
save_png(fig, '05_mean_fitness.png')

# ====================================================
# 6. Phenotype Frequencies (Stacked Area)
# ====================================================
fig = px.area(df_sample, x='Generation', y=pheno_cols,
              title='Phenotype Frequencies Over Time',
              labels={'value': 'Frequency', 'variable': 'Phenotype'})
new_names = {col: col[2:] for col in pheno_cols}
fig.for_each_trace(lambda t: t.update(name=new_names.get(t.name, t.name)))
fig.update_layout(hovermode='x unified')
save_html(fig, '06_phenotype_frequencies.html')

fig, ax = plt.subplots()
ax.stackplot(df['Generation'], *[df[col] for col in pheno_cols],
             labels=[col[2:] for col in pheno_cols], alpha=0.8)
ax.set_xlabel('Generation'); ax.set_ylabel('Frequency')
ax.set_title('Phenotype Frequencies Over Time (Stacked)')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
save_png(fig, '06_phenotype_frequencies.png')

# ====================================================
# 7. Dual-axis: p and mu_bar for each locus
# ====================================================
for locus in loci:
    # HTML
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=df_sample['Generation'], y=df_sample[f'p_{locus}'],
                   name=f'p_{locus}', line=dict(color='blue')),
        secondary_y=False
    )
    fig.add_trace(
        go.Scatter(x=df_sample['Generation'], y=df_sample[f'mu_bar_{locus}'],
                   name=f'mu_bar_{locus}', line=dict(color='red')),
        secondary_y=True
    )
    fig.update_xaxes(title_text="Generation")
    fig.update_yaxes(title_text="Allele Frequency (p)", secondary_y=False, color='blue')
    fig.update_yaxes(title_text="Mean Performance (μ̄)", secondary_y=True, color='red')
    fig.update_layout(title=f'Dual Axis: p_{locus} and mu_bar_{locus}', hovermode='x unified')
    save_html(fig, f'07_dual_p_mu_{locus}.html')
    
    # PNG
    fig_png, ax1 = plt.subplots()
    color1 = 'tab:blue'
    ax1.plot(df['Generation'], df[f'p_{locus}'], color=color1, label=f'p_{locus}')
    ax1.set_xlabel('Generation'); ax1.set_ylabel('Allele frequency', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.plot(df['Generation'], df[f'mu_bar_{locus}'], color=color2, label=f'mu_bar_{locus}')
    ax2.set_ylabel('Mean performance', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    plt.title(f'Dual Axis: p_{locus} and mu_bar_{locus}')
    fig_png.tight_layout()
    save_png(fig_png, f'07_dual_p_mu_{locus}.png')

# ====================================================
# 8. Dual-axis: alpha and mu_bar for each locus
# ====================================================
for locus in loci:
    # HTML
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=df_sample['Generation'], y=df_sample[f'alpha_{locus}'],
                   name=f'α_{locus}', line=dict(color='green')),
        secondary_y=False
    )
    fig.add_trace(
        go.Scatter(x=df_sample['Generation'], y=df_sample[f'mu_bar_{locus}'],
                   name=f'mu_bar_{locus}', line=dict(color='red')),
        secondary_y=True
    )
    fig.update_xaxes(title_text="Generation")
    fig.update_yaxes(title_text="Selection Pressure (α)", secondary_y=False, color='green')
    fig.update_yaxes(title_text="Mean Performance (μ̄)", secondary_y=True, color='red')
    fig.update_layout(title=f'Dual Axis: α_{locus} and mu_bar_{locus}', hovermode='x unified')
    save_html(fig, f'08_dual_alpha_mu_{locus}.html')
    
    # PNG
    fig_png, ax1 = plt.subplots()
    color1 = 'tab:green'
    ax1.plot(df['Generation'], df[f'alpha_{locus}'], color=color1, label=f'α_{locus}')
    ax1.set_xlabel('Generation'); ax1.set_ylabel('Selection pressure', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.plot(df['Generation'], df[f'mu_bar_{locus}'], color=color2, label=f'mu_bar_{locus}')
    ax2.set_ylabel('Mean performance', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    plt.title(f'Dual Axis: α_{locus} and mu_bar_{locus}')
    fig_png.tight_layout()
    save_png(fig_png, f'08_dual_alpha_mu_{locus}.png')

# ====================================================
# 9. Progress Ratio (R)
# ====================================================
if all(f'R_{l}' in df.columns for l in loci):
    r_cols = [f'R_{l}' for l in loci]
    
    # HTML
    fig = px.line(df_sample, x='Generation', y=r_cols,
                  title='Per-generation Progress Ratio',
                  labels={'value': 'Progress Ratio (R)', 'variable': 'Locus'})
    fig.update_layout(hovermode='x unified')
    save_html(fig, '09_progress_ratio.html')
    
    # PNG
    fig_png, ax = plt.subplots()
    for locus in loci:
        ax.plot(df['Generation'], df[f'R_{locus}'], label=f'R_{locus}')
    ax.set_xlabel('Generation'); ax.set_ylabel('Progress Ratio (R)')
    ax.set_title('Per-generation Progress Ratio')
    ax.legend()
    save_png(fig_png, '09_progress_ratio.png')

# ====================================================
# 10. Phase Plots: p_X vs p_Y (ALL combinations)
# ====================================================
if num_loci >= 2:
    for (l1, l2) in combinations(loci, 2):
        # HTML
        fig = px.scatter(df_sample, x=f'p_{l1}', y=f'p_{l2}',
                         color='Generation', color_continuous_scale='viridis',
                         title=f'Phase Plot: {l1} vs {l2}',
                         labels={f'p_{l1}': f'p_{l1}', f'p_{l2}': f'p_{l2}'})
        fig.update_traces(marker=dict(size=4, opacity=0.8))
        save_html(fig, f'10_phase_{l1}_vs_{l2}.html')
        
        # PNG
        fig_png, ax = plt.subplots()
        sc = ax.scatter(df[f'p_{l1}'], df[f'p_{l2}'],
                        c=df['Generation'], cmap='viridis', s=10, alpha=0.7)
        plt.colorbar(sc, label='Generation')
        ax.set_xlabel(f'p_{l1}'); ax.set_ylabel(f'p_{l2}')
        ax.set_title(f'Phase Plot: {l1} vs {l2}')
        ax.grid(True)
        save_png(fig_png, f'10_phase_{l1}_vs_{l2}.png')

# ====================================================
# 11. Phenotype Frequency Heatmap
# ====================================================
# HTML
pheno_heat = df_sample[pheno_cols].T
pheno_heat.index = [c[2:] for c in pheno_cols]
fig = px.imshow(pheno_heat, aspect='auto', color_continuous_scale='YlOrRd',
                labels=dict(x='Generation Index', y='Phenotype', color='Frequency'),
                title='Phenotype Frequency Heatmap')
save_html(fig, '11_phenotype_heatmap.html')

# PNG
max_heat = 200
if len(df) > max_heat:
    step = len(df) // max_heat
    hdf = df.iloc[::step].copy()
else:
    hdf = df.copy()
pheno_heat_png = hdf[pheno_cols].T
pheno_heat_png.index = [c[2:] for c in pheno_cols]
fig_png, ax = plt.subplots(figsize=(12, max(len(pheno_cols)*0.8, 4)))
sns.heatmap(pheno_heat_png, cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Frequency'})
ax.set_title('Phenotype Frequency Heatmap')
ax.set_ylabel('Phenotype'); ax.set_xlabel('Generation (sampled)')
ncol = pheno_heat_png.shape[1]
xtick_step = max(1, ncol//10)
ax.set_xticks(range(0, ncol, xtick_step))
ax.set_xticklabels(hdf['Generation'].iloc[::xtick_step].values, rotation=45)
save_png(fig_png, '11_phenotype_heatmap.png')

# ====================================================
# 12. Fixation Times
# ====================================================
fix_threshold = 0.99
fix_data = {'Locus': [], 'Fixation_Generation': []}
for l in loci:
    p = df[f'p_{l}']
    if p.max() >= fix_threshold:
        idx = p[p >= fix_threshold].index[0]
        fix_data['Locus'].append(l)
        fix_data['Fixation_Generation'].append(df.loc[idx, 'Generation'])
    else:
        fix_data['Locus'].append(l)
        fix_data['Fixation_Generation'].append(0)

fix_df = pd.DataFrame(fix_data)

# HTML
colors = ['salmon' if v > 0 else 'lightgray' for v in fix_df['Fixation_Generation']]
fig = px.bar(fix_df, x='Locus', y='Fixation_Generation',
             title='Fixation Time Comparison (p≥0.99)',
             labels={'Fixation_Generation': 'Generation to Reach p=0.99'},
             color='Locus', color_discrete_sequence=colors)
fig.update_traces(showlegend=False)
save_html(fig, '12_fixation_times.html')

# PNG
fig_png, ax = plt.subplots()
labels = list(fix_data['Locus'])
values = fix_data['Fixation_Generation']
bars = ax.bar(labels, values, color=['salmon' if v > 0 else 'lightgray' for v in values])
ax.axhline(y=0, color='black')
ax.set_ylabel('Generation to Reach p=0.99')
ax.set_title('Fixation Time Comparison (p≥0.99)')
for bar, gen in zip(bars, values):
    if gen > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+5, str(int(gen)), ha='center')
    else:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+5, 'Not fixed', ha='center')
save_png(fig_png, '12_fixation_times.png')

# ====================================================
# 13. Baseline Performance (mu0)
# ====================================================
# HTML
fig = px.line(df_sample, x='Generation', y=mu0_cols,
              title='Environmental Baseline (μ0) Over Time',
              labels={'value': 'Baseline Performance (μ0)', 'variable': 'Locus'})
fig.update_layout(hovermode='x unified')
save_html(fig, '13_baseline_mu0.html')

# PNG
fig_png, ax = plt.subplots()
for locus in loci:
    ax.plot(df['Generation'], df[f'mu0_{locus}'], label=f'mu0_{locus}')
ax.set_xlabel('Generation'); ax.set_ylabel('Baseline Performance (μ0)')
ax.set_title('Environmental Baseline Over Time')
ax.legend()
save_png(fig_png, '13_baseline_mu0.png')

from itertools import product





from itertools import product

# ====================================================
# 14. Population Count by Number of Dominant Alleles (LINE PLOT)
# ====================================================
k_list = list(range(num_loci + 1))

k_counts_full = np.zeros((len(df), len(k_list)))
for i, (_, row) in enumerate(df.iterrows()):
    N_total = float(row['N_Total'])
    for phen_col in pheno_cols:
        phen = phen_col[2:]
        k = count_dominant(phen)
        k_counts_full[i, k] += row[phen_col] * N_total

k_counts_sample = np.zeros((len(df_sample), len(k_list)))
for i, (_, row) in enumerate(df_sample.iterrows()):
    N_total = float(row['N_Total'])
    for phen_col in pheno_cols:
        phen = phen_col[2:]
        k = count_dominant(phen)
        k_counts_sample[i, k] += row[phen_col] * N_total

fig = go.Figure()
for ki, k in enumerate(k_list):
    fig.add_trace(go.Scatter(
        x=df_sample['Generation'],
        y=k_counts_sample[:, ki],
        name=f'{k} dominant{"s" if k != 1 else ""}',
        mode='lines'
    ))
fig.update_layout(
    title='Population Count by Number of Dominant Alleles Over Time',
    xaxis_title='Generation',
    yaxis_title='Number of Individuals',
    hovermode='x unified'
)
save_html(fig, '14_dominant_count_distribution.html')

fig_png, ax = plt.subplots()
for ki, k in enumerate(k_list):
    ax.plot(df['Generation'], k_counts_full[:, ki], label=f'{k} dominant{"s" if k != 1 else ""}')
ax.set_xlabel('Generation')
ax.set_ylabel('Number of Individuals')
ax.set_title('Population Count by Number of Dominant Alleles Over Time')
ax.legend()
ax.grid(True)
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
save_png(fig_png, '14_dominant_count_distribution.png')






# ====================================================
# 14c. Population of Each Genotype Group Over Time
# ====================================================

states_per_locus = []
for locus in loci:
    dom = locus.upper()
    rec = locus.lower()
    states_per_locus.append([dom*2, dom+rec, rec*2])

ALL_GENOTYPES = [''.join(combo) for combo in product(*states_per_locus)]

def get_dominant_pattern(genotype_str):
    pattern = ''
    for i, locus in enumerate(loci):
        start = i * 2
        state = genotype_str[start:start+2]
        rec_state = locus.lower() * 2
        if state == rec_state:
            pattern += locus.lower()
        else:
            pattern += locus.upper()
    return pattern

def genotype_to_group_name(genotype_str):
    return get_dominant_pattern(genotype_str)

group_names = sorted(list(set([genotype_to_group_name(g) for g in ALL_GENOTYPES])))
print(f"Found {len(group_names)} genotype groups: {group_names}")

genotype_counts_full = np.zeros((len(df), len(ALL_GENOTYPES)))
genotype_counts_sample = np.zeros((len(df_sample), len(ALL_GENOTYPES)))

for i, (_, row) in enumerate(df_sample.iterrows()):
    N_total = float(row['N_Total'])
    p_vals = {locus: float(row[f'p_{locus}']) for locus in loci}
    for j, genotype in enumerate(ALL_GENOTYPES):
        freq = 1.0
        for locus_idx, locus in enumerate(loci):
            p = p_vals[locus]
            q = 1.0 - p
            state = genotype[locus_idx*2:locus_idx*2+2]
            dom = locus.upper() * 2
            het = locus.upper() + locus.lower()
            if state == dom:
                freq *= p * p
            elif state == het:
                freq *= 2.0 * p * q
            else:
                freq *= q * q
        genotype_counts_sample[i, j] = freq * N_total

for i, (_, row) in enumerate(df.iterrows()):
    N_total = float(row['N_Total'])
    p_vals = {locus: float(row[f'p_{locus}']) for locus in loci}
    for j, genotype in enumerate(ALL_GENOTYPES):
        freq = 1.0
        for locus_idx, locus in enumerate(loci):
            p = p_vals[locus]
            q = 1.0 - p
            state = genotype[locus_idx*2:locus_idx*2+2]
            dom = locus.upper() * 2
            het = locus.upper() + locus.lower()
            if state == dom:
                freq *= p * p
            elif state == het:
                freq *= 2.0 * p * q
            else:
                freq *= q * q
        genotype_counts_full[i, j] = freq * N_total

group_counts_sample = {group: np.zeros(len(df_sample)) for group in group_names}
group_counts_full = {group: np.zeros(len(df)) for group in group_names}

for j, genotype in enumerate(ALL_GENOTYPES):
    group = genotype_to_group_name(genotype)
    group_counts_sample[group] += genotype_counts_sample[:, j]
    group_counts_full[group] += genotype_counts_full[:, j]

fig = go.Figure()
for group in group_names:
    fig.add_trace(go.Scatter(
        x=df_sample['Generation'],
        y=group_counts_sample[group],
        name=group,
        mode='lines',
        line=dict(width=1.5)
    ))
fig.update_layout(
    title='Population Size by Genotype Group (Based on Dominant Alleles) Over Time',
    xaxis_title='Generation',
    yaxis_title='Number of Individuals',
    hovermode='x unified',
    legend_title="Genotype Group"
)
save_html(fig, '14c_genotype_populations.html')

fig_png, ax = plt.subplots(figsize=(14, 8))
for group in group_names:
    ax.plot(df['Generation'], group_counts_full[group], label=group, linewidth=1.2)
ax.set_xlabel('Generation')
ax.set_ylabel('Number of Individuals')
ax.set_title('Population Size by Genotype Group (Based on Dominant Alleles) Over Time')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
ax.grid(True)
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
save_png(fig_png, '14c_genotype_populations.png')



# ====================================================
# 15. Effective Selection Coefficient (s)
# ====================================================
if all(f's_{l}' in df.columns for l in loci):
    s_cols = [f's_{l}' for l in loci]
    
    # HTML
    fig = px.line(df_sample, x='Generation', y=s_cols,
                  title='Effective Selection Strength Over Time',
                  labels={'value': 'Effective selection (α/C̄)', 'variable': 'Locus'})
    fig.update_layout(hovermode='x unified')
    save_html(fig, '15_effective_selection.html')
    
    # PNG
    fig_png, ax = plt.subplots()
    for locus in loci:
        ax.plot(df['Generation'], df[f's_{locus}'], label=f's_{locus}')
    ax.set_xlabel('Generation'); ax.set_ylabel('Effective selection (α/C̄)')
    ax.set_title('Effective Selection Strength Over Time')
    ax.legend()
    save_png(fig_png, '15_effective_selection.png')

# ====================================================
# 16. Fitness Variance
# ====================================================
if 'var_fitness' in df.columns:
    # HTML
    fig = px.line(df_sample, x='Generation', y='var_fitness',
                  title='Fitness Variance Across Phenotypes Over Time',
                  labels={'var_fitness': 'Variance of Fitness'})
    fig.update_traces(line_color='purple')
    fig.update_layout(hovermode='x unified')
    save_html(fig, '16_fitness_variance.html')
    
    # PNG
    fig_png, ax = plt.subplots()
    ax.plot(df['Generation'], df['var_fitness'], color='purple')
    ax.set_xlabel('Generation'); ax.set_ylabel('Variance of Fitness')
    ax.set_title('Fitness Variance Across Phenotypes Over Time')
    save_png(fig_png, '16_fitness_variance.png')

# ====================================================
# 17. 3D Trajectories – ALL combinations of 3 loci
# ====================================================
if num_loci >= 3:
    comb3 = list(combinations(loci, 3))
    print(f"Creating {len(comb3)} interactive 3D plot(s).")
    for (l1, l2, l3) in comb3:
        # HTML (interactive rotation)
        fig = px.scatter_3d(
            df_sample, x=f'p_{l1}', y=f'p_{l2}', z=f'p_{l3}',
            color='Generation', color_continuous_scale='viridis',
            title=f'3D Trajectory: {l1}-{l2}-{l3}',
            opacity=0.8,
            width=900, height=700
        )
        fig.update_traces(marker=dict(size=3))
        fig.update_layout(
            scene=dict(
                xaxis_title=f'p_{l1}',
                yaxis_title=f'p_{l2}',
                zaxis_title=f'p_{l3}'
            )
        )
        save_html(fig, f'17_3D_{l1}_{l2}_{l3}.html')
        
        # PNG (multiple angles)
        from mpl_toolkits.mplot3d import Axes3D
        angles = [(30,45), (30,135), (60,45), (20,-60)]
        for elev, azim in angles:
            fig_png = plt.figure()
            ax = fig_png.add_subplot(111, projection='3d')
            sc = ax.scatter(df[f'p_{l1}'], df[f'p_{l2}'], df[f'p_{l3}'],
                            c=df['Generation'], cmap='viridis', s=10, alpha=0.8)
            ax.set_xlabel(f'p_{l1}'); ax.set_ylabel(f'p_{l2}'); ax.set_zlabel(f'p_{l3}')
            ax.set_title(f'3D Trajectory: {l1}-{l2}-{l3} (elev={elev}, azim={azim})')
            ax.view_init(elev, azim)
            plt.colorbar(sc, ax=ax, pad=0.1, label='Generation')
            save_png(fig_png, f'17_3D_{l1}_{l2}_{l3}_elev{elev}_azim{azim}.png')
else:
    print("Not enough loci for 3D plots (need at least 3).")

# ====================================================
# 18. Parallel Coordinates
# ====================================================
# HTML
fig = px.parallel_coordinates(
    df_sample,
    dimensions=p_cols,
    color='Generation',
    color_continuous_scale=px.colors.sequential.Viridis,
    title='Parallel Coordinates of Allele Frequencies'
)
save_html(fig, '18_parallel_coordinates.html')

# PNG (static version using matplotlib)
fig_png, ax = plt.subplots(figsize=(10, 6))
norm_gen = (df['Generation'] - df['Generation'].min()) / (df['Generation'].max() - df['Generation'].min() + 1e-9)
colors = plt.cm.viridis(norm_gen)
for i in range(0, len(df), max(1, len(df)//1000)):
    row = df.iloc[i]
    ax.plot(p_cols, [row[f'p_{l}'] for l in loci], color=colors[i], alpha=0.3, linewidth=0.5)
ax.set_xlabel('Locus'); ax.set_ylabel('Allele Frequency')
ax.set_title('Parallel Coordinates of Allele Frequencies')
sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=df['Generation'].min(), vmax=df['Generation'].max()))
sm.set_array([])
plt.colorbar(sm, ax=ax, label='Generation')
save_png(fig_png, '18_parallel_coordinates.png')

# ====================================================
# 19. Allele Frequency Heatmap
# ====================================================
# HTML
p_heat = df_sample[['Generation'] + p_cols].set_index('Generation').T
fig = px.imshow(p_heat, aspect='auto', color_continuous_scale='YlOrRd',
                labels=dict(x='Generation', y='Locus', color='Frequency'),
                title='Allele Frequency Heatmap')
save_html(fig, '19_allele_frequency_heatmap.html')

# PNG
if len(df) > max_heat:
    hdf = df.iloc[::max(1, len(df)//max_heat)].copy()
else:
    hdf = df.copy()
p_heat_png = hdf[['Generation'] + p_cols].set_index('Generation').T
fig_png, ax = plt.subplots(figsize=(12, max(len(loci)*0.8, 4)))
sns.heatmap(p_heat_png, cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Frequency'})
ax.set_title('Allele Frequency Heatmap')
ax.set_ylabel('Locus'); ax.set_xlabel('Generation (sampled)')
ncol = p_heat_png.shape[1]
xtick_step = max(1, ncol//10)
ax.set_xticks(range(0, ncol, xtick_step))
ax.set_xticklabels(p_heat_png.columns[::xtick_step], rotation=45)
save_png(fig_png, '19_allele_frequency_heatmap.png')

# ====================================================
# 20. Cross-alpha scattering (α_X vs α_Y)
# ====================================================
if num_loci >= 2:
    for (l1, l2) in combinations(loci, 2):
        # HTML
        fig = px.scatter(df_sample, x=f'alpha_{l1}', y=f'alpha_{l2}',
                         color='Generation', color_continuous_scale='plasma',
                         title=f'Selection Pressure Cross: α_{l1} vs α_{l2}',
                         labels={f'alpha_{l1}': f'α_{l1}', f'alpha_{l2}': f'α_{l2}'})
        fig.update_traces(marker=dict(size=4, opacity=0.8))
        save_html(fig, f'20_cross_alpha_{l1}_vs_{l2}.html')
        
        # PNG
        fig_png, ax = plt.subplots()
        sc = ax.scatter(df[f'alpha_{l1}'], df[f'alpha_{l2}'],
                        c=df['Generation'], cmap='plasma', s=10, alpha=0.7)
        plt.colorbar(sc, label='Generation')
        ax.set_xlabel(f'α_{l1}'); ax.set_ylabel(f'α_{l2}')
        ax.set_title(f'Selection Pressure Cross: α_{l1} vs α_{l2}')
        ax.grid(True)
        save_png(fig_png, f'20_cross_alpha_{l1}_vs_{l2}.png')

# ====================================================
# 21. Rolling Window Correlation
# ====================================================
window_size = max(50, len(df)//200)
if num_loci >= 2 and len(df) > window_size:
    for (l1, l2) in combinations(loci, 2):
        p1 = df[f'p_{l1}'].values
        p2 = df[f'p_{l2}'].values
        rol_corr = np.full(len(df), np.nan)
        
        for i in range(window_size, len(df)):
            # Check if there's variation in both traits within the window
            std1 = np.std(p1[i-window_size:i])
            std2 = np.std(p2[i-window_size:i])
            if std1 > 1e-15 and std2 > 1e-15:
                rol_corr[i] = np.corrcoef(p1[i-window_size:i], p2[i-window_size:i])[0,1]
            else:
                rol_corr[i] = np.nan  # No variation, correlation undefined
        
        # HTML
        temp_df = pd.DataFrame({'Generation': df['Generation'], 'Rolling_Correlation': rol_corr})
        fig = px.line(temp_df, x='Generation', y='Rolling_Correlation',
                      title=f'Rolling Correlation of p_{l1} & p_{l2} (window={window_size})')
        fig.update_layout(hovermode='x unified')
        save_html(fig, f'21_rolling_corr_{l1}_{l2}.html')
        
        # PNG
        fig_png, ax = plt.subplots()
        ax.plot(df['Generation'], rol_corr, label=f'{l1}-{l2}')
        ax.set_xlabel('Generation'); ax.set_ylabel('Pearson Correlation')
        ax.set_title(f'Rolling Correlation of p_{l1} & p_{l2} (window={window_size})')
        ax.legend(); ax.grid(True)
        ax.set_ylim(-0.1, 1.1)  # Correlation range
        save_png(fig_png, f'21_rolling_corr_{l1}_{l2}.png')

# ====================================================
# 22. Acceleration of Allele Frequency
# ====================================================
for locus in loci:
    p = df[f'p_{locus}'].values
    acc = np.zeros_like(p)
    acc[2:] = np.diff(p, n=2)
    acc[:2] = np.nan
    
    # HTML
    temp_df = pd.DataFrame({'Generation': df['Generation'], 'Acceleration': acc})
    fig = px.line(temp_df, x='Generation', y='Acceleration',
                  title=f'Acceleration of {locus} Allele Frequency',
                  labels={'Acceleration': 'd²p/dt²'})
    fig.update_layout(hovermode='x unified')
    save_html(fig, f'22_acceleration_{locus}.html')
    
    # PNG
    fig_png, ax = plt.subplots()
    ax.plot(df['Generation'], acc, label=locus)
    ax.set_xlabel('Generation'); ax.set_ylabel('d²p/dt²')
    ax.set_title(f'Acceleration of {locus} Allele Frequency')
    ax.legend(); ax.grid(True)
    save_png(fig_png, f'22_acceleration_{locus}.png')

# ====================================================
# 23. Lag Plot
# ====================================================
lag = max(10, len(df)//1000)
if num_loci >= 2 and len(df) > lag:
    for (l1, l2) in combinations(loci, 2):
        p1 = df[f'p_{l1}'].values
        p2 = df[f'p_{l2}'].values
        lag_df = pd.DataFrame({
            f'p_{l1}(t-{lag})': p1[:-lag],
            f'p_{l2}(t)': p2[lag:],
            'Generation': df['Generation'].values[lag:]
        })
        
        # HTML
        fig = px.scatter(lag_df, x=f'p_{l1}(t-{lag})', y=f'p_{l2}(t)',
                         color='Generation', color_continuous_scale='viridis',
                         title=f'Lag Plot: {l1} leads {l2} by {lag} generations',
                         opacity=0.7)
        fig.update_traces(marker=dict(size=4))
        save_html(fig, f'23_lag_{l1}_lead_{l2}.html')
        
        # PNG
        fig_png, ax = plt.subplots()
        sc = ax.scatter(p1[:-lag], p2[lag:], c=df['Generation'].values[lag:],
                        cmap='viridis', s=10, alpha=0.6)
        plt.colorbar(sc, label='Generation')
        ax.set_xlabel(f'p_{l1}(t-{lag})'); ax.set_ylabel(f'p_{l2}(t)')
        ax.set_title(f'Lag Plot: {l1} leads {l2} by {lag} gen.')
        ax.grid(True)
        save_png(fig_png, f'23_lag_{l1}_lead_{l2}.png')

# ====================================================
# 24. α_X vs p_Y (Marginal fitness influence)
# ====================================================
if num_loci >= 2:
    for (l1, l2) in combinations(loci, 2):
        # HTML - α_l2 vs p_l1
        fig = px.scatter(df_sample, x=f'p_{l1}', y=f'alpha_{l2}',
                         color='Generation', color_continuous_scale='plasma',
                         title=f'Selection on {l2} vs frequency of {l1}',
                         labels={f'p_{l1}': f'p_{l1}', f'alpha_{l2}': f'α_{l2}'})
        fig.update_traces(marker=dict(size=4, opacity=0.8))
        save_html(fig, f'24_alpha_{l2}_vs_p_{l1}.html')
        
        # PNG - α_l2 vs p_l1
        fig_png, ax = plt.subplots()
        sc = ax.scatter(df[f'p_{l1}'], df[f'alpha_{l2}'],
                        c=df['Generation'], cmap='plasma', s=10, alpha=0.7)
        plt.colorbar(sc, label='Generation')
        ax.set_xlabel(f'p_{l1}'); ax.set_ylabel(f'α_{l2}')
        ax.set_title(f'Selection on {l2} vs frequency of {l1}')
        ax.grid(True)
        save_png(fig_png, f'24_alpha_{l2}_vs_p_{l1}.png')
        
        # HTML - α_l1 vs p_l2
        fig = px.scatter(df_sample, x=f'p_{l2}', y=f'alpha_{l1}',
                         color='Generation', color_continuous_scale='plasma',
                         title=f'Selection on {l1} vs frequency of {l2}',
                         labels={f'p_{l2}': f'p_{l2}', f'alpha_{l1}': f'α_{l1}'})
        fig.update_traces(marker=dict(size=4, opacity=0.8))
        save_html(fig, f'24_alpha_{l1}_vs_p_{l2}.html')
        
        # PNG - α_l1 vs p_l2
        fig_png, ax = plt.subplots()
        sc = ax.scatter(df[f'p_{l2}'], df[f'alpha_{l1}'],
                        c=df['Generation'], cmap='plasma', s=10, alpha=0.7)
        plt.colorbar(sc, label='Generation')
        ax.set_xlabel(f'p_{l2}'); ax.set_ylabel(f'α_{l1}')
        ax.set_title(f'Selection on {l1} vs frequency of {l2}')
        ax.grid(True)
        save_png(fig_png, f'24_alpha_{l1}_vs_p_{l2}.png')

# ====================================================
# 25. Adaptive Flux Φ = α * Var(has_dominant)
# ====================================================
for locus_idx, locus in enumerate(loci):
    flux = np.zeros(len(df_sample))
    for i, (_, row) in enumerate(df_sample.iterrows()):
        mean_I = 0.0
        mean_I2 = 0.0
        for phen_col in pheno_cols:
            phen = phen_col[2:]
            has_dom = 1 if phen[locus_idx].isupper() else 0
            freq = row[phen_col]
            mean_I += freq * has_dom
            mean_I2 += freq * has_dom
        var_I = mean_I2 - mean_I**2
        flux[i] = row[f'alpha_{locus}'] * var_I
    
    # HTML
    temp_df = pd.DataFrame({'Generation': df_sample['Generation'], 'Adaptive_Flux': flux})
    fig = px.line(temp_df, x='Generation', y='Adaptive_Flux',
                  title=f'Adaptive Flux for {locus}',
                  labels={'Adaptive_Flux': 'Φ'})
    fig.update_layout(hovermode='x unified')
    save_html(fig, f'25_adaptive_flux_{locus}.html')
    
    # PNG
    flux_full = np.zeros(len(df))
    for i, (_, row) in enumerate(df.iterrows()):
        mean_I = 0.0
        mean_I2 = 0.0
        for phen_col in pheno_cols:
            phen = phen_col[2:]
            has_dom = 1 if phen[locus_idx].isupper() else 0
            freq = row[phen_col]
            mean_I += freq * has_dom
            mean_I2 += freq * has_dom
        var_I = mean_I2 - mean_I**2
        flux_full[i] = row[f'alpha_{locus}'] * var_I
    fig_png, ax = plt.subplots()
    ax.plot(df['Generation'], flux_full, label=locus)
    ax.set_xlabel('Generation'); ax.set_ylabel('Adaptive Flux Φ')
    ax.set_title(f'Adaptive Flux for {locus}')
    ax.legend(); ax.grid(True)
    save_png(fig_png, f'25_adaptive_flux_{locus}.png')

# ====================================================
# 26. Phenotype Contributions to Mean Performance
# ====================================================
if all(f'epsilon_{l}' in df.columns for l in loci):
    for locus_idx, locus in enumerate(loci):
        # HTML
        contrib = np.zeros((len(df_sample), len(pheno_cols)))
        for i, (_, row) in enumerate(df_sample.iterrows()):
            mu0 = row[f'mu0_{locus}']
            eps = row[f'epsilon_{locus}']
            for j, phen_col in enumerate(pheno_cols):
                phen = phen_col[2:]
                has_dom = 1 if phen[locus_idx].isupper() else 0
                perf = mu0 + has_dom * eps
                contrib[i, j] = row[phen_col] * perf
        fig = go.Figure()
        for j, phen_col in enumerate(pheno_cols):
            fig.add_trace(go.Scatter(
                x=df_sample['Generation'], y=contrib[:, j],
                name=phen_col[2:],
                mode='lines',
                stackgroup='one'
            ))
        fig.update_layout(
            title=f'Phenotype Contributions to Mean Performance of {locus}',
            xaxis_title='Generation',
            yaxis_title='Contribution to μ̄',
            hovermode='x unified'
        )
        save_html(fig, f'26_phenotype_contributions_{locus}.html')
        
        # PNG
        contrib_full = np.zeros((len(df), len(pheno_cols)))
        for i, (_, row) in enumerate(df.iterrows()):
            mu0 = row[f'mu0_{locus}']
            eps = row[f'epsilon_{locus}']
            for j, phen_col in enumerate(pheno_cols):
                phen = phen_col[2:]
                has_dom = 1 if phen[locus_idx].isupper() else 0
                perf = mu0 + has_dom * eps
                contrib_full[i, j] = row[phen_col] * perf
        fig_png, ax = plt.subplots()
        ax.stackplot(df['Generation'], *[contrib_full[:, j] for j in range(len(pheno_cols))],
                     labels=[pc[2:] for pc in pheno_cols], alpha=0.8)
        ax.set_xlabel('Generation'); ax.set_ylabel('Contribution to μ̄')
        ax.set_title(f'Phenotype Contributions to Mean Performance of {locus}')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
        save_png(fig_png, f'26_phenotype_contributions_{locus}.png')

# ====================================================
# 27. C_bar vs p_X
# ====================================================
for locus in loci:
    # HTML
    fig = px.scatter(df_sample, x='C_bar', y=f'p_{locus}',
                     color='Generation', color_continuous_scale='viridis',
                     title=f'Fitness vs Allele Frequency of {locus}',
                     labels={'C_bar': 'Mean Fitness (C̄)', f'p_{locus}': f'p_{locus}'},
                     opacity=0.7)
    fig.update_traces(marker=dict(size=4))
    save_html(fig, f'27_Cbar_vs_p_{locus}.html')
    
    # PNG
    fig_png, ax = plt.subplots()
    sc = ax.scatter(df['C_bar'], df[f'p_{locus}'],
                    c=df['Generation'], cmap='viridis', s=10, alpha=0.7)
    plt.colorbar(sc, label='Generation')
    ax.set_xlabel('Mean Fitness (C̄)'); ax.set_ylabel(f'p_{locus}')
    ax.set_title(f'Fitness vs Allele Frequency of {locus}')
    ax.grid(True)
    save_png(fig_png, f'27_Cbar_vs_p_{locus}.png')

# ====================================================
# 28. Selection Response: Δp vs α
# ====================================================
for locus in loci:
    delta_p = np.diff(df[f'p_{locus}'].values, prepend=np.nan)
    resp_df = pd.DataFrame({
        'alpha': df[f'alpha_{locus}'],
        'delta_p': delta_p,
        'Generation': df['Generation']
    })
    
    # HTML
    fig = px.scatter(resp_df, x='alpha', y='delta_p',
                     color='Generation', color_continuous_scale='viridis',
                     title=f'Selection Response: Δp vs α for {locus}',
                     labels={'alpha': f'α_{locus}', 'delta_p': f'Δp_{locus}'},
                     opacity=0.7)
    fig.update_traces(marker=dict(size=4))
    save_html(fig, f'28_response_delta_p_vs_alpha_{locus}.html')
    
    # PNG
    fig_png, ax = plt.subplots()
    sc = ax.scatter(df[f'alpha_{locus}'], delta_p,
                    c=df['Generation'], cmap='viridis', s=10, alpha=0.7)
    plt.colorbar(sc, label='Generation')
    ax.set_xlabel(f'α_{locus}'); ax.set_ylabel(f'Δp_{locus}')
    ax.set_title(f'Selection Response: Δp vs α for {locus}')
    ax.grid(True)
    save_png(fig_png, f'28_response_delta_p_vs_alpha_{locus}.png')

# ====================================================
# 29. Total Selection Pressure (Σα) vs Mean Fitness
# ====================================================
total_alpha = df_sample[[f'alpha_{l}' for l in loci]].sum(axis=1)
total_alpha_full = df[[f'alpha_{l}' for l in loci]].sum(axis=1)

# HTML
fig = px.scatter(x=total_alpha, y=df_sample['C_bar'],
                 color=df_sample['Generation'], color_continuous_scale='viridis',
                 title='Total Selection Pressure vs Mean Fitness',
                 labels={'x': 'Total Selection Pressure (Σα)', 'y': 'Mean Fitness (C̄)'},
                 opacity=0.6)
fig.update_traces(marker=dict(size=4))
save_html(fig, '29_total_alpha_vs_Cbar.html')

# PNG
fig_png, ax = plt.subplots()
ax.plot(total_alpha_full, df['C_bar'], 'o', markersize=2, alpha=0.5)
ax.set_xlabel('Total selection pressure (Σα)'); ax.set_ylabel('Mean Fitness (C̄)')
ax.set_title('Total Selection vs Mean Fitness')
ax.grid(True)
save_png(fig_png, '29_total_alpha_vs_Cbar.png')

print(f"\n✅ All plots saved!")
print(f"   HTML (interactive): {output_dir_html}/")
print(f"   PNG (static): {output_dir_png}/")
print(f"   Total: 29 main plots + variants for all locus combinations")
