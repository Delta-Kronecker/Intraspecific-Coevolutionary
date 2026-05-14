import sys
import json
import csv
from decimal import Decimal, getcontext
from itertools import product
from tqdm import tqdm

# ====================================================
# Load configuration from JSON file or interactive input
# ====================================================

def load_params_from_json(json_file):
    with open(json_file, 'r') as f:
        params = json.load(f)
    
    return {
        'NUM_GENERATIONS': params.get('num_generations', 500),
        'TOTAL_POP': params.get('total_pop', 10000),
        'SAVE_INTERVAL': params.get('save_interval', 100),
        'C0': params.get('c0', '1.0'),
        'LAMBDA': params.get('lambda_rate', '0.8'),
        'LOCI': [l.strip() for l in params.get('loci_names', 'A,B').split(',')],
        'DECIMAL_PRECISION': params.get('decimal_precision', 30),
        'LOCI_PARAMS': params.get('loci_params', {})
    }

def get_params_interactive():
    print("=" * 60)
    print("Multi-locus co-evolution simulation setup")
    print("=" * 60)
    
    prec = int(input("Decimal precision: ") or "30")
    getcontext().prec = prec
    
    print("\n--- Basic Parameters ---")
    NUM_GENERATIONS = int(input("Number of generations: ") or "500")
    TOTAL_POP = int(input("Total initial population size: ") or "10000")
    SAVE_INTERVAL = int(input("Save interval (e.g., 100): ") or "100")
    
    print("\n--- Fitness Parameters ---")
    C0 = input("Base fitness C0 (non-selective survival): ") or "1.0"
    LAMBDA = input("Overall survival rate λ (e.g., 0.5): ") or "0.8"
    
    print("\n--- Loci Configuration ---")
    num_loci = int(input("Number of loci: ") or "2")
    LOCI = []
    for i in range(num_loci):
        name = input(f"Name of locus {i+1} (one letter, e.g., G): ").strip()
        while len(name) != 1:
            name = input("Please enter only one letter: ").strip()
        LOCI.append(name.upper())
    
    print("\n--- Locus-Specific Parameters ---")
    LOCI_PARAMS = {}
    for locus in LOCI:
        print(f"\n===== Locus {locus} =====")
        eps = input(f"  Additive advantage ε (benefit of dominant allele): ") or "0.1"
        alpha = input(f"  Initial selection pressure α: ") or "0.05"
        mu0 = input(f"  Initial baseline performance μ0: ") or "0.0"
        
        num_events = int(input(f"  Number of mutation events (0 if none): ") or "0")
        events = []
        for e in range(num_events):
            print(f"  Event {e+1}:")
            count = int(input(f"    Number of mutant individuals: "))
            gen = int(input(f"    Generation of occurrence: "))
            events.append([count, gen])
        
        LOCI_PARAMS[locus] = {
            'epsilon': float(eps),
            'alpha': float(alpha),
            'mu0': float(mu0),
            'mutations': events
        }
    
    return {
        'NUM_GENERATIONS': NUM_GENERATIONS,
        'TOTAL_POP': TOTAL_POP,
        'SAVE_INTERVAL': SAVE_INTERVAL,
        'C0': C0,
        'LAMBDA': LAMBDA,
        'LOCI': LOCI,
        'DECIMAL_PRECISION': prec,
        'LOCI_PARAMS': LOCI_PARAMS
    }

# ====================================================
# Load parameters (from JSON if provided, else interactive)
# ====================================================

if len(sys.argv) > 1 and sys.argv[1].endswith('.json'):
    print("Loading parameters from JSON file...")
    config = load_params_from_json(sys.argv[1])
else:
    config = get_params_interactive()

# Apply configuration
NUM_GENERATIONS = config['NUM_GENERATIONS']
TOTAL_POP = config['TOTAL_POP']
SAVE_INTERVAL = config['SAVE_INTERVAL']
C0 = Decimal(str(config['C0']))
LAMBDA = Decimal(str(config['LAMBDA']))
LOCI = config['LOCI']
DECIMAL_PRECISION = config['DECIMAL_PRECISION']
LOCI_PARAMS = config['LOCI_PARAMS']

getcontext().prec = DECIMAL_PRECISION

# Extract locus-specific parameters
EPSILON = {}
INIT_ALPHA = {}
INIT_MU0 = {}
MUTATION_EVENTS = {}

for locus in LOCI:
    lp = LOCI_PARAMS.get(locus, {})
    EPSILON[locus] = Decimal(str(lp.get('epsilon', 0.1)))
    INIT_ALPHA[locus] = Decimal(str(lp.get('alpha', 0.05)))
    INIT_MU0[locus] = Decimal(str(lp.get('mu0', 0.0)))
    MUTATION_EVENTS[locus] = lp.get('mutations', [])

print(f"\n✅ Configuration loaded: {NUM_GENERATIONS} generations, {len(LOCI)} loci: {LOCI}")

# ====================================================
# Simulation core (unchanged from original)
# ====================================================

states_per_locus = []
for locus in LOCI:
    dom = locus.upper()
    rec = locus.lower()
    states_per_locus.append([dom*2, dom+rec, rec*2])

ALL_GENOTYPES = [''.join(combo) for combo in product(*states_per_locus)]

def has_dominant(genotype_str, locus_idx):
    start = locus_idx * 2
    state = genotype_str[start:start+2]
    rec_state = LOCI[locus_idx].lower() * 2
    return 0 if state == rec_state else 1

phenotype_variants = []
for bits in product([0,1], repeat=len(LOCI)):
    pheno = ''
    for i, bit in enumerate(bits):
        pheno += LOCI[i] if bit else LOCI[i].lower()
    phenotype_variants.append(pheno)

def genotype_to_phenotype(genotype_str):
    pheno = ''
    for i, locus in enumerate(LOCI):
        if has_dominant(genotype_str, i) == 1:
            pheno += locus
        else:
            pheno += locus.lower()
    return pheno

def make_mutant_genotype(locus_idx):
    mut = ''
    for j, loc in enumerate(LOCI):
        if j == locus_idx:
            mut += loc.upper() + loc.lower()
        else:
            mut += loc.lower() * 2
    return mut

def get_initial_population():
    N_i = {g: Decimal('0') for g in ALL_GENOTYPES}
    all_rec = ''.join([locus.lower()*2 for locus in LOCI])
    N_i[all_rec] = Decimal(str(TOTAL_POP))
    return N_i

class SimulationState:
    def __init__(self, generation, N_total):
        self.generation = generation
        self.N_i = {}
        self.N_Total = N_total
        self.mu0 = {}
        self.alpha = {}
        self.mu_bar = {}
        self.p = {}
        self.C_bar = Decimal('0.0')
        self.f_i = {}
        self.R = {}

    def calculate_frequencies(self):
        if self.N_Total != Decimal('0'):
            for g in ALL_GENOTYPES:
                self.f_i[g] = self.N_i.get(g, Decimal('0')) / self.N_Total
        else:
            for g in ALL_GENOTYPES:
                self.f_i[g] = Decimal('0')

    def to_dict(self):
        self.calculate_frequencies()
        data = {'Generation': self.generation, 'N_Total': str(self.N_Total)}
        data['C0'] = str(C0)

        for locus in LOCI:
            data[f'mu0_{locus}'] = str(self.mu0.get(locus, Decimal('0')))
            data[f'alpha_{locus}'] = str(self.alpha.get(locus, Decimal('0')))
            data[f'mu_bar_{locus}'] = str(self.mu_bar.get(locus, Decimal('0')))
            data[f'p_{locus}'] = str(self.p.get(locus, Decimal('0')))
            data[f'epsilon_{locus}'] = str(EPSILON[locus])
            data[f'R_{locus}'] = str(self.R.get(locus, Decimal('1.0')))
            if self.C_bar != Decimal('0'):
                s = self.alpha[locus] / self.C_bar
            else:
                s = Decimal('0')
            data[f's_{locus}'] = str(s)

        data['C_bar'] = str(self.C_bar)

        phenotype_freqs = {p: Decimal('0') for p in phenotype_variants}
        for g in ALL_GENOTYPES:
            phen = genotype_to_phenotype(g)
            phenotype_freqs[phen] += self.f_i.get(g, Decimal('0'))
        for phen, freq in phenotype_freqs.items():
            data[f'f_{phen}'] = str(freq)

        var_fit = Decimal('0')
        for phen, freq in phenotype_freqs.items():
            c = C0
            for i, locus in enumerate(LOCI):
                if phen[i].isupper():
                    c += self.alpha[locus]
            var_fit += freq * (c - self.C_bar) ** 2
        data['var_fitness'] = str(var_fit)

        return data

def phase_1_analysis(current_state, previous_mu_bar):
    new_alpha = {}
    R_vals = {}
    for i, locus in enumerate(LOCI):
        if current_state.generation == 0:
            mu0_t = INIT_MU0[locus]
            mu_bar_prev = INIT_MU0[locus]
        else:
            mu0_t = previous_mu_bar[locus]
            mu_bar_prev = previous_mu_bar[locus]

        current_state.mu0[locus] = mu0_t

        N_has_X = Decimal('0')
        for genotype_str, count in current_state.N_i.items():
            N_has_X += count * Decimal(has_dominant(genotype_str, i))
        N_no_X = current_state.N_Total - N_has_X

        mu_bar_t = (N_no_X * mu0_t + N_has_X * (mu0_t + EPSILON[locus])) / current_state.N_Total
        current_state.mu_bar[locus] = mu_bar_t

        if mu_bar_prev == Decimal('0'):
            progress_ratio = Decimal('1.0')
        else:
            progress_ratio = mu_bar_t / mu_bar_prev
        R_vals[locus] = progress_ratio

        alpha_t = current_state.alpha[locus]
        new_alpha[locus] = alpha_t * progress_ratio

    return new_alpha, R_vals

def phase_2_reproduction(current_state):
    p_t = {}
    for i, locus in enumerate(LOCI):
        A_X = Decimal('0')
        T_X = Decimal('2') * current_state.N_Total
        for genotype_str, count in current_state.N_i.items():
            start = i*2
            state = genotype_str[start:start+2]
            if state == locus.upper()*2:
                allele_count = Decimal('2')
            elif state == locus.upper()+locus.lower():
                allele_count = Decimal('1')
            else:
                allele_count = Decimal('0')
            A_X += count * allele_count
        p_t[locus] = A_X / T_X
        current_state.p[locus] = p_t[locus]

    N_i_offspring = {}
    for genotype_str in ALL_GENOTYPES:
        f_i = Decimal('1.0')
        for i, locus in enumerate(LOCI):
            p = p_t[locus]
            q = Decimal('1') - p
            start = i*2
            state = genotype_str[start:start+2]
            dom_state = locus.upper()*2
            het_state = locus.upper()+locus.lower()
            if state == dom_state:
                f_i *= (p ** Decimal('2'))
            elif state == het_state:
                f_i *= (Decimal('2') * p * q)
            else:
                f_i *= (q ** Decimal('2'))
        N_i_offspring[genotype_str] = current_state.N_Total * f_i
    return N_i_offspring

def phase_3_selection(current_state, N_i_offspring):
    C_i = {}
    for genotype_str in ALL_GENOTYPES:
        sum_alpha = Decimal('0')
        for i, locus in enumerate(LOCI):
            sum_alpha += current_state.alpha[locus] * Decimal(has_dominant(genotype_str, i))
        C_i[genotype_str] = C0 + sum_alpha

    sum_weighted_C = Decimal('0')
    for genotype_str, count in N_i_offspring.items():
        sum_weighted_C += count * C_i[genotype_str]
    C_bar = sum_weighted_C / current_state.N_Total
    current_state.C_bar = C_bar

    N_i_after_selection = {}
    for genotype_str, count_offspring in N_i_offspring.items():
        N_i_after_selection[genotype_str] = count_offspring * (C_i[genotype_str] / C_bar)
    return N_i_after_selection

def phase_4_dynamics(current_state, N_i_after_selection):
    N_i_combined = {}
    for genotype_str in ALL_GENOTYPES:
        N_i_combined[genotype_str] = current_state.N_i[genotype_str] + N_i_after_selection[genotype_str]
    N_i_next = {}
    N_Total_next = Decimal('0')
    for genotype_str, count_combined in N_i_combined.items():
        N_next = LAMBDA * count_combined
        N_i_next[genotype_str] = N_next
        N_Total_next += N_next
    return N_i_next, N_Total_next

def run_simulation():
    OUTPUT_FILENAME = "results.csv"
    
    N_i_current = get_initial_population()
    N_Total_current = Decimal(str(TOTAL_POP))

    mu_bar_prev = INIT_MU0.copy()
    alpha_current = INIT_ALPHA.copy()

    temp_state = SimulationState(0, N_Total_current)
    temp_state.mu0 = INIT_MU0.copy()
    temp_state.alpha = INIT_ALPHA.copy()
    temp_state.calculate_frequencies()
    fieldnames = list(temp_state.to_dict().keys())

    with open(OUTPUT_FILENAME, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for t in tqdm(range(NUM_GENERATIONS + 1), desc="Simulating"):
            if N_Total_current == Decimal('0'):
                print("Population reached zero. Simulation stopped.")
                break

            # Apply mutations
            for i, locus in enumerate(LOCI):
                for count, gen in MUTATION_EVENTS.get(locus, []):
                    if gen == t:
                        mut_geno = make_mutant_genotype(i)
                        wild_type_geno = ''.join([locus.lower()*2 for locus in LOCI])
                        
                        converted = 0
                        for _ in range(count):
                            if N_i_current[wild_type_geno] > 0:
                                N_i_current[wild_type_geno] -= Decimal('1')
                                N_i_current[mut_geno] += Decimal('1')
                                converted += 1
                            else:
                                break
                        
                        print(f"Gen {t}: Converted {converted} individual(s) from {wild_type_geno} to {mut_geno}")

            current_state = SimulationState(t, N_Total_current)
            current_state.N_i = N_i_current
            current_state.alpha = alpha_current

            alpha_next, R_dict = phase_1_analysis(current_state, mu_bar_prev)
            current_state.R = R_dict
            current_state.alpha = alpha_next

            N_i_offspring = phase_2_reproduction(current_state)
            N_i_after_selection = phase_3_selection(current_state, N_i_offspring)
            N_i_next, N_Total_next = phase_4_dynamics(current_state, N_i_after_selection)

            if t % SAVE_INTERVAL == 0:
                writer.writerow(current_state.to_dict())

            mu_bar_prev = current_state.mu_bar.copy()
            alpha_current = alpha_next
            N_Total_current = N_Total_next
            N_i_current = N_i_next

    print(f"\n✅ Simulation finished successfully. Results saved in '{OUTPUT_FILENAME}'.")

# ====================================================
# Run
# ====================================================

if __name__ == "__main__":
    run_simulation()
