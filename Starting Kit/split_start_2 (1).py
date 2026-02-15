import numpy as np
import matplotlib.pyplot as plt
import json
import os
from matplotlib.colors import LogNorm

# ==========================================
# 1. DATA LOADING & PRE-PROCESSING
# ==========================================
print("Loading data...")
X = np.load("trainValTest_N=2197469_prod2_features_model=efficientnetB3_tsfrm=dataset_labeledOnly.npy")
metadata = np.load('trainValTest_N=2197469_prod2_metadata_labeledOnly.npz')['arr_0']
labels_data = np.load('trainValTest_N=2197469_prod2_nObs=4242_labels_labeledOnly.npz')
labels = labels_data['y']
angles = metadata[0]

with open('class_type_mapper.json', 'r') as f:
    name2id = json.load(f)
id2name = {v: k for k, v in name2id.items()}

# ==========================================
# 2. VERIFICATION & SAVING FUNCTIONS
# ==========================================
def verify_global_angle_isolation(a_train, a_p1, a_p2):
    """Prints a global report checking if any training angles appear in testing sets."""
    tr_set, p1_set, p2_set = set(a_train), set(a_p1), set(a_p2)
    print("\n" + "="*60)
    print("GLOBAL ANGLE ISOLATION VERIFICATION (ALL SPECIES)")
    print("="*60)
    
    for name, test_set in [("Phase 1", p1_set), ("Phase 2", p2_set)]:
        leak = tr_set.intersection(test_set)
        if not leak:
            print(f"✅ {name}: SUCCESS. No training angles found.")
        else:
            print(f"❌ {name}: FAILURE. {len(leak)} angles leaked: {sorted(list(leak))}")
    
    print("-" * 60)
    print(f"Unique Angles -> Train: {len(tr_set)} | P1: {len(p1_set)} | P2: {len(p2_set)}")
    print("="*60 + "\n")

def analyze_data_distribution(y, angles, u_lbls):
    """Calculates counts per species per angle for heatmap plotting."""
    dist_matrix = np.zeros((len(u_lbls), 31))
    id_to_idx = {int(lbl): i for i, lbl in enumerate(u_lbls)}
    for label, angle in zip(y, angles):
        if int(label) in id_to_idx:
            dist_matrix[id_to_idx[int(label)]][int(angle)] += 1
    return dist_matrix

def save_split_data(output_dir, datasets):
    """Saves splits as .npy and .json files."""
    for folder, feat, lab, ang, key in datasets:
        path = os.path.join(output_dir, folder)
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, 'X.npy'), feat)
        np.save(os.path.join(path, 'angles.npy'), ang)
        with open(os.path.join(path, 'y.json'), 'w') as f:
            json.dump({key: [int(l) for l in lab]}, f)

# ==========================================
# 3. SPECIALIZED DUMMY CREATION
# ==========================================
def create_starting_kit_dummy(output_dir, X_train, y_train, a_train, X_p1, y_p1, a_p1):
    """Creates a small dummy set with strict angle isolation and special species rules."""
    print("Creating specialized starting kit dummy...")
    dummy_dir = os.path.join(output_dir, 'starting_kit_dummy')
    dt_X, dt_y, dt_a = [], [], []
    dv_X, dv_y, dv_a = [], [], []

    special_species = ["Coleoptere", "Chenille", "Diptere"]
    
    for lbl in np.unique(y_train):
        name = id2name.get(lbl, "Unknown")
        idx_tr = np.where(y_train == lbl)[0]
        u_angs_tr = np.unique(a_train[idx_tr])
        
        if name in special_species:
            np.random.shuffle(idx_tr)
            subset_tr = idx_tr[:max(5, int(len(idx_tr) * 0.05))]
            dt_X.append(X_train[subset_tr]); dt_y.append(y_train[subset_tr]); dt_a.append(a_train[subset_tr])
            
            idx_p1 = np.where(y_p1 == lbl)[0]
            if len(idx_p1) > 0:
                np.random.shuffle(idx_p1)
                subset_p1 = idx_p1[:max(2, int(len(idx_p1) * 0.02))]
                dv_X.append(X_p1[subset_p1]); dv_y.append(y_p1[subset_p1]); dv_a.append(a_p1[subset_p1])
        else:
            if len(u_angs_tr) >= 2:
                np.random.shuffle(u_angs_tr)
                te_ang, tr_angs = u_angs_tr[0], u_angs_tr[1:]
                tr_idx = np.where((y_train == lbl) & (np.isin(a_train, tr_angs)))[0]
                te_idx = np.where((y_train == lbl) & (a_train == te_ang))[0]
                dt_X.append(X_train[tr_idx[:50]]); dt_y.append(y_train[tr_idx[:50]]); dt_a.append(a_train[tr_idx[:50]])
                dv_X.append(X_train[te_idx[:10]]); dv_y.append(y_train[te_idx[:10]]); dv_a.append(a_train[te_idx[:10]])

    dummy_data = [
        ('dummy_train', np.vstack(dt_X), np.concatenate(dt_y), np.concatenate(dt_a), 'y_train'),
        ('dummy_test', np.vstack(dv_X), np.concatenate(dv_y), np.concatenate(dv_a), 'y_test')
    ]
    save_split_data(dummy_dir, dummy_data)
    return dummy_data

# ==========================================
# 4. CORE SPLITTING LOGIC
# ==========================================
def split_label_multiphase(X, y, angles, label):
    trainMask, phase1Mask, phase2Mask = [np.zeros(len(X), dtype=bool) for _ in range(3)]
    label_name = id2name.get(label, "Unknown")
    idx_all = np.where(y == label)[0]
    u_angs, counts = np.unique(angles[idx_all], return_counts=True)
    ang_stats = sorted(zip(u_angs.astype(int), counts), key=lambda x: x[1], reverse=True)
    
    bottom_3_angs = []
    main_stats = ang_stats
    if label_name == "Syrphe" and len(ang_stats) > 3:
        bottom_3_angs = [a[0] for a in ang_stats[-3:]]
        main_stats = ang_stats[:-3]

    if len(main_stats) >= 3:
        tr_angs = [main_stats[i][0] for i in range(len(main_stats)) if i % 3 == 0] + bottom_3_angs
        p1_angs = [main_stats[i][0] for i in range(len(main_stats)) if i % 3 == 1]
        p2_angs = [main_stats[i][0] for i in range(len(main_stats)) if i % 3 == 2]
    else:
        tr_angs = [main_stats[0][0]] + bottom_3_angs
        p1_angs = p2_angs = [main_stats[1][0]]

    idx_tr = np.where((y == label) & (np.isin(angles, tr_angs)))[0]
    idx_p1_base = np.where((y == label) & (np.isin(angles, p1_angs)))[0]
    idx_p2_base = np.where((y == label) & (np.isin(angles, p2_angs)))[0]

    if len(main_stats) < 3:
        np.random.shuffle(idx_p1_base); mid = len(idx_p1_base) // 2
        p1_idx, p2_idx = idx_p1_base[:mid], idx_p1_base[mid:]
    else:
        p1_idx, p2_idx = idx_p1_base, idx_p2_base

    target_p = min(len(p1_idx), len(p2_idx)) if (len(p1_idx) > 0 and len(p2_idx) > 0) else 1
    target_tr = target_p * 3
    np.random.shuffle(idx_tr); trainMask[idx_tr[:target_tr]] = True
    np.random.shuffle(p1_idx); phase1Mask[p1_idx[:target_p]] = True
    np.random.shuffle(p2_idx); phase2Mask[p2_idx[:target_p]] = True
    return trainMask, phase1Mask, phase2Mask

def split_data_multiphase(X, y, angles):
    autre_id = next((k for k, v in id2name.items() if v == "Autre"), None)
    mask = (y != 0) & (y != autre_id)
    X_f, y_f, ang_f = X[mask], y[mask], angles[mask]
    u_classes = np.unique(y_f)
    labels_kept = [lbl for lbl in u_classes if len(np.unique(ang_f[y_f == lbl])) >= 2]
    final_m = np.isin(y_f, labels_kept)
    X_f, y_f, ang_f = X_f[final_m], y_f[final_m], ang_f[final_m]
    t_m, p1_m, p2_m = [np.zeros(len(X_f), bool) for _ in range(3)]
    for lbl in labels_kept:
        t, p1, p2 = split_label_multiphase(X_f, y_f, ang_f, lbl)
        t_m |= t; p1_m |= p1; p2_m |= p2
    return (X_f[t_m], X_f[p1_m], X_f[p2_m], y_f[t_m], y_f[p1_m], y_f[p2_m], 
            ang_f[t_m], ang_f[p1_m], ang_f[p2_m], X_f, y_f, ang_f)

# ==========================================
# 5. EXECUTION & VISUALIZATION
# ==========================================
if __name__ == "__main__":
    np.random.seed(42)
    output_dir = './split_data_final'
    os.makedirs(output_dir, exist_ok=True)

    # 1. Main Splits
    (X_tr, X_p1, X_p2, y_tr, y_p1, y_p2, a_tr, a_p1, a_p2, X_flt, y_flt, a_flt) = split_data_multiphase(X, labels, angles)
    
    # 2. Global Isolation Print
    verify_global_angle_isolation(a_tr, a_p1, a_p2)

    save_split_data(output_dir, [('train', X_tr, y_tr, a_tr, 'y_train'), 
                                 ('phase1', X_p1, y_p1, a_p1, 'y_test'), 
                                 ('phase2', X_p2, y_p2, a_p2, 'y_test')])

    # 3. Dummy Kit
    create_starting_kit_dummy(output_dir, X_tr, y_tr, a_tr, X_p1, y_p1, a_p1)

    # 4. Final Visual Report
    u_lbls = np.unique(y_flt)
    species_names = [id2name.get(int(l), str(l)) for l in u_lbls]
    fig, axes = plt.subplots(2, 4, figsize=(28, 14))
    x_pos = np.arange(len(u_lbls))
    
    axes[0, 0].bar(x_pos - 0.2, [np.sum(y_tr == l) for l in u_lbls], 0.2, label='Train')
    axes[0, 0].bar(x_pos, [np.sum(y_p1 == l) for l in u_lbls], 0.2, label='Phase 1')
    axes[0, 0].bar(x_pos + 0.2, [np.sum(y_p2 == l) for l in u_lbls], 0.2, label='Phase 2')
    axes[0, 0].set_xticks(x_pos); axes[0, 0].set_xticklabels(species_names, rotation=45, ha='right'); axes[0, 0].legend()
    
    dists = [(analyze_data_distribution(y_flt, a_flt, u_lbls), "Original"),
             (analyze_data_distribution(y_tr, a_tr, u_lbls), "Train"),
             (analyze_data_distribution(y_p1, a_p1, u_lbls), "Phase 1"),
             (analyze_data_distribution(y_p2, a_p2, u_lbls), "Phase 2")]

    for i, (mat, title) in enumerate(dists):
        im = axes[1, i].imshow(mat, aspect='auto', cmap='viridis', norm=LogNorm(vmin=1, vmax=np.max(mat)+1))
        axes[1, i].set_title(title); axes[1, i].set_xlabel("Angle ID")
        if i == 0: axes[1, i].set_yticks(range(len(species_names))); axes[1, i].set_yticklabels(species_names, fontsize=8)
        else: axes[1, i].set_yticks([])
        plt.colorbar(im, ax=axes[1, i])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'final_distribution_report.png'))
    plt.show()