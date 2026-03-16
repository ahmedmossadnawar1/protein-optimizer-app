"""
PlantProtein AI — ML Backend
Random Forest + Scipy SLSQP Optimization
Usage: python ml_model.py
"""
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import json, warnings
warnings.filterwarnings('ignore')

AMINO_ACIDS = ['Tryptophan','Threonine','Isoleucine','Leucine',
               'Lysine','Methionine','Phenylalanine','Valine','Histidine']

WHO_REF = {'Tryptophan':7,'Threonine':23,'Isoleucine':30,'Leucine':59,
           'Lysine':45,'Methionine':16,'Phenylalanine':38,'Valine':39,'Histidine':15}

EGG_REF = {'Tryptophan':0.153,'Threonine':0.604,'Isoleucine':0.671,
           'Leucine':1.086,'Lysine':0.904,'Methionine':0.392,
           'Phenylalanine':0.668,'Valine':0.767,'Histidine':0.298}


def load_and_prepare(filepath):
    df = pd.read_excel(filepath, sheet_name='essential amino acid')
    df.columns = ['food_group','food','amino_acid','qty']
    pivot = df.pivot_table(
        index=['food_group','food'], columns='amino_acid',
        values='qty', aggfunc='mean'
    ).reset_index()
    pivot.columns.name = None
    return pivot.dropna()


def compute_quality_score(row):
    """PDCAAS-inspired quality score vs WHO reference."""
    vals = np.array([row[aa] for aa in AMINO_ACIDS])
    total = vals.sum()
    if total == 0: return 0
    ref = np.array([WHO_REF[aa]*total/1000 for aa in AMINO_ACIDS])
    return float(np.mean(np.minimum(vals/(ref+1e-9), 1.0)) * 100)


def train_model(pivot):
    """Train Random Forest on amino acid features → quality score."""
    pivot['score'] = pivot.apply(compute_quality_score, axis=1)
    X = pivot[AMINO_ACIDS].values
    y = pivot['score'].values
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        'r2': float(r2_score(y_test, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
        'train_size': int(len(X_train)),
        'test_size': int(len(X_test))
    }
    return model, scaler, metrics


def optimize_blend(candidate_df, n_pick=5, total_grams=100):
    """SLSQP optimization to maximize cosine similarity with egg protein."""
    foods = candidate_df.nlargest(n_pick, 'predicted_score').reset_index(drop=True)
    n = len(foods)
    if n < 2: return None
    egg_vec = np.array([EGG_REF[aa] for aa in AMINO_ACIDS])

    def objective(weights):
        mix = sum(w/100 * foods.iloc[i][AMINO_ACIDS].values.astype(float)
                  for i,w in enumerate(weights))
        return -(np.dot(mix,egg_vec) /
                 (np.linalg.norm(mix)*np.linalg.norm(egg_vec)+1e-9))

    res = minimize(
        objective, np.ones(n)*total_grams/n,
        method='SLSQP',
        bounds=[(2, total_grams)]*n,
        constraints=[{'type':'eq','fun':lambda w:np.sum(w)-total_grams}],
        options={'ftol':1e-9,'maxiter':1000}
    )
    weights = res.x
    mix_profile = sum(weights[i]/100*foods.iloc[i][AMINO_ACIDS].values.astype(float)
                      for i in range(n))
    cos_sim = float(np.dot(mix_profile,egg_vec) /
                    (np.linalg.norm(mix_profile)*np.linalg.norm(egg_vec)+1e-9))
    return {
        'foods': [{'food':foods.iloc[i]['food'],'group':foods.iloc[i]['food_group'],
                   'grams':round(float(weights[i]),1)}
                  for i in range(n) if weights[i]>0.5],
        'mix_profile': {aa:round(float(mix_profile[j]),3) for j,aa in enumerate(AMINO_ACIDS)},
        'egg_similarity': round(cos_sim*100,1),
        'total_grams': total_grams
    }


def run_pipeline(filepath):
    print("📊 Loading data...")
    pivot = load_and_prepare(filepath)
    print(f"   {len(pivot)} foods · {pivot['food_group'].nunique()} groups")

    print("🧠 Training Random Forest (80/20 split)...")
    model, scaler, metrics = train_model(pivot)
    pivot['predicted_score'] = model.predict(scaler.transform(pivot[AMINO_ACIDS].values))
    print(f"   R²={metrics['r2']:.3f} | RMSE={metrics['rmse']:.3f}")

    print("⚗️  Optimizing blends (SLSQP)...")
    blends = {
        'cross_category': optimize_blend(
            pd.concat([pivot[pivot['food_group']==g].nlargest(3,'predicted_score')
                       for g in pivot['food_group'].unique()])),
        'legumes_nuts': optimize_blend(
            pivot[pivot['food_group'].isin(['16 Legumes and Legume Products','Nuts and Seeds'])].nlargest(12,'predicted_score')),
        'vegetables_seeds': optimize_blend(
            pivot[pivot['food_group'].isin(['Vegetables','Seeds'])].nlargest(12,'predicted_score')),
        'consumer_friendly': optimize_blend(
            pivot[pivot['food_group'].isin(['Fruits','Vegetables','16 Legumes and Legume Products'])].nlargest(10,'predicted_score'), n_pick=4)
    }

    for name, blend in blends.items():
        if blend:
            print(f"   [{name}] {blend['egg_similarity']}% egg similarity")

    results = {
        'model_metrics': metrics,
        'amino_acids': AMINO_ACIDS,
        'egg_reference': EGG_REF,
        'who_reference': WHO_REF,
        'feature_importance': {aa:round(float(imp),4) for aa,imp in
                               zip(AMINO_ACIDS,model.feature_importances_)},
        'top_foods': pivot.nlargest(10,'predicted_score')[
            ['food','food_group','predicted_score']].round(2).to_dict('records'),
        'blends': blends
    }
    with open('ml_results.json','w') as f:
        json.dump(results, f, indent=2)
    print("✅ Results saved to ml_results.json")
    return results


if __name__ == '__main__':
    import sys
    filepath = sys.argv[1] if len(sys.argv)>1 else 'merged.xlsx'
    run_pipeline(filepath)
