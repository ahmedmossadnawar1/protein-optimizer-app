"""
PlantProtein AI — Flask Backend Server
Aligned with Graduation Project Proposal

Changes vs previous version:
  1. Egg reference now uses g/100g protein (from proposal page 4)
  2. Blend output shows both % AND grams
  3. Protein_Content estimated and shown per food
  4. Output format matches proposal Table (Food | Category | % in Mixture)

Run:  python app.py
Open: http://localhost:5000
"""
import os, warnings
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_file
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings('ignore')

# ── API KEY ───────────────────────────────────────────────────────────────────
os.environ.setdefault('ANTHROPIC_API_KEY', os.environ.get('ANTHROPIC_API_KEY', ''))

app = Flask(__name__)

# ── GLOBAL STATE ──────────────────────────────────────────────────────────────
MODEL  = None
SCALER = None
PIVOT  = None
METRICS = {}
EXTRA_FILES = []  # list of extra excel files merged into training

AMINO_ACIDS = ['Histidine','Isoleucine','Leucine','Lysine',
               'Methionine','Phenylalanine','Threonine','Tryptophan','Valine']

# ── EGG REFERENCE — from proposal page 4 (g per 100g protein) ────────────────
EGG_REF = {
    'Histidine':    2.2,
    'Isoleucine':   5.4,
    'Leucine':      8.6,
    'Lysine':       7.0,
    'Methionine':   3.4,
    'Phenylalanine':5.7,
    'Threonine':    4.7,
    'Tryptophan':   1.6,
    'Valine':       6.6
}

# ── WHO/FAO Minimum requirements (mg per g protein) ──────────────────────────
WHO_REF = {
    'Histidine':15,'Isoleucine':30,'Leucine':59,'Lysine':45,
    'Methionine':16,'Phenylalanine':38,'Threonine':23,'Tryptophan':7,'Valine':39
}

EXCEL_FILE = 'merged.xlsx'

# ── CORS ──────────────────────────────────────────────────────────────────────
@app.after_request
def after_request(r):
    r.headers['Access-Control-Allow-Origin']  = '*'
    r.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    r.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    return r

# ── HELPERS ───────────────────────────────────────────────────────────────────
def estimate_protein_content(row):
    """Estimate total protein g/100g food from sum of amino acids (proxy)."""
    return round(float(sum(row[aa] for aa in AMINO_ACIDS) * 6.25), 1)

def convert_to_per_100g_protein(row):
    """
    Data is in g/100g food.
    Convert to g/100g protein using estimated protein content.
    Formula: (g/100g food) / (protein_content/100)
    """
    protein = estimate_protein_content(row)
    if protein < 0.5:
        return {aa: 0.0 for aa in AMINO_ACIDS}
    return {aa: round(row[aa] / (protein/100), 3) for aa in AMINO_ACIDS}

def compute_quality_score(row):
    """PDCAAS-inspired score: how well this food meets WHO requirements."""
    vals = np.array([row[aa] for aa in AMINO_ACIDS])
    total = vals.sum()
    if total == 0: return 0.0
    ref = np.array([WHO_REF[aa] * total / 1000 for aa in AMINO_ACIDS])
    return float(np.mean(np.minimum(vals / (ref + 1e-9), 1.0)) * 100)

def cosine_similarity(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0: return 0.0
    return float(np.dot(a, b) / (na * nb))

def optimize_blend(candidate_df, n_pick=5, total_grams=100):
    """
    SLSQP optimization:
    - Maximize cosine similarity of blend AA profile vs egg reference
    - Blend profile computed in g/100g protein space (matches proposal)
    """
    foods = candidate_df.nlargest(n_pick, 'predicted_score').reset_index(drop=True)
    n = len(foods)
    if n < 2: return None

    egg_vec = np.array([EGG_REF[aa] for aa in AMINO_ACIDS])

    def objective(w):
        # weighted sum in g/100g protein space
        mix = np.zeros(len(AMINO_ACIDS))
        total_w = w.sum()
        for i in range(n):
            frac = w[i] / (total_w + 1e-9)
            mix += frac * np.array([foods.iloc[i][f'p_{aa}'] for aa in AMINO_ACIDS])
        return -cosine_similarity(mix, egg_vec)

    res = minimize(
        objective,
        np.ones(n) * total_grams / n,
        method='SLSQP',
        bounds=[(1, total_grams)] * n,
        constraints=[{'type':'eq','fun': lambda w: w.sum() - total_grams}],
        options={'ftol':1e-9, 'maxiter':1000}
    )
    w = res.x
    total_w = w.sum()

    # Compute blend profile (g/100g protein) — weighted average
    mix_profile_protein = np.zeros(len(AMINO_ACIDS))
    for i in range(n):
        frac = w[i] / (total_w + 1e-9)
        mix_profile_protein += frac * np.array([foods.iloc[i][f'p_{aa}'] for aa in AMINO_ACIDS])

    # Also compute in g/100g food for display
    mix_profile_food = np.zeros(len(AMINO_ACIDS))
    for i in range(n):
        mix_profile_food += (w[i]/100) * foods.iloc[i][AMINO_ACIDS].values.astype(float)

    sim = cosine_similarity(mix_profile_protein, egg_vec)

    result_foods = []
    for i in range(n):
        if w[i] > 0.5:
            pct = round(float(w[i] / total_w * 100), 1)
            result_foods.append({
                'food':     foods.iloc[i]['food'],
                'category': foods.iloc[i]['food_group'],
                'grams':    round(float(w[i]), 1),
                'percentage': pct,                        # ← % as in proposal
                'protein_content': foods.iloc[i]['protein_content']
            })
    result_foods.sort(key=lambda x: -x['percentage'])

    # AA comparison table (g/100g protein — matches proposal units)
    aa_comparison = {}
    for j, aa in enumerate(AMINO_ACIDS):
        mv = round(float(mix_profile_protein[j]), 2)
        ev = EGG_REF[aa]
        aa_comparison[aa] = {
            'blend_per100g_protein': mv,
            'egg_per100g_protein':   ev,
            'ratio_pct':             round(mv / ev * 100, 1)
        }

    return {
        'foods':              result_foods,
        'mix_profile':        {aa: round(float(mix_profile_protein[j]),2) for j,aa in enumerate(AMINO_ACIDS)},
        'mix_profile_food':   {aa: round(float(mix_profile_food[j]),3)    for j,aa in enumerate(AMINO_ACIDS)},
        'aa_comparison':      aa_comparison,
        'egg_similarity':     round(sim * 100, 1),
        'total_grams':        total_grams,
        'unit_note':          'AA values in g per 100g protein (as per proposal)'
    }

# ── TRAINING PIPELINE ─────────────────────────────────────────────────────────
def load_excel_to_df(filepath):
    """Load excel file with essential amino acid sheet into long-format df."""
    df = pd.read_excel(filepath, sheet_name='essential amino acid')
    df.columns = ['food_group','food','amino_acid','qty']
    return df

def train_pipeline(base_filepath, extra_filepaths=None):
    global MODEL, SCALER, PIVOT, METRICS, EXTRA_FILES

    # ── 1. Load base data ─────────────────────────────────────────────────────
    df = load_excel_to_df(base_filepath)
    base_count = df['food'].nunique()

    # ── 2. Merge extra files if any ───────────────────────────────────────────
    extra_count = 0
    if extra_filepaths:
        for fp in extra_filepaths:
            if os.path.exists(fp):
                try:
                    extra_df = load_excel_to_df(fp)
                    df = pd.concat([df, extra_df], ignore_index=True)
                    extra_count += extra_df['food'].nunique()
                    print(f"  + Merged: {fp} ({extra_df['food'].nunique()} foods)")
                except Exception as e:
                    print(f"  ⚠ Could not load {fp}: {e}")

    EXTRA_FILES = extra_filepaths or []

    # ── 3. Pivot & clean ──────────────────────────────────────────────────────
    pivot = df.pivot_table(
        index=['food_group','food'], columns='amino_acid',
        values='qty', aggfunc='mean'
    ).reset_index()
    pivot.columns.name = None
    pivot = pivot.dropna(subset=AMINO_ACIDS)

    # Remove duplicate foods (keep mean if same food appears in base + extra)
    pivot = pivot.drop_duplicates(subset=['food'], keep='last').reset_index(drop=True)

    # Add Food-ID
    pivot.insert(0, 'food_id', ['F{:04d}'.format(i+1) for i in range(len(pivot))])

    # Protein content
    pivot['protein_content'] = pivot.apply(estimate_protein_content, axis=1)

    # Per-100g-protein columns
    for aa in AMINO_ACIDS:
        pivot[f'p_{aa}'] = pivot.apply(
            lambda row: row[aa] / (row['protein_content']/100 + 1e-9), axis=1
        ).round(3)

    # Quality score
    pivot['score'] = pivot.apply(compute_quality_score, axis=1)

    # ── 4. Train ML model ─────────────────────────────────────────────────────
    X = pivot[AMINO_ACIDS].values
    y = pivot['score'].values
    scaler = MinMaxScaler()
    X_sc = scaler.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(X_sc, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    pivot['predicted_score'] = model.predict(X_sc)

    MODEL, SCALER, PIVOT = model, scaler, pivot

    METRICS = {
        'r2':               round(float(r2_score(y_te, y_pred)), 3),
        'rmse':             round(float(np.sqrt(mean_squared_error(y_te, y_pred))), 3),
        'train_size':       int(len(X_tr)),
        'test_size':        int(len(X_te)),
        'total_foods':      int(len(pivot)),
        'base_foods':       int(base_count),
        'extra_foods':      int(extra_count),
        'extra_files':      [os.path.basename(f) for f in (extra_filepaths or [])],
        'food_groups':      pivot['food_group'].value_counts().to_dict(),
        'feature_importance': {aa: round(float(v),4) for aa,v in zip(AMINO_ACIDS, model.feature_importances_)},
    }
    print(f"✅ Trained | R²={METRICS['r2']} | {base_count} base + {extra_count} extra = {len(pivot)} total foods")
    return True

# ── API ROUTES ────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_file('index.html')

@app.route('/api/status')
def status():
    return jsonify({'trained': MODEL is not None, 'metrics': METRICS})

@app.route('/api/train', methods=['POST','OPTIONS'])
def train():
    if request.method == 'OPTIONS': return jsonify({}), 200
    filepath = request.json.get('file', EXCEL_FILE)
    if not os.path.exists(filepath):
        return jsonify({'error': f'File not found: {filepath}'}), 404
    try:
        train_pipeline(filepath, EXTRA_FILES if EXTRA_FILES else None)
        return jsonify({'success': True, 'metrics': METRICS})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/add_data', methods=['POST','OPTIONS'])
def add_data():
    """Upload an extra Excel file and retrain with it merged into the base data."""
    if request.method == 'OPTIONS': return jsonify({}), 200
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    f = request.files['file']
    if not f.filename.endswith(('.xlsx','.xls')):
        return jsonify({'error': 'Only .xlsx or .xls files are supported'}), 400
    save_path = os.path.join('extra_data', f.filename)
    os.makedirs('extra_data', exist_ok=True)
    f.save(save_path)
    # verify it has the right sheet
    try:
        test = pd.read_excel(save_path, sheet_name='essential amino acid')
        test.columns = ['food_group','food','amino_acid','qty']
        new_foods = test['food'].nunique()
    except Exception as e:
        os.remove(save_path)
        return jsonify({'error': f'Invalid file format: {e}'}), 400
    # add to extra files and retrain
    if save_path not in EXTRA_FILES:
        EXTRA_FILES.append(save_path)
    train_pipeline(EXCEL_FILE, EXTRA_FILES)
    return jsonify({'success': True, 'new_foods': new_foods, 'metrics': METRICS,
                    'message': f'Added {new_foods} new foods from {f.filename}'})

@app.route('/api/extra_files')
def extra_files():
    """List all extra files currently merged into the model."""
    return jsonify({'files': [{'name': os.path.basename(f), 'path': f} for f in EXTRA_FILES]})

@app.route('/api/remove_extra', methods=['POST','OPTIONS'])
def remove_extra():
    """Remove an extra file from disk and retrain without it."""
    if request.method == 'OPTIONS': return jsonify({}), 200
    filename = request.json.get('filename')
    global EXTRA_FILES
    # find the full path
    full_path = next((f for f in EXTRA_FILES if os.path.basename(f) == filename), None)
    # delete from disk
    if full_path and os.path.exists(full_path):
        os.remove(full_path)
        print(f"🗑 Deleted: {full_path}")
    # remove from list
    EXTRA_FILES = [f for f in EXTRA_FILES if os.path.basename(f) != filename]
    train_pipeline(EXCEL_FILE, EXTRA_FILES if EXTRA_FILES else None)
    return jsonify({'success': True, 'metrics': METRICS})

@app.route('/api/metrics')
def metrics():
    if MODEL is None: return jsonify({'error': 'Not trained'}), 400
    return jsonify(METRICS)

@app.route('/api/blends')
def blends():
    if MODEL is None or PIVOT is None:
        return jsonify({'error': 'Not trained'}), 400

    def get_uses_and_howto(key):
        if key == 'cross_category':
            uses = [
                {
                    'label': 'Complete Protein Supplement',
                    'steps': ['Dry all ingredients thoroughly at 55°C.','Grind each ingredient separately into fine powder.','Mix in exact percentages shown above.','Sieve to ensure uniform particle size.','Store in airtight container away from light.','Use 30–40g per serving in water or smoothies.'],
                    'tip': 'Best consumed within 3 months for optimal nutrition.'
                },
                {
                    'label': 'Multi-Source Protein Powder',
                    'steps': ['Dehydrate all ingredients fully until crispy.','Grind into powder using a high-speed blender.','Combine in blend percentages and mix well.','Add to protein shakes, oatmeal, or baked goods.','Store in a cool dry place in a sealed jar.','Recommended dose: 2 tablespoons per meal.'],
                    'tip': 'Works great blended into pancake or muffin batter.'
                },
                {
                    'label': 'Functional Food Blend',
                    'steps': ['Prepare each ingredient in its most bioavailable form.','Combine all ingredients in exact gram ratios.','Mix thoroughly to create a uniform blend.','Can be consumed raw, cooked, or as a powder.','Add to any meal as a protein and nutrient boost.','Refrigerate prepared blend up to 5 days.'],
                    'tip': 'This blend covers all 9 essential amino acids.'
                },
                {
                    'label': 'Nutritional Fortifier',
                    'steps': ['Grind or mash all ingredients into a uniform paste.','Mix in exact percentages for consistent nutrition.','Add small amounts to soups, sauces, or dips.','Use 20–30g per meal to fortify any dish.','Works in both savory and sweet applications.','Store fortifier paste in fridge up to 3 days.'],
                    'tip': 'Undetectable in most dishes — great for picky eaters.'
                },
            ]
        elif key == 'legumes_nuts':
            uses = [
                {
                    'label': 'Plant-Based Burger Patty',
                    'steps': ['Cook legumes until very soft, drain and mash well.','Chop or grind nuts/seeds coarsely.','Mix mashed legumes with nuts in blend percentages.','Add breadcrumbs, garlic, and seasoning to bind.','Form into patties and refrigerate 30 min.','Pan-fry or bake at 180°C for 15 min each side.'],
                    'tip': 'Add smoked paprika for a BBQ-like flavor.'
                },
                {
                    'label': 'Protein-Enriched Bread',
                    'steps': ['Grind nuts and seeds into a coarse flour.','Cook and mash legumes into a smooth paste.','Combine with wheat flour (50/50 ratio) and yeast.','Add water, olive oil, and salt — knead 10 min.','Let rise 1 hour in a warm place.','Bake at 190°C for 30–35 minutes.'],
                    'tip': 'This bread has 2× the protein of regular bread.'
                },
                {
                    'label': 'High-Protein Pasta',
                    'steps': ['Grind nuts and seeds into fine flour.','Cook and dry legumes, then grind to powder.','Mix both flours with eggs or flax eggs (vegan).','Knead dough until smooth — rest 30 minutes.','Roll thin and cut into desired pasta shape.','Cook in boiling salted water for 3–4 minutes.'],
                    'tip': 'Works best as tagliatelle or lasagne sheets.'
                },
                {
                    'label': 'Sports Recovery Bar',
                    'steps': ['Roast nuts and seeds at 160°C for 10 minutes.','Cook legumes until soft, mash into paste.','Mix roasted nuts with legume paste in blend ratios.','Add dates or honey as binder — 10–15% of weight.','Press firmly into a lined tray 1.5cm thick.','Refrigerate 2 hours, cut into bars. Keeps 1 week.'],
                    'tip': 'Add cocoa powder or vanilla for better flavor.'
                },
            ]
        elif key == 'vegetables_seeds':
            uses = [
                {
                    'label': 'Green Superfood Powder',
                    'steps': ['Wash and blanch vegetables 2 min, cool in ice water.','Dehydrate at 55°C for 6–8 hours until crispy.','Briefly crack seeds — do not fully grind.','Grind dried vegetables into fine powder.','Mix vegetable powder with seeds in blend percentages.','Add to smoothies, soups, or mix with water.'],
                    'tip': 'Blanching preserves the green color and nutrients.'
                },
                {
                    'label': 'Vegetable Protein Soup Base',
                    'steps': ['Roast all vegetables at 200°C for 25 minutes.','Toast seeds in dry pan until golden — 3 min.','Blend roasted vegetables with vegetable broth.','Add toasted seeds and blend briefly for texture.','Season with salt, pepper, and lemon juice.','Use as soup base or dilute for a light broth.'],
                    'tip': 'Freeze in ice cube trays for easy portioning.'
                },
                {
                    'label': 'Baby Food Fortifier',
                    'steps': ['Steam all vegetables until very soft — 15 min.','Grind seeds into very fine powder.','Blend steamed vegetables into smooth puree.','Mix in seed powder using exact blend percentages.','Ensure completely smooth texture for safety.','Refrigerate up to 48 hours or freeze in portions.'],
                    'tip': 'Always consult pediatrician before introducing new foods.'
                },
                {
                    'label': 'Detox Protein Blend',
                    'steps': ['Juice or blend all vegetables fresh.','Grind seeds finely and add to vegetable juice.','Mix in blend percentages for optimal nutrition.','Add lemon juice and ginger for detox effect.','Consume fresh immediately for best results.','Can be stored refrigerated for up to 24 hours.'],
                    'tip': 'Best consumed in the morning on an empty stomach.'
                },
            ]
        else:
            uses = [
                {
                    'label': 'Home Cooking Protein Boost',
                    'steps': ['Cook each ingredient separately to your preference.','Combine in exact gram ratios shown above.','Add to any dish — rice, salad, pasta, or stew.','Season with olive oil, lemon, and spices.','Can be eaten hot or cold as a complete meal.','Meal prep: refrigerate up to 4 days.'],
                    'tip': 'This blend covers all 9 essential amino acids in one meal.'
                },
                {
                    'label': 'Protein-Rich Soup & Stew',
                    'steps': ['Sauté onion and garlic in olive oil 3 minutes.','Add all blend ingredients and stir well.','Pour in 2 cups vegetable broth.','Bring to boil, then simmer 20 minutes.','Season with cumin, turmeric, salt and pepper.','Serve hot with whole grain bread.'],
                    'tip': 'Freezes well — make a big batch and freeze portions.'
                },
                {
                    'label': 'Everyday Meal Fortifier',
                    'steps': ['Cook all ingredients until tender.','Mash or blend into a smooth paste.','Add 2–3 tablespoons to any meal.','Works in soups, sauces, dips, and spreads.','Completely changes nutritional profile of any dish.','Store in fridge in sealed jar up to 5 days.'],
                    'tip': 'Completely undetectable in most sauces and stews.'
                },
                {
                    'label': 'Plant-Based Diet Staple',
                    'steps': ['Cook legumes: soak overnight, boil 45–60 minutes.','Prepare other ingredients per their method.','Combine all in exact ratios from the blend above.','Season generously and serve as main dish.','Pairs perfectly with whole grains and greens.','Great for weekly meal prep — stays fresh 4 days.'],
                    'tip': 'This is one of the most complete plant protein combinations possible.'
                },
            ]
        return uses

    all_top  = pd.concat([PIVOT[PIVOT['food_group']==g].nlargest(3,'predicted_score') for g in PIVOT['food_group'].unique()])
    leg_nuts = PIVOT[PIVOT['food_group'].isin(['16 Legumes and Legume Products','Nuts and Seeds'])].nlargest(12,'predicted_score')
    veg_seed = PIVOT[PIVOT['food_group'].isin(['Vegetables','Seeds'])].nlargest(12,'predicted_score')
    consumer = PIVOT[PIVOT['food_group'].isin(['Fruits','Vegetables','16 Legumes and Legume Products'])].nlargest(10,'predicted_score')

    result = {}
    configs = [
        ('cross_category',   all_top,  5, {'name':'Cross-Category Blend','tag':'Best Overall',   'color':'#4ade80','description':'Top ingredients from all 5 plant groups'}),
        ('legumes_nuts',     leg_nuts, 5, {'name':'Legumes & Nuts Blend', 'tag':'Classic Pairing','color':'#fbbf24','description':'Traditional legumes + nuts & seeds pairing'}),
        ('vegetables_seeds', veg_seed, 5, {'name':'Vegetables & Seeds',   'tag':'Whole Food',     'color':'#60a5fa','description':'Nutrient-dense whole-food blend'}),
        ('consumer_friendly',consumer, 4, {'name':'Smart Consumer Blend', 'tag':'Consumer Pick',  'color':'#a78bfa','description':'Easy everyday market ingredients'}),
    ]
    for key, df_sub, n_pick, meta in configs:
        b = optimize_blend(df_sub, n_pick=n_pick)
        if b:
            uses = get_uses_and_howto(key)
            result[key] = {**b, **meta, 'uses': uses}
    return jsonify(result)

@app.route('/api/top_foods')
def top_foods():
    if PIVOT is None: return jsonify({'error': 'Not trained'}), 400
    n = int(request.args.get('n', 10))
    cols = ['food_id','food','food_group','predicted_score','score','protein_content']
    foods = PIVOT.nlargest(n,'predicted_score')[cols].round(2)
    return jsonify(foods.to_dict('records'))

@app.route('/api/predict_custom', methods=['POST','OPTIONS'])
def predict_custom():
    """Consumer tool: optimize blend for user-selected home ingredients."""
    if request.method == 'OPTIONS': return jsonify({}), 200
    if MODEL is None: return jsonify({'error': 'Not trained'}), 400

    data        = request.json
    foods       = data.get('foods', [])
    total_grams = int(data.get('total_grams', 100))
    if len(foods) < 2: return jsonify({'error': 'Need at least 2 foods'}), 400

    egg_vec = np.array([EGG_REF[aa] for aa in AMINO_ACIDS])
    n = len(foods)

    def get_per_protein(food_item):
        aa_food = np.array([food_item['aa'].get(aa, 0) for aa in AMINO_ACIDS])
        protein = float(np.sum(aa_food) * 6.25)
        if protein < 0.5: return aa_food
        return aa_food / (protein/100 + 1e-9)

    def objective(w):
        total_w = w.sum()
        mix = sum((w[i]/total_w) * get_per_protein(foods[i]) for i in range(n))
        return -cosine_similarity(mix, egg_vec)

    res = minimize(objective, np.ones(n)*total_grams/n, method='SLSQP',
                   bounds=[(1,total_grams)]*n,
                   constraints=[{'type':'eq','fun':lambda w:w.sum()-total_grams}],
                   options={'ftol':1e-9,'maxiter':500})
    w = res.x
    total_w = w.sum()
    mix = sum((w[i]/total_w)*get_per_protein(foods[i]) for i in range(n))
    sim = cosine_similarity(mix, egg_vec)

    return jsonify({
        'weights': [{'food':foods[i]['name'],'group':foods[i].get('group',''),
                     'grams':round(float(w[i]),1),
                     'percentage':round(float(w[i]/total_w*100),1)} for i in range(n)],
        'mix_profile': {aa: round(float(mix[j]),2) for j,aa in enumerate(AMINO_ACIDS)},
        'egg_similarity': round(sim*100, 1),
        'total_grams': total_grams,
        'unit': 'g per 100g protein'
    })

@app.route('/api/all_foods')
def all_foods():
    if PIVOT is None: return jsonify({'error': 'Not trained'}), 400
    cols = ['food_id','food','food_group','predicted_score','score','protein_content'] + AMINO_ACIDS
    foods = PIVOT.sort_values('predicted_score', ascending=False)[cols].round(3)
    return jsonify(foods.to_dict('records'))

@app.route('/api/generate_recipe', methods=['POST','OPTIONS'])
def generate_recipe():
    if request.method == 'OPTIONS': return jsonify({}), 200
    try:
        import json as json_lib
        data = request.json
        ingredients = data.get('ingredients', [])
        total_grams = data.get('total_grams', 100)
        egg_similarity = data.get('egg_similarity', 0)
        protein_grams = data.get('protein_grams', round(total_grams * 0.12))
        target_label = data.get('target_label', '')

        # extract food names and groups
        foods = [i['food'].lower() for i in ingredients]
        groups = [i.get('group','').lower() for i in ingredients]
        main = ingredients[0]['food'] if ingredients else 'Mixed Plant Protein'

        # detect ingredient types from ALL ingredients
        has_legume   = any('legume' in g or 'bean' in f or 'lentil' in f or 'pea' in f or 'chickpea' in f or 'edamame' in f or 'okara' in f or 'cowpea' in f or 'soy' in f for f,g in zip(foods,groups))
        has_nut      = any('nut' in g or 'almond' in f or 'cashew' in f or 'walnut' in f or 'pistachio' in f or 'peanut' in f for f,g in zip(foods,groups))
        has_seed     = any('seed' in f or 'seed' in g or 'quinoa' in f or 'hemp' in f or 'chia' in f or 'flax' in f or 'sesame' in f or 'sunflower' in f or 'pumpkin' in f for f,g in zip(foods,groups))
        has_veg      = any('vegetable' in g or 'broccoli' in f or 'spinach' in f or 'kale' in f or 'potato' in f or 'amaranth' in f or 'carrot' in f or 'corn' in f or 'pea' in f for f,g in zip(foods,groups))
        has_fruit    = any('fruit' in g or 'avocado' in f or 'banana' in f or 'berry' in f or 'mango' in f or 'apple' in f for f,g in zip(foods,groups))
        has_avocado  = any('avocado' in f for f in foods)
        has_grain    = any('grain' in g or 'rice' in f or 'oat' in f or 'wheat' in f or 'quinoa' in f or 'barley' in f or 'millet' in f for f,g in zip(foods,groups))
        has_potato   = any('potato' in f for f in foods)
        has_smoothie_ok = has_fruit and (has_seed or has_nut) and not has_veg and not has_legume and not has_potato and not has_grain

        # choose recipe type — order matters: most specific first
        if has_legume and has_grain:
            recipe_type = 'grain_legume'
        elif has_legume and has_veg:
            recipe_type = 'legume_veg'
        elif has_avocado and has_legume:
            recipe_type = 'avocado_legume'
        elif has_smoothie_ok:
            recipe_type = 'smoothie'
        elif has_nut or (has_seed and not has_veg and not has_legume):
            recipe_type = 'nut_seed_salad'
        elif has_veg and has_seed:
            recipe_type = 'roasted_veg_seed'
        elif has_veg or has_potato:
            recipe_type = 'roasted_veg'
        elif has_legume:
            recipe_type = 'legume_stew'
        else:
            recipe_type = 'power_bowl'

        ing_list = [f"{i['food']} — {i['grams']}g ({i['percentage']}%)" for i in ingredients]
        ing_list += ['Olive oil — 1 tbsp', 'Salt & pepper to taste', 'Lemon juice — 1 tbsp']

        if recipe_type == 'smoothie':
            ing_list += ['1 cup plant milk', '1 tbsp honey or maple syrup', 'Fresh fruits for topping']
            recipe = {
                "name": "Protein Power Smoothie Bowl",
                "type": "Smoothie Bowl",
                "ingredients": ing_list,
                "steps": [
                    f"Blend {main} and fruits with 1 cup cold plant milk until completely smooth.",
                    "Add honey or maple syrup. Blend 30 more seconds — consistency should be thick.",
                    "Pour into a wide bowl. It should be thicker than a regular smoothie.",
                    "Top with seeds, fresh fruit slices, and a drizzle of honey.",
                    "Serve immediately — best enjoyed fresh for maximum nutrition."
                ],
                "nutrition_note": f"~{protein_grams}g plant protein — equivalent to {target_label} · {egg_similarity}% similarity to egg protein profile."
            }
        elif recipe_type == 'avocado_legume':
            ing_list += ['1 cup cooked brown rice or quinoa', 'Cherry tomatoes', 'Cucumber slices', 'Paprika & cumin']
            recipe = {
                "name": "Protein Avocado Power Bowl",
                "type": "Bowl",
                "ingredients": ing_list,
                "steps": [
                    "Cook rice or quinoa and let cool to room temperature.",
                    f"Mash avocado with lemon juice, salt, and a pinch of cumin until creamy.",
                    "Warm legumes with olive oil, paprika, and salt for 3-4 minutes.",
                    "Build the bowl: rice base → warm legumes → mashed avocado on top.",
                    "Add cherry tomatoes, cucumber, drizzle olive oil. Serve immediately."
                ],
                "nutrition_note": f"~{protein_grams}g plant protein — equivalent to {target_label} · {egg_similarity}% similarity to egg protein profile."
            }
        elif recipe_type == 'legume_veg':
            ing_list += ['2 cups vegetable broth', '1 onion diced', '2 garlic cloves', 'Cumin, turmeric, paprika']
            recipe = {
                "name": "Hearty Plant Protein Stew",
                "type": "Stew",
                "ingredients": ing_list,
                "steps": [
                    "Heat olive oil in a deep pot. Sauté onion and garlic 3 minutes until golden.",
                    "Add cumin, turmeric, and paprika. Stir 30 seconds until fragrant.",
                    "Add all vegetables and legumes. Stir well to coat with spices.",
                    "Pour in vegetable broth. Bring to boil then simmer 20 minutes.",
                    "Season with salt, pepper, and lemon. Serve hot with bread."
                ],
                "nutrition_note": f"~{protein_grams}g plant protein — equivalent to {target_label} · {egg_similarity}% similarity to egg protein profile."
            }
        elif recipe_type == 'grain_legume':
            ing_list += ['Fresh parsley or cilantro', 'Red onion diced', 'Lemon zest', 'Extra virgin olive oil']
            recipe = {
                "name": "High-Protein Grain & Legume Bowl",
                "type": "Bowl",
                "ingredients": ing_list,
                "steps": [
                    "Cook grains per package instructions. Warm legumes in a pan with olive oil.",
                    "Mix warm grains and legumes together in a large bowl.",
                    "Drizzle generously with olive oil and lemon juice.",
                    "Add diced red onion and fresh herbs. Toss well.",
                    "Serve warm or at room temperature. Great for meal prep — keeps 3 days."
                ],
                "nutrition_note": f"~{protein_grams}g plant protein — equivalent to {target_label} · {egg_similarity}% similarity to egg protein profile."
            }
        elif recipe_type == 'roasted_veg_seed':
            ing_list += ['Garlic powder', 'Dried thyme & rosemary', 'Tahini sauce for drizzle']
            recipe = {
                "name": "Roasted Vegetables & Seed Power Plate",
                "type": "Roasted Dish",
                "ingredients": ing_list,
                "steps": [
                    "Preheat oven to 200°C (400°F). Line baking sheet with parchment.",
                    "Chop vegetables evenly. Toss with olive oil, garlic powder, thyme, salt & pepper.",
                    "Spread in a single layer. Roast 25-30 min, flipping halfway.",
                    "In the last 5 minutes, scatter seeds on top to lightly toast them.",
                    "Drizzle with tahini sauce and serve warm over a bed of greens."
                ],
                "nutrition_note": f"~{protein_grams}g plant protein — equivalent to {target_label} · {egg_similarity}% similarity to egg protein profile."
            }
        elif recipe_type in ('roasted_veg', 'legume_stew'):
            ing_list += ['Garlic powder', 'Dried thyme & rosemary', 'Balsamic glaze']
            recipe = {
                "name": "Roasted Vegetable Protein Plate",
                "type": "Roasted Dish",
                "ingredients": ing_list,
                "steps": [
                    "Preheat oven to 200°C (400°F). Line baking sheet with parchment.",
                    "Chop all vegetables evenly. Toss with olive oil, garlic, thyme, salt & pepper.",
                    "Spread in single layer — don't overcrowd or they'll steam instead of roast.",
                    "Roast 25-30 min, flipping halfway, until golden and caramelized.",
                    "Drizzle with balsamic glaze and fresh herbs. Serve as main or side dish."
                ],
                "nutrition_note": f"~{protein_grams}g plant protein — equivalent to {target_label} · {egg_similarity}% similarity to egg protein profile."
            }
        elif recipe_type == 'nut_seed_salad':
            ing_list += ['Mixed greens (arugula, spinach)', 'Cherry tomatoes', 'Balsamic vinegar', 'Dijon mustard']
            recipe = {
                "name": "Protein Seed & Nut Energy Salad",
                "type": "Salad",
                "ingredients": ing_list,
                "steps": [
                    "Toast seeds/nuts in a dry pan 2-3 minutes until golden and fragrant. Set aside.",
                    "Whisk olive oil, balsamic vinegar, Dijon mustard, salt, and pepper for dressing.",
                    "Arrange mixed greens and cherry tomatoes in a large bowl.",
                    "Sprinkle toasted seeds and nuts generously on top.",
                    "Drizzle dressing just before serving. Toss gently and enjoy immediately."
                ],
                "nutrition_note": f"~{protein_grams}g plant protein — equivalent to {target_label} · {egg_similarity}% similarity to egg protein profile."
            }
        else:
            ing_list += ['Mixed greens', 'Cherry tomatoes', 'Tahini dressing', 'Fresh herbs']
            recipe = {
                "name": "Complete Plant Protein Power Bowl",
                "type": "Bowl",
                "ingredients": ing_list,
                "steps": [
                    "Prepare all ingredients — rinse, chop, and measure according to amounts above.",
                    "Warm protein components gently in a pan with olive oil for 3-4 minutes.",
                    "Arrange mixed greens as base in a wide bowl.",
                    "Layer all protein blend ingredients on top.",
                    "Finish with tahini dressing, fresh herbs, and lemon squeeze. Serve immediately."
                ],
                "nutrition_note": f"~{protein_grams}g plant protein — equivalent to {target_label} · {egg_similarity}% similarity to egg protein profile."
            }

        return jsonify({'success': True, 'recipe': recipe})

    except Exception as e:
        import traceback
        print(f"Recipe error: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict_food', methods=['POST','OPTIONS'])
def predict_food():
    """Predict quality score for a new food given its AA values."""
    if request.method == 'OPTIONS': return jsonify({}), 200
    if MODEL is None or SCALER is None: return jsonify({'error':'Not trained'}), 400
    aa_values = request.json.get('amino_acids', {})
    vec = np.array([[aa_values.get(aa, 0) for aa in AMINO_ACIDS]])
    score = float(MODEL.predict(SCALER.transform(vec))[0])
    protein = float(sum(aa_values.get(aa,0) for aa in AMINO_ACIDS) * 6.25)
    return jsonify({'predicted_score': round(score,2), 'estimated_protein_content': round(protein,1)})

# ── STARTUP ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 55)
    print("  🌱 PlantProtein AI — Graduation Project Server")
    print("=" * 55)
    print(f"  Base file:  {EXCEL_FILE}")
    print(f"  Extra data: extra_data/ folder (auto-merged)")
    print("=" * 55)
    # auto-load any previously saved extra files
    if os.path.exists('extra_data'):
        saved = [os.path.join('extra_data',f) for f in os.listdir('extra_data') if f.endswith(('.xlsx','.xls'))]
        EXTRA_FILES.extend(saved)
    if os.path.exists(EXCEL_FILE):
        print(f"\n📊 Training on {EXCEL_FILE}" + (f" + {len(EXTRA_FILES)} extra file(s)" if EXTRA_FILES else "") + " ...")
        train_pipeline(EXCEL_FILE, EXTRA_FILES if EXTRA_FILES else None)
    else:
        print(f"\n⚠️  {EXCEL_FILE} not found. Place it here then restart.\n")
    print(f"\n🚀 Open browser: http://localhost:5000\n")
    app.run(debug=False, port=5000)
