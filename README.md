# 🌱 PlantProtein AI — Plant-Based Protein Optimization System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-2.3+-black?style=for-the-badge&logo=flask)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2+-orange?style=for-the-badge&logo=scikit-learn)
![Chart.js](https://img.shields.io/badge/Chart.js-4.4-pink?style=for-the-badge&logo=chart.js)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A Machine Learning system that predicts optimal plant-based protein blends matching the amino acid profile of animal protein.**

[🚀 Live Demo](#) • [📖 Documentation](#how-it-works) • [🧬 Proposal](docs/proposal.pdf)

![PlantProtein AI Screenshot](docs/screenshot.png)

</div>

---

## 📌 Overview

PlantProtein AI is a **Graduation Project** that uses Machine Learning and Mathematical Optimization to find the best combinations of plant-based ingredients that match the nutritional quality of complete animal proteins (like eggs).

The system analyzes **1,074+ plant foods** across 5 categories, predicts protein quality scores using **Random Forest**, and optimizes blends using **Scipy SLSQP** to maximize similarity to the egg protein reference — achieving **99%+ amino acid similarity**.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🧠 **ML Model** | Random Forest (200 trees) trained on 9 essential amino acids |
| ⚗️ **Optimizer** | Scipy SLSQP finds optimal gram weights per ingredient |
| 🥚 **Egg Reference** | Compares blends to WHO egg protein standard (g/100g protein) |
| 📊 **Live Dashboard** | Real-time predictions via Flask REST API |
| ➕ **Data Upload** | Add new Excel datasets — model retrains automatically |
| 🍽️ **Recipe Generator** | AI-suggested preparation recipes per blend |
| 🏠 **Consumer Tool** | Home Kitchen Calculator with 5 protein reference targets |
| 📦 **Industry Uses** | Per-blend manufacturing suggestions with preparation steps |

---

## 🧬 How It Works

```
Excel Dataset (1,074 foods)
        │
        ▼
┌─────────────────────┐
│   Data Preprocessing │  → Pivot table, protein content estimation
│   (pandas)           │  → Convert to g/100g protein units
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   Random Forest      │  → Trains on 80/20 split
│   ML Model           │  → Predicts quality score vs WHO reference
│   R² = 0.93+         │  → Feature importance per amino acid
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   Scipy SLSQP        │  → Maximizes cosine similarity to egg
│   Optimizer          │  → Outputs % and grams per ingredient
│                      │  → 4 optimized blends generated
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   Flask REST API     │  → Serves all predictions live
│   + HTML Frontend    │  → Interactive dashboard + consumer tool
└─────────────────────┘
```

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| **Algorithm** | Random Forest (n_estimators=200, max_depth=10) |
| **Train/Test Split** | 80% / 20% (859 train · 215 test) |
| **R² Score** | 0.934 |
| **RMSE** | 0.153 |
| **Total Foods** | 1,074 plant ingredients |
| **Best Blend Similarity** | 99.6% to egg protein |

### Top Feature Importance (Amino Acids)

| Amino Acid | Importance |
|---|---|
| Methionine | 22.8% |
| Phenylalanine | 18.6% |
| Tryptophan | 15.6% |

---

## 🧪 Optimized Blends

The system generates 4 scientifically optimized blends:

| Blend | Category Mix | Egg Similarity |
|---|---|---|
| 🟢 **Cross-Category** | All 5 plant groups | 99.6% |
| 🟡 **Legumes & Nuts** | Legumes + Nuts/Seeds | 99.5% |
| 🔵 **Vegetables & Seeds** | Vegetables + Seeds | 99.4% |
| 🟣 **Smart Consumer** | Everyday market foods | 99.1% |

Each blend output matches the **proposal specification** (Section 6):

```
Food              | Category  | % in Mixture | Grams | Protein/100g
─────────────────────────────────────────────────────────────────
Boiled Chestnuts  | Nuts      |    64.2%     | 64.2g | ~3g
Okara             | Legumes   |    29.8%     | 29.8g | ~4g
Blackeyed Peas    | Legumes   |     2.0%     |  2.0g | ~2g
...
```

---

## 🗂️ Dataset

- **Source:** USDA FoodData Central
- **Sheet name:** `essential amino acid`
- **Format:**

| food_group | food | amino_acid | qty (g/100g food) |
|---|---|---|---|
| Vegetables | Broccoli | Leucine | 0.190 |
| Legumes | Lentils | Lysine | 0.589 |

**Categories:** Vegetables · Legumes · Fruits · Nuts & Seeds · Seeds · Grains

---

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.10+
pip
```

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/plantprotein-ai.git
cd plantprotein-ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place your dataset
# Put merged.xlsx in the project root

# 4. Run the server
python app.py
```

### Open in Browser

```
http://127.0.0.1:5000
```

---

## 📁 Project Structure

```
plantprotein-ai/
│
├── app.py                  # Flask server + ML pipeline + API endpoints
├── ml_model.py             # Standalone ML training script
├── index.html              # Frontend web application
├── merged.xlsx             # Main dataset (USDA plant foods)
├── requirements.txt        # Python dependencies
│
├── extra_data/             # Auto-created folder for additional datasets
│   └── *.xlsx              # Extra food datasets (auto-merged on startup)
│
└── README.md
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/status` | Server status + training state |
| `POST` | `/api/train` | Retrain model on dataset |
| `GET` | `/api/metrics` | Model performance metrics |
| `GET` | `/api/blends` | All 4 optimized protein blends |
| `GET` | `/api/top_foods?n=10` | Top N foods by ML score |
| `GET` | `/api/all_foods` | All foods with AA values |
| `POST` | `/api/predict_custom` | Optimize custom ingredient list |
| `POST` | `/api/add_data` | Upload extra Excel dataset |
| `GET` | `/api/extra_files` | List merged extra datasets |
| `POST` | `/api/remove_extra` | Remove extra dataset |
| `POST` | `/api/generate_recipe` | Get recipe for a blend |

---

## ➕ Adding New Data

You can extend the dataset without retraining from scratch:

1. Prepare an Excel file with the same format (`essential amino acid` sheet)
2. Open the web app → **Model Performance** → **➕ Add New Data**
3. Upload the file — model retrains automatically
4. New foods appear in All Foods table and blends update

---

## 🏠 Consumer Tool — Smart Protein Builder

The consumer module allows anyone to:

1. **Select a protein target** (Eggs / Beef / Chicken / Tuna / Milk)
2. **Set quantity** (e.g. 3 eggs, 150g chicken)
3. **Choose ingredients** from 12 quick-picks or search all 1,074 foods
4. **Get optimal blend** with % and grams per ingredient
5. **Generate recipe** with preparation steps

---

## 📦 Industry Applications

Each blend comes with category-specific manufacturing suggestions:

- 🟢 **Cross-Category** → Complete Protein Supplement · Protein Powder
- 🟡 **Legumes & Nuts** → Plant-Based Burger · High-Protein Pasta
- 🔵 **Vegetables & Seeds** → Green Superfood Powder · Soup Base
- 🟣 **Smart Consumer** → Home Cooking Boost · Meal Fortifier

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Python · Flask · Scipy · Scikit-learn · Pandas · NumPy |
| **ML** | Random Forest · MinMaxScaler · SLSQP Optimization |
| **Frontend** | HTML5 · CSS3 · Vanilla JavaScript · Chart.js |
| **Data** | USDA FoodData Central · WHO/FAO/UNU 2007 |

---

## 📚 References

- WHO/FAO/UNU (2007). *Protein and Amino Acid Requirements in Human Nutrition*
- USDA FoodData Central — [fdc.nal.usda.gov](https://fdc.nal.usda.gov)
- FAO Food Composition Tables
- Dietary Reference Intakes for Macronutrients (National Academies, 2005)

---

## 👨‍💻 Author

**Graduation Project — Computer Science / AI Track**

Built with ❤️ using Python, Flask, Scikit-learn, and Scipy SLSQP Optimization.

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
<strong>🌱 PlantProtein AI</strong> — Making plant-based complete protein accessible to everyone.
</div>
