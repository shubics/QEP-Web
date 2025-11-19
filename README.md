# QEP-Web
A modern, web-based visualization suite for Quantum ESPRESSO electronic structure calculations. Features interactive plotting for Band Structures, Fatbands (Projected Bands), DOS/PDOS, and 2D layer stacking analysis.
# ⚛️ QEPlotter Pro

A modern, web-based visualization suite for **Quantum ESPRESSO** electronic structure calculations. Built with **Python** and **Streamlit**.

## 🌟 Features

* **Band Structure:** Plot clean, publication-ready band structures.
* **Fatbands (Projected Bands):** Visualize atomic and orbital contributions with Bubble, Line, and Heatmap modes.
* **DOS & PDOS:** Analyze Total and Projected Density of States.
* **Overlay Mode:** Compare two different band structures on top of each other.
* **Tools:**
    * `proj.out` to `.pdos` converter (Standard & SOC support).
    * Band Gap Detector (VBM, CBM, Direct/Indirect).
    * Bilayer Stacking Analyzer.

## 🛠 Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/QEPlotter.git](https://github.com/your-username/QEPlotter.git)
    cd QEPlotter
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage

Run the application locally using Streamlit:

```bash
streamlit run gui.py
