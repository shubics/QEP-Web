# QEP-Web

Streamlit deployment repository for the modular QEPlotter 2.0 application.
The scientific engines live in `qeplotter/`, the interface pages live in
`gui/`, and Streamlit continues to start from `gui.py`.

The maintained source project is
[QEPlotter](https://github.com/shubics/QEPlotter).

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run gui.py
```

The deployed interface provides:

- band, DOS, PDOS, overlay, and fatband plotting;
- plot title, axis-label, legend/colour-scale, grid, size, and DPI controls;
- crystal structure, bond, angle, periodic-cell, and stacking analysis;
- native high-symmetry K-path and first-Brillouin-zone tools;
- irreducible K-grid reduction from spglib symmetry operations;
- Γ-point symmetry and orbital-representation analysis;
- QE conversion and band-gap utilities.
