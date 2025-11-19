import streamlit as st
import os
import tempfile
import matplotlib
import matplotlib.pyplot as plt
from io import StringIO, BytesIO
from contextlib import redirect_stdout
import ast

# --- 1. APP CONFIGURATION ---
st.set_page_config(
    page_title="QEPlotter Pro",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Force non-interactive backend (Critical for macOS)
matplotlib.use("Agg")

# Import Backend
try:
    import qep5
except ImportError:
    st.error("❌ Critical Error: `qep5.py` not found. Please put it in the same folder.")
    st.stop()

# --- 2. STYLING ---
st.markdown("""
<style>
    .stButton button { width: 100%; border-radius: 8px; font-weight: bold; padding: 0.6rem; }
    button[kind="primary"] { background: linear-gradient(90deg, #FF4B4B 0%, #D13333 100%); border: none; }
    .stExpander { border: 1px solid #e0e0e0; border-radius: 8px; }
    /* Input alanlarını biraz daha modern yapalım */
    .stTextInput input, .stNumberInput input { border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# --- 3. FILE HANDLING ---
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = tempfile.mkdtemp()


def save_file(uploaded_file, subdir=None):
    if uploaded_file is None: return None
    target_dir = st.session_state.temp_dir
    if subdir:
        target_dir = os.path.join(target_dir, subdir)
        os.makedirs(target_dir, exist_ok=True)
    file_path = os.path.join(target_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


# --- 4. MAIN LAYOUT ---
def main():
    with st.sidebar:
        st.title("⚛️ QEPlotter")
        st.caption("Full-Feature Interface v3.2")
        mode = st.radio("Navigation", ["📊 Visualization Dashboard", "🛠 Tools & Utilities"])

    if mode == "📊 Visualization Dashboard":
        render_dashboard()
    else:
        render_tools()


def render_dashboard():
    # --- STEP 1: PLOT TYPE SELECTION ---
    c_type, c_space = st.columns([2, 3])
    with c_type:
        plot_type_ui = st.selectbox("Select Plot Type",
                                    ["Band Structure", "Fatbands (Projected)", "Total DOS", "PDOS Only",
                                     "Overlay Comparison"])

    # Map to backend 'plot_type'
    p_map = {
        "Band Structure": "band", "Fatbands (Projected)": "fatbands",
        "Total DOS": "dos", "PDOS Only": "pdos", "Overlay Comparison": "overlay_band"
    }
    pt = p_map[plot_type_ui]

    # Initialize Args Dictionary
    args = {'plot_type': pt, 'savefig': None}
    paths = {}

    # ==========================================
    # LEFT COLUMN: DYNAMIC INPUTS
    # ==========================================
    col_inputs, col_preview = st.columns([1, 1.5])

    with col_inputs:
        # --- A. FILE INPUTS (Context Aware) ---
        with st.expander("📂 1. Input Files", expanded=True):

            # 1. Band File & Kpath (All except pure DOS/PDOS)
            if pt in ["band", "fatbands", "overlay_band"]:
                f_band = st.file_uploader("Band File (.gnu)", type=["gnu", "dat"], key="u_band")
                paths['band_file'] = save_file(f_band)

                f_kpath = st.file_uploader("K-Path File", type=["kpath", "in", "txt"], key="u_kpath")
                paths['kpath_file'] = save_file(f_kpath)

            # 2. Fatband/PDOS Directory (Fatbands/PDOS only)
            if pt in ["fatbands", "pdos"]:
                f_pdos = st.file_uploader("PDOS Files (Select Multiple)", accept_multiple_files=True, key="u_pdos")
                if f_pdos:
                    subdir = "pdos_data"
                    for f in f_pdos: save_file(f, subdir=subdir)
                    paths['fatband_dir'] = os.path.join(st.session_state.temp_dir, subdir)
                    paths['pdos_dir'] = paths['fatband_dir']
                else:
                    paths['fatband_dir'] = None

            # 3. DOS File
            f_dos = st.file_uploader("Total DOS File (Optional)", type=["dos", "dat"], key="u_dos")
            paths['dos_file'] = save_file(f_dos)

            # 4. Overlay Specifics
            if pt == "overlay_band":
                st.markdown("---")
                st.caption("Comparison Data")
                f_b2 = st.file_uploader("Band File 2", key="u_b2")
                paths['band_file2'] = save_file(f_b2)
                f_k2 = st.file_uploader("K-Path File 2", key="u_k2")
                paths['kpath_file2'] = save_file(f_k2)

        # --- B. DATA SETTINGS ---
        with st.expander("⚙️ 2. Data Settings", expanded=True):
            # Global Settings
            c1, c2 = st.columns(2)
            args['fermi_level'] = c1.number_input("Fermi Level (eV)", value=0.0, format="%.4f")
            args['shift_fermi'] = c2.toggle("Shift Fermi to 0", value=True)

            # --- PDOS İÇİN ÖZEL Y-RANGE MANTIĞI ---
            # PDOS/DOS seçiliyse Y-Limitlerini gizleyelim veya opsiyonel yapalım
            if pt in ["pdos", "dos"]:
                st.markdown("---")
                use_custom_y = st.checkbox("Set Custom Y-Limits (Density)", value=False)
                if use_custom_y:
                    c3, c4 = st.columns(2)
                    ymin = c3.number_input("Y-Min (DOS)", value=0.0)
                    ymax = c4.number_input("Y-Max (DOS)", value=10.0)
                    args['y_range'] = (ymin, ymax)
                else:
                    # Backend'e None göndererek otomatik ölçeklenmesini sağla
                    args['y_range'] = None
            else:
                # Band yapısı için standart Y (Enerji) limitleri
                c3, c4 = st.columns(2)
                ymin = c3.number_input("Y-Min (Energy)", value=-3.0)
                ymax = c4.number_input("Y-Max (Energy)", value=3.0)
                args['y_range'] = (ymin, ymax)

            # Specific Toggles
            if pt in ["band", "fatbands"]:
                args['spin'] = st.checkbox("Spin Polarized", help="Enable if calculation used spin polarization")
                args['sub_orb'] = st.checkbox("Sub-Orbital Analysis", help="Parse specific orbitals (px, py, dxy...)")

        # --- C. VISUAL & MODE SETTINGS ---
        with st.expander("🎨 3. Visual & Mode Config", expanded=True):

            # --- GRAFİK BOYUT AYARI (YENİ) ---
            st.markdown("### Figure Layout")
            col_w, col_h = st.columns(2)
            fig_width = col_w.number_input("Plot Width", value=12, min_value=4, max_value=20)
            fig_height = col_h.number_input("Plot Height", value=6, min_value=4, max_value=20)

            # --- 1. FATBAND MODES ---
            if pt == "fatbands":
                st.markdown("### Fatband Mode")
                fb_mode = st.selectbox("Display Mode", [
                    "most", "atomic", "orbital", "element_orbital",
                    "normal", "o_atomic", "o_orbital", "o_element_orbital",
                    "heat_total", "heat_atomic", "heat_orbital", "heat_element_orbital", "layer"
                ])
                args['fatbands_mode'] = fb_mode

                # Logic Flags
                is_line_mode = fb_mode in ['normal', 'o_atomic', 'o_orbital', 'o_element_orbital']
                is_heat_mode = "heat" in fb_mode
                is_layer_mode = fb_mode == "layer"
                needs_highlight = (is_line_mode and fb_mode != 'normal') or is_heat_mode

                if needs_highlight:
                    hl = st.text_input("Highlight Channel", help="e.g. 'Mo', 'd', or 'Mo-d'")

                    # Dual Mode Logic
                    if is_line_mode:
                        args['dual'] = st.checkbox("Dual Mode",
                                                   help="Interpolate 2 channels. Enter 'Mo,S' in highlight.")
                        if args['dual'] and "," in hl:
                            args['highlight_channel'] = [x.strip() for x in hl.split(',')]
                        else:
                            args['highlight_channel'] = hl
                    else:
                        args['highlight_channel'] = hl

                # Heatmap Specifics
                if is_heat_mode:
                    st.caption("Heatmap Intensity")
                    hc1, hc2 = st.columns(2)
                    h_min = hc1.number_input("Heat Min", value=0.0)
                    h_max = hc2.number_input("Heat Max", value=0.0)
                    if h_max > 0:
                        args['heat_vmin'] = h_min
                        args['heat_vmax'] = h_max

                    args['overlay_bands_in_heat'] = st.checkbox("Overlay Lines on Heatmap", value=True)

                # Layer Specifics
                if is_layer_mode:
                    l_str = st.text_area("Layer Dictionary", placeholder="{'Mo1':'top', 'S3':'bottom'}")
                    if l_str:
                        try:
                            args['layer_assignment'] = ast.literal_eval(l_str)
                        except:
                            st.error("Invalid Dictionary syntax")

                # Bubble Specifics
                if not (is_line_mode or is_heat_mode or is_layer_mode):
                    st.caption("Bubble Sizes")
                    sc1, sc2 = st.columns(2)
                    args['s_min'] = sc1.number_input("Size Min", 10)
                    args['s_max'] = sc2.number_input("Size Max", 100)
                    args['weight_threshold'] = st.number_input("Weight Threshold", 0.01)

                st.markdown("---")
                args['plot_total_dos'] = st.checkbox("Plot Side DOS")

            # --- 2. BAND MODES ---
            elif pt == "band":
                b_mode = st.selectbox("Band Color Mode", ["normal", "atomic", "orbital", "element_orbital", "most"])
                args['band_mode'] = b_mode
                if b_mode != "normal":
                    st.info("Upload PDOS files above for coloring.")
                    args['fatband_dir'] = paths.get('fatband_dir')

            # --- 3. PDOS SETTINGS (YENİLENMİŞ) ---
            elif pt == "pdos":
                args['pdos_mode'] = st.selectbox("Grouping Mode", ["atomic", "orbital", "element_orbital"])
                st.caption("Tip: Y-Axis is now autoscaled. Check 'Data Settings' to set manually.")

            # Common Visuals
            st.markdown("---")
            vc1, vc2 = st.columns(2)
            args['cmap_name'] = vc1.selectbox("Colormap", ["tab10", "magma", "viridis", "plasma", "jet"])
            args['dpi'] = vc2.number_input("DPI", value=200)

    # ==========================================
    # RIGHT COLUMN: EXECUTION & OUTPUT
    # ==========================================
    with col_preview:
        st.subheader("🚀 Execution")

        if st.button("GENERATE VISUALIZATION", type="primary"):
            # 1. Merge Paths
            args.update(paths)

            # 2. Validation
            valid = True
            msg = ""
            if pt in ["band", "fatbands", "overlay_band"]:
                if not args.get('band_file') or not args.get('kpath_file'):
                    valid = False;
                    msg = "Missing Band or K-Path file."
            elif pt in ["fatbands", "pdos"] and not args.get('fatband_dir'):
                valid = False;
                msg = "Missing PDOS files."
            elif pt == "fatbands" and args.get('plot_total_dos') and not args.get('dos_file'):
                valid = False;
                msg = "You enabled Side DOS but didn't upload a DOS file."
            elif pt == "fatbands" and "heat" in args.get('fatbands_mode', '') and not args.get('highlight_channel'):
                valid = False;
                msg = "Heatmap requires a Highlight Channel."

            if not valid:
                st.error(f"❌ {msg}")
            else:
                # 3. Run Backend
                with st.spinner("Processing..."):
                    log_io = StringIO()
                    try:
                        plt.close('all')
                        with redirect_stdout(log_io):
                            qep5.plot_from_file(**args)

                        if plt.get_fignums():
                            # --- CRITICAL FIX: RESIZE FIGURE ---
                            fig = plt.gcf()
                            # Backend (qep5) 6x6 gibi kare bir boyut oluşturuyor olabilir.
                            # Burada kullanıcının seçtiği 'Geniş' boyutu zorluyoruz.
                            fig.set_size_inches(fig_width, fig_height)

                            st.pyplot(fig, use_container_width=True)

                            # Download
                            buf = BytesIO()
                            fig.savefig(buf, format="png", dpi=args['dpi'], bbox_inches='tight')
                            st.download_button("💾 Download Image", buf, file_name="plot.png", mime="image/png")
                        else:
                            st.warning("Backend finished but produced no figure.")
                    except Exception as e:
                        st.error(f"Backend Error: {e}")

                    with st.expander("View Logs"):
                        st.text(log_io.getvalue())


# ==========================================
# TOOLS PAGE (UNIQUE KEYS FIXED)
# ==========================================
def render_tools():
    st.title("🛠️ Computational Utilities")

    tab1, tab2, tab3 = st.tabs(["Fatband Converter", "Gap Detector", "Bilayer Analysis"])

    # --- 1. CONVERTER ---
    with tab1:
        st.markdown("#### `proj.out` → `.pdos` Converter")
        f = st.file_uploader("Upload proj.out", key="t_p_uploader")
        if f:
            p = save_file(f)
            out_d = os.path.join(st.session_state.temp_dir, "converted")

            c1, c2 = st.columns(2)
            if c1.button("Convert (Standard)", key="btn_conv_std"):
                run_tool(qep5.convert_consistent, p, outdir=out_d)
            if c2.button("Convert (SOC Mode)", key="btn_conv_soc"):
                run_tool(qep5.convert_soc_proj_to_ml, p, outdir=out_d)

    # --- 2. GAP DETECTOR ---
    with tab2:
        st.markdown("#### Band Gap Analysis")
        c1, c2 = st.columns(2)
        fb = c1.file_uploader("Band File", key="t_bg_uploader")
        fk = c2.file_uploader("K-Path File", key="t_kg_uploader")
        fermi = st.number_input("Fermi Level", key="t_fermi_input")

        if st.button("Analyze Gap", key="btn_analyze_gap") and fb and fk:
            run_tool(qep5.detect_band_gap, save_file(fb), save_file(fk), fermi)

    # --- 3. BILAYER ---
    with tab3:
        st.markdown("#### Structure Analyzer")
        fs = st.file_uploader("Input/Output File", key="t_s_uploader")
        if st.button("Analyze Structure", key="btn_analyze_struc") and fs:
            run_tool(qep5.analyse_file, save_file(fs))


def run_tool(func, *args, **kwargs):
    log = StringIO()
    try:
        with redirect_stdout(log):
            func(*args, **kwargs)
        st.success("Execution Complete")
        st.code(log.getvalue())
    except Exception as e:
        st.error(f"Error: {e}")


if __name__ == "__main__":
    main()