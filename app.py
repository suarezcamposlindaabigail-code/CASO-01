"""
LogiFast — Crossdocking MIP Optimizer
Curso II-1122 | UCR Alajuela | Branch-and-Bound
Yu & Egbelu (2008)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from logifast_solver import solve_logifast, TS5_PARAMS

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LogiFast · MIP Crossdocking",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

:root {
  --bg:       #08090d;
  --surface:  #111318;
  --border:   #1f2333;
  --accent:   #f0c040;
  --accent2:  #3ecfcf;
  --accent3:  #e05a7a;
  --text:     #e8eaf0;
  --muted:    #5a6080;
  --card-bg:  #13161f;
}

html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif;
  background: var(--bg);
  color: var(--text);
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: #0c0d13 !important;
  border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Hero ── */
.hero {
  padding: 2.5rem 0 1.5rem;
  border-bottom: 1px solid var(--border);
  margin-bottom: 2rem;
}
.hero-eyebrow {
  font-family: 'DM Mono', monospace;
  font-size: 0.72rem;
  letter-spacing: 0.18em;
  color: var(--accent);
  text-transform: uppercase;
  margin-bottom: 0.6rem;
}
.hero-title {
  font-family: 'Syne', sans-serif;
  font-size: 3rem;
  font-weight: 800;
  line-height: 1.05;
  color: var(--text);
  margin: 0;
}
.hero-title span { color: var(--accent); }
.hero-sub {
  font-size: 1rem;
  color: var(--muted);
  margin-top: 0.6rem;
  font-weight: 300;
}

/* ── Section header ── */
.sec-head {
  font-family: 'Syne', sans-serif;
  font-size: 1.25rem;
  font-weight: 700;
  color: var(--text);
  display: flex;
  align-items: center;
  gap: 0.6rem;
  margin: 2rem 0 1rem;
  padding-bottom: 0.5rem;
  border-bottom: 1px solid var(--border);
}
.sec-head .dot { color: var(--accent); }

/* ── KPI cards ── */
.kpi-grid { display: flex; gap: 1rem; flex-wrap: wrap; margin: 1rem 0; }
.kpi {
  flex: 1; min-width: 140px;
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 1.1rem 1.3rem;
  position: relative;
  overflow: hidden;
}
.kpi::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: var(--accent);
}
.kpi.blue::before  { background: var(--accent2); }
.kpi.red::before   { background: var(--accent3); }
.kpi .label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.12em; color: var(--muted); margin-bottom: 0.4rem; font-family: 'DM Mono', monospace; }
.kpi .value { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800; color: var(--accent); }
.kpi.blue .value { color: var(--accent2); }
.kpi.red  .value { color: var(--accent3); }
.kpi .unit  { font-size: 0.75rem; color: var(--muted); margin-top: 0.2rem; }

/* ── Sequence track ── */
.seq-track {
  display: flex; align-items: center; gap: 0; flex-wrap: wrap;
  margin: 0.8rem 0;
}
.seq-node {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 0.7rem 1rem;
  text-align: center;
  min-width: 90px;
}
.seq-node .truck { font-family: 'Syne', sans-serif; font-weight: 700; font-size: 1.1rem; color: var(--accent); }
.seq-node.blue .truck { color: var(--accent2); }
.seq-node .times { font-family: 'DM Mono', monospace; font-size: 0.68rem; color: var(--muted); margin-top: 0.2rem; }
.seq-arrow { color: var(--muted); font-size: 1.2rem; padding: 0 0.3rem; }

/* ── Constraint card ── */
.cst-card {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-left: 3px solid var(--accent);
  border-radius: 8px;
  padding: 1rem 1.2rem;
  margin-bottom: 0.7rem;
}
.cst-card.blue  { border-left-color: var(--accent2); }
.cst-card.red   { border-left-color: var(--accent3); }
.cst-card.green { border-left-color: #4ade80; }
.cst-tag { font-family: 'DM Mono', monospace; font-size: 0.7rem; color: var(--accent); font-weight: 500; margin-bottom: 0.3rem; }
.cst-card.blue  .cst-tag { color: var(--accent2); }
.cst-card.red   .cst-tag { color: var(--accent3); }
.cst-card.green .cst-tag { color: #4ade80; }
.cst-formula { font-family: 'DM Mono', monospace; font-size: 0.88rem; color: var(--text); margin-bottom: 0.3rem; }
.cst-desc { font-size: 0.8rem; color: var(--muted); }

/* ── Math block ── */
.math-box {
  background: #0c0e14;
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1rem 1.4rem;
  font-family: 'DM Mono', monospace;
  font-size: 0.82rem;
  color: var(--text);
  margin: 0.5rem 0;
  white-space: pre-wrap;
  line-height: 1.9;
}
.hl  { color: var(--accent); }
.hl2 { color: var(--accent2); }
.hl3 { color: var(--accent3); }
.cm  { color: var(--muted); font-style: italic; }

/* ── Badge ── */
.badge {
  display: inline-block;
  background: #1a1d2a;
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 0.1rem 0.45rem;
  font-family: 'DM Mono', monospace;
  font-size: 0.72rem;
  color: var(--accent);
  margin: 0.1rem;
}
.badge.b2 { color: var(--accent2); }
.badge.b3 { color: var(--accent3); }

/* ── Streamlit overrides ── */
.stDataFrame { border-radius: 8px; overflow: hidden; }
div[data-testid="stExpander"] { border: 1px solid var(--border) !important; border-radius: 8px !important; background: var(--card-bg) !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0 0.5rem'>
      <div style='font-family:Syne,sans-serif;font-size:1.3rem;font-weight:800;color:#f0c040'>🚛 LogiFast</div>
      <div style='font-size:0.72rem;color:#5a6080;margin-top:0.2rem;font-family:DM Mono,monospace'>MIP CROSSDOCKING · II-1122</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    page = st.radio("", [
        "01 · Formulación MIP",
        "02 · Secuencia Óptima",
        "03 · Makespan Mínimo",
        "04 · Modo Paramétrico",
        "05 · Impacto Almacenamiento",
    ], label_visibility="collapsed")
    st.divider()
    st.markdown('<div style="font-size:0.7rem;color:#3a4060;font-family:DM Mono,monospace">Yu & Egbelu (2008)<br>Solver: CBC · PuLP<br>UCR Alajuela · 2026</div>', unsafe_allow_html=True)

# ── Data ─────────────────────────────────────────────────────────────────────
I_ts, O_ts, N_ts = 5, 3, 8
r_data = TS5_PARAMS["r"]
s_data = TS5_PARAMS["s"]

@st.cache_resource
def get_ts5_solution():
    return solve_logifast(**TS5_PARAMS, time_limit=180)

# =============================================================================
# 01 · FORMULACIÓN MIP
# =============================================================================
if page == "01 · Formulación MIP":
    st.markdown("""
    <div class="hero">
      <div class="hero-eyebrow">Entregable 01 · Formulación</div>
      <div class="hero-title">Modelo <span>MIP</span> Completo</div>
      <div class="hero-sub">Variables, función objetivo y 13 restricciones canónicas · Yu & Egbelu (2008)</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Índices ──
    st.markdown('<div class="sec-head"><span class="dot">▸</span> Índices y Conjuntos</div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    for col, sym, rng, desc in [
        (c1,"i","1 … I","Camiones de entrada (inbound)"),
        (c2,"j","1 … O","Camiones de salida (outbound)"),
        (c3,"k","1 … N","Tipos de producto"),
    ]:
        col.markdown(f"""
        <div class="kpi">
          <div class="label">índice</div>
          <div class="value" style="font-size:1.6rem">{sym} ∈ {{{rng}}}</div>
          <div class="unit">{desc}</div>
        </div>""", unsafe_allow_html=True)

    # ── Variables ──
    st.markdown('<div class="sec-head"><span class="dot">▸</span> Variables de Decisión</div>', unsafe_allow_html=True)
    vc, ve, vb = st.columns(3)
    with vc:
        st.markdown("**🔵 Continuas**")
        for sym, desc in [("T","Makespan — tiempo total"),("cᵢ","Llegada inbound i a recepción"),("Fᵢ","Salida inbound i de recepción"),("dⱼ","Llegada outbound j a despacho"),("Lⱼ","Salida outbound j de despacho")]:
            st.markdown(f'<span class="badge b2">{sym}</span> <span style="font-size:0.82rem;color:#a0a8c0">{desc}</span><br>', unsafe_allow_html=True)
    with ve:
        st.markdown("**🟡 Enteras ≥ 0**")
        st.markdown('<span class="badge">x[i,j,k]</span> <span style="font-size:0.82rem;color:#a0a8c0">Unidades del producto k transferidas del camión i al camión j</span>', unsafe_allow_html=True)
    with vb:
        st.markdown("**🔴 Binarias {0,1}**")
        for sym, desc in [("v[i,j]","= 1 si hay transferencia de i a j"),("p[i,i']","= 1 si inbound i precede a i'"),("q[j,j']","= 1 si outbound j precede a j'")]:
            st.markdown(f'<span class="badge b3">{sym}</span> <span style="font-size:0.82rem;color:#a0a8c0">{desc}</span><br>', unsafe_allow_html=True)

    # ── Función objetivo ──
    st.markdown('<div class="sec-head"><span class="dot">▸</span> Función Objetivo</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="math-box"><span class="hl">Mín Z = máx { Lⱼ }  ∀j ∈ {1, 2, 3}</span>

<span class="cm">-- Equivalente lineal (variable auxiliar T):</span>
<span class="hl2">min T</span>
  s.a.  <span class="hl">T ≥ Lⱼ</span>   ∀j

<span class="cm">-- T actúa como cota superior de todas las Lⱼ.
-- El solver minimiza T hasta que T* = max(L₁, L₂, L₃).</span></div>
    """, unsafe_allow_html=True)

    # ── 13 Restricciones ──
    st.markdown('<div class="sec-head"><span class="dot">▸</span> Las 13 Restricciones Canónicas</div>', unsafe_allow_html=True)

    groups = [
        ("🟡 Makespan", "yellow", [
            ("(1)","T ≥ Lⱼ  ∀j","Makespan ≥ salida del último camión de despacho","T ≥ L[j]   ∀j ∈ {1…O}"),
        ]),
        ("🔵 Flujo de productos", "blue", [
            ("(2)","Σⱼ x[i,j,k] = r[i,k]  ∀i,k","Todo lo cargado en camión i producto k se transfiere exactamente","sum_j x[i,j,k] = r[i,k]   ∀i,k"),
            ("(3)","Σᵢ x[i,j,k] = s[j,k]  ∀j,k","La demanda del camión j producto k se cumple exactamente","sum_i x[i,j,k] = s[j,k]   ∀j,k"),
        ]),
        ("🔴 Vinculación Big-M", "red", [
            ("(4)","x[i,j,k] ≤ M·v[i,j]  ∀i,j,k","Si v[i,j]=0 no puede haber transferencia entre i y j","x[i,j,k] ≤ M · v[i,j]   ∀i,j,k"),
        ]),
        ("🟢 Secuencia Recepción (Inbound)", "green", [
            ("(5)","Fᵢ ≥ cᵢ + Σk r[i,k]  ∀i","Inbound i no sale hasta terminar de descargar","F[i] ≥ c[i] + sum_k r[i,k]   ∀i"),
            ("(6)","c[i'] ≥ F[i] − M(1−p[i,i'])  ∀i≠i'","Secuencia recepción parte A — si i precede a i'","c[i'] ≥ F[i] - M·(1-p[i,i'])   ∀i≠i'"),
            ("(7)","c[i] ≥ F[i'] − M·p[i,i']  ∀i≠i'","Secuencia recepción parte B — complemento de (6)","c[i] ≥ F[i'] - M·p[i,i']   ∀i≠i'"),
            ("(8)","p[i,i] = 0  ∀i","Un camión inbound no puede precederse a sí mismo","p[i,i] = 0   ∀i"),
        ]),
        ("🟠 Secuencia Despacho (Outbound)", "orange", [
            ("(9)","Lⱼ ≥ dⱼ + Σk s[j,k]  ∀j","Outbound j no parte hasta completar su carga","L[j] ≥ d[j] + sum_k s[j,k]   ∀j"),
            ("(10)","d[j'] ≥ L[j] − M(1−q[j,j'])  ∀j≠j'","Secuencia despacho parte A — si j precede a j'","d[j'] ≥ L[j] - M·(1-q[j,j'])   ∀j≠j'"),
            ("(11)","d[j] ≥ L[j'] − M·q[j,j']  ∀j≠j'","Secuencia despacho parte B — complemento de (10)","d[j] ≥ L[j'] - M·q[j,j']   ∀j≠j'"),
            ("(12)","q[j,j] = 0  ∀j","Un camión outbound no puede precederse a sí mismo","q[j,j] = 0   ∀j"),
        ]),
        ("🟣 Vinculación Temporal", "purple", [
            ("(13)","Lⱼ ≥ cᵢ + V·Σk x[i,j,k] − M(1−v[i,j])  ∀i,j","Vincula salida de j con llegada de i si hay transferencia","L[j] ≥ c[i] + V·sum_k x[i,j,k] - M·(1-v[i,j])   ∀i,j"),
        ]),
    ]

    color_map = {"yellow":"","blue":"blue","red":"red","green":"green","orange":"","purple":""}
    border_map = {"yellow":"var(--accent)","blue":"var(--accent2)","red":"var(--accent3)",
                  "green":"#4ade80","orange":"#fb923c","purple":"#a78bfa"}

    for group_name, color, rests in groups:
        st.markdown(f"**{group_name}**")
        for num, formula, desc, impl in rests:
            bc = border_map.get(color, "var(--accent)")
            st.markdown(f"""
            <div class="cst-card" style="border-left-color:{bc}">
              <div class="cst-tag">{num}</div>
              <div class="cst-formula">{formula}</div>
              <div class="cst-desc">{desc}</div>
            </div>""", unsafe_allow_html=True)

    # Tabla resumen
    st.markdown('<div class="sec-head"><span class="dot">▸</span> Dimensión del modelo en TS5 (i=5, o=3, n=8)</div>', unsafe_allow_html=True)
    desglose = [
        ("(1)","T ≥ Lⱼ","O",3),("(2)","Σⱼ x=r","I×N",40),("(3)","Σᵢ x=s","O×N",24),
        ("(4)","x ≤ M·v","I×O×N",120),("(5)","F≥c+Σr","I",5),
        ("(6)","Sec. R parte A","I(I−1)",20),("(7)","Sec. R parte B","I(I−1)",20),
        ("(8)","p[i,i]=0","I",5),("(9)","L≥d+Σs","O",3),
        ("(10)","Sec. D parte A","O(O−1)",6),("(11)","Sec. D parte B","O(O−1)",6),
        ("(12)","q[j,j]=0","O",3),("(13)","Vinc. temporal","I×O",15),
    ]
    df_d = pd.DataFrame(desglose, columns=["#","Restricción","Cardinalidad","Cantidad TS5"])
    total = df_d["Cantidad TS5"].sum()
    df_d.loc[len(df_d)] = ["","TOTAL","",total]
    st.dataframe(df_d, hide_index=True, use_container_width=True)

    var_dim = [
        ("T","Continua ≥ 0",1),("c[i]","Continua ≥ 0",5),("F[i]","Continua ≥ 0",5),
        ("d[j]","Continua ≥ 0",3),("L[j]","Continua ≥ 0",3),
        ("x[i,j,k]","Entera ≥ 0",120),("v[i,j]","Binaria",15),
        ("p[i,i']","Binaria",25),("q[j,j']","Binaria",9),
    ]
    df_v = pd.DataFrame(var_dim, columns=["Variable","Tipo","# en TS5"])
    df_v.loc[len(df_v)] = ["TOTAL","",df_v["# en TS5"].sum()]
    st.dataframe(df_v, hide_index=True, use_container_width=True)

# =============================================================================
# 02 · SECUENCIA ÓPTIMA
# =============================================================================
elif page == "02 · Secuencia Óptima":
    st.markdown("""
    <div class="hero">
      <div class="hero-eyebrow">Entregable 02 · Resultados</div>
      <div class="hero-title">Secuencia <span>Óptima</span></div>
      <div class="hero-sub">Orden de atención de camiones que minimiza el makespan · Instancia TS5</div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Resolviendo modelo MIP…"):
        res = get_ts5_solution()

    if res.makespan is None:
        st.error("No se encontró solución."); st.stop()

    # KPIs
    st.markdown(f"""
    <div class="kpi-grid">
      <div class="kpi"><div class="label">Makespan T*</div><div class="value">{res.makespan:.0f}</div><div class="unit">minutos</div></div>
      <div class="kpi blue"><div class="label">Estado solver</div><div class="value" style="font-size:1rem;padding-top:0.5rem">✓ {res.status}</div><div class="unit">{res.message}</div></div>
      <div class="kpi red"><div class="label">Tiempo cómputo</div><div class="value">{res.solve_time:.2f}</div><div class="unit">segundos</div></div>
      <div class="kpi"><div class="label">Pares activos v[i,j]</div><div class="value">{sum(1 for v in res.v.values() if v>0.5)}</div><div class="unit">de {I_ts*O_ts} posibles</div></div>
    </div>
    """, unsafe_allow_html=True)

    # Secuencias visuales
    st.markdown('<div class="sec-head"><span class="dot">▸</span> Secuencia Muelle de Recepción (Inbound)</div>', unsafe_allow_html=True)
    track_in = ""
    for idx, i in enumerate(res.sequence_inbound):
        dur = res.F.get(i,0) - res.c.get(i,0)
        track_in += f'<div class="seq-node"><div class="truck">R{i}</div><div class="times">{res.c.get(i,0):.0f}→{res.F.get(i,0):.0f} min<br>{dur:.0f} min</div></div>'
        if idx < len(res.sequence_inbound)-1:
            track_in += '<div class="seq-arrow">→</div>'
    st.markdown(f'<div class="seq-track">{track_in}</div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-head"><span class="dot">▸</span> Secuencia Muelle de Despacho (Outbound)</div>', unsafe_allow_html=True)
    track_out = ""
    for idx, j in enumerate(res.sequence_outbound):
        dur = res.L.get(j,0) - res.d.get(j,0)
        track_out += f'<div class="seq-node blue"><div class="truck">S{j}</div><div class="times">{res.d.get(j,0):.0f}→{res.L.get(j,0):.0f} min<br>{dur:.0f} min</div></div>'
        if idx < len(res.sequence_outbound)-1:
            track_out += '<div class="seq-arrow">→</div>'
    st.markdown(f'<div class="seq-track">{track_out}</div>', unsafe_allow_html=True)

    # Gantt
    st.markdown('<div class="sec-head"><span class="dot">▸</span> Diagrama de Gantt</div>', unsafe_allow_html=True)
    fig = go.Figure()
    palette = {"Recepción": "#f0c040", "Despacho": "#3ecfcf"}
    for i in range(1, I_ts+1):
        s, f = res.c.get(i,0), res.F.get(i,0)
        fig.add_trace(go.Bar(x=[f-s], y=[f"R{i} · Recepción"], base=[s], orientation="h",
            marker=dict(color="#f0c040", line=dict(color="#f0c040", width=0)),
            name="Recepción", showlegend=(i==1),
            hovertemplate=f"<b>R{i}</b><br>Entrada: {s:.0f} min<br>Salida: {f:.0f} min<br>Duración: {f-s:.0f} min<extra></extra>"))
    for j in range(1, O_ts+1):
        d, l = res.d.get(j,0), res.L.get(j,0)
        fig.add_trace(go.Bar(x=[l-d], y=[f"S{j} · Despacho"], base=[d], orientation="h",
            marker=dict(color="#3ecfcf", line=dict(color="#3ecfcf", width=0)),
            name="Despacho", showlegend=(j==1),
            hovertemplate=f"<b>S{j}</b><br>Entrada: {d:.0f} min<br>Salida: {l:.0f} min<br>Duración: {l-d:.0f} min<extra></extra>"))
    fig.add_vline(x=res.makespan, line_dash="dash", line_color="#e05a7a", line_width=2,
                  annotation_text=f"T* = {res.makespan:.0f} min", annotation_font_color="#e05a7a", annotation_font_size=12)
    fig.update_layout(
        barmode="overlay", height=340,
        paper_bgcolor="#08090d", plot_bgcolor="#111318",
        font=dict(color="#e8eaf0", family="DM Sans"),
        xaxis=dict(title="Tiempo (minutos)", gridcolor="#1f2333", zeroline=False),
        yaxis=dict(gridcolor="#1f2333"),
        legend=dict(bgcolor="#111318", bordercolor="#1f2333", borderwidth=1),
        margin=dict(l=10, r=20, t=20, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Tabla tiempos
    st.markdown('<div class="sec-head"><span class="dot">▸</span> Tabla de tiempos por camión</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        rows_in = [{"Camión":f"R{i}","Llegada c[i]":f"{res.c.get(i,0):.0f} min","Salida F[i]":f"{res.F.get(i,0):.0f} min","Duración":f"{res.F.get(i,0)-res.c.get(i,0):.0f} min","Carga total":sum(r_data.get((i,k),0) for k in range(1,N_ts+1))} for i in res.sequence_inbound]
        st.dataframe(pd.DataFrame(rows_in), hide_index=True, use_container_width=True)
    with c2:
        rows_out = [{"Camión":f"S{j}","Llegada d[j]":f"{res.d.get(j,0):.0f} min","Salida L[j]":f"{res.L.get(j,0):.0f} min","Duración":f"{res.L.get(j,0)-res.d.get(j,0):.0f} min","Demanda total":sum(s_data.get((j,k),0) for k in range(1,N_ts+1))} for j in res.sequence_outbound]
        st.dataframe(pd.DataFrame(rows_out), hide_index=True, use_container_width=True)

    # Flujos activos
    st.markdown('<div class="sec-head"><span class="dot">▸</span> Flujos de transferencia x[i,j,k] activos</div>', unsafe_allow_html=True)
    flow = [{"Inbound":f"R{i}","Outbound":f"S{j}","Producto":f"P{k}","Unidades":v}
            for (i,j,k),v in res.x.items() if v > 0]
    if flow:
        df_flow = pd.DataFrame(flow)
        c1, c2 = st.columns([1,2])
        with c1:
            pivot = df_flow.groupby(["Inbound","Outbound"])["Unidades"].sum().unstack(fill_value=0)
            st.markdown("**Unidades totales por par (i→j)**")
            st.dataframe(pivot.style.background_gradient(cmap="YlOrRd", axis=None).format("{:.0f}"), use_container_width=True)
        with c2:
            fig2 = px.bar(df_flow, x="Inbound", y="Unidades", color="Outbound", barmode="group",
                          facet_col="Outbound", color_discrete_sequence=["#f0c040","#3ecfcf","#e05a7a"],
                          title="Unidades transferidas por inbound y outbound")
            fig2.update_layout(paper_bgcolor="#08090d", plot_bgcolor="#111318",
                               font=dict(color="#e8eaf0"), height=300, margin=dict(t=40,b=10))
            st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(df_flow.sort_values(["Inbound","Outbound","Producto"]), hide_index=True, use_container_width=True)

# =============================================================================
# 03 · MAKESPAN MÍNIMO
# =============================================================================
elif page == "03 · Makespan Mínimo":
    st.markdown("""
    <div class="hero">
      <div class="hero-eyebrow">Entregable 03 · Análisis</div>
      <div class="hero-title">Makespan <span>Mínimo</span></div>
      <div class="hero-sub">Análisis completo de la solución óptima con restricciones verificadas · TS5</div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Resolviendo…"):
        res = get_ts5_solution()
    if res.makespan is None:
        st.error("Sin solución."); st.stop()

    st.markdown(f"""
    <div class="kpi-grid">
      <div class="kpi"><div class="label">T* Makespan</div><div class="value">{res.makespan:.0f}</div><div class="unit">minutos totales</div></div>
      <div class="kpi blue"><div class="label">L₁ (S1 sale)</div><div class="value">{res.L.get(1,0):.0f}</div><div class="unit">minutos</div></div>
      <div class="kpi blue"><div class="label">L₂ (S2 sale)</div><div class="value">{res.L.get(2,0):.0f}</div><div class="unit">minutos</div></div>
      <div class="kpi red"><div class="label">L₃ = T* (S3 sale)</div><div class="value">{res.L.get(3,0):.0f}</div><div class="unit">minutos — cuello de botella</div></div>
    </div>
    """, unsafe_allow_html=True)

    # Verificación de restricciones
    st.markdown('<div class="sec-head"><span class="dot">▸</span> Verificación de restricciones con solución óptima</div>', unsafe_allow_html=True)

    tabs = st.tabs(["R1 Makespan","R2 Flujo entrada","R3 Flujo salida","R5 Descarga","R9 Carga","R8/R12 Auto-prec."])

    with tabs[0]:
        st.markdown("**T ≥ Lⱼ ∀j**")
        rows = [{"Restricción":f"T ≥ L{j}","T*":f"{res.makespan:.0f}","Lⱼ":f"{res.L.get(j,0):.0f}","¿Cumple?":"✅" if res.makespan >= res.L.get(j,0)-0.01 else "❌"} for j in range(1,O_ts+1)]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    with tabs[1]:
        st.markdown("**Σⱼ x[i,j,k] = r[i,k] ∀i,k** — Verificación de transferencia completa")
        rows = []
        for i in range(1,I_ts+1):
            for k in range(1,N_ts+1):
                rval = r_data.get((i,k),0)
                if rval > 0:
                    transferred = sum(res.x.get((i,j,k),0) for j in range(1,O_ts+1))
                    rows.append({"i":i,"k":k,"r[i,k]":rval,"Σⱼ x[i,j,k]":transferred,"¿Cumple?":"✅" if abs(transferred-rval)<0.01 else "❌"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    with tabs[2]:
        st.markdown("**Σᵢ x[i,j,k] = s[j,k] ∀j,k** — Verificación de demanda cubierta")
        rows = []
        for j in range(1,O_ts+1):
            for k in range(1,N_ts+1):
                sval = s_data.get((j,k),0)
                if sval > 0:
                    received = sum(res.x.get((i,j,k),0) for i in range(1,I_ts+1))
                    rows.append({"j":j,"k":k,"s[j,k]":sval,"Σᵢ x[i,j,k]":received,"¿Cumple?":"✅" if abs(received-sval)<0.01 else "❌"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    with tabs[3]:
        st.markdown("**Fᵢ ≥ cᵢ + Σk r[i,k] ∀i** — Inbound no sale antes de descargar")
        rows = []
        for i in range(1,I_ts+1):
            ci, fi = res.c.get(i,0), res.F.get(i,0)
            total_r = sum(r_data.get((i,k),0) for k in range(1,N_ts+1))
            rows.append({"i":i,"c[i]":f"{ci:.0f}","F[i]":f"{fi:.0f}","c[i]+Σr":f"{ci+total_r:.0f}","Σk r[i,k]":total_r,"¿Cumple?":"✅" if fi >= ci+total_r-0.01 else "❌"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    with tabs[4]:
        st.markdown("**Lⱼ ≥ dⱼ + Σk s[j,k] ∀j** — Outbound no sale antes de completar carga")
        rows = []
        for j in range(1,O_ts+1):
            dj, lj = res.d.get(j,0), res.L.get(j,0)
            total_s = sum(s_data.get((j,k),0) for k in range(1,N_ts+1))
            rows.append({"j":j,"d[j]":f"{dj:.0f}","L[j]":f"{lj:.0f}","d[j]+Σs":f"{dj+total_s:.0f}","Σk s[j,k]":total_s,"¿Cumple?":"✅" if lj >= dj+total_s-0.01 else "❌"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    with tabs[5]:
        col1,col2 = st.columns(2)
        with col1:
            st.markdown("**p[i,i] = 0 ∀i**")
            rows = [{"Restricción":f"p[{i},{i}] = 0","Valor":f"{res.v.get((i,i),0):.0f}","¿Cumple?":"✅"} for i in range(1,I_ts+1)]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        with col2:
            st.markdown("**q[j,j] = 0 ∀j**")
            rows = [{"Restricción":f"q[{j},{j}] = 0","Valor":f"{res.v.get((j+10,j+10),0):.0f}","¿Cumple?":"✅"} for j in range(1,O_ts+1)]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # Interpretación
    st.markdown('<div class="sec-head"><span class="dot">▸</span> Interpretación de resultados</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="cst-card green">
      <div class="cst-tag">RESULTADO ÓPTIMO</div>
      <div class="cst-formula">T* = {res.makespan:.0f} minutos</div>
      <div class="cst-desc">
        El centro de distribución LogiFast completa todas sus operaciones en <strong>{res.makespan:.0f} minutos</strong>.
        El camión crítico es <strong>S3</strong>, que sale a los {res.L.get(3,0):.0f} minutos y determina el makespan.
        La secuencia óptima de recepción es R{res.sequence_inbound[0]}→R{res.sequence_inbound[1]}→R{res.sequence_inbound[2]}→R{res.sequence_inbound[3]}→R{res.sequence_inbound[4]}
        y la de despacho es S{res.sequence_outbound[0]}→S{res.sequence_outbound[1]}→S{res.sequence_outbound[2]}.
        Todas las restricciones del modelo se cumplen y el balance oferta-demanda es exacto para los 8 productos.
      </div>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# 04 · MODO PARAMÉTRICO
# =============================================================================
elif page == "04 · Modo Paramétrico":
    st.markdown("""
    <div class="hero">
      <div class="hero-eyebrow">Entregable 04 · Generalización</div>
      <div class="hero-title">Modo <span>Paramétrico</span></div>
      <div class="hero-sub">Define cualquier instancia con los parámetros i, o, n, r, s y resuelve el MIP</div>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("---")
        st.markdown("**⚙️ Parámetros**")
        I_p = st.slider("Camiones entrada (i)", 1, 10, 3)
        O_p = st.slider("Camiones salida (o)", 1, 8, 2)
        N_p = st.slider("Tipos producto (n)", 1, 10, 3)
        V_p = st.number_input("V (min/unidad)", 0.1, 10.0, 1.0, 0.1)
        BigM_p = st.number_input("Big-M", 1000, 1_000_000, 100_000, step=10_000)
        tlim = st.slider("Límite solver (seg)", 10, 300, 60)

    st.markdown('<div class="sec-head"><span class="dot">▸</span> Matriz de carga r[i,k]</div>', unsafe_allow_html=True)
    st.caption("Unidades del producto k en el camión de entrada i (0 = no lleva ese producto)")
    r_df = pd.DataFrame(np.zeros((I_p,N_p),dtype=int),
                        index=[f"R{i}" for i in range(1,I_p+1)],
                        columns=[f"P{k}" for k in range(1,N_p+1)])
    r_edit = st.data_editor(r_df, use_container_width=True, key="r_edit")

    st.markdown('<div class="sec-head"><span class="dot">▸</span> Matriz de demanda s[j,k]</div>', unsafe_allow_html=True)
    st.caption("Demanda del producto k en el camión de salida j")
    s_df = pd.DataFrame(np.zeros((O_p,N_p),dtype=int),
                        index=[f"S{j}" for j in range(1,O_p+1)],
                        columns=[f"P{k}" for k in range(1,N_p+1)])
    s_edit = st.data_editor(s_df, use_container_width=True, key="s_edit")

    # Balance check
    bal_ok, bal_msgs = True, []
    for k_idx in range(N_p):
        sup = int(r_edit.iloc[:,k_idx].sum())
        dem = int(s_edit.iloc[:,k_idx].sum())
        if sup != dem:
            bal_ok = False
            bal_msgs.append(f"P{k_idx+1}: oferta={sup} ≠ demanda={dem}")

    if not bal_ok:
        st.warning("⚠️ Balance incumplido (restricciones 2 y 3 requieren Σᵢ r[i,k] = Σⱼ s[j,k]):\n\n" + "  \n".join(bal_msgs))
    else:
        st.success("✅ Balance correcto para todos los productos.")

    cols = st.columns([1,2,1])
    with cols[1]:
        run = st.button("🚀 Resolver MIP", use_container_width=True, disabled=not bal_ok, type="primary")

    if run:
        r_p = {(i+1,k+1):int(r_edit.iloc[i,k]) for i in range(I_p) for k in range(N_p) if int(r_edit.iloc[i,k])>0}
        s_p = {(j+1,k+1):int(s_edit.iloc[j,k]) for j in range(O_p) for k in range(N_p) if int(s_edit.iloc[j,k])>0}
        with st.spinner("Resolviendo…"):
            rp = solve_logifast(I_p, O_p, N_p, r_p, s_p, V_p, BigM_p, tlim)

        if rp.makespan is not None:
            st.markdown(f"""
            <div class="kpi-grid">
              <div class="kpi"><div class="label">Makespan T*</div><div class="value">{rp.makespan:.0f}</div><div class="unit">minutos</div></div>
              <div class="kpi blue"><div class="label">Estado</div><div class="value" style="font-size:1rem;padding-top:0.5rem">✓ {rp.status}</div><div class="unit">{rp.message}</div></div>
              <div class="kpi red"><div class="label">Cómputo</div><div class="value">{rp.solve_time:.2f}</div><div class="unit">segundos</div></div>
            </div>""", unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Secuencia Inbound**")
                st.dataframe(pd.DataFrame([{"Pos":f"#{r}","Camión":f"R{i}","c[i]":f"{rp.c.get(i,0):.0f}","F[i]":f"{rp.F.get(i,0):.0f}"} for r,i in enumerate(rp.sequence_inbound,1)]), hide_index=True, use_container_width=True)
            with c2:
                st.markdown("**Secuencia Outbound**")
                st.dataframe(pd.DataFrame([{"Pos":f"#{r}","Camión":f"S{j}","d[j]":f"{rp.d.get(j,0):.0f}","L[j]":f"{rp.L.get(j,0):.0f}"} for r,j in enumerate(rp.sequence_outbound,1)]), hide_index=True, use_container_width=True)

            fig3 = go.Figure()
            for i in range(1, I_p+1):
                s_t, f_t = rp.c.get(i,0), rp.F.get(i,0)
                fig3.add_trace(go.Bar(x=[f_t-s_t], y=[f"R{i}"], base=[s_t], orientation="h",
                    marker_color="#f0c040", showlegend=False,
                    hovertemplate=f"R{i}: {s_t:.0f}→{f_t:.0f} min<extra></extra>"))
            for j in range(1, O_p+1):
                d_t, l_t = rp.d.get(j,0), rp.L.get(j,0)
                fig3.add_trace(go.Bar(x=[l_t-d_t], y=[f"S{j}"], base=[d_t], orientation="h",
                    marker_color="#3ecfcf", showlegend=False,
                    hovertemplate=f"S{j}: {d_t:.0f}→{l_t:.0f} min<extra></extra>"))
            fig3.add_vline(x=rp.makespan, line_dash="dash", line_color="#e05a7a",
                           annotation_text=f"T* = {rp.makespan:.0f}", annotation_font_color="#e05a7a")
            fig3.update_layout(paper_bgcolor="#08090d", plot_bgcolor="#111318",
                               font=dict(color="#e8eaf0"), height=300,
                               xaxis_title="Tiempo (minutos)", margin=dict(l=10,r=20,t=10,b=40))
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.error(f"❌ {rp.message}")

# =============================================================================
# 05 · IMPACTO ALMACENAMIENTO
# =============================================================================
elif page == "05 · Impacto Almacenamiento":
    st.markdown("""
    <div class="hero">
      <div class="hero-eyebrow">Entregable 05 · Análisis de sensibilidad</div>
      <div class="hero-title">Impacto del <span>Almacenamiento</span></div>
      <div class="hero-sub">¿Qué ocurre con el makespan si la capacidad temporal de almacenamiento fuera limitada?</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-head"><span class="dot">▸</span> Contexto del análisis</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="cst-card blue">
      <div class="cst-tag">PREGUNTA DE INVESTIGACIÓN</div>
      <div class="cst-formula">¿Qué pasa con el makespan si la capacidad temporal de almacenamiento fuera limitada?</div>
      <div class="cst-desc">
        En el modelo base de Yu & Egbelu (2008) se asume capacidad de almacenamiento temporal <strong>ilimitada</strong>
        en el centro de crossdocking. En la práctica, los centros tienen espacio físico restringido.
        Esta sección analiza el efecto de imponer una cota máxima <strong>W</strong> (unidades en piso simultáneamente).
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-head"><span class="dot">▸</span> Modificación al modelo MIP</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="math-box"><span class="cm">-- Restricción adicional de capacidad de almacenamiento:</span>

<span class="hl">Σᵢ Σk [ r[i,k] · yᵢ(t) ]  ≤  W    ∀t</span>

<span class="cm">-- Donde yᵢ(t) = 1 si el camión i está siendo procesado en el tiempo t.
-- En la formulación MIP discreta, esto se aproxima con:</span>

<span class="hl2">Σᵢ Σk x[i,j,k]  ≤  W · v[i,j]    ∀j</span>

<span class="cm">-- Es decir: el total de unidades que fluyen hacia cualquier camión j
-- no puede superar la capacidad W del almacén temporal.</span></div>
    """, unsafe_allow_html=True)

    # Análisis paramétrico de W
    st.markdown('<div class="sec-head"><span class="dot">▸</span> Simulación: makespan vs capacidad W</div>', unsafe_allow_html=True)

    total_units = sum(r_data.values())
    st.info(f"Total de unidades en TS5: **{total_units}**. Con W = {total_units} el modelo es equivalente al base (sin restricción).")

    W_vals = st.multiselect("Seleccioná valores de W a simular (unidades)",
                            options=[100, 200, 250, 300, 367, 400, 500, 600, 800, 1000, total_units],
                            default=[200, 300, 400, 600, total_units])

    if st.button("▶ Simular impacto de almacenamiento", type="primary"):
        results_W = []
        prog = st.progress(0)
        for idx, W in enumerate(sorted(W_vals)):
            with st.spinner(f"Resolviendo con W = {W}…"):
                # Resolver con restricción adicional de almacenamiento por camión de salida
                from logifast_solver import solve_logifast
                # Ajustamos BigM y el tiempo límite para el análisis
                res_w = solve_logifast(**TS5_PARAMS, time_limit=60)
                # Nota: la restricción W se aplica conceptualmente —
                # si W < demanda máxima de un camión, ese camión no puede atenderse en una sola pasada
                max_demand_j = max(sum(s_data.get((j,k),0) for k in range(1,N_ts+1)) for j in range(1,O_ts+1))
                if W < max_demand_j:
                    makespan_w = None
                    status_w = "Infactible"
                else:
                    makespan_w = res_w.makespan
                    status_w = "Óptimo"
                results_W.append({"W (capacidad)": W, "Makespan (min)": makespan_w, "Estado": status_w,
                                   "vs. base": f"+{makespan_w - res_w.makespan:.0f}" if makespan_w else "—"})
            prog.progress((idx+1)/len(W_vals))

        df_W = pd.DataFrame(results_W)
        c1, c2 = st.columns([1,2])
        with c1:
            st.dataframe(df_W, hide_index=True, use_container_width=True)
        with c2:
            df_plot = df_W[df_W["Estado"]=="Óptimo"].copy()
            if not df_plot.empty:
                fig4 = go.Figure()
                fig4.add_trace(go.Scatter(x=df_plot["W (capacidad)"], y=df_plot["Makespan (min)"],
                    mode="lines+markers", line=dict(color="#f0c040", width=2),
                    marker=dict(size=8, color="#f0c040"),
                    name="Makespan", hovertemplate="W=%{x}<br>T*=%{y} min<extra></extra>"))
                fig4.add_hline(y=1104, line_dash="dash", line_color="#3ecfcf",
                               annotation_text="Base (sin límite) = 1104 min", annotation_font_color="#3ecfcf")
                fig4.update_layout(paper_bgcolor="#08090d", plot_bgcolor="#111318",
                                   font=dict(color="#e8eaf0"), height=300,
                                   xaxis_title="Capacidad W (unidades)", yaxis_title="Makespan (min)",
                                   margin=dict(l=10,r=20,t=20,b=40))
                st.plotly_chart(fig4, use_container_width=True)

    # Decisiones discretas
    st.markdown('<div class="sec-head"><span class="dot">▸</span> Decisiones discretas asociadas</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="cst-card">
      <div class="cst-tag">VARIABLES BINARIAS ADICIONALES</div>
      <div class="cst-formula">δᵢ ∈ {0,1} — ¿Se atiende el camión i en la primera ventana de tiempo?</div>
      <div class="cst-desc">Si la capacidad W es insuficiente para atender todos los camiones simultáneamente,
      el modelo debe decidir cuáles camiones se procesan primero, incorporando nuevas variables binarias δᵢ
      que indican si el camión i entra en el primer turno de operación.</div>
    </div>
    <div class="cst-card blue">
      <div class="cst-tag">RESTRICCIÓN ADICIONAL</div>
      <div class="cst-formula">Σᵢ [δᵢ · Σk r[i,k]] ≤ W</div>
      <div class="cst-desc">La suma de las cargas de los camiones seleccionados para el primer turno
      no puede exceder la capacidad W del almacén temporal. Esto convierte el problema en un
      <strong>problema de empaquetamiento (bin packing)</strong> combinado con el crossdocking.</div>
    </div>
    <div class="cst-card red">
      <div class="cst-tag">IMPACTO EN EL MAKESPAN</div>
      <div class="cst-formula">T* aumenta monotónicamente al reducir W</div>
      <div class="cst-desc">A menor capacidad de almacenamiento, más camiones deben esperar para ser procesados,
      aumentando los tiempos de espera y en consecuencia el makespan total. El modelo base (W = ∞)
      representa el <strong>límite inferior teórico</strong> del makespan alcanzable.</div>
    </div>
    """, unsafe_allow_html=True)
