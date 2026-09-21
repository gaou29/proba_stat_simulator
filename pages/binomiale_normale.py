import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from scipy.stats import binom, norm

st.set_page_config(
    page_title="Binomiale → Normale",
    layout="wide"
)


st.title("Convergence de la loi binomiale vers la loi normale")

st.markdown("""
Cette simulation permet d'observer comment la loi binomiale
se rapproche progressivement d'une loi normale lorsque le
nombre d'essais augmente.
""")


st.sidebar.header("Paramètres pour la loi binomial B(n,p)")

n = st.sidebar.slider(
    "Nombre d'essais (n)",
    min_value=1,
    max_value=200,
    value=10
)

p = st.sidebar.slider(
    "Probabilité de succès (p)",
    min_value=0.01,
    max_value=0.99,
    value=0.50,
    step=0.01
)

# Binomial

x = np.arange(0, n + 1)

probabilites = binom.pmf(x, n, p)

moyenne = n * p
variance = n * p * (1 - p)
ecart_type = np.sqrt(variance)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Moyenne E(X)",
        f"{moyenne:.2f}"
    )

with col2:
    st.metric(
        "Variance",
        f"{variance:.2f}"
    )

with col3:
    st.metric(
        "Écart-type σ",
        f"{ecart_type:.2f}"
    )
    
# Graph

# Axe continu pour la loi normale
x_normal = np.linspace(
    max(0, moyenne - 4 * ecart_type),
    min(n, moyenne + 4 * ecart_type),
    500
)

# Densité de la loi normale
densite_normale = norm.pdf(
    x_normal,
    loc=moyenne,
    scale=ecart_type
)

fig, ax = plt.subplots(figsize=(12, 6))

ax.bar(
    x,
    probabilites,
    width=0.8,
    alpha=0.6,
    label=f"Binomiale B({n}, {p})"
)

ax.plot(
    x_normal,
    densite_normale,
    linewidth=2,
    label=f"Normale N({moyenne:.2f}, {variance:.2f})"
)

ax.axvline(
    moyenne,
    linestyle="--",
    linewidth=1.5,
    label=f"μ = {moyenne:.2f}"
)


ax.set_title(
    "Convergence de la loi binomiale vers la loi normale"
)

ax.set_xlabel("Nombre de succès x")
ax.set_ylabel("Probabilité / densité")

ax.legend()

ax.grid(
    axis="y",
    alpha=0.3
)

st.pyplot(fig)