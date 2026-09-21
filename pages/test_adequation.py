import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import (
    chisquare,
    norm,
    poisson,
    binom
)


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Test d'adéquation",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# TITRE
# ============================================================

st.title("📊 Test d'adéquation d'une loi")

st.markdown("""
Le test d'adéquation permet de déterminer si des données
observées sont compatibles avec une loi théorique donnée.

Nous utilisons ici le **test du Khi-deux d'adéquation**.
""")


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Paramètres")


law = st.sidebar.selectbox(
    "Loi théorique",
    [
        "Normale",
        "Poisson",
        "Binomiale"
    ]
)


sample_size = st.sidebar.slider(
    "Nombre d'observations",
    min_value=50,
    max_value=5000,
    value=1000,
    step=50
)


alpha = st.sidebar.selectbox(
    "Niveau de signification α",
    [0.01, 0.05, 0.10],
    index=1
)


st.sidebar.markdown("---")


# ============================================================
# PARAMÈTRES DE LA LOI
# ============================================================

st.sidebar.subheader("Paramètres de la loi")


if law == "Normale":

    mu = st.sidebar.number_input(
        "Moyenne μ",
        value=50.0
    )

    sigma = st.sidebar.number_input(
        "Écart-type σ",
        min_value=0.1,
        value=10.0
    )


elif law == "Poisson":

    lambda_param = st.sidebar.number_input(
        "Paramètre λ",
        min_value=0.1,
        value=5.0
    )


else:

    n_binom = st.sidebar.number_input(
        "Nombre d'essais n",
        min_value=1,
        value=10
    )

    p_binom = st.sidebar.slider(
        "Probabilité p",
        min_value=0.01,
        max_value=0.99,
        value=0.50,
        step=0.01
    )


# ============================================================
# GÉNÉRATION DES DONNÉES
# ============================================================

st.subheader("Génération des données")

generate = st.button(
    "Générer les données"
)


if generate:

    if law == "Normale":

        data = np.random.normal(
            loc=mu,
            scale=sigma,
            size=sample_size
        )

    elif law == "Poisson":

        data = np.random.poisson(
            lam=lambda_param,
            size=sample_size
        )

    else:

        data = np.random.binomial(
            n=n_binom,
            p=p_binom,
            size=sample_size
        )

    st.session_state["data"] = data
    st.session_state["law"] = law


# ============================================================
# VÉRIFIER SI DES DONNÉES EXISTENT
# ============================================================

if "data" not in st.session_state:

    st.info(
        "Cliquez sur **Générer les données** pour commencer la simulation."
    )

    st.stop()


data = st.session_state["data"]


# ============================================================
# INFORMATIONS SUR LES DONNÉES
# ============================================================

st.subheader("📋 Données générées")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Nombre d'observations",
        len(data)
    )

with col2:
    st.metric(
        "Moyenne observée",
        f"{np.mean(data):.2f}"
    )

with col3:
    st.metric(
        "Écart-type observé",
        f"{np.std(data, ddof=1):.2f}"
    )


# ============================================================
# CRÉATION DES CLASSES
# ============================================================


if law == "Normale":
    number_classes = 10

    edges = np.linspace(
        np.min(data),
        np.max(data),
        number_classes + 1
    )

    observed, _ = np.histogram(
        data,
        bins=edges
    )

    theoretical_probabilities = (
        norm.cdf(edges[1:], mu, sigma)
        - norm.cdf(edges[:-1], mu, sigma)
    )

    expected = theoretical_probabilities * sample_size


elif law == "Poisson":

    max_value = int(np.max(data))

    values = np.arange(
        0,
        max_value + 1
    )

    observed = np.bincount(
        data.astype(int),
        minlength=max_value + 1
    )

    theoretical_probabilities = poisson.pmf(
        values,
        lambda_param
    )

    expected = theoretical_probabilities * sample_size

    edges = values


else:

    values = np.arange(
        0,
        n_binom + 1
    )

    observed = np.bincount(
        data.astype(int),
        minlength=n_binom + 1
    )

    theoretical_probabilities = binom.pmf(
        values,
        n_binom,
        p_binom
    )

    expected = theoretical_probabilities * sample_size

    edges = values


# ============================================================
# REGROUPEMENT DES PETITS EFFECTIFS
# ============================================================

mask = expected >= 5

if np.sum(mask) >= 2:

    observed_test = observed[mask]
    expected_test = expected[mask]

else:

    observed_test = observed
    expected_test = expected


# ============================================================
# NORMALISATION DES EFFECTIFS THÉORIQUES
# ============================================================

expected_test = (
    expected_test
    * observed_test.sum()
    / expected_test.sum()
)


# ============================================================
# TEST DU KHI-DEUX
# ============================================================

chi2_stat, p_value = chisquare(
    f_obs=observed_test,
    f_exp=expected_test
)


# ============================================================
# RÉSULTATS
# ============================================================

st.subheader("Résultat du test")


col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "χ²",
        f"{chi2_stat:.4f}"
    )

with col2:

    st.metric(
        "p-value",
        f"{p_value:.4f}"
    )

with col3:

    st.metric(
        "α",
        f"{alpha:.2f}"
    )


# ============================================================
# DÉCISION
# ============================================================

st.subheader("Décision statistique")


if p_value <= alpha:

    st.error(
        f"""
        ### ❌ Rejet de H₀

        La p-value est :

        **{p_value:.4f}**

        et

        **{p_value:.4f} ≤ {alpha:.2f}**

        On rejette donc l'hypothèse :

        > H₀ : les données suivent la loi théorique choisie.

        Les données observées présentent une différence
        statistiquement significative avec la loi théorique.
        """
    )

else:

    st.success(
        f"""
        ### ✅ On ne rejette pas H₀

        La p-value est :

        **{p_value:.4f}**

        et

        **{p_value:.4f} > {alpha:.2f}**

        On ne rejette donc pas l'hypothèse :

        > H₀ : les données sont compatibles avec la loi théorique choisie.

        Il n'y a pas suffisamment d'éléments statistiques
        pour conclure à une différence significative.
        """
    )


# ============================================================
# GRAPHIQUE
# ============================================================

st.subheader("📈 Observé vs théorique")


fig, ax = plt.subplots(
    figsize=(12, 6)
)


if law == "Normale":

    positions = np.arange(
        len(observed)
    )

    labels = [
        f"{edges[i]:.1f}–{edges[i+1]:.1f}"
        for i in range(len(edges) - 1)
    ]

    ax.bar(
        positions - 0.2,
        observed,
        width=0.4,
        alpha=0.7,
        label="Observé"
    )

    ax.bar(
        positions + 0.2,
        expected,
        width=0.4,
        alpha=0.7,
        label="Théorique"
    )

    ax.set_xticks(
        positions
    )

    ax.set_xticklabels(
        labels,
        rotation=45,
        ha="right"
    )


else:

    positions = np.arange(
        len(observed)
    )

    ax.bar(
        positions - 0.2,
        observed,
        width=0.4,
        alpha=0.7,
        label="Observé"
    )

    ax.bar(
        positions + 0.2,
        expected,
        width=0.4,
        alpha=0.7,
        label="Théorique"
    )

    ax.set_xticks(
        positions
    )

    ax.set_xticklabels(
        edges
    )


ax.set_xlabel("Classes / valeurs")
ax.set_ylabel("Effectifs")

ax.set_title(
    f"Adéquation à la loi {law}"
)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.3
)

st.pyplot(fig)


# ============================================================
# INTERPRÉTATION
# ============================================================

st.subheader("💡 Interprétation")


st.markdown(f"""
### Hypothèses

**H₀ :** les données suivent la loi {law}.

**H₁ :** les données ne suivent pas la loi {law}.

Le test compare les effectifs observés aux effectifs
théoriquement attendus sous H₀.

La statistique utilisée est :

$$
\\chi^2 =
\\sum_i \\frac{{(O_i-E_i)^2}}{{E_i}}
$$

où :

- $O_i$ représente l'effectif observé ;
- $E_i$ représente l'effectif théorique.

Le seuil utilisé ici est :

$$
\\alpha = {alpha}
$$
""")