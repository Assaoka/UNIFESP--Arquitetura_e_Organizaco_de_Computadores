import streamlit as st
import numpy as np
import plotly.express as px

from numpy import sqrt, pi, exp
from scipy.special import gamma, beta

st.set_page_config(page_title="Distribuições e Testes", layout="wide")
st.html('''
<style>
    #MainMenu {visibility: collapsed;}
    footer {visibility: hidden;}
    header {visibility: hidden;} 
</style>''')

# Funções lambda para as distribuições
normal_pdf = lambda x, loc=0, scale=1: 1/(scale * sqrt(2*pi)) * exp(-0.5 * ((x - loc) / scale)**2)
t_pdf = lambda x, df: gamma((df + 1) / 2) / (sqrt(df * pi) * gamma(df / 2)) * (1 + (x**2) / df) ** (-(df + 1) / 2)
chi2_pdf = lambda x, k: np.where(x > 0, (1 / (2 ** (k / 2) * gamma(k / 2))) * x ** (k / 2 - 1) * np.exp(-x / 2), 0)
F_pdf = lambda x, dfn, dfd: np.where(x > 0, ((dfn / dfd) ** (dfn / 2) * x ** (dfn / 2 - 1)) / (beta(dfn / 2, dfd / 2) * (1 + (dfn / dfd) * x) ** ((dfn + dfd) / 2)), 0)

# Dicionário de distribuições e fórmulas em LaTeX
distribuicoes = {
    'Normal': normal_pdf,
    't de Student': t_pdf,
    'Qui-quadrado': chi2_pdf,
    'F': F_pdf
}

latex_formulas = {
    'Normal': r'f(x) = \frac{1}{\sigma \sqrt{2\pi}} \exp\left(-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2\right)',
    't de Student': r'f(x) = \frac{\Gamma\left(\frac{\nu+1}{2}\right)}{\sqrt{\nu\pi}\,\Gamma\left(\frac{\nu}{2}\right)} \left(1+\frac{x^2}{\nu}\right)^{-\frac{\nu+1}{2}}',
    'Qui-quadrado': r'f(x) = \frac{1}{2^{k/2}\Gamma\left(\frac{k}{2}\right)} x^{\frac{k}{2}-1} \exp\left(-\frac{x}{2}\right), \quad x>0',
    'F': r'f(x) = \frac{\left(\frac{d_1}{d_2}\right)^{\frac{d_1}{2}} x^{\frac{d_1}{2}-1}}{B\left(\frac{d_1}{2}, \frac{d_2}{2}\right) \left(1+\frac{d_1}{d_2}x\right)^{\frac{d_1+d_2}{2}}}, \quad x>0'
}

# Interface na sidebar
with st.sidebar:
    with st.container(border=True):
        st.markdown("## Parametros da Distribuição")
        dist_selected = st.selectbox('Distribuição', list(distribuicoes.keys()))

        # Parâmetros específicos para cada distribuição
        if dist_selected == 'Normal':
            cols = st.columns(2)
            loc = cols[0].number_input('Média ($μ$)', value=0.0)
            scale = cols[1].number_input('Desvio padrão ($σ$)', value=1.0, min_value=0.1)
            f = lambda x: distribuicoes['Normal'](x, loc, scale)
        elif dist_selected == 't de Student':
            df_val = st.number_input('Graus de liberdade ($ν$)', value=1, min_value=1)
            f = lambda x: distribuicoes['t de Student'](x, df_val)
        elif dist_selected == 'Qui-quadrado':
            k = st.number_input('Graus de liberdade ($k$)', value=1, min_value=1)
            f = lambda x: distribuicoes['Qui-quadrado'](x, k)
        elif dist_selected == 'F':
            cols = st.columns(2)
            dfn = cols[0].number_input('Graus de liberdade do numerador ($d_1$)', value=1, min_value=1)
            dfd = cols[1].number_input('Graus de liberdade do denominador ($d_2$)', value=1, min_value=1)
            f = lambda x: distribuicoes['F'](x, dfn, dfd)

    with st.container(border=True):
        st.markdown("## Intervalo e Teste de Hipóteses")

        # Tipo de teste e nível de significância
        tipo = st.selectbox('Tipo de Teste', ['Bilateral', 'Unilateral à direita', 'Unilateral à esquerda'])
        alpha = st.number_input('Nível de significância ($α$)', value=0.05, step=0.01, min_value=0.001, max_value=0.5)

    with st.container(border=True):
        st.markdown("## Visualização da Distribuição")
        n_div = st.number_input('Número de divisões', value=1000, step=100, max_value=10000, format="%d")

        # Definição do intervalo (para distribuições com suporte em todo o real ou somente x>0)
        cols = st.columns(2)
        if dist_selected in ['Normal', 't de Student']:
            xini = cols[0].number_input('Início do intervalo', value=-5.0, step=1.0)
            xfim = cols[1].number_input('Fim do intervalo', value=5.0, step=1.0, min_value=xini)
        else:
            xini = cols[0].number_input('Início do intervalo', value=0.0, step=1.0, min_value=0.0)
            xfim = cols[1].number_input('Fim do intervalo', value=10.0, step=1.0, min_value=xini)


# Cria o vetor x e calcula y
x = np.linspace(xini, xfim, int(n_div))
y = f(x)

st.write(f"# Distribuição {dist_selected}:")

# Exibe a fórmula da distribuição selecionada
st.markdown("##### Fórmula da Distribuição:")
st.latex(latex_formulas[dist_selected])

# Função de integração pelo método do Trapézio Composto
def trapezoidal_composta(x, y, n):
    h = (x[-1] - x[0]) / n
    return h * (y[0] + 2 * np.sum(y[1:-1]) + y[-1]) / 2

# Funções de bisseção para encontrar os valores críticos
def critical_value_right(x, y, alpha):
    low, high = x[0], x[-1]
    tol = x[1] - x[0]
    while high - low > tol:
        mid = (low + high) / 2
        mask = (x >= mid) & (x <= x[-1])
        area = trapezoidal_composta(x[mask], y[mask], len(x[mask]) - 1)
        if area > alpha:
            low = mid
        else:
            high = mid
    return (low + high) / 2

def critical_value_left(x, y, alpha):
    low, high = x[0], x[-1]
    tol = x[1] - x[0]
    while high - low > tol:
        mid = (low + high) / 2
        mask = (x >= x[0]) & (x <= mid)
        area = trapezoidal_composta(x[mask], y[mask], len(x[mask]) - 1)
        if area < alpha:
            low = mid
        else:
            high = mid
    return (low + high) / 2



# Cálculo dos valores críticos conforme o tipo de teste
if tipo == 'Unilateral à direita':
    crit = critical_value_right(x, y, alpha)
    st.write(f'##### Valor crítico (Unilateral à direita): ${crit:.4f}$')
elif tipo == 'Unilateral à esquerda':
    crit = critical_value_left(x, y, alpha)
    st.write(f'##### Valor crítico (Unilateral à esquerda): ${crit:.4f}$')
else:
    crit_left = critical_value_left(x, y, alpha/2)
    crit_right = critical_value_right(x, y, alpha/2)
    st.write(f'##### Valores críticos (Bilateral): ${crit_left:.4f}$ e ${crit_right:.4f}$')

st.write(f'##### Área sob a curva: ${trapezoidal_composta(x, y, int(n_div)):.4f}$')

# Plot da distribuição com a região crítica destacada
fig = px.line(x=x, y=y, labels={'x': 'x', 'y': 'f(x)'}, title=f'Distribuição {dist_selected}')
if tipo == 'Unilateral à direita':
    crit = critical_value_right(x, y, alpha)
    mask = x >= crit
    fig.add_scatter(x=x[mask], y=y[mask], fill='tozeroy', mode='none', name='Região Crítica')
elif tipo == 'Unilateral à esquerda':
    crit = critical_value_left(x, y, alpha)
    mask = x <= crit
    fig.add_scatter(x=x[mask], y=y[mask], fill='tozeroy', mode='none', name='Região Crítica')
elif tipo == 'Bilateral':
    crit_left = critical_value_left(x, y, alpha/2)
    crit_right = critical_value_right(x, y, alpha/2)
    mask_left = x <= crit_left
    mask_right = x >= crit_right
    fig.add_scatter(x=x[mask_left], y=y[mask_left], fill='tozeroy', mode='none', name='Região Crítica Esquerda')
    fig.add_scatter(x=x[mask_right], y=y[mask_right], fill='tozeroy', mode='none', name='Região Crítica Direita')

st.plotly_chart(fig)
