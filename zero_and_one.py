import numpy as np
from scipy import stats, special, optimize

def indicator_function(cs: set[float]):
    def evaluate(y: float) -> float:
        if y in cs: return 1
        return 0
    return np.vectorize(evaluate)

def params_to_ab(mu: float, phi: float) -> tuple[float, float]:
    a = mu * phi
    b = (1 - mu) * phi
    return a, b

def params_from_ab(a: float, b: float) -> tuple[float, float]:
    mu = a / (a + b)
    phi = a + b
    return mu, phi

ZERO_AND_ONE: set[float] = {0.0, 1.0}
indicator = indicator_function(ZERO_AND_ONE)

SufficientStatistic: type = tuple[int, list[float], float, float, float, float]
LikelihoodResult: type = tuple[float, float, float, float]
Parameters: type = tuple[float, float, float, float]

def compute_sufficient_statistic(y: list[float]) -> SufficientStatistic:
    n: int = len(y)
    y_filtered = y[~np.isin(y, list(ZERO_AND_ONE))]
    
    T1: float = indicator(y).sum()
    T2: float = np.multiply(y, indicator(y)).sum()
    T3: float = np.log(y_filtered).sum()
    T4: float = np.log(1 - y_filtered).sum()

    return n, y_filtered, T1, T2, T3, T4

def likelihood(y: list[float], *,
               alpha: float = None,
               gamma: float = None,
               mu: float = None,
               phi: float = None) -> LikelihoodResult:
    a, b = params_to_ab(mu, phi)
    beta_rv: stats.beta_gen = stats.beta(a, b)

    n, yf, T1, T2, *_ = compute_sufficient_statistic(y)

    L1: float = np.power(alpha, T1) * np.power(1 - alpha, n - T1)
    L2: float = np.power(gamma, T2) * np.power(1 - gamma, T1 - T2)
    L3: float = np.product(beta_rv.pdf(yf))
    L = L1 * L2 * L3

    return L, L1, L2, L3

def log_likelihood(y: list[float], *,
                   alpha: float = None,
                   gamma: float = None,
                   mu: float = None,
                   phi: float = None) -> LikelihoodResult:
    a, b = params_to_ab(mu, phi)

    n, _, T1, T2, T3, T4 = compute_sufficient_statistic(y)

    l1: float = T1 * np.log(alpha) + (n - T1) * np.log(1 - alpha)
    l2: float = T2 * np.log(gamma) + (T1 - T2) * np.log(1 - gamma)
    l3: float = (n - T1) * np.log(1 / special.beta(a, b)) + T3 * (a - 1) + T4 * (b - 1)
    l = l1 + l2 + l3

    return l, l1, l2, l3

def score_vector(y: list[float], *,
                 alpha: float = None,
                 gamma: float = None,
                 mu: float = None,
                 phi: float = None) -> LikelihoodResult:
    a, b = params_to_ab(mu, phi)

    n, _, T1, T2, T3, T4 = compute_sufficient_statistic(y)

    U_alpha: float = T1 / alpha - (n - T1) / (1 - alpha)
    U_gamma: float = T2 / gamma - (T1 - T2) / (1 - gamma)
    U_mu: float = phi * (
        (n - T1) * (special.digamma(b) - special.digamma(a)) + T3 - T4
    )
    U_phi: float = (
        (n - T1) * (special.digamma(phi) - mu * special.digamma(a) - (1 - mu) * special.digamma(b))
        + mu * T3
        - (1 - mu) * T4
    )

    return U_alpha, U_gamma, U_mu, U_phi

def find_params(y: list[float]) -> Parameters:
    n, yf, T1, T2, *_ = compute_sufficient_statistic(y)
    
    alpha0 = T1 / n
    gamma0 = T2 / T1
    mu0 = np.mean(yf)
    phi0 = 1 / np.var(yf)

    def func(params: list[float], data = y) -> list[float]:
        alpha, gamma, mu, phi = params
        l, *_ = log_likelihood(data, alpha=alpha, gamma=gamma, mu=mu, phi=phi)
        return -l
    
    x0 = [alpha0, gamma0, mu0, phi0]
    alpha_bounds = (0, 1)
    gamma_bounds = (0, 1)
    mu_bounds = (0, 1)
    phi_bounds = (1E-6, None)
    bounds = (alpha_bounds, gamma_bounds, mu_bounds, phi_bounds)
    solution = optimize.minimize(func, x0, args=(y,),
                                 bounds=bounds,
                                 options={'gtol': 1e-6, 'disp': False})
    alpha, gamma, mu, phi  = solution.x

    return alpha, gamma, mu, phi, solution



if __name__ == '__main__':
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    palette = sns.color_palette()

    def plot_continuous(df, *, name: str  =None ,**kwplot):
        fig, (ax0, ax1) = plt.subplots(2,
                                    figsize=(8, 6),
                                    sharex=True,
                                    gridspec_kw={"height_ratios": (.15, .85)})

        whis = 1.5
        series = df[name]
        sns.boxplot(x=series, whis=whis, color=palette[0], ax=ax0)
        sns.histplot(series, stat='density', color=palette[0], ax=ax1, **kwplot)

        _count, mean, _std, min_, q1, q2, q3, max_ = (series.describe().to_list())
        iqr = q3 - q1
        whis1 = np.nan_to_num(series[series < q1 - whis * iqr].max(), nan=min_)
        whis2 = np.nan_to_num(series[series > q3 + whis * iqr].min(), nan=max_)

        ax1.axvline(mean, color=palette[3], linestyle='dashed')
        ax1.axvline(q2, color=palette[1], linestyle='dashdot')
        ax1.axvline(q1, color=palette[1], linewidth=.7, linestyle='dashdot')
        ax1.axvline(q3, color=palette[1], linewidth=.7, linestyle='dashdot')
        ax1.axvline(whis1, color=palette[1], linewidth=.5, linestyle='dashdot')
        ax1.axvline(whis2, color=palette[1], linewidth=.5, linestyle='dashdot')

        ax0.set_xlabel('')
        ax1.set_ylabel('density')

        ax0.set_yticks([], [])
        ax0.set_xticks([], [])

        fig.suptitle(name)

        return fig
    
    df = (
        pd
        .read_csv('https://raw.githubusercontent.com/Cizika/tiktok-popularity-analysis/main/tiktok.csv')
        .drop_duplicates('track_id')
        .assign(popularity=lambda df_: df_['popularity'] / 100)
    )
    y = df['popularity']

    alpha, gamma, mu, phi, solution = find_params(y)
    a, b = params_to_ab(mu, phi)
    
    print(f"Result {alpha = :.6f} {gamma = :.6f} {mu = :.6f} {phi = :.6f}")
    print(solution)

    # Aqui inicia a parte onde o plot acontece - usando a função que já existe e editando a imagem
    fig = plot_continuous(df, name='popularity')
    ax0, ax1 = fig.get_axes()

    bernoulli_rv = stats.bernoulli(gamma)
    beta_rv = stats.beta(a, b)

    x = np.linspace(0, 1, 100)[1:-1]
    y = (1 - alpha) * beta_rv.pdf(x)
    sns.lineplot(x=x, y=y, ax=ax1, color=palette[4])

    plt.show()