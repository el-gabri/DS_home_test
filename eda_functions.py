import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from math import floor
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# séries temporais

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import kpss, adfuller, pacf, acf


def data_info(df: pd.DataFrame):
    return pd.concat([pd.DataFrame(df.dtypes).T.rename(index={0: 'tipo da coluna'}),
                      pd.DataFrame(df.isnull().sum()).T.rename(index={0: 'nº nulls'}),
                      pd.DataFrame(df.isnull().sum() / df.shape[0] * 100).T.
                     rename(index={0: '% nulls'}), pd.DataFrame(df.nunique()).T.rename(index={0: 'nº únicos'})])


def corr_func(x: pd.Series, y: pd.Series):
    """
    Calcula a correlação de Pearson, Kendall e Spearman entre x e y.

    :param x: Primeira série de valores numéricos de mesmo tamanho.
    :param y: Segunda série de valores numéricos de mesmo tamanho.
    :return: dicionário com os valores de cada correlação.
    """

    print('Correlação básica:\n')

    corr_dict = {}
    for method in ['kendall', 'spearman', 'pearson']:
        corr = x.corr(y, method=method)
        corr_dict[method] = [corr]
        print('{}: {:.4f}'.format(method.title(), corr))

    return corr_dict


def corr_movel(x: pd.Series, y: pd.Series, z: pd.Series, method_list=None, window=30):
    """
    Calculo e visualização das correlações desejadas  dentro da janela window.

    :param x: Primeira série de valores numéricos de mesmo tamanho.
    :param y: Segunda série de valores numéricos de mesmo tamanho.
    :param z: Terceira série de valores datetime de mesmo tamanho.
    :param method_list: Lista de métodos a ser aplicada, os pontos de min e max da visualização.
    se referem ao primeiro método da lista. Default=['pearson', 'kendall', 'spearman'].
    :param window: Janela a qual será calculada a correlação. Default=30.
    :return: dataframe com o valor das correlações.
    """

    print('Correlação em janelas:\n')

    # Lista de métodos
    if method_list is None:
        method_list = ['kendall', 'spearman', 'pearson']

    # Criando dataframe das correlações
    df = pd.DataFrame()
    for method in method_list:
        df[method] = x.rolling(window=window).corr(y, method=method)

    # Retirando valores nulos
    df = df.loc[df[method_list[0]].notna()].reset_index(drop=True)
    df.index = z.head(z.shape[0] - window + 1)
    df.reset_index(inplace=True)

    # Visualização
    plt.figure(figsize=(13, 5))

    for method in method_list:
        sns.lineplot(x='index', y=method, data=df, label=method)

    ## Annotate valor mín e max de Pearson
    ### Minimo
    min_value_geral = df[method_list[0]].min()
    min_geral = df.loc[df[method_list[0]] == min_value_geral, 'index'].tolist()[0]
    plt.annotate(' {:.4f}'.format(min_value_geral), xy=(min_geral, min_value_geral),
                 ha='left', va='top')

    ### Maximo
    max_value_geral = df[method_list[0]].max()
    max_geral = df.loc[df[method_list[0]] == max_value_geral, 'index'].tolist()[0]
    plt.annotate(' {:.4f}'.format(max_value_geral), xy=(max_geral, max_value_geral),
                 ha='left', va='bottom')

    plt.xlabel('Janela de {} Periodos'.format(window))
    plt.ylabel('Correlações')
    plt.ylim(-1.1, 1.1)
    plt.show()

    return df[method_list]


def corr_deslocada(x: pd.Series, y: pd.Series, method='kendall', desloc=30, limit=True):
    """
    Calculo e visualização da correlação de séries deslocadas

    :param x: Primeira série de valores numéricos de mesmo tamanho.
    :param y: Série a ser deslocada.
    :param method: Qual tipo de correlação deve ser calculada.
    :param desloc: Quantos Periodos deve se deslocar para ambos os sentido.
    :param limit: Booleano se deve limitar ou não o eixo y do gráfico de correlação.
    :return: Dataframe contendo o número de Periodos deslocados e a correlação
    """

    print('Correlação deslocada:\n')

    # Range de deslocamento
    range_shift = range(-desloc, 0)

    # Calculo da correlação deslocada
    correlacao = pd.Series(map(lambda i: x.corr(y.shift(-i), method=method), range_shift))

    # Dataframe com deslocamente e correlação
    df = pd.DataFrame({'deslocamento': range_shift, method: correlacao})

    # Visualização
    fig, ax = plt.subplots(2, 1, figsize=(12, 6))

    # Dados originais normalizados
    sns.lineplot(x=x.index, y=(x - x.mean()) / x.std(), label='{} normalizado'.format(x.name), ax=ax[0])
    sns.lineplot(x=y.index, y=(y - y.mean()) / y.std(), label='{} normalizado'.format(y.name), ax=ax[0])

    ax[0].set_title('Dados normalizados para comparação')
    ax[0].set_xlabel('Tempo')
    ax[0].set_ylabel('Valores para comparação')

    # Correlação
    sns.lineplot(x='deslocamento', y=method, data=df, ax=ax[1])

    # Demarcação da correlação máxima atingida
    if df[method].max() > abs(df[method].min()):
        maximo = df[method].max()
    else:
        maximo = df[method].min()

    x_maximo = df.loc[df[method] == maximo, 'deslocamento'].tolist()[0]
    plt.vlines(x_maximo, ymin=-0.9, ymax=0.9, color='red', linestyles='dashed')
    plt.annotate('{} Periodos\n{:.4f}'.format(x_maximo, maximo), xy=(x_maximo + desloc / 100, 0.9), ha='left', va='top')

    # Set plot
    ax[1].set_title(f'Correlação dos dados com deslocamento da(o) "{y.name}" \n'
                    f'em {desloc} periodos para trás e para frente')
    ax[1].set_xlabel('Periodos deslocados')
    ax[1].set_ylabel('Correlação')
    ax[1].set_xlim(-desloc - 1, 0)
    if limit:
        ax[1].set_ylim(-1, 1)

    plt.tight_layout()
    plt.show()

    return df


def moving_average(series: pd.Series, n: int):
    """
        Calculate average of last n observations
    """
    return np.average(series[-n:])


def standard(serie: pd.Series):
    """
    Método de normalização min_max dos dados.

    :param serie: série a ser normalizada
    :return: série normalizada
    """
    min_serie = serie.min()
    max_serie = serie.max()
    return serie.apply(lambda x: (x - min_serie) / (max_serie - min_serie))


def mean_absolute_error(y_true, predictions):
    y_true, predictions = np.array(y_true), np.array(predictions)
    return np.mean(np.abs(y_true - predictions))


def mean_absolute_percentage_error(y_real, y_prev):
    """
    Calcula o erro absoluto médio em porcentagem

    :param y_real: valor observado
    :param y_prev: valor previsto
    :return: MAPE calculado
    """
    return np.mean(np.abs((np.subtract(y_real, y_prev) / y_real))) * 100


def plotMovingAverage(series: pd.Series, window: int, plot_intervals=False, scale=1.96, plot_anomalies=False):
    """
        series - dataframe with timeseries
        window - rolling window size
        plot_intervals - show confidence intervals
        plot_anomalies - show anomalies

    """
    rolling_mean = series.rolling(window=window).mean()

    plt.figure(figsize=(15, 5))
    plt.title("Média móvel\n tamanho da janela = {}".format(window))
    plt.plot(rolling_mean, "g", label="Tendência da média móvel")

    # Plot confidence intervals for smoothed values
    if plot_intervals:
        mae = mean_absolute_error(series[window:], rolling_mean[window:])
        deviation = np.std(series[window:] - rolling_mean[window:])
        lower_bond = rolling_mean - (mae + scale * deviation)
        upper_bond = rolling_mean + (mae + scale * deviation)
        plt.plot(upper_bond, "r--", label="Limite superior / Limite inferior")
        plt.plot(lower_bond, "r--")

        # Having the intervals, find abnormal values
        if plot_anomalies:
            anomalies = pd.DataFrame(index=series.index, columns=series)
            anomalies[series < lower_bond] = series[series < lower_bond]
            anomalies[series > upper_bond] = series[series > upper_bond]
            plt.plot(anomalies, "ro", markersize=10)

    plt.plot(series[window:], label="Valores observados")
    plt.legend(loc="upper left")
    plt.grid(True)


def remove_collinear_features(x, threshold=0.9):
    """
    Objective:
        Remove collinear features in a dataframe with a correlation coefficient
        greater than the threshold. Removing collinear features can help a model
        to generalize and improves the interpretability of the model.

    Inputs:
        x: features dataframe
        threshold: features with correlations greater than this value are removed

    Output:
        dataframe that contains only the non-highly-collinear features
    """

    # Calculate the correlation matrix
    corr_matrix = x.corr()
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []

    # Iterate through the correlation matrix and compare correlations
    for i in iters:
        for j in range(i + 1):
            item = corr_matrix.iloc[j:(j + 1), (i + 1):(i + 2)]
            col = item.columns
            row = item.index
            val = abs(item.values)

            # If correlation exceeds the threshold
            if val >= threshold:
                # Print the correlated features and the correlation value
                print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
                drop_cols.append(col.values[0])

    # Drop one of each pair of correlated columns
    drops = set(drop_cols)
    x = x.drop(columns=drops)

    return x


def corr_func(x: pd.Series, y: pd.Series):
    """
    Calcula a correlação de Pearson, Kendall e Spearman entre x e y.

    :param x: Primeira série de valores numéricos de mesmo tamanho.
    :param y: Segunda série de valores numéricos de mesmo tamanho.
    :return: dicionário com os valores de cada correlação.
    """

    print('Correlação básica:\n')

    corr_dict = {}
    for method in ['kendall', 'spearman', 'pearson']:
        corr = x.corr(y, method=method)
        corr_dict[method] = [corr]
        print('{}: {:.4f}'.format(method.title(), corr))

    return corr_dict


def corr_movel(x: pd.Series, y: pd.Series, z: pd.Series, method_list=None, window=30):
    """
    Calculo e visualização das correlações desejadas  dentro da janela window.

    :param x: Primeira série de valores numéricos de mesmo tamanho.
    :param y: Segunda série de valores numéricos de mesmo tamanho.
    :param z: Terceira série de valores datetime de mesmo tamanho.
    :param method_list: Lista de métodos a ser aplicada, os pontos de min e max da visualização.
    se referem ao primeiro método da lista. Default=['pearson', 'kendall', 'spearman'].
    :param window: Janela a qual será calculada a correlação. Default=30.
    :return: dataframe com o valor das correlações.
    """

    print('Correlação em janelas:\n')

    # Lista de métodos
    if method_list is None:
        method_list = ['kendall', 'spearman', 'pearson']

    # Criando dataframe das correlações
    df = pd.DataFrame()
    for method in method_list:
        df[method] = x.rolling(window=window).corr(y, method=method)

    # Retirando valores nulos
    df = df.loc[df[method_list[0]].notna()].reset_index(drop=True)
    df.index = z.head(z.shape[0] - window + 1)
    df.reset_index(inplace=True)

    # Visualização
    plt.figure(figsize=(13, 5))

    for method in method_list:
        sns.lineplot(x='index', y=method, data=df, label=method)

    ## Annotate valor mín e max de Pearson
    ### Minimo
    min_value_geral = df[method_list[0]].min()
    min_geral = df.loc[df[method_list[0]] == min_value_geral, 'index'].tolist()[0]
    plt.annotate(' {:.4f}'.format(min_value_geral), xy=(min_geral, min_value_geral),
                 ha='left', va='top')

    ### Maximo
    max_value_geral = df[method_list[0]].max()
    max_geral = df.loc[df[method_list[0]] == max_value_geral, 'index'].tolist()[0]
    plt.annotate(' {:.4f}'.format(max_value_geral), xy=(max_geral, max_value_geral),
                 ha='left', va='bottom')

    plt.xlabel('Janela de {} Periodos'.format(window))
    plt.ylabel('Correlações')
    plt.ylim(-1.1, 1.1)
    plt.show()

    return df[method_list]


def corr_deslocada(x: pd.Series, y: pd.Series, method='kendall', desloc=30, limit=True):
    """
    Calculo e visualização da correlação de séries deslocadas

    :param x: Primeira série de valores numéricos de mesmo tamanho.
    :param y: Série a ser deslocada.
    :param method: Qual tipo de correlação deve ser calculada.
    :param desloc: Quantos Periodos deve se deslocar para ambos os sentido.
    :param limit: Booleano se deve limitar ou não o eixo y do gráfico de correlação.
    :return: Dataframe contendo o número de Periodos deslocados e a correlação
    """

    print('Correlação deslocada:\n')

    # Range de deslocamento
    range_shift = range(-desloc, 0)

    # Calculo da correlação deslocada
    correlacao = pd.Series(map(lambda i: x.corr(y.shift(-i), method=method), range_shift))

    # Dataframe com deslocamente e correlação
    df = pd.DataFrame({'deslocamento': range_shift, method: correlacao})

    # Visualização
    fig, ax = plt.subplots(2, 1, figsize=(12, 6))

    # Dados originais normalizados
    sns.lineplot(x=x.index, y=(x - x.mean()) / x.std(), label='{} normalizado'.format(x.name), ax=ax[0])
    sns.lineplot(x=y.index, y=(y - y.mean()) / y.std(), label='{} normalizado'.format(y.name), ax=ax[0])

    ax[0].set_title('Dados normalizados para comparação')
    ax[0].set_xlabel('Tempo')
    ax[0].set_ylabel('Valores para comparação')

    # Correlação
    sns.lineplot(x='deslocamento', y=method, data=df, ax=ax[1])

    # Demarcação da correlação máxima atingida
    if df[method].max() > abs(df[method].min()):
        maximo = df[method].max()
    else:
        maximo = df[method].min()

    x_maximo = df.loc[df[method] == maximo, 'deslocamento'].tolist()[0]
    plt.vlines(x_maximo, ymin=-0.9, ymax=0.9, color='red', linestyles='dashed')
    plt.annotate('{} Periodos\n{:.4f}'.format(x_maximo, maximo), xy=(x_maximo + desloc / 100, 0.9), ha='left', va='top')

    # Set plot
    ax[1].set_title(f'Correlação dos dados com deslocamento da(o) "{y.name}" \n'
                    f'em {desloc} periodos para trás e para frente')
    ax[1].set_xlabel('Periodos deslocados')
    ax[1].set_ylabel('Correlação')
    ax[1].set_xlim(-desloc - 1, 0)
    if limit:
        ax[1].set_ylim(-1, 1)

    plt.tight_layout()
    plt.show()

    return df


def mean_absolute_percentage_error(y_real, y_prev):
    """
    Calcula o erro absoluto médio em porcentagem

    :param y_real: valor observado
    :param y_prev: valor previsto
    :return: MAPE calculado
    """
    return np.mean(np.abs((np.subtract(y_real, y_prev) / y_real))) * 100


def MAPE(y_true, y_pred):
    """
    Calcula o erro absoluto médio em porcentagem

    :param y_true: valor observado
    :param y_pred: valor previsto
    :return: MAPE calculado
    """
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def plotMovingAverage(series: pd.Series, window: int, plot_intervals=False, scale=1.96, plot_anomalies=False):
    """
        series - dataframe with timeseries
        window - rolling window size
        plot_intervals - show confidence intervals
        plot_anomalies - show anomalies

    """
    rolling_mean = series.rolling(window=window).mean()

    plt.figure(figsize=(15, 5))
    plt.title("Média móvel\n tamanho da janela = {}".format(window))
    plt.plot(rolling_mean, "g", label="Tendência da média móvel")

    # Plot confidence intervals for smoothed values
    if plot_intervals:
        mae = mean_absolute_error(series[window:], rolling_mean[window:])
        deviation = np.std(series[window:] - rolling_mean[window:])
        lower_bond = rolling_mean - (mae + scale * deviation)
        upper_bond = rolling_mean + (mae + scale * deviation)
        plt.plot(upper_bond, "r--", label="Limite superior / Limite inferior")
        plt.plot(lower_bond, "r--")

        # Having the intervals, find abnormal values
        if plot_anomalies:
            anomalies = pd.DataFrame(index=series.index, columns=series)
            anomalies[series < lower_bond] = series[series < lower_bond]
            anomalies[series > upper_bond] = series[series > upper_bond]
            plt.plot(anomalies, "ro", markersize=10)

    plt.plot(series[window:], label="Valores observados")
    plt.legend(loc="upper left")
    plt.grid(True)


def weighted_average(series, weights):
    """
        Calculate weighted average on the series.
        Assuming weights are sorted in descending order
        (larger weights are assigned to more recent observations).
    """
    result = 0.0
    for n in range(len(weights)):
        result += series.iloc[-n - 1] * weights[n]
    return float(result)


def compare_ts(ts1, ts2, colors=('b', 'r'), ls='--', ax=None):
    ts1 = ts1.dropna()
    ts2 = ts2.dropna()

    # 1ª série
    if ax == None:
        ax = sns.lineplot(x=ts1.index, y=ts1, color=colors[0], label=ts1.name, legend=None)
    else:
        sns.lineplot(x=ts1.index, y=ts1, color=colors[0], ax=ax, label=ts1.name, legend=None)
    z1 = np.polyfit(pd.factorize(ts1.index)[0] + 1, ts1, 1)
    p1 = np.poly1d(z1)
    ax.plot(ts1.index, p1(pd.factorize(ts1.index)[0] + 1), ls='--', color=colors[0])
    ax.grid(None)

    # 2ª série
    ax2 = ax.twinx()
    z2 = np.polyfit(pd.factorize(ts2.index)[0] + 1, ts2, 1)
    p2 = np.poly1d(z2)
    sns.lineplot(x=ts2.index, y=ts2, color=colors[1], ax=ax2, label=ts2.name, legend=None)
    ax2.plot(ts2.index, p2(pd.factorize(ts2.index)[0] + 1), ls='--', color=colors[1])
    ax2.grid(None)

    # legenda
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    plt.legend(h1 + h2, l1 + l2)


def predict_sarima(data, pred_col, cutoff_col, cutoff_ref,
                   steps, series_order, seasonal_order, exog_col=None):
    slice_values = data.loc[data[cutoff_col] >= cutoff_ref, cutoff_col].unique()
    results = pd.DataFrame()

    for slice_value in slice_values:

        if slice_value == slice_values[-1]:
            continue

        train = data[data[cutoff_col] <= slice_value].copy()
        size = train.shape[0]

        model = SARIMAX(train[pred_col],
                        order=series_order,
                        seasonal_order=seasonal_order)

        res = model.fit(method='powell')
        res = res.predict(size, size + steps - 1)

        res = res.to_frame(name=slice_value).T
        pred_cols = [f'vol_pred_step{i + 1}' for i in range(res.shape[1])]
        res.columns = pred_cols

        results = pd.concat([results, res])

    return results


def multiple_compare_ts(df, main_var, var_array, linhas, colunas, figsize):
    fig, ax = plt.subplots(linhas, colunas, figsize=figsize)
    var_array = [var for var in var_array if var != main_var]
    for i, var in enumerate(var_array):
        compare_ts(df[main_var], df[var], ax=ax[i // colunas, i % colunas])
    plt.tight_layout()


# Visualização da decomposição de uma série temporal com plotly
# Visualização da decomposição de uma série temporal com plotly
def plot_decomp(dec, col=None, shared_x=True, shared_y=False):
    """Imprime os quatro gráficos resultantes
    (série, tendência, sazonalidade e resíduos)."""

    fig = make_subplots(rows=4, cols=1, shared_xaxes=shared_x, shared_yaxes=shared_y, vertical_spacing=0.05,
                        subplot_titles=('Série', 'Tendência', 'Sazonalidade', 'Residual'))

    # necessário para permitir uso de STL e seasonal_decompose
    if col:
        # STL
        observado = dec.observed[col]
    else:
        # seasonal_decompose
        observado = dec.observed

    fig.add_scatter(row=1, col=1, x=dec.observed.index, y=observado, name='Série')
    fig.add_scatter(row=2, col=1, x=dec.trend.index, y=dec.trend, name='Tendência')
    fig.add_scatter(row=3, col=1, x=dec.seasonal.index, y=dec.seasonal, name='Sazonalidade')
    fig.add_scatter(row=4, col=1, x=dec.resid.index, y=dec.resid, name='Residual')

    fig.update_layout(
        title=f'Decomposição',
        title_x=0.5,
        showlegend=False,
        height=800,
        width=1600,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis_type="date",
        xaxis_rangeslider_visible=False,
        xaxis1=dict(
            rangeselector=dict(
                buttons=list(
                    [
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(count=2, label="2y", step="year", stepmode="backward"),
                        dict(count=3, label="3y", step="year", stepmode="backward"),
                        dict(count=4, label="4y", step="year", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),

                        dict(step="all"),
                    ]
                )
            ),
            type="date",
        ),
        xaxis4_rangeslider_visible=True,
    )

    return fig


def is_stationary(df, col, dif=None, alpha=0.05):
    series = df[col]

    if dif:
        series = series.diff(dif).dropna()

    st_ad = adfuller(series)
    p_ad = st_ad[1]

    if p_ad >= alpha:
        res_ad = 'Não-estacionária'
    else:
        res_ad = 'Estacionária'

    st_kpss = kpss(series)
    p_kpss = st_kpss[1]

    if p_kpss >= alpha:
        res_kpss = 'Estacionária'
    else:
        res_kpss = 'Não-Estacionária'

    print(f'Teste de estacionariedade', end='\n\n')
    print(f'ADFuller: {res_ad} (p-valor: {p_ad})')
    print(f'KPSS: {res_kpss} (p-valor: {p_kpss})')


def plot_correlations(series, n_lags):
    def corr_plot(series, fig, row, n_lags=45, plot_pacf=False):
        # Sem a Autocorrelation do Lag 0
        corr_array = pacf(series.dropna(), alpha=0.05, nlags=n_lags) if plot_pacf else acf(series.dropna(), alpha=0.05,
                                                                                           nlags=n_lags)
        lower_y = corr_array[1][:, 0] - corr_array[0]
        upper_y = corr_array[1][:, 1] - corr_array[0]

        # deveria ser feito com vline ou hline
        [fig.add_scatter(row=row, col=1, x=(x, x), y=(0, corr_array[0][x]), mode='lines', line_color='#3f3f3f',
                         hoverinfo='skip')
         for x in range(1, len(corr_array[0]))]

        fig.add_scatter(row=row, col=1, x=[1, len(corr_array[0]) - 1], y=[0, 0], mode='lines', line_color='#CCCCCC',
                        hoverinfo='skip')

        fig.add_scatter(row=row, col=1, x=np.arange(1, len(corr_array[0])), y=corr_array[0][1:], mode='markers',
                        marker_color='#1f77b4',
                        marker_size=12)
        fig.add_scatter(row=row, col=1, x=np.arange(len(corr_array[0])) + 1, y=upper_y[1:], mode='lines',
                        line_color='rgba(255,255,255,0)', hoverinfo='skip')
        fig.add_scatter(row=row, col=1, x=np.arange(len(corr_array[0])) + 1, y=lower_y[1:], mode='lines',
                        fillcolor='rgba(32, 146, 230,0.3)', hoverinfo='skip',
                        fill='tonexty', line_color='rgba(255,255,255,0)')

    n_lags = n_lags if n_lags < 30 else min(29, floor(series.shape[0] / 2) - 1)

    fig = make_subplots(rows=2, cols=1,
                        shared_xaxes=True, shared_yaxes=False,
                        vertical_spacing=0.05,
                        subplot_titles=('ACF', 'PACF'))

    corr_plot(series, fig, 1, n_lags, plot_pacf=False, )
    corr_plot(series, fig, 2, n_lags, plot_pacf=True)

    fig.update_traces(showlegend=False)
    fig.update_layout(height=800,
                      plot_bgcolor='rgba(0,0,0,0)',
                      paper_bgcolor='rgba(0,0,0,0)')

    return fig
