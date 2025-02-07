import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import skew, kurtosis
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def caract_df(df: pd.DataFrame):
    """
    Características básicas de um dataframe.

    :param df: Dataframe a ser análisado
    :return: None
    """
    print('Nº linhas: {}\nNº colunas: {}'.format(df.shape[0], df.shape[1]))
    print('Nº linhas duplicadas: {}'.format(df.duplicated().sum()))
    print('\nNº de vazios*:')
    for col in df.columns:
        # n_na = pd.isna(df[col]).sum()
        n_na = df[col].isnull().sum()
        if n_na > 1:
            print('\t{}: {} - {}%'.format(col,
                                          n_na,
                                          round(100 * n_na / df.shape[0], 2)))
    print('(*) Antes do processamento.')
    return


def check_catg(serie: pd.Series, s=7):
    """
    Checa se existe alguma observação fora do padrão, só funciona para séries de números de mesmo tamanho

    :param s:
    :param serie: Série que será checada a procura de valoresfora do padrão.
    :return: None
    """
    cond1 = serie.apply(lambda x: False if pd.isna(x) else True if len(x) != s else False)
    cond2 = serie.apply(
        lambda x: False if pd.isna(x) else True if ''.join([num for num in x if not num.isnumeric()]) else False)

    print(cond1.sum())
    print(cond2.sum())
    return cond1, cond2


def barplot(serie: pd.Series, c='Green'):
    """
    Função automática de criação de sns.barplot
    :param serie: Série que origininará o gráfico
    :param c: Cor do gráfico
    :return: None
    """

    # Contabilizando Na's
    serie = serie.replace(np.nan, 'Faltante', regex=True).copy()

    # Gerando dados para plot
    df = serie.value_counts().reset_index()
    df = df.astype({'index': str, serie.name: int})

    # Separando dados missings
    miss = df[df['index'] == 'Faltante']

    # criando categoria "outros"
    df = df[df['index'] != 'Faltante'].reset_index(drop=True)
    size_outros = 0
    if df.shape[0] > 10:
        size_outros = df.shape[0] - 6
        df_top = df.head(5).copy()
        df = pd.concat([df_top, pd.DataFrame({'index': ['Outros'],
                                              serie.name: df.drop([0, 1, 2, 3, 4])[serie.name].sum()})])

    df = pd.concat([df, miss]).reset_index(drop=True)

    # Display dados
    df_display = df.copy()
    df_display.columns = [serie.name, 'Freq']
    if size_outros > 0:
        print("Mostrando as maiores categorias de {}".format(df_display.shape[0] + size_outros))

    # Setup Plot
    palette = {catg: c if (catg != 'Outros') & (catg != 'Faltante') else 'gray' if catg != 'Faltante' else 'black'
               for catg in df['index']}
    altura = 4 if df.shape[0] < 16 else int(len(df['index']) / 4)
    plt.figure(figsize=(14, altura))

    # Plot
    sns.barplot(x=serie.name, y='index', data=df, palette=palette)
    for y, x in enumerate(df[serie.name]):
        dif = df[serie.name].max() * 0.005
        percent = 100 * x / df[serie.name].sum()

        # Posição das legendas
        if x > 7 * dif:
            plt.annotate(x, xy=(x - dif, y), ha='right', va='center', color='white')
            plt.annotate('{:.2f}%'.format(percent), xy=(x + dif, y), ha='left', va='center', color='black')
        else:
            plt.annotate('{} - {:.2f}%'.format(x, percent), xy=(x + dif, y), ha='left', va='center', color='black')

    # layout
    plt.xlim(0, df[serie.name].max() * 1.1)

    plt.title('Frequência da coluna {}'.format(serie.name))
    plt.xlabel('Frequência')
    plt.ylabel(serie.name.title().replace('_', ' '))
    plt.show()

    return


def time_plot(serie: pd.Series, c='Green', stats=True):
    """
    Função de criação automatica de séries temporais e análise.
    :param serie: Série que origininará o gráfico
    :param c: Cor do gráfico
    :param stats: Indicador se traz ou não as estatísticas/plots estatísticos da série temporal
    :return: None
    """

    if serie.dtype != 'datetime64[ns]':
        serie = serie.astype(str)
        serie = pd.to_datetime(serie, format='%Y%m%d', errors='coerce')

    # Contabilizando Na's
    print('Número de missings: {}'.format(pd.isna(serie).sum()))
    print('Range da série: {} - {}'.format(serie.min().strftime('%d/%b/%Y'), serie.max().strftime('%d/%b/%Y')))

    dias = (serie.max() - serie.min()).days
    if dias // 365 == 1:
        print('\t- {}Ano {}Meses {}dias'.format(dias // 365, (dias % 365) // 30, (dias % 365) % 30))
    else:
        print('\t- {}Anos {}Meses {}dias'.format(dias // 365, (dias % 365) // 30, (dias % 365) % 30))

    # Destrinchando a série em dia, mês e ano
    serie_mes = serie.dt.strftime('%b/%Y').copy()
    serie_ano = serie.dt.strftime('%Y').copy()

    # Gerando dados para plots
    df_dia = serie.value_counts()  # frequencia
    df_dia = df_dia.reindex(pd.date_range(serie.min(), serie.max())).fillna(0)  # arrumando os dias
    df_dia = df_dia.reset_index()
    df_dia['media movel (30)'] = df_dia[serie.name].rolling(window=30).mean()  # média movel

    df_mes = serie_mes.value_counts().reset_index()
    df_mes['date'] = pd.to_datetime(df_mes['index'])
    df_mes.sort_values('date', inplace=True)

    df_ano = serie_ano.value_counts().reset_index()
    df_ano['date'] = pd.to_datetime(df_ano['index'])
    df_ano.sort_values('date', inplace=True)

    # Plots
    fig, ax = plt.subplots(3, figsize=(15, 8))

    sns.lineplot(x='index', y=serie.name, color=c, data=df_dia, ax=ax[0])

    sns.barplot(x='index', y=serie.name, color=c, data=df_mes, ax=ax[1])
    for x, y in enumerate(df_mes[serie.name]):
        ax[1].annotate(y, xy=(x, y), ha='center', va='bottom')

    sns.barplot(x='index', y=serie.name, color=c, data=df_ano, ax=ax[2])
    for x, y in enumerate(df_ano[serie.name]):
        ax[2].annotate(y, xy=(x, y), ha='center', va='bottom')

    # Setup plots
    ax[0].set_title('Distribuição de {} por dia'.format(serie.name))
    ax[1].set_title('Distribuição de {} por mes'.format(serie.name))
    ax[2].set_title('Distribuição de {} por ano'.format(serie.name))

    ax[0].set_xlabel('Dias')
    ax[1].set_xlabel('Meses')
    ax[2].set_xlabel('Anos')

    ax[0].set_ylabel('Frequência')
    ax[1].set_ylabel('Frequência')
    ax[2].set_ylabel('Frequência')

    ax[0].set_ylim(0, df_dia[serie.name].max() * 1.2)
    ax[1].set_ylim(0, df_mes[serie.name].max() * 1.2)
    ax[2].set_ylim(0, df_ano[serie.name].max() * 1.2)

    plt.tight_layout()
    plt.show()

    if stats:
        # Estacionaridade
        p_value = sm.tsa.stattools.adfuller(df_dia[serie.name])[1]
        print('\n\n')
        print(27 * '##', 'Estatísticas', 27 * '##', '\n')

        # Plots estatisticos
        plt.figure(figsize=(15, 8))
        ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
        ax2 = plt.subplot2grid((2, 2), (1, 0))
        ax3 = plt.subplot2grid((2, 2), (1, 1))

        sns.lineplot(x='index', y=serie.name, color=c, data=df_dia, label='Dados', ax=ax1)
        sns.lineplot(x='index', y='media movel (30)', color='firebrick', data=df_dia, label='Média movel (30 dias)',
                     ax=ax1)

        plot_acf(df_dia[serie.name], ax=ax2)
        plot_pacf(df_dia[serie.name], ax=ax3)

        ax1.set_title('Análise da série temporal {}\n Dickey-fuler: p={:.5f}'.format(serie.name, p_value))

        plt.tight_layout()
        plt.show()

    return


def iqr(serie: pd.Series, multiplicador=1.5):
    """
    Análise de outliers da série numérica

    :param serie: Série de valores numéricos
    :param multiplicador: Intervalo do que é considerado aceitavel, sugestão 3.0
    :return: série sem outliers
    """
    # Valores outliers
    # q1, q3 = np.quantile(serie, [0.25, 0.75])
    # IQR = (q3 - q1) * multiplicador
    # limit_lower = q1 - IQR if q1 > IQR else 0
    # limit_upper = q3 + IQR
    factor = multiplicador
    limit_upper = serie.mean() + serie.std() * factor
    limit_lower = serie.mean() - serie.std() * factor

    # Outliers
    outliers = [x for x in serie if (x > limit_upper) | (x < limit_lower)]
    in_limits = [x if (x <= limit_upper) & (x >= limit_lower) else np.nan for x in serie]

    print('Número de outliers (excluídos): {} ({}% do total)'.format(len(outliers),
                                                                     round(len(outliers) * 100 / len(serie),
                                                                           2)))
    print('Número de registros considerados: {}'.format(len(serie) - len(outliers)))

    return pd.Series(in_limits, name=serie.name)


def numeric_plot(serie: pd.Series, c='Green', outliers=True, mult=1.5):
    """
    Análise de dados numéricos.

    :param serie: Série a ser analisada
    :param c: cor do gráfico
    :param outliers: Indicador dos outliers
    :param mult: Intervalo do que é considerado aceitavel, sugestão 1.5 ou 2.5.
    :return: None
    """

    # Outliers
    if outliers:
        serie = iqr(serie.copy(), multiplicador=mult)

    serie = serie.loc[pd.notna(serie)].copy()
    df = serie.describe().reset_index()
    df = pd.concat([df, pd.DataFrame({'index': ['skewness', 'Kurtosis'],
                                      serie.name: [skew(serie), kurtosis(serie)]})])

    df.columns = ['', 'Valor']
    df.set_index('', inplace=True)

    # Plot
    fig, ax = plt.subplots(2, figsize=(15, 6), sharex=True)

    violin = sns.violinplot(x=serie, color=c, inner=None, ax=ax[0])
    plt.setp(violin.collections, alpha=.3)
    sns.boxplot(x=serie, color=c, ax=ax[0])

    sns.histplot(x=serie, color=c, ax=ax[1])

    # Plot layout
    ax[0].set_xlabel('Valor')
    ax[1].set_xlabel('Valor')

    ax[0].set_ylabel('{}'.format(serie.name.title().replace('_', ' ')))
    ax[1].set_ylabel('Frequência')

    ax[0].set_title('Distribuição da variável {}'.format(serie.name.title().replace('_', ' ')))
    ax[1].set_title('Distribuição da variável {}'.format(serie.name.title().replace('_', ' ')))

    plt.tight_layout()
    plt.show()
    return
