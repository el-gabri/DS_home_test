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
        n_na = df[col].isnull().sum()
        if n_na > 0:
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
    Função automática de criação de sns.barplot.
    :param serie: Série que origininará o gráfico.
    :param c: Cor do gráfico.
    :return: None.
    """
    # Contabilizando Na's: substitui NaN por 'Faltante'
    serie = serie.replace(np.nan, 'Faltante', regex=True).copy()

    # Gerando dados para plot
    df = serie.value_counts().reset_index()
    # Force column names so that we know exactly what they are:
    # The first column will be the category (named 'index') and the second column is the frequency.
    df.columns = ['index', serie.name]
    # Now we can safely cast types:
    df = df.astype({'index': str, serie.name: int})

    # Separando dados missings
    miss = df[df['index'] == 'Faltante']

    # Criando categoria "Outros"
    df = df[df['index'] != 'Faltante'].reset_index(drop=True)
    size_outros = 0
    if df.shape[0] > 10:
        size_outros = df.shape[0] - 6
        df_top = df.head(5).copy()
        # Sum the rest of the values
        sum_outros = df.drop([0, 1, 2, 3, 4])[serie.name].sum()
        df_outros = pd.DataFrame({'index': ['Outros'], serie.name: [sum_outros]})
        df = pd.concat([df_top, df_outros])

    df = pd.concat([df, miss]).reset_index(drop=True)

    # Display dados (para debug)
    df_display = df.copy()
    df_display.columns = [serie.name, 'Freq']
    if size_outros > 0:
        print("Mostrando as maiores categorias de {}".format(df_display.shape[0] + size_outros))

    # Setup Plot
    palette = {catg: c if (catg != 'Outros') and (catg != 'Faltante') else ('gray' if catg == 'Outros' else 'black')
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

    # Layout
    plt.xlim(0, df[serie.name].max() * 1.1)
    plt.title('Frequência da coluna {}'.format(serie.name))
    plt.xlabel('Frequência')
    plt.ylabel(serie.name.title().replace('_', ' '))
    plt.show()


def time_plot(serie: pd.Series, c='Green', stats=True):
    """
    Função de criação automatica de séries temporais e análise.
    :param serie: Série que origininará o gráfico
    :param c: Cor do gráfico
    :param stats: Indicador se traz ou não as estatísticas/plots estatísticos da série temporal
    :return: None
    """

    # Garantir que a série esteja em datetime
    if serie.dtype != 'datetime64[ns]':
        serie = serie.astype(str)
        serie = pd.to_datetime(serie, format='%Y%m%d', errors='coerce')

    # Contabilizando Na's e mostrando o range
    print('Número de missings: {}'.format(pd.isna(serie).sum()))
    print('Range da série: {} - {}'.format(serie.min().strftime('%d/%b/%Y'),
                                           serie.max().strftime('%d/%b/%Y')))

    dias = (serie.max() - serie.min()).days
    if dias // 365 == 1:
        print('\t- {}Ano {}Meses {}dias'.format(dias // 365, (dias % 365) // 30, (dias % 365) % 30))
    else:
        print('\t- {}Anos {}Meses {}dias'.format(dias // 365, (dias % 365) // 30, (dias % 365) % 30))

    # Destrinchando a série em dia, mês e ano
    serie_dia = serie.dt.strftime('%d/%m/%Y').copy()
    serie_mes = serie.dt.strftime('%b/%Y').copy()
    serie_ano = serie.dt.strftime('%Y').copy()

    # Para os dados diários:
    df_dia = serie_dia.value_counts().rename(serie.name)
    df_dia.index.name = "day"  # define um nome diferente para o índice
    df_dia = df_dia.reset_index()
    df_dia['date'] = pd.to_datetime(df_dia['day'], format='%d/%m/%Y')
    df_dia['media movel (30)'] = df_dia[serie.name].rolling(window=30).mean()
    df_dia.sort_values('date', inplace=True)

    # Para os dados mensais:
    df_mes = serie_mes.value_counts().rename(serie.name)
    df_mes.index.name = "month"  # define um nome diferente para o índice
    df_mes = df_mes.reset_index()
    df_mes['date'] = pd.to_datetime(df_mes['month'], format='%b/%Y')
    df_mes.sort_values('date', inplace=True)

    # Para os dados anuais:
    df_ano = serie_ano.value_counts().rename(serie.name)
    df_ano.index.name = "year"  # define um nome diferente para o índice
    df_ano = df_ano.reset_index()
    df_ano['date'] = pd.to_datetime(df_ano['year'], format='%Y')
    df_ano.sort_values('date', inplace=True)

    # Plots
    fig, ax = plt.subplots(3, figsize=(15, 8))

    sns.lineplot(x='date', y=serie.name, color=c, data=df_dia, ax=ax[0])

    sns.barplot(x='month', y=serie.name, color=c, data=df_mes, ax=ax[1])
    for x, y in enumerate(df_mes[serie.name]):
        ax[1].annotate(y, xy=(x, y), ha='center', va='bottom')

    sns.barplot(x='year', y=serie.name, color=c, data=df_ano, ax=ax[2])
    for x, y in enumerate(df_ano[serie.name]):
        ax[2].annotate(y, xy=(x, y), ha='center', va='bottom')

    # Setup dos plots
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
        # Estacionaridade: Teste Dickey-Fuller
        p_value = sm.tsa.stattools.adfuller(df_dia[serie.name])[1]
        print('\n\n' + 27 * '##' + ' Estatísticas ' + 27 * '##' + '\n')

        # Plots estatísticos
        plt.figure(figsize=(15, 8))
        ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
        ax2 = plt.subplot2grid((2, 2), (1, 0))
        ax3 = plt.subplot2grid((2, 2), (1, 1))

        sns.lineplot(x='date', y=serie.name, color=c, data=df_dia, label='Dados', ax=ax1)
        sns.lineplot(x='date', y='media movel (30)', color='firebrick',
                     data=df_dia, label='Média movel (30 dias)', ax=ax1)

        plot_acf(df_dia[serie.name], ax=ax2)
        plot_pacf(df_dia[serie.name], ax=ax3)

        ax1.set_title('Análise da série temporal {}\n Dickey-fuler: p={:.5f}'.format(serie.name, p_value))

        plt.tight_layout()
        plt.show()

    return


def remove_outliers_zscore(serie: pd.Series, multiplicador=1.5):
    """
    Análise de outliers da série numérica via desvio em relação à média
    (mean ± multiplicador * desvio padrão). Note que isso NÃO é o método do
    intervalo interquartil (IQR) apesar do nome antigo desta função sugerir —
    para IQR de verdade, use quantis (q1, q3) e ``(q3 - q1) * multiplicador``.

    :param serie: Série de valores numéricos
    :param multiplicador: Intervalo do que é considerado aceitavel, sugestão 3.0
    :return: série sem outliers
    """
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
        serie = remove_outliers_zscore(serie.copy(), multiplicador=mult)

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
