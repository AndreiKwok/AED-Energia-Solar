import matplotlib.pyplot as plt
import pandas as pd
import os

# statsmodels imports
from statsmodels.tsa.seasonal import seasonal_decompose

# Configuração de estilo
plt.style.use("dark_background")


def exportar_figura(nome: str) -> None:
    output_dir = os.path.join(os.getcwd(), "images")
    os.makedirs(output_dir, exist_ok=True)
    output_imagem = os.path.join(output_dir, nome)
    plt.savefig(output_imagem, bbox_inches="tight")


def plotar_distribuicao_horaria(
    data: pd.DataFrame, salvar_imagem: bool = False, **kwargs
) -> None:
    # Agrupando por hora e somando a energia
    energia_por_hora = data.groupby(data.index.hour).sum()
    # Plotando o gráfico
    plt.figure(figsize=(18, 6))
    energia_por_hora.plot(kind="bar", **kwargs)
    plt.title("Distribuição Horária de Energia Solar")
    plt.xlabel("Hora do Dia")
    plt.ylabel("Energia Solar (kWh)")
    plt.xticks(rotation=0)
    # Salvar gráfico como arquivo .png
    if salvar_imagem:
        exportar_figura(nome="distribuicao_horaria.png")
    plt.show()


def plotar_top_anos(
    data: pd.DataFrame, n: int = 3, salvar_imagem: bool = False, **kwargs
) -> None:
    # Agrupando por ano e somando a energia
    energia_por_ano = data.groupby(data.index.year).sum()

    # Selecionando os n anos de maior geração
    top_anos = energia_por_ano.nlargest(n, "Energia")

    # Plotando o gráfico
    plt.figure(figsize=(18, 6))
    top_anos.plot(kind="barh", **kwargs)
    plt.title(f"Top {n} Anos de Maior Geração de Energia Solar")
    plt.xlabel("Energia Solar (kWh)")
    plt.ylabel("Ano")
    plt.gca().invert_yaxis()  # Inverte a ordem dos anos

    # Adicionar rótulos de dados
    for index, value in enumerate(top_anos["Energia"]):
        plt.text(
            value,
            index,
            f"{value:.0f} MWh / year ",
            va="center",
            ha="right",
            fontweight="bold",
            fontsize=12,
        )
    # Salvar gráfico como arquivo .png
    if salvar_imagem:
        exportar_figura(nome=f"top{n}_anos_geracao.png")
    plt.show()


def plotar_distribuicao_mensal(
    data: pd.DataFrame, salvar_imagem: bool = False, **kwargs
) -> None:
    # Extraindo o ano e o mês do índice
    df_mensal = data.groupby(data.index.month).mean()

    # Plotando o gráfico
    plt.figure(figsize=(18, 6))
    df_mensal.plot(kind="line", **kwargs)
    plt.title("Distribuição Mensal de Energia Solar")
    plt.xlabel("Ano e Mês")
    plt.ylabel("Energia Solar (kWh)")
    plt.xticks(rotation=45)
    plt.grid(False)
    # Salvar gráfico como arquivo .png
    if salvar_imagem:
        exportar_figura(nome="distribuicao_mensal.png")
    plt.show()


def plotar_decomposicao(data: pd.DataFrame, salvar_imagem: bool = False, **kwargs) -> None:
    # Calcula decomposição da série temporal usando statsmodels
    decomposition = seasonal_decompose(
        x=data, model="additive", extrapolate_trend="freq", **kwargs
    )
    # Cria figura e áreas de plotagem
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, figsize=(18, 12))
    fig.tight_layout()

    # Plota componentes
    decomposition.observed.plot(ax=ax0, ylabel="observed")
    decomposition.trend.plot(ax=ax1, ylabel="trend")
    decomposition.seasonal.plot(ax=ax2, ylabel="seasonal")
    decomposition.resid.plot(ax=ax3, ylabel="resid")
    # Salvar gráfico como arquivo .png
    if salvar_imagem:
        exportar_figura(nome="decomposicao_temporal.png")
    plt.show()