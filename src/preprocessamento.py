# Importação de bibliotecas
import os
import pandas as pd


def listar_arquivos(diretorio: os.PathLike, base_de_dados: str = "Solcast") -> list:
    # Lista os arquivos do diretório especificado
    arquivos = [os.path.join(diretorio, arquivo) for arquivo in os.listdir(diretorio)]
    # Retorna uma lista com os arquivos da base de dados especificada
    return list(filter(lambda x: base_de_dados in x, arquivos))


def ler_e_formatar_dados(arquivo: os.PathLike, **kwargs) -> pd.DataFrame:
    # Lê o arquivo .csv e converte em dataframe (tabela no Python)
    df = pd.read_csv(arquivo, **kwargs)
    # Renomeia a coluna para um nome mais sugestivo
    df = df.rename(columns={"kW": "Energia"})
    # Converte os valores de kW para MWh
    df /= 1000
    # Converte valores negativos em zero (geração nula durante o período noturno)
    df = df.applymap(lambda x: 0 if x < 0 else x)
    # Transforma os valores do índice em objetos datetime (datas no Python devem ser objetos datetime)
    df.index = pd.to_datetime(df.index, format="%d/%m/%y %H:%M")
    return df


def processar_arquivo(arquivo: os.PathLike) -> bool:
    # Se arquivo existe, conferir com usuário se deseja processá-lo novamente
    if os.path.exists(arquivo):
        check_usuario = input(
            "Arquivo já existe, deseja processar e salvar outra versão? (S/N)"
        )
        if check_usuario.upper() == "S":
            return True
        elif check_usuario.upper() == "N":
            return False
        else:
            raise TypeError("Por favor, insira uma resposta válida (S/N)")
    else:
        return True


def compilar_dados(
    diretorio: os.PathLike,
    base_de_dados: str = "Solcast",
    perdas_transformador_alta_tensao: float = 0.003,
    perdas_linha_alta_tensao: float = 0.01,
    salvar_dados: bool = False,
) -> pd.DataFrame:

    # Verifica se arquivo já existe
    output_dir = os.path.join(os.getcwd(), "data", "output")
    output_arquivo = os.path.join(output_dir, f"dados_compilados_{base_de_dados}.csv")

    # Se arquivo existe, conferir com usuário se deseja processá-lo novamente
    processar = processar_arquivo(arquivo=output_arquivo)

    if processar:
        # Busca arquivos com dados a serem concatenados
        arquivos = listar_arquivos(diretorio, base_de_dados)
        # Lista para armazenar os dataframes obtidos para cada arquivo (séries temporais de energia solar)
        # A lista começa vazia e vai sendo preenchida conforme o loop é executado
        dados_por_ano = []
        for arquivo in arquivos:
            df = ler_e_formatar_dados(
                arquivo,
                skiprows=11,
                sep=";",
                encoding="latin-1",
                index_col=0,
                decimal=",",
            )
            # Cada dataframe tratado é adicionado à lista inicialmente vazia
            dados_por_ano.append(df)

        # Junta todas as tabelas em uma única série temporal
        df_concat = pd.concat(dados_por_ano, axis=0)
        # Aplica as perdas aos dados de energia
        df_concat *= (1 - perdas_transformador_alta_tensao) * (
            1 - perdas_linha_alta_tensao
        )
        # Salva os dados em um arquivo .csv
        if salvar_dados:
            os.makedirs(output_dir, exist_ok=True)
            df_concat.to_csv(output_arquivo)

        return df_concat
    else:
        return pd.read_csv(output_arquivo, index_col=0, parse_dates=True)
