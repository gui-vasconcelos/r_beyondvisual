# -*- coding: utf-8 -*-
# Certifique-se de ter as bibliotecas instaladas:
# pip install pandas numpy scipy
import pandas as pd
import numpy as np
from scipy import stats

# --- 1. CONFIGURAÇÃO E CARREGAMENTO DE DADOS ---
try:
    # Carrega o conjunto de dados.
    # Garanta que o arquivo 'resultados_formatados.csv' esteja na mesma pasta que este script.
    df = pd.read_csv("resultados_formatados.csv")
    print("Arquivo 'resultados_formatados.csv' carregado com sucesso.")
except FileNotFoundError:
    print("ERRO: O arquivo 'resultados_formatados.csv' não foi encontrado.")
    print("Por favor, coloque o arquivo no mesmo diretório do script e tente novamente.")
    # Encerra o script se o arquivo não for encontrado
    exit()

# --- 2. PREPARAÇÃO DA MATRIZ DE CONTAGEM ---
# Esta matriz é a base para a maioria das análises.
# Linhas: Imagens, Colunas: Sons, Valores: contagem de escolhas.
try:
    contingency_matrix = df.pivot_table(
        index='image_id',
        columns='sound_id',
        values='participant',
        aggfunc='count',
        fill_value=0
    )
except KeyError as e:
    print(f"ERRO: A coluna esperada '{e}' não foi encontrada no CSV.")
    print("Verifique se o seu arquivo tem as colunas 'image_id', 'sound_id', e 'participant'.")
    exit()

# Parâmetros básicos do estudo
n_participants = df['participant'].nunique()
n_images = df['image_id'].nunique()
n_sounds = df['sound_id'].nunique()

print("-" * 50)
print(f"Análise baseada em {n_participants} participantes, {n_images} imagens e {n_sounds} sons.")
print("-" * 50)


# --- 3. CÁLCULO DA TABELA 1: Análise por Imagem ---
results_per_image = []

# Distribuição esperada para o Chi-quadrado (distribuição uniforme)
# Se 12 participantes votam, o esperado é que cada um dos 6 sons receba 12/6 = 2 votos.
expected_distribution = [n_participants / n_sounds] * n_sounds

for image_id, row in contingency_matrix.iterrows():
    observed_counts = row.values

    # a) Cálculo do Chi-Quadrado (χ²) e p-value
    # Compara as contagens observadas com as contagens esperadas.
    chi2_stat, p_value = stats.chisquare(f_obs=observed_counts, f_exp=expected_distribution)

    # b) Cálculo da Entropia de Shannon (H) em bits
    # Mede a incerteza/dispersão das escolhas. Menor entropia = maior consenso.
    probabilities = observed_counts / n_participants
    # Filtra probabilidades iguais a zero para evitar erro de log2(0)
    non_zero_probs = probabilities[probabilities > 0]
    entropy = -np.sum(non_zero_probs * np.log2(non_zero_probs))

    # c) Cálculo da Proporção do Voto Máximo (max_prop)
    # Qual a proporção de votos concentrada no som mais popular?
    max_prop = observed_counts.max() / n_participants

    results_per_image.append({
        "Image ID": image_id,
        "Chi-Square (χ²)": chi2_stat,
        "p-value": p_value,
        "Shannon Entropy (bits)": entropy,
        "Max Proportion": max_prop
    })

# Cria um DataFrame com os resultados para visualização clara
analysis_table_1 = pd.DataFrame(results_per_image)


# --- 4. CÁLCULO DA TABELA 2: Correlação Consenso vs. Confiança ---
# Verifica se a coluna de confiança existe antes de prosseguir
if 'confidence' not in df.columns:
    print("\nAVISO: A coluna 'confidence' não foi encontrada no CSV.")
    print("A análise de correlação não pode ser executada.")
    spearman_rho = "Dados de Confiança Ausentes"
    spearman_p = "Dados de Confiança Ausentes"
else:
    # Agrega a confiança média por imagem
    mean_confidence_per_image = df.groupby('image_id')['confidence'].mean()

    # Extrai a medida de consenso da Tabela 1 (usaremos Max Proportion como métrica de consenso)
    consensus_metric = analysis_table_1.set_index('Image ID')['Max Proportion']

    # Garante que os dados de confiança e consenso estão alinhados pelo 'image_id'
    aligned_confidence = mean_confidence_per_image.loc[consensus_metric.index]

    # Calcula a Correlação de Spearman
    spearman_rho, spearman_p = stats.spearmanr(consensus_metric, aligned_confidence)


# --- 5. EXIBIÇÃO DOS RESULTADOS FINAIS ---
print("\n--- RESULTADOS: Análise de Consenso por Imagem ---")
# Formata os números para melhor visualização antes de imprimir
pd.options.display.float_format = '{:.3f}'.format
print(analysis_table_1.to_string())

print("\n\n--- RESULTADOS: Correlação entre Consenso e Confiança ---")
print(f"Métrica de Consenso Utilizada: Proporção Máxima de Votos (Max Proportion)")
if isinstance(spearman_rho, str):
    print(spearman_rho)
else:
    print(f"Correlação de Spearman (ρ) = {spearman_rho:.3f}")
    print(f"p-value = {spearman_p:.3f}")

