import json
import os

import pandas as pd
import streamlit as st

from src.config import BASE_DIR  # <-- CORREÇÃO AQUI


class DashboardApp:
    def __init__(self):
        self.results_path = os.path.join(BASE_DIR, "results", "experiment_results.json")
        st.set_page_config(layout="wide", page_title="Análise de Pruning - LibrasNow")

    # ... (o resto do arquivo continua exatamente igual) ...
    def _render_header(self):
        st.title("Análise Acadêmica de Técnicas de Pruning para Tradução de Libras")
        st.markdown(
            """
        Este painel apresenta uma análise comparativa de diferentes técnicas de compressão de modelos 
        de redes neurais, aplicadas a um Transformer para reconhecimento da Língua Brasileira de Sinais (Libras). 
        O objetivo deste trabalho é investigar o trade-off entre **tamanho do modelo**, **latência** e **acurácia**.
        """
        )

    def _render_theory(self):
        st.header("1. Fundamentos Teóricos de Pruning")
        st.markdown(
            """
        O pruning (poda) é uma família de técnicas para reduzir a complexidade de uma rede neural removendo 
        parâmetros "desnecessários". Isso resulta em modelos menores, mais rápidos e com menor consumo de 
        energia, sendo crucial para o deployment em dispositivos com recursos limitados (edge).
        """
        )

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Magnitude Pruning (Não Estruturado)")
            st.markdown(
                """
            - **Descrição:** Remove pesos individuais com base em sua magnitude absoluta. A intuição é que pesos com valores próximos de zero contribuem menos para a saída da rede.
            - **Referência:** Han et al., *Deep Compression* (2015).
            """
            )
            st.latex(
                r"""W' = M \odot W, \quad \text{onde} \quad M_{ij} = \begin{cases} 1 & \text{se } |W_{ij}| \ge \tau \\ 0 & \text{caso contrário} \end{cases}"""
            )

        with col2:
            st.subheader("LN-Structured Pruning (Estruturado)")
            st.markdown(
                """
            - **Descrição:** Remove estruturas inteiras, como neurônios ou filtros, com base em sua norma $L_n$. Isso resulta em matrizes de peso densas e menores, acelerando a inferência.
            - **Referência:** Li et al., *Pruning Filters for Efficient ConvNets* (2017).
            """
            )
            st.latex(
                r"""\text{Importância}(\text{neurônio}_k) = \left( \sum_{j} |W_{kj}|^n \right)^{1/n}"""
            )

    def _render_results(self):
        st.header("2. Análise Comparativa dos Resultados")
        try:
            with open(self.results_path, "r") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            st.error(
                f"Arquivo de resultados não encontrado ou inválido em '{self.results_path}'. Execute 'pipeline.py --action run-experiments' primeiro."
            )
            return

        processed_data = []
        for name, values in data.items():
            parts = name.split("_")
            pruning_type = " ".join(parts[:-1]).replace("Pruning", "")
            pruning_level = int(parts[-1].replace("p", ""))
            processed_data.append(
                {
                    "Estratégia": pruning_type,
                    "Nível de Pruning (%)": pruning_level,
                    "Acurácia": values["accuracy"],
                    "Tamanho (MB)": values["size_mb"],
                }
            )

        if not processed_data:
            st.warning("Nenhum resultado de experimento encontrado no arquivo.")
            return

        df = pd.DataFrame(processed_data)

        st.subheader("Tabela de Resultados")
        st.dataframe(df.style.format({"Acurácia": "{:.2%}", "Tamanho (MB)": "{:.2f}"}))

        st.subheader("Gráfico: Acurácia vs. Tamanho do Modelo")
        chart = st.scatter_chart(
            df,
            x="Tamanho (MB)",
            y="Acurácia",
            color="Estratégia",
            size="Nível de Pruning (%)",
            height=500,
        )
        st.info(
            "Passe o mouse sobre os pontos para ver os detalhes. O tamanho do ponto indica o nível de agressividade da poda."
        )

    def run(self):
        self._render_header()
        self._render_theory()
        self._render_results()
        st.sidebar.info("Projeto de TCC\n\nSamuel Mauli")
