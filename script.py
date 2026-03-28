"""
Projeto ODS 04 - Educação de Qualidade
Solução em Ciência de Dados (Python + Pandas)

Melhorias aplicadas vs versão original:
  - Configurações centralizadas em dataclass (imutável, documentada)
  - pathlib.Path no lugar de os.path (mais pythonico e seguro)
  - logging estruturado no lugar de print statements
  - Leitura defensiva do Excel (sheet_name=None retornava dict, bug corrigido)
  - Detecção de colunas case-insensitive (evita falhas por capitalização)
  - Relatório de qualidade de dados (nulos, outliers, duplicatas)
  - Novos gráficos: heatmap de correlação + boxplots + distribuições
  - Estatísticas descritivas completas no resumo
  - Score de criticidade normalizado (0-100)
  - Exportação de gráficos em alta resolução com layout melhorado
  - Proteção contra divisão por zero e edge cases
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns

# =========================
# LOGGING
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# =========================
# CONFIGURAÇÕES (imutável)
# =========================
@dataclass(frozen=True)
class Config:
    input_excel: Path = Path("dataset_acompanhamento_educacional.xlsx")
    output_dir: Path = Path("outputs")

    # Corte para risco ALTO (Em Risco = Sim)
    nota_corte_alto: float = 6.0
    freq_corte_alto: float = 70.0

    # Corte para risco MÉDIO
    nota_corte_medio: float = 7.0
    freq_corte_medio: float = 80.0

    # Mapeamento de nomes de colunas aceitos (case-insensitive)
    col_aluno: tuple = ("aluno", "nome", "estudante")
    col_serie: tuple = ("série", "serie", "turma", "ano")
    col_nota: tuple = ("média das notas", "media das notas", "nota", "média", "media")
    col_freq: tuple = ("frequência (%)", "frequencia (%)", "frequência", "frequencia", "freq")
    col_part: tuple = ("participação", "participacao", "engajamento")

    # Paleta de cores para gráficos
    palette_risco: dict = field(default_factory=lambda: {
        "Sim": "#E74C3C",
        "Não": "#2ECC71",
        "Indefinido": "#95A5A6",
    })
    palette_nivel: dict = field(default_factory=lambda: {
        "Alto": "#E74C3C",
        "Médio": "#F39C12",
        "Baixo": "#2ECC71",
        "Indefinido": "#95A5A6",
    })


CFG = Config()


# =========================
# DETECÇÃO ROBUSTA DE COLUNAS
# =========================
def detect_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> Optional[str]:
    """
    Retorna o nome real da coluna no DataFrame que coincide
    com algum dos candidatos (comparação case-insensitive, strip).
    """
    lower_map = {c.strip().lower(): c for c in df.columns}
    for candidate in candidates:
        match = lower_map.get(candidate.strip().lower())
        if match is not None:
            return match
    return None


def rename_to_standard(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renomeia colunas detectadas para os nomes canônicos usados no restante do código.
    Lança ValueError se alguma coluna obrigatória não for encontrada.
    """
    mapping_spec = {
        "Aluno":           CFG.col_aluno,
        "Série":           CFG.col_serie,
        "Média das Notas": CFG.col_nota,
        "Frequência (%)":  CFG.col_freq,
        "Participação":    CFG.col_part,
    }

    rename_map: dict[str, str] = {}
    missing: list[str] = []

    for canonical, candidates in mapping_spec.items():
        found = detect_column(df, candidates)
        if found is None:
            missing.append(canonical)
        elif found != canonical:
            rename_map[found] = canonical

    if missing:
        raise ValueError(
            f"Colunas obrigatórias não encontradas: {missing}\n"
            f"Colunas disponíveis no Excel: {list(df.columns)}"
        )

    return df.rename(columns=rename_map)


# =========================
# NORMALIZAÇÃO
# =========================
def normalize_numeric_col(series: pd.Series) -> pd.Series:
    """
    Converte coluna para float, aceitando:
      '68', '68%', '68,5', '68,5%', 6.2, '6,2'
    """
    return (
        series.astype(str)
        .str.strip()
        .str.replace("%", "", regex=False)
        .str.replace(",", ".", regex=False)
        .pipe(pd.to_numeric, errors="coerce")
    )


# =========================
# QUALIDADE DE DADOS
# =========================
def data_quality_report(df: pd.DataFrame) -> str:
    """
    Gera relatório de qualidade: nulos, duplicatas e outliers numéricos.
    """
    lines = ["── RELATÓRIO DE QUALIDADE DOS DADOS ──"]

    # Duplicatas
    dups = df.duplicated(subset=["Aluno"]).sum()
    lines.append(f"Registros duplicados (por 'Aluno'): {dups}")

    # Nulos por coluna
    null_counts = df.isnull().sum()
    if null_counts.any():
        lines.append("Valores nulos por coluna:")
        for col, cnt in null_counts[null_counts > 0].items():
            pct = cnt / len(df) * 100
            lines.append(f"  {col}: {cnt} ({pct:.1f}%)")
    else:
        lines.append("Sem valores nulos ✓")

    # Outliers via IQR para colunas numéricas
    for col in ["Média das Notas", "Frequência (%)"]:
        s = df[col].dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = df[(df[col] < low) | (df[col] > high)][["Aluno", col]]
        if not outliers.empty:
            lines.append(f"Outliers em '{col}' ({len(outliers)} aluno(s)):")
            lines.append(outliers.to_string(index=False))

    return "\n".join(lines)


# =========================
# CLASSIFICAÇÃO DE RISCO
# =========================
def classify_risk_sim_nao(nota: float, freq: float) -> str:
    if pd.isna(nota) or pd.isna(freq):
        return "Indefinido"
    return "Sim" if (nota < CFG.nota_corte_alto or freq < CFG.freq_corte_alto) else "Não"


def classify_risk_level(nota: float, freq: float) -> str:
    if pd.isna(nota) or pd.isna(freq):
        return "Indefinido"
    if nota < CFG.nota_corte_alto or freq < CFG.freq_corte_alto:
        return "Alto"
    if nota < CFG.nota_corte_medio or freq < CFG.freq_corte_medio:
        return "Médio"
    return "Baixo"


def compute_critical_score(nota: float, freq: float) -> float:
    """
    Score de criticidade normalizado de 0 a 100.
    Quanto maior, mais crítico o aluno.
    Peso 60% para nota e 40% para frequência.
    """
    if pd.isna(nota) or pd.isna(freq):
        return np.nan
    nota_max = 10.0
    freq_max = 100.0
    score_nota = max(0.0, (nota_max - nota) / nota_max) * 100
    score_freq = max(0.0, (freq_max - freq) / freq_max) * 100
    return round(0.60 * score_nota + 0.40 * score_freq, 2)


# =========================
# VISUALIZAÇÕES
# =========================
def _save_fig(fig: plt.Figure, path: Path, dpi: int = 200) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("Gráfico salvo: %s", path)


def plot_scatter(df: pd.DataFrame, outpath: Path) -> None:
    """Scatter: Frequência vs Média, colorido por Nível de Risco."""
    fig, ax = plt.subplots(figsize=(9, 6))
    order = ["Alto", "Médio", "Baixo", "Indefinido"]
    palette = CFG.palette_nivel

    for nivel in order:
        sub = df[df["Nível de Risco"] == nivel]
        if sub.empty:
            continue
        ax.scatter(
            sub["Frequência (%)"],
            sub["Média das Notas"],
            label=nivel,
            color=palette.get(nivel, "#7F8C8D"),
            alpha=0.80,
            edgecolors="white",
            linewidths=0.5,
            s=70,
        )

    # Linhas de corte
    ax.axvline(CFG.freq_corte_alto, color="#E74C3C", linestyle="--", linewidth=1,
               label=f"Corte freq. {CFG.freq_corte_alto}%")
    ax.axhline(CFG.nota_corte_alto, color="#E67E22", linestyle="--", linewidth=1,
               label=f"Corte nota {CFG.nota_corte_alto}")

    ax.set_xlabel("Frequência (%)", fontsize=12)
    ax.set_ylabel("Média das Notas", fontsize=12)
    ax.set_title("Frequência vs Média das Notas por Nível de Risco", fontsize=13, fontweight="bold")
    ax.legend(title="Nível de Risco", framealpha=0.9)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.7)
    _save_fig(fig, outpath)


def plot_bar_risco(df: pd.DataFrame, outpath: Path) -> None:
    """Barra horizontal: distribuição Em Risco (Sim/Não) com percentual."""
    counts = df["Em Risco"].value_counts()
    total = counts.sum()
    order = [k for k in ["Sim", "Não", "Indefinido"] if k in counts.index]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(
        order,
        [counts[k] for k in order],
        color=[CFG.palette_risco.get(k, "#95A5A6") for k in order],
        edgecolor="white",
        height=0.5,
    )
    for bar, key in zip(bars, order):
        pct = counts[key] / total * 100
        ax.text(
            bar.get_width() + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f"{counts[key]}  ({pct:.1f}%)",
            va="center", fontsize=11,
            )
    ax.set_xlabel("Quantidade de Alunos", fontsize=11)
    ax.set_title("Alunos em Risco (Sim / Não)", fontsize=13, fontweight="bold")
    ax.set_xlim(0, total * 1.25)
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", linewidth=0.4, alpha=0.7)
    _save_fig(fig, outpath)


def plot_bar_nivel(df: pd.DataFrame, outpath: Path) -> None:
    """Barra: distribuição por Nível de Risco."""
    order = ["Alto", "Médio", "Baixo", "Indefinido"]
    counts = df["Nível de Risco"].value_counts()
    order = [k for k in order if k in counts.index]
    total = counts.sum()

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(
        order,
        [counts[k] for k in order],
        color=[CFG.palette_nivel.get(k, "#95A5A6") for k in order],
        edgecolor="white",
        width=0.5,
    )
    for bar, key in zip(bars, order):
        pct = counts[key] / total * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{counts[key]}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=10,
            )
    ax.set_ylabel("Quantidade de Alunos", fontsize=11)
    ax.set_title("Distribuição por Nível de Risco", fontsize=13, fontweight="bold")
    ax.set_ylim(0, counts.max() * 1.25)
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.7)
    _save_fig(fig, outpath)


def plot_boxplots(df: pd.DataFrame, outpath: Path) -> None:
    """Boxplot de Nota e Frequência por Nível de Risco."""
    nivel_order = [n for n in ["Alto", "Médio", "Baixo"] if n in df["Nível de Risco"].unique()]
    palette = {k: CFG.palette_nivel[k] for k in nivel_order if k in CFG.palette_nivel}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, col in zip(axes, ["Média das Notas", "Frequência (%)"]):
        sns.boxplot(
            data=df[df["Nível de Risco"] != "Indefinido"],
            x="Nível de Risco",
            y=col,
            order=nivel_order,
            palette=palette,
            width=0.5,
            linewidth=1.2,
            ax=ax,
        )
        sns.stripplot(
            data=df[df["Nível de Risco"] != "Indefinido"],
            x="Nível de Risco",
            y=col,
            order=nivel_order,
            color="black",
            alpha=0.35,
            size=3,
            jitter=True,
            ax=ax,
        )
        ax.set_title(f"{col} por Nível de Risco", fontsize=12, fontweight="bold")
        ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.7)

    fig.suptitle("Distribuição das Variáveis Principais por Nível de Risco", fontsize=13)
    _save_fig(fig, outpath)


def plot_correlation_heatmap(df: pd.DataFrame, outpath: Path) -> None:
    """Heatmap de correlação entre variáveis numéricas."""
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if len(num_cols) < 2:
        log.warning("Menos de 2 colunas numéricas; heatmap ignorado.")
        return

    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(max(6, len(num_cols)), max(5, len(num_cols) - 1)))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        linewidths=0.5,
        ax=ax,
        annot_kws={"size": 10},
    )
    ax.set_title("Mapa de Correlação – Variáveis Numéricas", fontsize=13, fontweight="bold")
    _save_fig(fig, outpath)


def plot_serie_risco(df: pd.DataFrame, outpath: Path) -> None:
    """Barras empilhadas: % Em Risco por Série."""
    pivot = (
        df.groupby(["Série", "Em Risco"])
        .size()
        .unstack(fill_value=0)
    )
    # Garante colunas na ordem certa
    for col in ["Sim", "Não", "Indefinido"]:
        if col not in pivot.columns:
            pivot[col] = 0

    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
    pivot_pct = pivot_pct[["Sim", "Não", "Indefinido"]]

    colors = [CFG.palette_risco[c] for c in ["Sim", "Não", "Indefinido"]]
    fig, ax = plt.subplots(figsize=(9, 5))
    pivot_pct.plot(kind="bar", stacked=True, ax=ax, color=colors, edgecolor="white", width=0.6)

    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_ylabel("Proporção de Alunos (%)", fontsize=11)
    ax.set_xlabel("Série", fontsize=11)
    ax.set_title("Proporção Em Risco por Série", fontsize=13, fontweight="bold")
    ax.legend(title="Em Risco", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.xticks(rotation=0)
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.7)
    _save_fig(fig, outpath)


# =========================
# RESUMO TEXTUAL
# =========================
def summarize(df: pd.DataFrame, quality_report: str) -> str:
    total = len(df)
    risco_sim  = (df["Em Risco"] == "Sim").sum()
    risco_nao  = (df["Em Risco"] == "Não").sum()
    indef      = (df["Em Risco"] == "Indefinido").sum()

    desc = df[["Média das Notas", "Frequência (%)", "Score Criticidade"]].describe().round(2)

    por_serie = (
        df.groupby(["Série", "Em Risco"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )

    top_risco = (
        df[df["Em Risco"] == "Sim"]
        .sort_values("Score Criticidade", ascending=False)
        .head(5)[["Aluno", "Série", "Média das Notas", "Frequência (%)", "Participação", "Score Criticidade"]]
    )

    lines = [
        "RESUMO DO PROJETO – ODS 04: Educação de Qualidade",
        "=" * 55,
        f"Total de alunos analisados : {total}",
        f"Em risco (Sim)             : {risco_sim}  ({risco_sim/total*100:.1f}%)",
        f"Fora de risco (Não)        : {risco_nao}  ({risco_nao/total*100:.1f}%)",
        ]
    if indef:
        lines.append(f"Indefinidos (dados faltantes): {indef}")

    lines += [
        "",
        "Critério de classificação:",
        f"  SIM    → Média < {CFG.nota_corte_alto} OU Frequência < {CFG.freq_corte_alto}%",
        f"  Nível Alto  → idem acima",
        f"  Nível Médio → Média < {CFG.nota_corte_medio} OU Frequência < {CFG.freq_corte_medio}%",
        "  Nível Baixo → caso contrário",
        "",
        "Estatísticas descritivas:",
        desc.to_string(),
        "",
        "Distribuição por Série:",
        por_serie.to_string(),
        "",
        "Top 5 alunos em risco (maior Score de Criticidade):",
    ]

    if top_risco.empty:
        lines.append("  (Nenhum aluno classificado como 'Sim')")
    else:
        lines.append(top_risco.to_string(index=False))

    lines += ["", quality_report]

    return "\n".join(lines)


# =========================
# PIPELINE PRINCIPAL
# =========================
def main() -> int:
    CFG.output_dir.mkdir(parents=True, exist_ok=True)
    log.info("Diretório de saída: %s", CFG.output_dir.resolve())

    # ── Leitura do Excel ──────────────────────────────────────────────────────
    if not CFG.input_excel.exists():
        log.error("Arquivo não encontrado: %s", CFG.input_excel.resolve())
        log.error("Ajuste 'input_excel' em Config ou coloque o arquivo na mesma pasta.")
        return 1

    log.info("Lendo arquivo: %s", CFG.input_excel)
    try:
        # sheet_name=0 → lê apenas a primeira aba (sheet_name=None retornaria dict)
        df = pd.read_excel(CFG.input_excel, sheet_name=0)
    except Exception as exc:
        log.error("Falha ao ler Excel: %s", exc)
        return 1

    log.info("Arquivo lido com sucesso. Shape inicial: %s", df.shape)

    # ── Normalização de colunas ───────────────────────────────────────────────
    try:
        df = rename_to_standard(df)
    except ValueError as exc:
        log.error("%s", exc)
        return 1

    df["Média das Notas"] = normalize_numeric_col(df["Média das Notas"])
    df["Frequência (%)"]  = normalize_numeric_col(df["Frequência (%)"])
    df["Participação"]    = df["Participação"].astype(str).str.strip().str.title()
    df["Série"]           = df["Série"].astype(str).str.strip()

    # Remove linhas completamente vazias que às vezes aparecem no Excel
    df.dropna(subset=["Aluno", "Média das Notas", "Frequência (%)"], how="all", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ── Qualidade dos dados ───────────────────────────────────────────────────
    quality = data_quality_report(df)
    log.info("\n%s", quality)

    # ── Classificações ────────────────────────────────────────────────────────
    df["Em Risco"] = df.apply(
        lambda r: classify_risk_sim_nao(r["Média das Notas"], r["Frequência (%)"]), axis=1
    )
    df["Nível de Risco"] = df.apply(
        lambda r: classify_risk_level(r["Média das Notas"], r["Frequência (%)"]), axis=1
    )
    df["Score Criticidade"] = df.apply(
        lambda r: compute_critical_score(r["Média das Notas"], r["Frequência (%)"]), axis=1
    )

    # ── Gráficos ──────────────────────────────────────────────────────────────
    sns.set_theme(style="whitegrid", font_scale=1.05)

    plot_scatter(df,            CFG.output_dir / "scatter_frequencia_vs_notas.png")
    plot_bar_risco(df,          CFG.output_dir / "bar_risco_sim_nao.png")
    plot_bar_nivel(df,          CFG.output_dir / "bar_risco_nivel.png")
    plot_boxplots(df,           CFG.output_dir / "boxplot_nivel_risco.png")
    plot_correlation_heatmap(df,CFG.output_dir / "heatmap_correlacao.png")
    plot_serie_risco(df,        CFG.output_dir / "bar_serie_risco.png")

    # ── Exportação Excel ──────────────────────────────────────────────────────
    out_excel = CFG.output_dir / "resultados_alunos.xlsx"
    try:
        with pd.ExcelWriter(out_excel, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Resultados")

            # Resumo por série
            resumo_serie = (
                df.groupby(["Série", "Em Risco"])
                .size()
                .unstack(fill_value=0)
                .reset_index()
            )
            resumo_serie.to_excel(writer, index=False, sheet_name="Resumo_por_Serie")

            # Top alunos críticos
            top_criticos = (
                df[df["Em Risco"] == "Sim"]
                .sort_values("Score Criticidade", ascending=False)
                [["Aluno", "Série", "Média das Notas", "Frequência (%)", "Participação",
                  "Nível de Risco", "Score Criticidade"]]
            )
            top_criticos.to_excel(writer, index=False, sheet_name="Alunos_Criticos")

        log.info("Excel de resultados salvo: %s", out_excel)
    except Exception as exc:
        log.error("Falha ao salvar Excel: %s", exc)

    # ── Resumo textual ────────────────────────────────────────────────────────
    resumo_txt = summarize(df, quality)
    out_txt = CFG.output_dir / "resumo.txt"
    out_txt.write_text(resumo_txt, encoding="utf-8")
    log.info("Resumo salvo: %s", out_txt)

    # ── Relatório final no terminal ───────────────────────────────────────────
    print("\n" + "=" * 55)
    print("✅  PIPELINE CONCLUÍDO COM SUCESSO")
    print("=" * 55)
    print(resumo_txt)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())