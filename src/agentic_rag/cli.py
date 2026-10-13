"""命令列工具：rag ingest / rag ask / rag eval"""

from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from agentic_rag.config import get_settings

app = typer.Typer(help="Agentic RAG 學習專案的命令列工具", no_args_is_help=True)
console = Console()


@app.command()
def ingest(path: Path = typer.Argument(Path("data/handbook"), help="要匯入的資料夾")) -> None:
    """把資料夾內的文件切塊、embedding 後寫入 pgvector。"""
    from agentic_rag.ingestion.pipeline import ingest_directory
    from agentic_rag.llm import OllamaEmbedder
    from agentic_rag.store.pgvector import PgVectorStore

    s = get_settings()
    store = PgVectorStore(s.database_url)
    report = ingest_directory(path, OllamaEmbedder(s), store, s.chunk_size, s.chunk_overlap)
    console.print(
        f"匯入 {report.documents} 份文件、{report.chunks} 個 chunk；索引內共 {store.count()} 個 chunk"
    )


@app.command()
def ask(
    question: str,
    current_only: bool = typer.Option(False, "--current-only", help="只查現行版本的文件"),
) -> None:
    """問一個問題，印出答案與引用來源。"""
    from agentic_rag.rag import build_pipeline

    filters = {"status": "現行"} if current_only else None
    answer = build_pipeline(get_settings()).ask(question, filters)
    console.print(f"\n[bold]答案[/bold]\n{answer.text}\n")
    table = Table("#", "文件", "章節", "分數")
    for i, r in enumerate(answer.sources, start=1):
        table.add_row(
            str(i), f"{r.chunk.title}（{r.chunk.doc_id}）", r.chunk.section_path, f"{r.score:.3f}"
        )
    console.print(table)


@app.command("eval")
def run_eval(
    golden: Path = typer.Option(Path("eval/golden.jsonl"), help="評估集"),
    phase: int = typer.Option(1, help="只跑 phase 小於等於此值的題目"),
    generate: bool = typer.Option(True, help="是否也呼叫 LLM 生成答案（較慢）"),
    name: str = typer.Option("", help="結果檔名，預設為時間戳記"),
) -> None:
    """跑評估集，把逐題結果寫到 eval/results/。"""
    from agentic_rag.evaluation.runner import evaluate, load_golden, save_report
    from agentic_rag.rag import build_pipeline

    s = get_settings()
    items = load_golden(golden, phase)
    report = evaluate(build_pipeline(s), items, s.top_k, generate)
    out = save_report(
        report, Path("eval/results"), name or datetime.now().strftime("%Y%m%d-%H%M%S")
    )

    table = Table("指標", "分數")
    for key, value in report.summary.items():
        table.add_row(key, str(value))
    console.print(table)
    console.print(f"逐題結果：{out}")


if __name__ == "__main__":
    app()
