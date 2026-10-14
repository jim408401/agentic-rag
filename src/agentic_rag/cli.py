"""命令列工具：rag ingest / rag ask / rag eval"""

from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from agentic_rag.config import Settings, get_settings

app = typer.Typer(help="Agentic RAG 學習專案的命令列工具", no_args_is_help=True)
console = Console()

MODE_HELP = "檢索方式：vector 或 hybrid，預設依 .env 的 RAG_RETRIEVAL_MODE"
RERANK_HELP = "是否使用 Reranker，預設依 .env 的 RAG_RERANKER_URL 是否有值"
DEFAULT_RERANKER_URL = "http://localhost:8080"


def _settings(mode: str | None, rerank: bool | None) -> Settings:
    """命令列參數覆蓋 .env，方便同一份資料比較不同檢索設定。"""
    s = get_settings()
    update: dict = {}
    if mode is not None:
        if mode not in ("vector", "hybrid"):
            raise typer.BadParameter("只能是 vector 或 hybrid", param_hint="--mode")
        update["retrieval_mode"] = mode
    if rerank is False:
        update["reranker_url"] = ""
    elif rerank and not s.reranker_url:
        update["reranker_url"] = DEFAULT_RERANKER_URL
    return s.model_copy(update=update)


def _pipeline_or_exit(s: Settings):
    """組裝 RAG 管線；Ollama、模型或資料庫有問題時印出原因並結束，而不是卡住。"""
    from agentic_rag.llm import ServiceUnavailable, check_ollama
    from agentic_rag.rag import build_pipeline
    from agentic_rag.retrieval.rerank import check_reranker

    rerank = "＋Reranker" if s.reranker_url else ""
    console.print(f"[dim]檢索方式：{s.retrieval_mode}{rerank}[/dim]")
    try:
        check_ollama(s, [s.embed_model, s.llm_model])
        if s.reranker_url:
            check_reranker(s)
        return build_pipeline(s)
    except ServiceUnavailable as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc


@app.command()
def ingest(path: Path = typer.Argument(Path("data/handbook"), help="要匯入的資料夾")) -> None:
    """把資料夾內的文件切塊、embedding 後寫入 pgvector。"""
    from agentic_rag.ingestion.pipeline import ingest_directory
    from agentic_rag.llm import OllamaEmbedder, ServiceUnavailable, check_ollama
    from agentic_rag.store.pgvector import PgVectorStore

    s = get_settings()
    try:
        console.print("連線資料庫…")
        store = PgVectorStore(s.database_url)
        console.print(f"檢查 Ollama 與模型 {s.embed_model}…")
        check_ollama(s, [s.embed_model])
    except ServiceUnavailable as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    console.print("開始匯入（第一份文件需要載入 embedding 模型，會比較久）")

    def progress(i: int, n: int, doc, chunks: int) -> None:
        console.print(f"  [{i}/{n}] {doc.title}（{doc.doc_id}）：{chunks} 個 chunk")

    report = ingest_directory(
        path, OllamaEmbedder(s), store, s.chunk_size, s.chunk_overlap, on_progress=progress
    )
    console.print(
        f"完成：匯入 {report.documents} 份文件、{report.chunks} 個 chunk；"
        f"索引內共 {store.count()} 個 chunk"
    )


@app.command()
def ask(
    question: str,
    all_versions: bool = typer.Option(
        False, "--all-versions", help="連已廢止的舊版文件也一起查（預設只查現行版）"
    ),
    mode: str | None = typer.Option(None, help=MODE_HELP),
    rerank: bool | None = typer.Option(None, "--rerank/--no-rerank", help=RERANK_HELP),
) -> None:
    """問一個問題，印出答案與引用來源。"""
    from agentic_rag.llm import ServiceUnavailable

    pipeline = _pipeline_or_exit(_settings(mode, rerank))
    try:
        hits = pipeline.retrieve(question, all_versions=all_versions)
        console.print("[dim]檢索完成，模型生成中（第一次需要載入模型，可能較久）…[/dim]")
        console.print("\n[bold]答案[/bold]")
        answer = pipeline.generate(question, hits, on_token=lambda t: print(t, end="", flush=True))
    except ServiceUnavailable as exc:
        console.print(f"\n[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    print("\n")
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
    mode: str | None = typer.Option(None, help=MODE_HELP),
    rerank: bool | None = typer.Option(None, "--rerank/--no-rerank", help=RERANK_HELP),
) -> None:
    """跑評估集，把逐題結果寫到 eval/results/。"""
    from agentic_rag.evaluation.runner import evaluate, load_golden, save_report

    s = _settings(mode, rerank)
    items = load_golden(golden, phase)
    console.print(f"評估 {len(items)} 題，每題完成會印一行…")
    from agentic_rag.llm import ServiceUnavailable

    try:
        report = evaluate(
            _pipeline_or_exit(s),
            items,
            generate,
            on_item=lambda r: console.print(
                f"  {r.id} hit={r.hit} section={r.section_hit} fact={r.fact_recall} {r.latency_s}s"
            ),
        )
    except ServiceUnavailable as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
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
