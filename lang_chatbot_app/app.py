from flask import Flask, render_template, request, jsonify, g
from typing import Literal
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from serpapi import GoogleSearch
import os
import config

# === Flask アプリケーションの設定 ===
app = Flask(__name__)

# === PDFとChromaのセットアップ ===
main_path = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(main_path, "LangChain株式会社IR資料.pdf")
loader = PyPDFLoader(pdf_path)
documents = loader.load()
# OpenAIの埋め込みモデルを設定
embeddings_model = OpenAIEmbeddings(api_key=config.OPENAI_API_KEY,model="text-embedding-3-small")
# Chromaデータベースを作成
db = Chroma.from_documents(documents=documents,embedding=embeddings_model)

# === プロンプトテンプレート ===
template = """
あなたは提供された文書抜粋とツールを活用して質問に答えるアシスタントです。
以下の文書抜粋を参照して質問に答えるか、必要に応じて、"search"ツールを使用してください。

文書抜粋：{document_snippet}

質問：{question}

答え：

"""


# === ツールの設定 ===
@tool
def search(query: str):
    """SerpAPIを使用してウェブ検索を実行します。"""
    params = {
        "q": query,  # 検索クエリ
        "hl": "ja",  # 言語設定（日本語）
        "gl": "jp",  # 地域設定（日本）
        "api_key": config.SERP_API_KEY  # SerpAPIのAPIキー
    }
    
    search = GoogleSearch(params)  # SerpAPIの検索オブジェクトを作成
    result = search.get_dict()  # 検索結果を辞書形式で取得
    
    results_list = result.get("organic_results", [])  # オーガニック検索結果を取得
    search_results = [
        f"{res['title']}: {res['snippet']} - {res['link']}" for res in results_list[:3]
    ]  # 最初の3件の結果をフォーマットしてリストに格納

    g.search_results = search_results if search_results else ["検索結果が見つかりませんでした。"]

    return g.search_results

tools = [search]  # 使用するツールのリスト

tool_node = ToolNode(tools)  # ツールノードを作成

# === モデルのセットアップ ===
model = ChatOpenAI(api_key=config.OPENAI_API_KEY, model_name="gpt-4o-mini").bind_tools(tools)

# === 条件判定 ===
def should_continue(state: MessagesState) -> Literal["tools", END]:
    messages = state['messages']  # 現在のメッセージ状態を取得
    last_message = messages[-1]  # last_messageを取得    
    # LLMがツールを呼び出した場合、"tools"ノードに遷移
    if last_message.tool_calls:
        return "tools"

    if "search" in last_message.content:
        return "tools"

    # それ以外の場合、終了 (END)
    return END


# === モデルの応答生成関数 ===
def call_model(state: MessagesState):
    messages = state['messages']  # 現在のメッセージ状態を取得
    response = model.invoke(messages)  # モデルを呼び出して応答を取得
    return {"messages": [response]}


# === RAG用ロジック ===
def rag_retieve(question:str):
    question_embedding = embeddings_model.embed_query(question)
    docs = db.similarity_search_by_vector(question_embedding,k=3)
    return "\n".join([doc.page_content for doc in docs])

# === メッセージの前処理 ===
def preprocess_message(question:str):
    document_snippet = rag_retieve(question)
    content = template.format(document_snippet=document_snippet,question=question)
    return [HumanMessage(content=content)]

# === ワークフローの構築 ===
workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)  # agentノードを追加
workflow.add_node("tools", tool_node)  # toolsノードを追加
workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,  # 次のノードを判定する関数を指定
)
workflow.add_edge("tools", 'agent')
checkpointer = MemorySaver()
app_flow = workflow.compile(checkpointer=checkpointer)

# === Flaskルート設定 ===
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask",methods=["POST"])
def ask():
    question = request.form["question"]
    inputs = preprocess_message(question)
    # スレッドIDを設定
    thread = {"configurable": {"thread_id": "42"}}

    # ワークフローをストリームモードで実行し、応答を逐次処理
    for event in app_flow.stream({"messages": inputs}, thread, stream_mode="values"):
        response = event["messages"][-1].content  # 応答を整形して出力

    search_results = getattr(g,"search_results",[])
    links = "\n".join(search_results) if search_results else "関連するリンクは見つかりませんでした"

    return jsonify({"answer":response,"links":links})


# === Flaskの起動===
if __name__ == "__main__":
    app.run(debug=True)
