import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os

# 環境変数の読み込み
load_dotenv()

# OpenAI APIキーの設定
openai_api_key = os.getenv("OPENAI_API_KEY")

# 専門家の種類とシステムメッセージの定義
EXPERT_TYPES = {
    "医療専門家": "あなたは経験豊富な医療専門家です。医学的な知識に基づいて、正確で分かりやすい回答を提供してください。",
    "法律専門家": "あなたは法律の専門家です。法的な観点から正確で詳細な情報を提供してください。",
    "金融専門家": "あなたは金融・投資の専門家です。金融市場や投資戦略について専門的なアドバイスを提供してください。",
    "ITエンジニア": "あなたは経験豊富なITエンジニアです。技術的な問題に対して実践的な解決策を提供してください。",
    "マーケティング専門家": "あなたはマーケティングの専門家です。効果的なマーケティング戦略やビジネス成長のためのアドバイスを提供してください。"
}


def get_llm_response(user_input: str, expert_type: str) -> tuple:
    """
    LLMからの回答を取得する関数
    
    Args:
        user_input (str): ユーザーからの入力テキスト
        expert_type (str): 選択された専門家の種類
    
    Returns:
        tuple: (回答テキスト, 入力トークン数, 出力トークン数, 合計トークン数)
    """
    try:
        # ChatOpenAIインスタンスの作成
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            openai_api_key=openai_api_key
        )
        
        # システムメッセージとユーザーメッセージの作成
        messages = [
            SystemMessage(content=EXPERT_TYPES[expert_type]),
            HumanMessage(content=user_input)
        ]
        
        # LLMに問い合わせ
        response = llm.invoke(messages)
        
        # トークン数を取得
        token_usage = response.response_metadata.get('token_usage', {})
        prompt_tokens = token_usage.get('prompt_tokens', 0)
        completion_tokens = token_usage.get('completion_tokens', 0)
        total_tokens = token_usage.get('total_tokens', 0)
        
        return response.content, prompt_tokens, completion_tokens, total_tokens
    
    except Exception as e:
        return f"エラーが発生しました: {str(e)}", 0, 0, 0


# Streamlitアプリケーションのメイン部分
def main():
    # ページ設定
    st.set_page_config(
        page_title="AI専門家相談アプリ",
        page_icon="🤖",
        layout="wide"
    )
    
    # タイトル
    st.title("🤖 AI専門家相談アプリ")
    
    # アプリの概要説明
    st.markdown("""
    ### 📖 アプリの概要
    このアプリは、様々な分野の専門家としてAIに相談できるサービスです。
    質問したい分野の専門家を選択して、質問を入力してください。
    
    ### 🔧 使い方
    1. ラジオボタンから相談したい専門家の種類を選択してください
    2. テキストボックスに質問や相談内容を入力してください
    3. 「送信」ボタンをクリックすると、AIが専門家として回答します
    
    ---
    """)
    
    # サイドバーに専門家選択
    st.sidebar.header("専門家を選択")
    expert_type = st.sidebar.radio(
        "相談したい専門家の種類:",
        list(EXPERT_TYPES.keys())
    )
    
    # 選択された専門家の説明を表示
    st.sidebar.info(f"**選択中:** {expert_type}")
    st.sidebar.write(EXPERT_TYPES[expert_type])
    
    # メインエリアに入力フォーム
    st.subheader(f"💬 {expert_type}への相談")
    
    # 入力フォーム
    user_input = st.text_area(
        "質問や相談内容を入力してください:",
        height=150,
        placeholder="ここに質問を入力してください..."
    )
    
    # 送信ボタン
    if st.button("送信", type="primary"):
        if not user_input.strip():
            st.warning("⚠️ 質問を入力してください。")
        elif not openai_api_key:
            st.error("❌ OpenAI APIキーが設定されていません。.envファイルを確認してください。")
        else:
            # ローディング表示
            with st.spinner("回答を生成中..."):
                # LLMからの回答を取得
                response, prompt_tokens, completion_tokens, total_tokens = get_llm_response(user_input, expert_type)
            
            # 回答を表示
            st.success("✅ 回答が生成されました!")
            st.markdown("### 📝 回答:")
            st.markdown(response)
            
            # トークン数を表示
            st.markdown("### 📊 トークン使用量:")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("入力トークン", f"{prompt_tokens:,}")
            with col2:
                st.metric("出力トークン", f"{completion_tokens:,}")
            with col3:
                st.metric("合計トークン", f"{total_tokens:,}")
            
            # 追加情報
            st.info(f"💡 この回答は{expert_type}の視点から提供されています。")


if __name__ == "__main__":
    main()
